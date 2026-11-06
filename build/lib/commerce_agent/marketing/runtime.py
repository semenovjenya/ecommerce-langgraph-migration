"""Four-node read-only Marketing subgraph. Dependencies are trusted process resources."""
from __future__ import annotations

import asyncio
from copy import deepcopy
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from langgraph.graph import START, END, StateGraph
from langgraph.runtime import Runtime
from typing_extensions import TypedDict

from commerce_agent.context import CommerceRuntimeContext
from commerce_agent.models import AgentStatus, DataGap, DomainAnalysis, DomainName, EvidenceItem, QueryPlan, WorkflowError
from commerce_agent.marketing.contracts import MarketingDiagnosticResult
from commerce_agent.marketing.evidence import collect_marketing_evidence
from commerce_agent.marketing.policy import MarketingRequest, resolve_marketing_request
from commerce_agent.marketing.prompts import build_marketing_instructions
from commerce_agent.marketing.publication import validate_marketing_publication
from commerce_agent.marketing.report import render_marketing_report


class MarketingModel(Protocol):
    async def analyze(self, *, instructions: str, prepared: dict) -> MarketingDiagnosticResult: ...


@dataclass(frozen=True)
class MarketingDependencies:
    client_factory: Callable = field(repr=False)
    model: MarketingModel = field(repr=False)
    workflow_version: Literal["v1", "v2_prompt", "v2_optimized"] = "v1"
    model_profile: str = "default"
    collection_timeout: float = 90
    model_timeout: float = 60

    def __post_init__(self):
        if self.workflow_version not in {"v1", "v2_prompt", "v2_optimized"}:
            raise ValueError("invalid Marketing workflow profile")
        if not 0 < self.collection_timeout <= 300 or not 0 < self.model_timeout <= 180:
            raise ValueError("invalid Marketing deadline")


class MarketingInput(TypedDict, total=False):
    messages: list[Any]
    query_plan: QueryPlan | None
    agent_statuses: dict[str, AgentStatus]
    marketing_analysis: DomainAnalysis | None


class MarketingOutput(TypedDict, total=False):
    agent_statuses: dict[str, AgentStatus]
    marketing_analysis: DomainAnalysis | None
    evidence_matrix: list[EvidenceItem]
    data_gaps: list[DataGap]
    errors: list[WorkflowError]


class MarketingState(MarketingInput, MarketingOutput, total=False):
    marketing_request: dict | None
    prepared_marketing: dict | None
    marketing_candidate: dict | None
    marketing_failure: str | None
    marketing_skipped: bool
    marketing_base_attempt: int
    marketing_revision: int


def _status(state, node, stage, status="running", detail=None):
    return {"agent_statuses": {"marketing": AgentStatus(domain=DomainName.MARKETING, status=status,
        attempt=state.get("marketing_base_attempt", 0) + stage, node=node, detail=detail)}}


def build_marketing_subgraph(dependencies: MarketingDependencies, *, checkpointer=None, interrupt_after=None):
    def prepare(state: MarketingState, runtime: Runtime[CommerceRuntimeContext]):
        prior = state.get("agent_statuses", {}).get("marketing")
        prior = AgentStatus.model_validate(prior) if prior else None
        previous = state.get("marketing_analysis")
        previous = DomainAnalysis.model_validate(previous) if previous else None
        base = {"marketing_base_attempt": prior.attempt if prior else 0,
                "marketing_revision": previous.revision + 1 if previous else 1,
                "marketing_request": None, "prepared_marketing": None, "marketing_candidate": None,
                "marketing_failure": None, "marketing_skipped": False}
        plan = state.get("query_plan")
        plan = QueryPlan.model_validate(plan) if plan else QueryPlan()
        if DomainName.MARKETING not in plan.required_domains:
            return {**base, **_status(base, "prepare", 1, "skipped", "domain_not_required"), "marketing_skipped": True}
        try:
            if runtime.context.model_profile != dependencies.model_profile:
                raise ValueError("Marketing model profile mismatch")
            messages = state.get("messages", [])
            query = next(message.content for message in reversed(messages) if message.type == "human")
            request = resolve_marketing_request(query, plan.constraints.get("marketing"))
            return {**base, **_status(base, "prepare", 1), "marketing_request": request.model_dump(mode="json")}
        except (ValueError, TypeError, StopIteration, AttributeError):
            return {**base, **_status(base, "prepare", 1, "failed", "invalid_request"), "marketing_failure": "invalid_request"}

    async def collect_evidence(state: MarketingState, runtime: Runtime[CommerceRuntimeContext]):
        if state.get("marketing_skipped") or state.get("marketing_failure"):
            return {}
        try:
            async with asyncio.timeout(dependencies.collection_timeout):
                async with dependencies.client_factory() as client:
                    prepared = await collect_marketing_evidence(client, runtime.context,
                        MarketingRequest.model_validate(state["marketing_request"]))
            return {**_status(state, "collect_evidence", 2), "prepared_marketing": prepared}
        except Exception:
            # Provider/transport exception bodies may contain secrets. Static
            # codes only; cancellation is BaseException and propagates.
            return {**_status(state, "collect_evidence", 2, "failed", "needs_data"), "marketing_failure": "needs_data"}

    async def analyze(state: MarketingState):
        if state.get("marketing_skipped") or state.get("marketing_failure"):
            return {}
        prepared = state["prepared_marketing"]
        try:
            instructions = build_marketing_instructions(workflow_version=dependencies.workflow_version,
                user_query=prepared["request"]["user_query"], today=prepared["today"])
            async with asyncio.timeout(dependencies.model_timeout):
                candidate = await dependencies.model.analyze(instructions=instructions, prepared=deepcopy(prepared))
            validated = validate_marketing_publication(candidate, prepared)
            return {**_status(state, "analyze", 3), "marketing_candidate": validated.model_dump(mode="json")}
        except Exception:
            return {**_status(state, "analyze", 3, "failed", "analysis_rejected"), "marketing_failure": "analysis_rejected"}

    def validate_and_publish(state: MarketingState, runtime: Runtime[CommerceRuntimeContext]):
        if state.get("marketing_skipped"):
            return {}
        failure = state.get("marketing_failure")
        prepared = state.get("prepared_marketing")
        if not failure:
            try:
                result = validate_marketing_publication(state["marketing_candidate"], prepared)
                report = render_marketing_report(result, prepared)
            except Exception:
                failure = "analysis_rejected"
        identity = f"marketing:{runtime.context.request_id}"
        if failure:
            return {**_status(state, "validate_and_publish", 4, "failed", failure),
                "marketing_candidate": None, "marketing_failure": failure,
                "marketing_analysis": DomainAnalysis(domain=DomainName.MARKETING, revision=state["marketing_revision"],
                    status="failed", summary="Анализ рекламы недоступен: недостаточно проверенных данных или результат не прошёл проверку."),
                "data_gaps": [DataGap(gap_id=f"{identity}:{failure}", domain=DomainName.MARKETING, reason=failure)],
                "errors": [WorkflowError(error_id=f"{identity}:{failure}", code=f"MARKETING_{failure.upper()}",
                    message="Marketing read-only branch did not produce a validated result.", node="marketing.validate_and_publish")]}
        refs = tuple(dict.fromkeys(item["evidence_ref"] for item in prepared["evidence"]))
        warnings = prepared["warnings"]
        gaps = [DataGap(gap_id=f"{identity}:limitation:{index}", domain=DomainName.MARKETING,
                       reason=warning, blocking=warning.startswith("needs_data:"),
                       status="open" if warning.startswith("needs_data:") else "accepted_limitation")
                for index, warning in enumerate(warnings)]
        gaps.extend(DataGap(gap_id=f"{identity}:model-gap:{index}", domain=DomainName.MARKETING, reason=gap.reason)
                    for index, gap in enumerate(result.data_gaps))
        return {**_status(state, "validate_and_publish", 4, "completed"),
            "marketing_analysis": DomainAnalysis(domain=DomainName.MARKETING, revision=state["marketing_revision"],
                status="partial" if warnings or result.data_gaps else "completed", summary=report,
                result=result, evidence_refs=refs),
            "evidence_matrix": [EvidenceItem.model_validate(item) for item in prepared["evidence"]],
            "data_gaps": gaps, "errors": []}

    builder = StateGraph(MarketingState, input_schema=MarketingInput, output_schema=MarketingOutput,
                         context_schema=CommerceRuntimeContext)
    for name, node in (("prepare", prepare), ("collect_evidence", collect_evidence),
                       ("analyze", analyze), ("validate_and_publish", validate_and_publish)):
        builder.add_node(name, node)
    for left, right in ((START, "prepare"), ("prepare", "collect_evidence"),
                        ("collect_evidence", "analyze"), ("analyze", "validate_and_publish"),
                        ("validate_and_publish", END)):
        builder.add_edge(left, right)
    return builder.compile(name="marketing_subgraph", checkpointer=checkpointer, interrupt_after=interrupt_after)
