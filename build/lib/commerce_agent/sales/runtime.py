"""Four-node read-only Sales subgraph. Dependencies are trusted process resources."""
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
from commerce_agent.sales.contracts import SalesDiagnosticResult
from commerce_agent.sales.evidence import collect_sales_evidence
from commerce_agent.sales.policy import SalesRequest, resolve_sales_request
from commerce_agent.sales.prompts import build_sales_instructions
from commerce_agent.sales.publication import validate_sales_publication
from commerce_agent.sales.report import render_sales_report


class SalesModel(Protocol):
    async def analyze(self, *, instructions: str, prepared: dict) -> SalesDiagnosticResult: ...


@dataclass(frozen=True)
class SalesDependencies:
    client_factory: Callable = field(repr=False)
    model: SalesModel = field(repr=False)
    workflow_version: Literal["v1", "v2_prompt", "v2_optimized"] = "v1"
    model_profile: str = "default"
    collection_timeout: float = 90
    model_timeout: float = 60

    def __post_init__(self):
        if self.workflow_version not in {"v1", "v2_prompt", "v2_optimized"}:
            raise ValueError("invalid Sales workflow profile")
        if not 0 < self.collection_timeout <= 300 or not 0 < self.model_timeout <= 180:
            raise ValueError("invalid Sales deadline")


class SalesInput(TypedDict, total=False):
    messages: list[Any]
    query_plan: QueryPlan | None
    agent_statuses: dict[str, AgentStatus]
    sales_analysis: DomainAnalysis | None


class SalesOutput(TypedDict, total=False):
    agent_statuses: dict[str, AgentStatus]
    sales_analysis: DomainAnalysis | None
    evidence_matrix: list[EvidenceItem]
    data_gaps: list[DataGap]
    errors: list[WorkflowError]


class SalesState(SalesInput, SalesOutput, total=False):
    sales_request: dict | None
    prepared_sales: dict | None
    sales_candidate: dict | None
    sales_failure: str | None
    sales_skipped: bool
    sales_base_attempt: int
    sales_revision: int


def _status(state, node, stage, status="running", detail=None):
    return {"agent_statuses": {"sales": AgentStatus(domain=DomainName.SALES, status=status,
        attempt=state.get("sales_base_attempt", 0) + stage, node=node, detail=detail)}}


def build_sales_subgraph(dependencies: SalesDependencies, *, checkpointer=None, interrupt_after=None):
    def prepare(state: SalesState, runtime: Runtime[CommerceRuntimeContext]):
        prior = state.get("agent_statuses", {}).get("sales")
        prior = AgentStatus.model_validate(prior) if prior else None
        previous = state.get("sales_analysis")
        previous = DomainAnalysis.model_validate(previous) if previous else None
        base = {"sales_base_attempt": prior.attempt if prior else 0,
                "sales_revision": previous.revision + 1 if previous else 1,
                "sales_request": None, "prepared_sales": None, "sales_candidate": None,
                "sales_failure": None, "sales_skipped": False}
        plan = state.get("query_plan")
        plan = QueryPlan.model_validate(plan) if plan else QueryPlan()
        if DomainName.SALES not in plan.required_domains:
            return {**base, **_status(base, "prepare", 1, "skipped", "domain_not_required"), "sales_skipped": True}
        try:
            if runtime.context.model_profile != dependencies.model_profile:
                raise ValueError("Sales model profile mismatch")
            messages = state.get("messages", [])
            query = next(message.content for message in reversed(messages) if message.type == "human")
            request = resolve_sales_request(query, plan.constraints.get("sales"))
            return {**base, **_status(base, "prepare", 1), "sales_request": request.model_dump(mode="json")}
        except (ValueError, TypeError, StopIteration, AttributeError):
            return {**base, **_status(base, "prepare", 1, "failed", "invalid_request"), "sales_failure": "invalid_request"}

    async def collect_evidence(state: SalesState, runtime: Runtime[CommerceRuntimeContext]):
        if state.get("sales_skipped") or state.get("sales_failure"):
            return {}
        try:
            async with asyncio.timeout(dependencies.collection_timeout):
                async with dependencies.client_factory() as client:
                    prepared = await collect_sales_evidence(client, runtime.context,
                        SalesRequest.model_validate(state["sales_request"]))
            return {**_status(state, "collect_evidence", 2), "prepared_sales": prepared}
        except Exception:
            # Provider/transport exception bodies may contain secrets. Static
            # codes only; cancellation is BaseException and propagates.
            return {**_status(state, "collect_evidence", 2, "failed", "needs_data"), "sales_failure": "needs_data"}

    async def analyze(state: SalesState):
        if state.get("sales_skipped") or state.get("sales_failure"):
            return {}
        prepared = state["prepared_sales"]
        try:
            instructions = build_sales_instructions(workflow_version=dependencies.workflow_version,
                user_query=prepared["request"]["user_query"], today=prepared["today"])
            async with asyncio.timeout(dependencies.model_timeout):
                candidate = await dependencies.model.analyze(instructions=instructions, prepared=deepcopy(prepared))
            validated = validate_sales_publication(candidate, prepared)
            return {**_status(state, "analyze", 3), "sales_candidate": validated.model_dump(mode="json")}
        except Exception:
            return {**_status(state, "analyze", 3, "failed", "analysis_rejected"), "sales_failure": "analysis_rejected"}

    def validate_and_publish(state: SalesState, runtime: Runtime[CommerceRuntimeContext]):
        if state.get("sales_skipped"):
            return {}
        failure = state.get("sales_failure")
        prepared = state.get("prepared_sales")
        if not failure:
            try:
                result = validate_sales_publication(state["sales_candidate"], prepared)
                report = render_sales_report(result, prepared)
            except Exception:
                failure = "analysis_rejected"
        identity = f"sales:{runtime.context.request_id}"
        if failure:
            return {**_status(state, "validate_and_publish", 4, "failed", failure),
                "sales_candidate": None, "sales_failure": failure,
                "sales_analysis": DomainAnalysis(domain=DomainName.SALES, revision=state["sales_revision"],
                    status="failed", summary="Анализ продаж недоступен: недостаточно проверенных данных или результат не прошёл проверку."),
                "data_gaps": [DataGap(gap_id=f"{identity}:{failure}", domain=DomainName.SALES, reason=failure)],
                "errors": [WorkflowError(error_id=f"{identity}:{failure}", code=f"SALES_{failure.upper()}",
                    message="Sales read-only branch did not produce a validated result.", node="sales.validate_and_publish")]}
        refs = tuple(dict.fromkeys(item["evidence_ref"] for item in prepared["evidence"]))
        warnings = prepared["warnings"]
        gaps = [DataGap(gap_id=f"{identity}:limitation:{index}", domain=DomainName.SALES,
                       reason=warning, blocking=warning.startswith("needs_data:"),
                       status="open" if warning.startswith("needs_data:") else "accepted_limitation")
                for index, warning in enumerate(warnings)]
        gaps.extend(DataGap(gap_id=f"{identity}:model-gap:{index}", domain=DomainName.SALES, reason=gap.reason)
                    for index, gap in enumerate(result.data_gaps))
        return {**_status(state, "validate_and_publish", 4, "completed"),
            "sales_analysis": DomainAnalysis(domain=DomainName.SALES, revision=state["sales_revision"],
                status="partial" if warnings or result.data_gaps else "completed", summary=report,
                result=result, evidence_refs=refs),
            "evidence_matrix": [EvidenceItem.model_validate(item) for item in prepared["evidence"]],
            "data_gaps": gaps, "errors": []}

    builder = StateGraph(SalesState, input_schema=SalesInput, output_schema=SalesOutput,
                         context_schema=CommerceRuntimeContext)
    for name, node in (("prepare", prepare), ("collect_evidence", collect_evidence),
                       ("analyze", analyze), ("validate_and_publish", validate_and_publish)):
        builder.add_node(name, node)
    for left, right in ((START, "prepare"), ("prepare", "collect_evidence"),
                        ("collect_evidence", "analyze"), ("analyze", "validate_and_publish"),
                        ("validate_and_publish", END)):
        builder.add_edge(left, right)
    return builder.compile(name="sales_subgraph", checkpointer=checkpointer, interrupt_after=interrupt_after)
