"""Four-node read-only Finance subgraph. Dependencies are trusted process resources."""
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
from commerce_agent.conversation_context import memory_update, memory_block
from commerce_agent.conversation_guard import guard_conversation
from commerce_agent.models import AgentStatus, DataGap, DomainAnalysis, DomainName, EvidenceItem, QueryPlan, WorkflowError
from commerce_agent.finance.contracts import AgentDiagnosticResult
from commerce_agent.finance.evidence import collect_finance_evidence
from commerce_agent.finance.policy import FinanceRequest, resolve_finance_request
from commerce_agent.finance.prompts import build_finance_instructions
from commerce_agent.finance.publication import validate_finance_publication
from commerce_agent.finance.report import render_finance_report


class FinanceModel(Protocol):
    async def analyze(self, *, instructions: str, prepared: dict) -> AgentDiagnosticResult: ...


@dataclass(frozen=True)
class FinanceDependencies:
    client_factory: Callable = field(repr=False)
    model: FinanceModel = field(repr=False)
    workflow_version: Literal["v1", "v2_prompt", "v2_optimized"] = "v1"
    model_profile: str = "default"
    collection_timeout: float = 90
    model_timeout: float = 60

    enable_conversation_memory: bool = False

    def __post_init__(self):
        if type(self.enable_conversation_memory) is not bool:
            raise ValueError("conversation capability must be bool")
        if self.workflow_version not in {"v1", "v2_prompt", "v2_optimized"}:
            raise ValueError("invalid Finance workflow profile")
        if not 0 < self.collection_timeout <= 300 or not 0 < self.model_timeout <= 180:
            raise ValueError("invalid Finance deadline")


class FinanceInput(TypedDict, total=False):
    messages: list[Any]
    query_plan: QueryPlan | None
    agent_statuses: dict[str, AgentStatus]
    finance_analysis: DomainAnalysis | None


class FinanceOutput(TypedDict, total=False):
    agent_statuses: dict[str, AgentStatus]
    finance_analysis: DomainAnalysis | None
    evidence_matrix: list[EvidenceItem]
    data_gaps: list[DataGap]
    errors: list[WorkflowError]


class FinanceState(FinanceInput, FinanceOutput, total=False):
    finance_conversation: dict | None
    finance_conversation_version: int | None
    finance_request: dict | None
    prepared_finance: dict | None
    finance_candidate: dict | None
    finance_failure: str | None
    finance_skipped: bool
    finance_base_attempt: int
    finance_revision: int


def _status(state, node, stage, status="running", detail=None):
    return {"agent_statuses": {"finance": AgentStatus(domain=DomainName.FINANCE, status=status,
        attempt=state.get("finance_base_attempt", 0) + stage, node=node, detail=detail)}}


def build_finance_subgraph(dependencies: FinanceDependencies, *, checkpointer=None, interrupt_after=None):
    def prepare(state: FinanceState, runtime: Runtime[CommerceRuntimeContext]):
        prior = state.get("agent_statuses", {}).get("finance")
        prior = AgentStatus.model_validate(prior) if prior else None
        previous = state.get("finance_analysis")
        previous = DomainAnalysis.model_validate(previous) if previous else None
        base = {"finance_base_attempt": prior.attempt if prior else 0,
                "finance_revision": previous.revision + 1 if previous else 1,
                "finance_request": None, "prepared_finance": None, "finance_candidate": None,
                "finance_failure": None, "finance_skipped": False}
        if dependencies.enable_conversation_memory:
            base.update(finance_conversation=None, finance_conversation_version=None)
        plan = state.get("query_plan")
        plan = QueryPlan.model_validate(plan) if plan else QueryPlan()
        if DomainName.FINANCE not in plan.required_domains:
            return {**base, **_status(base, "prepare", 1, "skipped", "domain_not_required"), "finance_skipped": True}
        try:
            if runtime.context.model_profile != dependencies.model_profile:
                raise ValueError("Finance model profile mismatch")
            messages = state.get("messages", [])
            query = next(message.content for message in reversed(messages) if message.type == "human")
            constraints = dict(plan.constraints.get("finance") or {})
            request = resolve_finance_request(query, constraints or None)
            base.update(memory_update(dependencies.enable_conversation_memory,state,runtime,"finance",dependencies.workflow_version))
            return {**base, **_status(base, "prepare", 1), "finance_request": request.model_dump(mode="json")}
        except (ValueError, TypeError, StopIteration, AttributeError):
            return {**base, **_status(base, "prepare", 1, "failed", "invalid_request"), "finance_failure": "invalid_request"}

    async def collect_evidence(state: FinanceState, runtime: Runtime[CommerceRuntimeContext]):
        if state.get("finance_skipped") or state.get("finance_failure"):
            return {}
        try:
            async with asyncio.timeout(dependencies.collection_timeout):
                async with dependencies.client_factory() as client:
                    prepared = await collect_finance_evidence(client, runtime.context,
                        FinanceRequest.model_validate(state["finance_request"]))
            if not set(prepared["required_operations"]) <= {c["operation"] for c in prepared["claims"]}:
                raise ValueError("Finance requested analysis has no verified data")
            return {**_status(state, "collect_evidence", 2), "prepared_finance": prepared}
        except Exception:
            # Provider/transport exception bodies may contain secrets. Static
            # codes only; cancellation is BaseException and propagates.
            return {**_status(state, "collect_evidence", 2, "failed", "needs_data"), "finance_failure": "needs_data"}

    async def analyze(state: FinanceState, runtime: Runtime[CommerceRuntimeContext]):
        if state.get("finance_skipped") or state.get("finance_failure"):
            return {}
        prepared = state["prepared_finance"]
        try:
            instructions = build_finance_instructions(workflow_version=dependencies.workflow_version,
                user_query=prepared["request"]["user_query"], today=prepared["today"],
                memory_block=memory_block(state,runtime,"finance",dependencies.workflow_version))
            async with asyncio.timeout(dependencies.model_timeout):
                candidate = await dependencies.model.analyze(instructions=instructions, prepared=deepcopy(prepared))
            validated = validate_finance_publication(candidate, prepared)
            return {**_status(state, "analyze", 3), "finance_candidate": validated.model_dump(mode="json")}
        except Exception:
            return {**_status(state, "analyze", 3, "failed", "analysis_rejected"), "finance_failure": "analysis_rejected"}

    def validate_and_publish(state: FinanceState, runtime: Runtime[CommerceRuntimeContext]):
        if state.get("finance_skipped"):
            return {}
        failure = state.get("finance_failure")
        prepared = state.get("prepared_finance")
        if not failure:
            try:
                result = validate_finance_publication(state["finance_candidate"], prepared)
                report = render_finance_report(result, prepared)
            except Exception:
                failure = "analysis_rejected"
        identity = f"finance:{runtime.context.request_id}"
        if failure:
            return {**_status(state, "validate_and_publish", 4, "failed", failure),
                "finance_candidate": None, "finance_failure": failure,
                "finance_analysis": DomainAnalysis(domain=DomainName.FINANCE, revision=state["finance_revision"],
                    status="failed", summary="Финансовый анализ недоступен: недостаточно проверенных данных или результат не прошёл проверку."),
                "data_gaps": [DataGap(gap_id=f"{identity}:{failure}", domain=DomainName.FINANCE, reason=failure)],
                "errors": [WorkflowError(error_id=f"{identity}:{failure}", code=f"FINANCE_{failure.upper()}",
                    message="Finance read-only branch did not produce a validated result.", node="finance.validate_and_publish")]}
        refs = tuple(dict.fromkeys(item["evidence_ref"] for item in prepared["evidence"]))
        warnings = prepared["warnings"]
        gaps = [DataGap(gap_id=f"{identity}:limitation:{index}", domain=DomainName.FINANCE,
                       reason=warning, blocking=warning.startswith("needs_data:"),
                       status="open" if warning.startswith("needs_data:") else "accepted_limitation")
                for index, warning in enumerate(warnings)]
        # Publication only permits the exact native warnings. Do not create a
        # second blocking copy of accepted simulation/provenance limitations.
        return {**_status(state, "validate_and_publish", 4, "completed"),
            "finance_analysis": DomainAnalysis(domain=DomainName.FINANCE, revision=state["finance_revision"],
                status="partial" if warnings or result.data_gaps else "completed", summary=report,
                result=result, evidence_refs=refs),
            "evidence_matrix": [EvidenceItem.model_validate(item) for item in prepared["evidence"]],
            "data_gaps": gaps, "errors": []}

    builder = StateGraph(FinanceState, input_schema=FinanceInput, output_schema=FinanceOutput,
                         context_schema=CommerceRuntimeContext)
    for name, node in (("prepare", prepare), ("collect_evidence", collect_evidence),
                       ("analyze", analyze), ("validate_and_publish", validate_and_publish)):
        builder.add_node(name, node)
    for left, right in ((START, "prepare"), ("prepare", "collect_evidence"),
                        ("collect_evidence", "analyze"), ("analyze", "validate_and_publish"),
                        ("validate_and_publish", END)):
        builder.add_edge(left, right)
    compiled = builder.compile(name="finance_subgraph", checkpointer=checkpointer, interrupt_after=interrupt_after)
    return guard_conversation(compiled,enabled=dependencies.enable_conversation_memory,domain="finance",workflow=dependencies.workflow_version)
