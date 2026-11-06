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
from commerce_agent.conversation_context import memory_update, memory_block
from commerce_agent.conversation_guard import guard_conversation
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
    enable_targeted_reports: bool = False
    enable_daily_reports: bool = False
    enable_historical_funnel: bool = False
    scoped_policy_version: int = 3

    enable_conversation_memory: bool = False

    def __post_init__(self):
        if type(self.enable_historical_funnel) is not bool or self.enable_historical_funnel and not self.enable_targeted_reports:
            raise ValueError("Historical funnel requires targeted Sales")
        if type(self.enable_conversation_memory) is not bool:
            raise ValueError("conversation capability must be bool")
        if self.workflow_version not in {"v1", "v2_prompt", "v2_optimized"}:
            raise ValueError("invalid Sales workflow profile")
        if not 0 < self.collection_timeout <= 300 or not 0 < self.model_timeout <= 180:
            raise ValueError("invalid Sales deadline")
        if type(self.enable_targeted_reports) is not bool or type(self.enable_daily_reports) is not bool:
            raise ValueError("invalid Sales targeted capability")
        if type(self.scoped_policy_version) is not int or self.scoped_policy_version not in (1, 2, 3):
            raise ValueError("unsupported Sales scoped writer policy")


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


class SalesHistoricalOutput(SalesOutput, total=False):
    # Private typed neighbor receipt; root CommerceState does not publish it.
    historical_funnel_evidence: dict | None


class SalesState(SalesInput, SalesHistoricalOutput, total=False):
    sales_conversation: dict | None
    sales_conversation_version: int | None
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
    def guard_targeted_debug():
        from commerce_agent.debug_privacy import development_debug_enabled
        if dependencies.enable_targeted_reports and development_debug_enabled():
            raise ValueError("Sales targeted sources require separate debug acceptance")
        if (dependencies.enable_daily_reports or dependencies.enable_historical_funnel) and development_debug_enabled():
            raise ValueError("Sales daily sources require separate debug acceptance")

    guard_targeted_debug()
    def prepare(state: SalesState, runtime: Runtime[CommerceRuntimeContext]):
        guard_targeted_debug()
        prior = state.get("agent_statuses", {}).get("sales")
        prior = AgentStatus.model_validate(prior) if prior else None
        previous = state.get("sales_analysis")
        previous = DomainAnalysis.model_validate(previous) if previous else None
        base = {"sales_base_attempt": prior.attempt if prior else 0,
                "sales_revision": previous.revision + 1 if previous else 1,
                "sales_request": None, "prepared_sales": None, "sales_candidate": None,
                "sales_failure": None, "sales_skipped": False}
        if "historical_funnel_evidence" in state:
            base["historical_funnel_evidence"] = None
        if dependencies.enable_conversation_memory:
            base.update(sales_conversation=None, sales_conversation_version=None)
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
            if request.scope == "portfolio" and request.targets and not dependencies.enable_targeted_reports:
                raise ValueError("Sales targeted capability disabled")
            base.update(memory_update(dependencies.enable_conversation_memory,state,runtime,"sales",dependencies.workflow_version))
            return {**base, **_status(base, "prepare", 1), "sales_request": request.model_dump(mode="json")}
        except (ValueError, TypeError, StopIteration, AttributeError):
            return {**base, **_status(base, "prepare", 1, "failed", "invalid_request"), "sales_failure": "invalid_request"}

    async def collect_evidence(state: SalesState, runtime: Runtime[CommerceRuntimeContext]):
        guard_targeted_debug()
        if state.get("sales_skipped") or state.get("sales_failure"):
            return {}
        try:
            async with asyncio.timeout(dependencies.collection_timeout):
                async with dependencies.client_factory() as client:
                    prepared = await collect_sales_evidence(client, runtime.context,
                        SalesRequest.model_validate(state["sales_request"]),
                        publish_targeted=dependencies.enable_targeted_reports,publish_daily=dependencies.enable_daily_reports,
                        scoped_policy_version=dependencies.scoped_policy_version)
                    if dependencies.enable_historical_funnel:
                        from commerce_agent.sales.historical_funnel import collect
                        prepared = await collect(client, runtime.context, SalesRequest.model_validate(state["sales_request"]), prepared)
            return {**_status(state, "collect_evidence", 2), "prepared_sales": prepared}
        except Exception:
            # Provider/transport exception bodies may contain secrets. Static
            # codes only; cancellation is BaseException and propagates.
            return {**_status(state, "collect_evidence", 2, "failed", "needs_data"), "sales_failure": "needs_data"}

    async def analyze(state: SalesState, runtime: Runtime[CommerceRuntimeContext]):
        guard_targeted_debug()
        if state.get("sales_skipped") or state.get("sales_failure"):
            return {}
        prepared = state["prepared_sales"]
        try:
            instructions = build_sales_instructions(workflow_version=dependencies.workflow_version,
                user_query=prepared["request"]["user_query"], today=prepared["today"],
                memory_block=memory_block(state,runtime,"sales",dependencies.workflow_version))
            async with asyncio.timeout(dependencies.model_timeout):
                candidate = await dependencies.model.analyze(instructions=instructions, prepared=deepcopy(prepared))
            validated = validate_sales_publication(candidate, prepared)
            return {**_status(state, "analyze", 3), "sales_candidate": validated.model_dump(mode="json")}
        except Exception:
            return {**_status(state, "analyze", 3, "failed", "analysis_rejected"), "sales_failure": "analysis_rejected"}

    def validate_and_publish(state: SalesState, runtime: Runtime[CommerceRuntimeContext]):
        guard_targeted_debug()
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
            **({"historical_funnel_evidence": deepcopy(prepared["historical_funnel"])}
                if prepared.get("historical_funnel_version") == 2 else {}),
            "data_gaps": gaps, "errors": []}

    builder = StateGraph(SalesState, input_schema=SalesInput,
                         output_schema=SalesHistoricalOutput if dependencies.enable_historical_funnel else SalesOutput,
                         context_schema=CommerceRuntimeContext)
    for name, node in (("prepare", prepare), ("collect_evidence", collect_evidence),
                       ("analyze", analyze), ("validate_and_publish", validate_and_publish)):
        builder.add_node(name, node)
    for left, right in ((START, "prepare"), ("prepare", "collect_evidence"),
                        ("collect_evidence", "analyze"), ("analyze", "validate_and_publish"),
                        ("validate_and_publish", END)):
        builder.add_edge(left, right)
    from commerce_agent.sales.capability_guard import guard_sales_capabilities
    compiled = guard_sales_capabilities(builder.compile(name="sales_subgraph", checkpointer=checkpointer,
        interrupt_after=interrupt_after),enabled=dependencies.enable_targeted_reports,daily_enabled=dependencies.enable_daily_reports,historical_funnel_enabled=dependencies.enable_historical_funnel)
    return guard_conversation(compiled,enabled=dependencies.enable_conversation_memory,domain="sales",workflow=dependencies.workflow_version)
