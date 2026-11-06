"""Four-node read-only Supply subgraph. Dependencies are trusted process resources."""
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
from commerce_agent.supply.contracts import AgentDiagnosticResult
from commerce_agent.supply.evidence import collect_supply_evidence
from commerce_agent.supply.policy import SupplyRequest, resolve_supply_request
from commerce_agent.supply.prompts import build_supply_instructions
from commerce_agent.supply.publication import validate_supply_publication
from commerce_agent.supply.report import render_supply_report


class SupplyModel(Protocol):
    async def analyze(self, *, instructions: str, prepared: dict) -> AgentDiagnosticResult: ...


@dataclass(frozen=True)
class SupplyDependencies:
    client_factory: Callable = field(repr=False)
    model: SupplyModel = field(repr=False)
    workflow_version: Literal["v1", "v2_prompt", "v2_optimized"] = "v1"
    model_profile: str = "default"
    collection_timeout: float = 90
    model_timeout: float = 60

    enable_conversation_memory: bool = False
    enable_alternative_methods: bool = False
    alternative_boundary_tolerance_days: int = 0

    def __post_init__(self):
        if type(self.enable_conversation_memory) is not bool:
            raise ValueError("conversation capability must be bool")
        if (type(self.enable_alternative_methods) is not bool
            or type(self.alternative_boundary_tolerance_days) is not int
            or not 0 <= self.alternative_boundary_tolerance_days <= 31):
            raise ValueError("invalid Supply alternative capability")
        if self.workflow_version not in {"v1", "v2_prompt", "v2_optimized"}:
            raise ValueError("invalid Supply workflow profile")
        if not 0 < self.collection_timeout <= 300 or not 0 < self.model_timeout <= 180:
            raise ValueError("invalid Supply deadline")


class SupplyInput(TypedDict, total=False):
    messages: list[Any]
    query_plan: QueryPlan | None
    agent_statuses: dict[str, AgentStatus]
    supply_analysis: DomainAnalysis | None


class SupplyOutput(TypedDict, total=False):
    agent_statuses: dict[str, AgentStatus]
    supply_analysis: DomainAnalysis | None
    evidence_matrix: list[EvidenceItem]
    data_gaps: list[DataGap]
    errors: list[WorkflowError]


class SupplyState(SupplyInput, SupplyOutput, total=False):
    supply_conversation: dict | None
    supply_conversation_version: int | None
    supply_request: dict | None
    prepared_supply: dict | None
    supply_candidate: dict | None
    supply_failure: str | None
    supply_skipped: bool
    supply_base_attempt: int
    supply_revision: int


def _status(state, node, stage, status="running", detail=None):
    return {"agent_statuses": {"supply": AgentStatus(domain=DomainName.SUPPLY, status=status,
        attempt=state.get("supply_base_attempt", 0) + stage, node=node, detail=detail)}}


def build_supply_subgraph(dependencies: SupplyDependencies, *, checkpointer=None, interrupt_after=None):
    def guard_alternative_debug():
        from commerce_agent.debug_privacy import development_debug_enabled
        if dependencies.enable_alternative_methods and development_debug_enabled():
            raise ValueError("Supply alternative sources require separate debug acceptance")

    def prepare(state: SupplyState, runtime: Runtime[CommerceRuntimeContext]):
        guard_alternative_debug()
        prior = state.get("agent_statuses", {}).get("supply")
        prior = AgentStatus.model_validate(prior) if prior else None
        previous = state.get("supply_analysis")
        previous = DomainAnalysis.model_validate(previous) if previous else None
        base = {"supply_base_attempt": prior.attempt if prior else 0,
                "supply_revision": previous.revision + 1 if previous else 1,
                "supply_request": None, "prepared_supply": None, "supply_candidate": None,
                "supply_failure": None, "supply_skipped": False}
        if dependencies.enable_conversation_memory:
            base.update(supply_conversation=None, supply_conversation_version=None)
        plan = state.get("query_plan")
        plan = QueryPlan.model_validate(plan) if plan else QueryPlan()
        if DomainName.SUPPLY not in plan.required_domains:
            return {**base, **_status(base, "prepare", 1, "skipped", "domain_not_required"), "supply_skipped": True}
        try:
            if runtime.context.model_profile != dependencies.model_profile:
                raise ValueError("Supply model profile mismatch")
            messages = state.get("messages", [])
            query = next(message.content for message in reversed(messages) if message.type == "human")
            constraints = dict(plan.constraints.get("supply") or {})
            requested = set(plan.requested_analyses)
            if "operation" not in constraints:
                if "inventory_matrix" in requested:
                    constraints["operation"] = "matrix"
                elif "inventory_turnover" in requested and "inventory_value" in requested:
                    constraints["operation"] = "matrix"
                elif "inventory_turnover" in requested:
                    constraints["operation"] = "turnover"
                elif "inventory_value" in requested:
                    constraints["operation"] = "value"
            request = resolve_supply_request(query, constraints or None)
            base.update(memory_update(dependencies.enable_conversation_memory,state,runtime,"supply",dependencies.workflow_version))
            return {**base, **_status(base, "prepare", 1), "supply_request": request.model_dump(mode="json")}
        except (ValueError, TypeError, StopIteration, AttributeError):
            return {**base, **_status(base, "prepare", 1, "failed", "invalid_request"), "supply_failure": "invalid_request"}

    async def collect_evidence(state: SupplyState, runtime: Runtime[CommerceRuntimeContext]):
        guard_alternative_debug()
        if state.get("supply_skipped") or state.get("supply_failure"):
            return {}
        try:
            async with asyncio.timeout(dependencies.collection_timeout):
                async with dependencies.client_factory() as client:
                    prepared = await collect_supply_evidence(client, runtime.context,
                        SupplyRequest.model_validate(state["supply_request"]),
                        enable_alternative_methods=dependencies.enable_alternative_methods,
                        alternative_boundary_tolerance_days=dependencies.alternative_boundary_tolerance_days)
            if not set(prepared["required_operations"]) <= {c["operation"] for c in prepared["claims"]}:
                raise ValueError("Supply requested analysis has no verified data")
            return {**_status(state, "collect_evidence", 2), "prepared_supply": prepared}
        except Exception:
            # Provider/transport exception bodies may contain secrets. Static
            # codes only; cancellation is BaseException and propagates.
            return {**_status(state, "collect_evidence", 2, "failed", "needs_data"), "supply_failure": "needs_data"}

    async def analyze(state: SupplyState, runtime: Runtime[CommerceRuntimeContext]):
        guard_alternative_debug()
        if state.get("supply_skipped") or state.get("supply_failure"):
            return {}
        prepared = state["prepared_supply"]
        try:
            instructions = build_supply_instructions(workflow_version=dependencies.workflow_version,
                user_query=prepared["request"]["user_query"], today=prepared["today"],
                memory_block=memory_block(state,runtime,"supply",dependencies.workflow_version))
            async with asyncio.timeout(dependencies.model_timeout):
                candidate = await dependencies.model.analyze(instructions=instructions, prepared=deepcopy(prepared))
            validated = validate_supply_publication(candidate, prepared)
            return {**_status(state, "analyze", 3), "supply_candidate": validated.model_dump(mode="json")}
        except Exception:
            return {**_status(state, "analyze", 3, "failed", "analysis_rejected"), "supply_failure": "analysis_rejected"}

    def validate_and_publish(state: SupplyState, runtime: Runtime[CommerceRuntimeContext]):
        guard_alternative_debug()
        if state.get("supply_skipped"):
            return {}
        failure = state.get("supply_failure")
        prepared = state.get("prepared_supply")
        if not failure:
            try:
                result = validate_supply_publication(state["supply_candidate"], prepared)
                report = render_supply_report(result, prepared)
            except Exception:
                failure = "analysis_rejected"
        identity = f"supply:{runtime.context.request_id}"
        if failure:
            return {**_status(state, "validate_and_publish", 4, "failed", failure),
                "supply_candidate": None, "supply_failure": failure,
                "supply_analysis": DomainAnalysis(domain=DomainName.SUPPLY, revision=state["supply_revision"],
                    status="failed", summary="Анализ запасов недоступен: недостаточно проверенных данных или результат не прошёл проверку."),
                "data_gaps": [DataGap(gap_id=f"{identity}:{failure}", domain=DomainName.SUPPLY, reason=failure)],
                "errors": [WorkflowError(error_id=f"{identity}:{failure}", code=f"SUPPLY_{failure.upper()}",
                    message="Supply read-only branch did not produce a validated result.", node="supply.validate_and_publish")]}
        refs = tuple(dict.fromkeys(item["evidence_ref"] for item in prepared["evidence"]))
        warnings = prepared["warnings"]
        gaps = [DataGap(gap_id=f"{identity}:limitation:{index}", domain=DomainName.SUPPLY,
                       reason=warning, blocking=warning.startswith("needs_data:"),
                       status="open" if warning.startswith("needs_data:") else "accepted_limitation")
                for index, warning in enumerate(warnings)]
        # Publication only permits the exact native warnings. Do not create a
        # second blocking copy of accepted simulation/provenance limitations.
        return {**_status(state, "validate_and_publish", 4, "completed"),
            "supply_analysis": DomainAnalysis(domain=DomainName.SUPPLY, revision=state["supply_revision"],
                status="partial" if warnings or result.data_gaps else "completed", summary=report,
                result=result, evidence_refs=refs),
            "evidence_matrix": [EvidenceItem.model_validate(item) for item in prepared["evidence"]],
            "data_gaps": gaps, "errors": []}

    builder = StateGraph(SupplyState, input_schema=SupplyInput, output_schema=SupplyOutput,
                         context_schema=CommerceRuntimeContext)
    for name, node in (("prepare", prepare), ("collect_evidence", collect_evidence),
                       ("analyze", analyze), ("validate_and_publish", validate_and_publish)):
        builder.add_node(name, node)
    for left, right in ((START, "prepare"), ("prepare", "collect_evidence"),
                        ("collect_evidence", "analyze"), ("analyze", "validate_and_publish"),
                        ("validate_and_publish", END)):
        builder.add_edge(left, right)
    from commerce_agent.supply.capability_guard import guard_supply_capabilities
    compiled = guard_supply_capabilities(builder.compile(name="supply_subgraph", checkpointer=checkpointer,
        interrupt_after=interrupt_after), enabled=dependencies.enable_alternative_methods)
    return guard_conversation(compiled,enabled=dependencies.enable_conversation_memory,domain="supply",workflow=dependencies.workflow_version)
