"""Fixed four-node read-only Price subgraph; dependencies never checkpointed."""
from __future__ import annotations
import asyncio
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol
from typing_extensions import TypedDict
from langgraph.graph import START, END, StateGraph
from langgraph.runtime import Runtime
from commerce_agent.context import CommerceRuntimeContext
from commerce_agent.conversation_context import memory_update, memory_block
from commerce_agent.conversation_guard import guard_conversation
from commerce_agent.models import AgentStatus, DataGap, DomainAnalysis, DomainName, EvidenceItem, QueryPlan, WorkflowError
from commerce_agent.price.analysis import prepare_price_analysis
from commerce_agent.price.contracts import AgentDiagnosticResult
from commerce_agent.price.evidence import collect_price_evidence
from commerce_agent.price.policy import PriceRequest, resolve_price_request
from commerce_agent.price.prompts import build_price_instructions
from commerce_agent.price.publication import validate_price_publication
from commerce_agent.price.report import render_price_report


class PriceModel(Protocol):
    async def analyze(self, *, instructions: str, prepared: dict) -> AgentDiagnosticResult: ...


@dataclass(frozen=True)
class PriceDependencies:
    client_factory: Callable = field(repr=False)
    model: PriceModel = field(repr=False)
    workflow_version: Literal["v1", "v2_prompt", "v2_optimized"] = "v1"
    model_profile: str = "default"
    collection_timeout: float = 90
    model_timeout: float = 60

    enable_conversation_memory: bool = False
    enable_history: bool = False

    def __post_init__(self):
        if type(self.enable_history) is not bool or self.enable_history and self.enable_conversation_memory:
            raise ValueError("Price history requires isolated read-only capability")
        if type(self.enable_conversation_memory) is not bool:
            raise ValueError("conversation capability must be bool")
        if self.workflow_version not in {"v1", "v2_prompt", "v2_optimized"}:
            raise ValueError("invalid Price workflow profile")
        if not 0 < self.collection_timeout <= 300 or not 0 < self.model_timeout <= 180:
            raise ValueError("invalid Price deadline")


class PriceInput(TypedDict, total=False):
    messages: list[Any]
    query_plan: QueryPlan | None
    agent_statuses: dict[str, AgentStatus]
    price_analysis: DomainAnalysis | None


class PriceOutput(TypedDict, total=False):
    price_history_evidence: dict | None
    agent_statuses: dict[str, AgentStatus]
    price_analysis: DomainAnalysis | None
    evidence_matrix: list[EvidenceItem]
    data_gaps: list[DataGap]
    errors: list[WorkflowError]


class PriceState(PriceInput, PriceOutput, total=False):
    price_conversation: dict | None
    price_conversation_version: int | None
    price_request: dict | None
    prepared_price: dict | None
    price_candidate: dict | None
    price_failure: str | None
    price_skipped: bool
    price_base_attempt: int
    price_revision: int


def _status(state, node, stage, status="running", detail=None):
    return {"agent_statuses": {"price": AgentStatus(domain=DomainName.PRICE, status=status,
        attempt=state.get("price_base_attempt", 0) + stage, node=node, detail=detail)}}


def build_price_subgraph(dependencies: PriceDependencies, *, checkpointer=None, interrupt_after=None):
    def prepare(state: PriceState, runtime: Runtime[CommerceRuntimeContext]):
        prior = state.get("agent_statuses", {}).get("price")
        prior = AgentStatus.model_validate(prior) if prior else None
        previous = state.get("price_analysis")
        previous = DomainAnalysis.model_validate(previous) if previous else None
        base = {"price_base_attempt": prior.attempt if prior else 0,
            "price_revision": previous.revision + 1 if previous else 1,
            "price_request": None, "prepared_price": None, "price_candidate": None,
            "price_history_evidence": None,
            "price_failure": None, "price_skipped": False}
        if dependencies.enable_conversation_memory:
            base.update(price_conversation=None, price_conversation_version=None)
        plan = state.get("query_plan")
        plan = QueryPlan.model_validate(plan) if plan else QueryPlan()
        if DomainName.PRICE not in plan.required_domains:
            return {**base, **_status(base, "prepare", 1, "skipped", "domain_not_required"), "price_skipped": True}
        try:
            if runtime.context.model_profile != dependencies.model_profile:
                raise ValueError("Price model profile mismatch")
            query = next(message.content for message in reversed(state.get("messages", [])) if message.type == "human")
            constraints = dict(plan.constraints.get("price") or {})
            request = resolve_price_request(query, constraints or None)
            if request.operation == "history" and not dependencies.enable_history:
                raise ValueError("Price history capability disabled")
            base.update(memory_update(dependencies.enable_conversation_memory,state,runtime,"price",dependencies.workflow_version))
            return {**base, **_status(base, "prepare", 1), "price_request": request.model_dump(mode="json")}
        except (ValueError, TypeError, StopIteration, AttributeError):
            return {**base, **_status(base, "prepare", 1, "failed", "invalid_request"), "price_failure": "invalid_request"}

    async def collect_evidence(state: PriceState, runtime: Runtime[CommerceRuntimeContext]):
        if state.get("price_skipped") or state.get("price_failure"):
            return {}
        try:
            async with asyncio.timeout(dependencies.collection_timeout):
                async with dependencies.client_factory() as client:
                    request = PriceRequest.model_validate(state["price_request"])
                    if request.operation == "history":
                        from commerce_agent.price.history import collect_history
                        if not dependencies.enable_history:
                            raise ValueError("Price history capability disabled")
                        prepared = await collect_history(client, runtime.context, request)
                    else:
                        collection = await collect_price_evidence(client, runtime.context, request)
                        prepared = prepare_price_analysis(collection)
            return {**_status(state, "collect_evidence", 2), "prepared_price": prepared}
        except Exception:
            # Never persist transport exception bodies; caller cancellation
            # derives from BaseException and deliberately propagates.
            return {**_status(state, "collect_evidence", 2, "failed", "needs_data"), "price_failure": "needs_data"}

    async def analyze(state: PriceState, runtime: Runtime[CommerceRuntimeContext]):
        if state.get("price_skipped") or state.get("price_failure"):
            return {}
        prepared = state["prepared_price"]
        try:
            instructions = build_price_instructions(workflow_version=dependencies.workflow_version,
                user_query=prepared["request"]["user_query"], today=prepared["today"],
                memory_block=memory_block(state,runtime,"price",dependencies.workflow_version))
            async with asyncio.timeout(dependencies.model_timeout):
                candidate = await dependencies.model.analyze(instructions=instructions, prepared=deepcopy(prepared))
            validated = validate_price_publication(candidate, prepared)
            return {**_status(state, "analyze", 3), "price_candidate": validated.model_dump(mode="json")}
        except Exception:
            return {**_status(state, "analyze", 3, "failed", "analysis_rejected"), "price_failure": "analysis_rejected"}

    def validate_and_publish(state: PriceState, runtime: Runtime[CommerceRuntimeContext]):
        if state.get("price_skipped"):
            return {}
        failure, prepared = state.get("price_failure"), state.get("prepared_price")
        if not failure:
            try:
                result = validate_price_publication(state["price_candidate"], prepared)
                report = render_price_report(result, prepared)
            except Exception:
                failure = "analysis_rejected"
        identity = f"price:{runtime.context.request_id}"
        if failure:
            return {**_status(state, "validate_and_publish", 4, "failed", failure),
                "price_candidate": None, "price_failure": failure,
                "price_analysis": DomainAnalysis(domain=DomainName.PRICE, revision=state["price_revision"],
                    status="failed", summary="Ценовой анализ недоступен: недостаточно проверенных данных или результат отклонён."),
                "data_gaps": [DataGap(gap_id=f"{identity}:{failure}", domain=DomainName.PRICE, reason=failure)],
                "errors": [WorkflowError(error_id=f"{identity}:{failure}", code=f"PRICE_{failure.upper()}",
                    message="Price read-only branch did not produce a validated result.", node="price.validate_and_publish")]}
        refs = tuple(dict.fromkeys(item["evidence_ref"] for item in prepared["evidence"]))
        gaps = [DataGap(gap_id=f"{identity}:limitation:{index}", domain=DomainName.PRICE, reason=warning,
            blocking=warning.startswith("needs_data:"), status="open" if warning.startswith("needs_data:") else "accepted_limitation")
            for index, warning in enumerate(prepared["warnings"])]
        return {**_status(state, "validate_and_publish", 4, "completed"),
            "price_history_evidence": ({key: prepared[key] for key in
                ("request", "today", "history_version", "history_envelope")} if prepared.get("history_version") else None),
            "price_analysis": DomainAnalysis(domain=DomainName.PRICE, revision=state["price_revision"],
                status="partial" if gaps else "completed", summary=report, result=result, evidence_refs=refs),
            "evidence_matrix": [EvidenceItem.model_validate(item) for item in prepared["evidence"]],
            "data_gaps": gaps, "errors": []}

    builder = StateGraph(PriceState, input_schema=PriceInput, output_schema=PriceOutput, context_schema=CommerceRuntimeContext)
    for name, node in (("prepare", prepare), ("collect_evidence", collect_evidence), ("analyze", analyze), ("validate_and_publish", validate_and_publish)):
        builder.add_node(name, node)
    for left, right in ((START, "prepare"), ("prepare", "collect_evidence"), ("collect_evidence", "analyze"),
                        ("analyze", "validate_and_publish"), ("validate_and_publish", END)):
        builder.add_edge(left, right)
    compiled = builder.compile(name="price_subgraph", checkpointer=checkpointer, interrupt_after=interrupt_after)
    from commerce_agent.price.history_guard import guard_history
    return guard_history(guard_conversation(compiled,enabled=dependencies.enable_conversation_memory,
        domain="price",workflow=dependencies.workflow_version), dependencies.enable_history)
