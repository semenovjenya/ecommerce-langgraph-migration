"""Read-only Marketing subgraph with an opt-in bounded investigation loop."""
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
from commerce_agent.marketing.contracts import MarketingDiagnosticResult
from commerce_agent.marketing.evidence import collect_marketing_evidence
from commerce_agent.marketing.policy import parse_marketing_request, resolve_marketing_request
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
    enable_diagnostics: bool = False
    enable_diagnostic_debug: bool = False
    enable_composites: bool = False
    composite_storage_version: int | None = None
    composite_policy_version: int = 3
    enable_scoped_reports: bool = False
    enable_scoped_debug: bool = False
    enable_scoped_brief: bool = False
    enable_overview: bool = False
    enable_investigation: bool = False
    investigator: Any = field(default=None, repr=False)
    feedback_neighbors: Any = field(default=None, repr=False)
    enable_feedback_details: bool = False
    feedback_report_style: Literal["technical", "readable", "compact"] = "technical"
    enable_price_feedback: bool = False
    enable_research_v6: bool = False
    # Selects the wire format for NEW journals. The v6 reader keeps older formats.
    enable_research_phases: bool = False
    enable_research_owned: bool = False
    enable_historical_sources: bool = False

    enable_conversation_memory: bool = False

    def __post_init__(self):
        if type(self.enable_historical_sources) is not bool or self.enable_historical_sources and (
                not self.enable_research_owned or self.enable_scoped_debug or self.enable_conversation_memory):
            raise ValueError("Historical sources require isolated code-owned research")
        if type(self.enable_research_owned) is not bool or self.enable_research_owned and (
                not self.enable_research_v6 or self.enable_research_phases):
            raise ValueError("Code-owned research requires v6 and an unambiguous writer")
        if type(self.enable_research_phases) is not bool or self.enable_research_phases and not self.enable_research_v6:
            raise ValueError("Research phases require the v6 reader")
        if type(self.enable_research_v6) is not bool or self.enable_research_v6 and (
                self.feedback_report_style != "compact" or not self.enable_feedback_details
                or not self.enable_investigation or not hasattr(self.investigator, "decide")):
            raise ValueError("Research v6 requires compact details and a v6 investigator")
        if type(self.enable_price_feedback) is not bool or self.enable_price_feedback and (
                self.feedback_report_style != "compact" or getattr(self.feedback_neighbors, "price", None) is None):
            raise ValueError("Price feedback requires compact mode and a historical Price neighbor")
        if self.feedback_report_style not in ("technical", "readable", "compact") or (
                self.feedback_report_style in ("readable", "compact") and not self.enable_feedback_details):
            raise ValueError("Readable feedback report requires details")
        if self.feedback_report_style == "compact" and self.enable_conversation_memory:
            raise ValueError("Compact memory requires separate acceptance")
        if type(self.enable_feedback_details) is not bool or self.enable_feedback_details and self.feedback_neighbors is None:
            raise ValueError("Feedback details require fixed neighbors")
        if self.feedback_neighbors is not None:
            from commerce_agent.marketing.feedback import NeighborExecutor
            if (not self.enable_investigation or not isinstance(self.feedback_neighbors, NeighborExecutor)
                or self.feedback_neighbors.profile != self.model_profile
                or self.feedback_neighbors.workflow != self.workflow_version):
                raise ValueError("Marketing feedback requires matching fixed neighbors and investigation")
        if type(self.enable_investigation) is not bool:
            raise ValueError("Marketing investigation flag must be bool")
        if self.enable_investigation and (not self.enable_scoped_reports or self.investigator is None
                or self.enable_scoped_debug or self.enable_diagnostic_debug):
            raise ValueError("Marketing investigation requires scoped capability and an investigator, debug disabled")
        if type(self.enable_overview) is not bool:
            raise ValueError("Marketing overview capability must be bool")
        if type(self.enable_scoped_brief) is not bool or self.enable_scoped_brief and not self.enable_scoped_reports:
            raise ValueError("Marketing brief presentation requires scoped capability")
        if type(self.enable_conversation_memory) is not bool:
            raise ValueError("conversation capability must be bool")
        if self.workflow_version not in {"v1", "v2_prompt", "v2_optimized"}:
            raise ValueError("invalid Marketing workflow profile")
        if not 0 < self.collection_timeout <= 300 or not 0 < self.model_timeout <= 180:
            raise ValueError("invalid Marketing deadline")
        if type(self.enable_diagnostics) is not bool or type(self.enable_diagnostic_debug) is not bool or type(self.enable_composites) is not bool or type(self.enable_scoped_reports) is not bool or type(self.enable_scoped_debug) is not bool:
            raise ValueError("invalid Marketing diagnostic feature flag")
        if self.enable_diagnostic_debug and not self.enable_diagnostics:
            raise ValueError("Marketing diagnostic debug requires capability")
        if self.enable_scoped_debug and (not self.enable_scoped_reports or self.enable_diagnostics or self.enable_composites or self.enable_overview):
            raise ValueError("Marketing scoped debug requires isolated scoped capability")
        if self.composite_storage_version is not None and (type(self.composite_storage_version) is not int
            or self.composite_storage_version!=1 or not self.enable_composites):
            raise ValueError("invalid Marketing composite storage version")
        if (type(self.composite_policy_version) is not int or self.composite_policy_version not in (3,4)
            or self.composite_policy_version==4 and (not self.enable_composites or self.composite_storage_version!=1)):
            raise ValueError("invalid Marketing composite analytical policy")


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
    marketing_conversation: dict | None
    marketing_conversation_version: int | None
    marketing_request: dict | None
    prepared_marketing: dict | None
    marketing_candidate: dict | None
    marketing_failure: str | None
    marketing_skipped: bool
    marketing_base_attempt: int
    marketing_revision: int
    marketing_report_presentation_version: int | None
    marketing_investigation_version: int | None
    marketing_investigation_decisions: list[dict]
    marketing_investigation_reads: list[dict]
    marketing_investigation_report_style: str
    marketing_investigation_v6: dict


def _validate(candidate, prepared):
    if prepared.get("capability")=="latest_marketing_overview":
        from commerce_agent.marketing.overview import validate_overview
        return validate_overview(candidate,prepared)
    if prepared.get("capability")=="scoped_report":
        from commerce_agent.marketing.scoped import validate_scoped_publication
        return validate_scoped_publication(candidate,prepared)
    if prepared.get("capability")=="latest_native_diagnostic":
        from commerce_agent.marketing.composite_analysis import validate_composite_publication
        return validate_composite_publication(candidate,prepared)
    if prepared.get("capability") == "sku_diagnostic":
        from commerce_agent.marketing.diagnostic_analysis import validate_diagnostic_publication
        return validate_diagnostic_publication(candidate, prepared)
    return validate_marketing_publication(candidate, prepared)


def _status(state, node, stage, status="running", detail=None):
    return {"agent_statuses": {"marketing": AgentStatus(domain=DomainName.MARKETING, status=status,
        attempt=state.get("marketing_base_attempt", 0) + stage, node=node, detail=detail)}}


def build_marketing_subgraph(dependencies: MarketingDependencies, *, checkpointer=None, interrupt_after=None):
    from commerce_agent.marketing import investigation as investigation_api
    feedback_enabled = dependencies.feedback_neighbors is not None
    if feedback_enabled:
        from commerce_agent.marketing import feedback as investigation_api
        if dependencies.enable_feedback_details:
            from commerce_agent.marketing.feedback_details import api
            investigation_api = api()
        if dependencies.feedback_report_style == "compact":
            from commerce_agent.marketing.compact import api
            investigation_api = api(price=dependencies.enable_price_feedback)

    def prepare(state: MarketingState, runtime: Runtime[CommerceRuntimeContext]):
        if dependencies.enable_research_v6 and state.get("marketing_investigation_v6"):
            from commerce_agent.marketing.research import assert_owner
            assert_owner(state["marketing_investigation_v6"], runtime.context)
        prior = state.get("agent_statuses", {}).get("marketing")
        prior = AgentStatus.model_validate(prior) if prior else None
        previous = state.get("marketing_analysis")
        previous = DomainAnalysis.model_validate(previous) if previous else None
        base = {"marketing_base_attempt": prior.attempt if prior else 0,
                "marketing_revision": previous.revision + 1 if previous else 1,
                "marketing_request": None, "prepared_marketing": None, "marketing_candidate": None,
                "marketing_failure": None, "marketing_skipped": False}
        base["marketing_report_presentation_version"]=None
        if dependencies.enable_investigation:
            base.update(marketing_investigation_version=6 if dependencies.enable_research_v6 else 5 if dependencies.enable_price_feedback else 4 if dependencies.feedback_report_style == "compact" else 3 if dependencies.enable_feedback_details else 2 if feedback_enabled else 1,
                marketing_investigation_decisions=[], marketing_investigation_reads=[])
            base["marketing_investigation_report_style"] = dependencies.feedback_report_style
            if dependencies.enable_research_v6:
                from commerce_agent.marketing.research import new_journal
                base["marketing_investigation_v6"] = new_journal(runtime.context,
                    wire_version=3 if dependencies.enable_research_owned else 2 if dependencies.enable_research_phases else 1)
        if dependencies.enable_conversation_memory:
            base.update(marketing_conversation=None, marketing_conversation_version=None)
        plan = state.get("query_plan")
        plan = QueryPlan.model_validate(plan) if plan else QueryPlan()
        if DomainName.MARKETING not in plan.required_domains:
            return {**base, **_status(base, "prepare", 1, "skipped", "domain_not_required"), "marketing_skipped": True}
        try:
            if runtime.context.model_profile != dependencies.model_profile:
                raise ValueError("Marketing model profile mismatch")
            messages = state.get("messages", [])
            query = next(message.content for message in reversed(messages) if message.type == "human")
            request = resolve_marketing_request(query, plan.constraints.get("marketing"), allow_diagnostics=dependencies.enable_diagnostics,
                allow_composites=dependencies.enable_composites,allow_scoped=dependencies.enable_scoped_reports,allow_overview=dependencies.enable_overview)
            if dependencies.enable_investigation and getattr(request, "capability", None) != "scoped_report":
                raise ValueError("investigation currently requires explicit scoped targets")
            if feedback_enabled and len(request.targets) != 1:
                raise ValueError("feedback currently requires one canonical SKU")
            if dependencies.feedback_report_style == "compact" and request.scope != "single_sku":
                raise ValueError("compact currently requires single_sku scope")
            if getattr(request, "capability", None) == "sku_diagnostic":
                from commerce_agent.debug_privacy import development_debug_enabled
                if development_debug_enabled():
                    if not dependencies.enable_diagnostic_debug:
                        raise ValueError("Marketing diagnostic debug not accepted")
                    from commerce_agent.marketing.debug_export import assert_diagnostic_debug_request
                    assert_diagnostic_debug_request(request)
            base.update(memory_update(dependencies.enable_conversation_memory,state,runtime,"marketing",dependencies.workflow_version))
            if dependencies.enable_scoped_brief and getattr(request,"capability",None)=="scoped_report":
                base["marketing_report_presentation_version"]=1
            return {**base, **_status(base, "prepare", 1), "marketing_request": request.model_dump(mode="json")}
        except (ValueError, TypeError, StopIteration, AttributeError):
            return {**base, **_status(base, "prepare", 1, "failed", "invalid_request"), "marketing_failure": "invalid_request"}

    async def collect_evidence(state: MarketingState, runtime: Runtime[CommerceRuntimeContext]):
        if dependencies.enable_research_v6:
            from commerce_agent.marketing.research_runtime import checked
            checked(state, runtime.context)
        if state.get("marketing_skipped") or state.get("marketing_failure"):
            return {}
        try:
            async with asyncio.timeout(dependencies.collection_timeout):
                async with dependencies.client_factory() as client:
                    request = parse_marketing_request(state["marketing_request"])
                    if getattr(request,"capability",None)=="latest_marketing_overview":
                        from commerce_agent.marketing.overview import collect_overview
                        prepared=await collect_overview(client,runtime.context,request)
                    elif getattr(request,"capability",None)=="scoped_report":
                        from commerce_agent.marketing.scoped import collect_scoped_report
                        prepared=await collect_scoped_report(client,runtime.context,request,allow_debug=dependencies.enable_scoped_debug)
                        if dependencies.enable_historical_sources:
                            from commerce_agent.marketing.historical_sources import collect
                            prepared = await collect(client, runtime.context, prepared)
                        if state.get("marketing_report_presentation_version")==1:
                            from commerce_agent.marketing.scoped import build_scoped_catalog
                            prepared=build_scoped_catalog({**prepared,"scoped_report_presentation_version":1})
                    elif getattr(request,"capability",None)=="latest_native_diagnostic":
                        from commerce_agent.marketing.composite import collect_marketing_composite
                        from commerce_agent.marketing.composite_storage import build_composite_storage
                        from commerce_agent.marketing.composite_analysis import build_composite_catalog
                        native=await collect_marketing_composite(client,runtime.context,request,policy_version=dependencies.composite_policy_version)
                        prepared=(build_composite_storage(native) if dependencies.composite_storage_version==1
                            else build_composite_catalog(native))
                    elif getattr(request, "capability", None) == "sku_diagnostic":
                        from commerce_agent.marketing.diagnostics import collect_marketing_diagnostic
                        from commerce_agent.marketing.diagnostic_analysis import build_diagnostic_catalog
                        prepared = build_diagnostic_catalog(await collect_marketing_diagnostic(client, runtime.context, request,
                            allow_debug=dependencies.enable_diagnostic_debug))
                        from commerce_agent.debug_privacy import development_debug_enabled
                        if development_debug_enabled():
                            from commerce_agent.marketing.debug_export import assert_diagnostic_debug_prepared
                            assert_diagnostic_debug_prepared(prepared)
                    else:
                        prepared = await collect_marketing_evidence(client, runtime.context, request)
            if dependencies.enable_price_feedback:
                prepared["marketing_price_feedback_version"] = 1
            return {**_status(state, "collect_evidence", 2), "prepared_marketing": prepared}
        except Exception:
            # Provider/transport exception bodies may contain secrets. Static
            # codes only; cancellation is BaseException and propagates.
            return {**_status(state, "collect_evidence", 2, "failed", "needs_data"), "marketing_failure": "needs_data"}

    async def analyze(state: MarketingState, runtime: Runtime[CommerceRuntimeContext]):
        if state.get("marketing_skipped") or state.get("marketing_failure"):
            return {}
        prepared = state["prepared_marketing"]
        try:
            instructions = build_marketing_instructions(workflow_version=dependencies.workflow_version,
                user_query=prepared["request"]["user_query"], today=prepared["today"],
                memory_block=memory_block(state,runtime,"marketing",dependencies.workflow_version))
            if dependencies.enable_investigation:
                # Preserve the complete numeric baseline without an unnecessary
                # model call that only selects already mandatory observations.
                from commerce_agent.marketing.scoped import assemble_scoped_result
                candidate = assemble_scoped_result({"summary_claim_indices": [0],
                    "findings": [{"claim_indices": [0]}]}, prepared)
            else:
                async with asyncio.timeout(dependencies.model_timeout):
                    candidate = await dependencies.model.analyze(instructions=instructions, prepared=deepcopy(prepared))
            validated = _validate(candidate, prepared)
            return {**_status(state, "analyze", 3), "marketing_candidate": validated.model_dump(mode="json")}
        except Exception:
            return {**_status(state, "analyze", 3, "failed", "analysis_rejected"), "marketing_failure": "analysis_rejected"}

    async def investigate(state: MarketingState, runtime: Runtime[CommerceRuntimeContext]):
        if state.get("marketing_skipped") or state.get("marketing_failure"):
            return {}
        if dependencies.enable_research_v6:
            from commerce_agent.marketing.research_runtime import investigate as investigate_v6
            return await investigate_v6(state, runtime.context, dependencies)
        validate_decision = investigation_api.validate_decision
        decisions = state.get("marketing_investigation_decisions", [])
        reads = state.get("marketing_investigation_reads", [])
        try:
            if len(decisions) >= 3:
                raise ValueError("investigation model budget exhausted")
            remaining = 2 - len(decisions)
            async with asyncio.timeout(dependencies.model_timeout):
                raw = await dependencies.investigator.investigate(prepared=deepcopy(state["prepared_marketing"]),
                    reads=deepcopy(reads), previous=deepcopy(decisions[-1]) if decisions else None,
                    remaining_rounds=remaining, history=memory_block(state, runtime, "marketing", dependencies.workflow_version))
            decision = validate_decision(raw, state["prepared_marketing"], reads,
                decisions[-1] if decisions else None, remaining_rounds=remaining)
            if state.get("marketing_investigation_report_style", "technical") == "readable":
                from commerce_agent.marketing.feedback_readable import validate_readable_decisions
                validate_readable_decisions([decision.model_dump(mode="json")])
            if not decision.checks and any(h.assessment == "untested" for h in decision.hypotheses):
                raise ValueError("stopped investigation must qualify missing checks")
            if dependencies.feedback_report_style == "compact":
                investigation_api.validate_journal(state["prepared_marketing"],
                    [*decisions, decision.model_dump(mode="json")], reads)
            return {**_status(state, "investigate", 4 + 2 * len(decisions)),
                "marketing_investigation_decisions": [*decisions, decision.model_dump(mode="json")]}
        except Exception:
            return {"marketing_failure": "investigation_rejected"}

    async def execute_checks(state: MarketingState, runtime: Runtime[CommerceRuntimeContext], *, neighbor_graph=None):
        if dependencies.enable_research_v6:
            from commerce_agent.marketing.research_runtime import execute
            return await execute(state, runtime.context, dependencies, neighbor_graph)
        InvestigationDecision, execute_check = investigation_api.InvestigationDecision, investigation_api.execute_check
        reads = deepcopy(state["marketing_investigation_reads"])
        decision = InvestigationDecision.model_validate(state["marketing_investigation_decisions"][-1])
        try:
            async with asyncio.timeout(dependencies.collection_timeout):
                async with dependencies.client_factory() as client:
                    if feedback_enabled:
                        # One answer per superstep: checkpoint before another
                        # neighbor or model is called. Never replay completed reads.
                        done = {(r["operation"], r["target"]) for r in reads}
                        check = next(c for c in decision.checks if (c.operation, c.target) not in done)
                        try:
                            async with asyncio.timeout(dependencies.collection_timeout - 1):
                                read = await execute_check(client, runtime.context, check, state["prepared_marketing"],
                                    neighbors=dependencies.feedback_neighbors, graph=neighbor_graph)
                        except TimeoutError:
                            read = {**check.model_dump(mode="json"),
                                "arguments": investigation_api.check_arguments(check, state["prepared_marketing"]),
                                "status": "timeout", "envelope": None}
                        if dependencies.enable_feedback_details:
                            from commerce_agent.marketing.feedback_details import fit_read_budget
                            read = fit_read_budget(read, state["prepared_marketing"], reads)
                        reads.append(read)
                    else:
                        for check in decision.checks:
                            reads.append(await execute_check(client, runtime.context, check, state["prepared_marketing"]))
            return {**_status(state, "execute_checks", 3 + 2 * len(state["marketing_investigation_decisions"])),
                "marketing_investigation_reads": reads}
        except Exception as exc:
            from langgraph.errors import GraphBubbleUp, NodeCancelledError
            if isinstance(exc, (GraphBubbleUp, NodeCancelledError)):
                raise
            return {"marketing_failure": "investigation_check_rejected"}

    def route_investigation(state):
        if dependencies.enable_research_v6:
            from commerce_agent.marketing.research_runtime import next_node
            return next_node(state)
        if state.get("marketing_skipped") or state.get("marketing_failure"):
            return "validate_and_publish"
        decisions = state.get("marketing_investigation_decisions", [])
        return pending_check_node(state) if decisions and decisions[-1]["checks"] else "validate_and_publish"

    def pending_check_node(state):
        if not feedback_enabled:
            return "execute_checks"
        done = {(r["operation"], r["target"]) for r in state["marketing_investigation_reads"]}
        op = next(c["operation"] for c in state["marketing_investigation_decisions"][-1]["checks"]
            if (c["operation"], c["target"]) not in done)
        return {"sales": "check_sales", "supply": "check_supply", "price": "check_price"}.get(op, "execute_checks")

    def route_check(state):
        if dependencies.enable_research_v6:
            from commerce_agent.marketing.research_runtime import next_node
            return next_node(state)
        if state.get("marketing_failure"):
            return "validate_and_publish"
        done = {(r["operation"], r["target"]) for r in state["marketing_investigation_reads"]}
        if any((c["operation"], c["target"]) not in done for c in state["marketing_investigation_decisions"][-1]["checks"]):
            return pending_check_node(state)
        return "investigate"

    def validate_and_publish(state: MarketingState, runtime: Runtime[CommerceRuntimeContext]):
        if state.get("marketing_skipped"):
            return {}
        failure = state.get("marketing_failure")
        research_failure = failure if dependencies.enable_research_v6 else None
        if dependencies.enable_research_v6:
            from commerce_agent.marketing.research_runtime import checked
            checked(state, runtime.context)
            if state.get("marketing_candidate") and research_failure:
                failure = None  # retain validated baseline, mark partial + explicit error below
        prepared = state.get("prepared_marketing")
        if not failure:
            try:
                result = _validate(state["marketing_candidate"], prepared)
                if prepared.get("capability")=="latest_marketing_overview":
                    from commerce_agent.marketing.overview import render_overview
                    report=render_overview(result,prepared)
                elif prepared.get("capability")=="scoped_report":
                    from commerce_agent.marketing.scoped import render_scoped_report
                    report=render_scoped_report(result,prepared)
                elif prepared.get("capability")=="latest_native_diagnostic":
                    from commerce_agent.marketing.composite_analysis import render_composite_report
                    report=render_composite_report(result,prepared)
                elif prepared.get("capability") == "sku_diagnostic":
                    from commerce_agent.marketing.diagnostic_analysis import render_diagnostic_report
                    report = render_diagnostic_report(result, prepared)
                else:
                    report = render_marketing_report(result, prepared)
                if dependencies.enable_investigation:
                    publish_investigation = investigation_api.publish_investigation
                    if dependencies.enable_research_v6:
                        from commerce_agent.marketing.research_runtime import publish
                        result, analytical_report = publish(result, prepared, state["marketing_investigation_reads"],
                            state["marketing_investigation_v6"], failure=research_failure)
                    else:
                        result, analytical_report = publish_investigation(result, prepared,
                            state["marketing_investigation_decisions"], state["marketing_investigation_reads"])
                    report = analytical_report + "\n\n" + report
                    if state.get("marketing_investigation_report_style", "technical") == "readable":
                        from commerce_agent.marketing.feedback_readable import render_readable
                        report = render_readable(result, prepared, state["marketing_investigation_decisions"],
                            state["marketing_investigation_reads"])
                        # Keep the full native appendix in the typed result;
                        # the UI summary is the business-facing explanation.
                        result.summary = report + "\n\n## Техническое приложение\n\n" + analytical_report
                        from commerce_agent.marketing.policy import assert_bounded_safe
                        assert_bounded_safe(result.model_dump(mode="json"), max_bytes=524288)
            except Exception:
                failure = "analysis_rejected"
        identity = f"marketing:{runtime.context.request_id}"
        if failure:
            return {**_status(state, "validate_and_publish", 10 if dependencies.enable_investigation else 4, "failed", failure),
                "marketing_candidate": None, "marketing_failure": failure,
                "marketing_analysis": DomainAnalysis(domain=DomainName.MARKETING, revision=state["marketing_revision"],
                    status="failed", summary="Анализ рекламы недоступен: недостаточно проверенных данных или результат не прошёл проверку."),
                "data_gaps": [DataGap(gap_id=f"{identity}:{failure}", domain=DomainName.MARKETING, reason=failure)],
                "errors": [WorkflowError(error_id=f"{identity}:{failure}", code=f"MARKETING_{failure.upper()}",
                    message="Marketing read-only branch did not produce a validated result.", node="marketing.validate_and_publish")]}
        evidence = list(prepared["evidence"])
        if dependencies.enable_investigation:
            investigation_evidence = investigation_api.investigation_evidence
            if dependencies.enable_research_v6:
                from commerce_agent.marketing.research_runtime import evidence as research_evidence
                evidence.extend(research_evidence(prepared, state["marketing_investigation_reads"]).values())
            else:
                evidence.extend(investigation_evidence(prepared, state["marketing_investigation_decisions"],
                    state["marketing_investigation_reads"]).values())
        refs = tuple(dict.fromkeys(item["evidence_ref"] for item in evidence))
        warnings = prepared["warnings"]
        gaps = [DataGap(gap_id=f"{identity}:limitation:{index}", domain=DomainName.MARKETING,
                       reason=warning, blocking=warning.startswith("needs_data:"),
                       status="open" if warning.startswith("needs_data:") else "accepted_limitation")
                for index, warning in enumerate(warnings)]
        gaps.extend(DataGap(gap_id=f"{identity}:model-gap:{index}", domain=DomainName.MARKETING, reason=gap.reason)
                    for index, gap in enumerate(result.data_gaps))
        return {**_status(state, "validate_and_publish", 10 if dependencies.enable_investigation else 4, "failed" if research_failure else "completed", research_failure),
            "marketing_analysis": DomainAnalysis(domain=DomainName.MARKETING, revision=state["marketing_revision"],
                status="partial" if warnings or result.data_gaps else "completed", summary=report,
                result=result, evidence_refs=refs),
            "evidence_matrix": [EvidenceItem.model_validate(item) for item in evidence],
            "data_gaps": gaps, "errors": ([WorkflowError(error_id=f"{identity}:{research_failure}",
                code="MARKETING_RESEARCH_INCOMPLETE", message="Verified data retained; research did not complete.",
                node="marketing.validate_and_publish")] if research_failure else [])}

    builder = StateGraph(MarketingState, input_schema=MarketingInput, output_schema=MarketingOutput,
                         context_schema=CommerceRuntimeContext)
    for name, node in (("prepare", prepare), ("collect_evidence", collect_evidence),
                       ("analyze", analyze), ("validate_and_publish", validate_and_publish)):
        builder.add_node(name, node)
    for left, right in ((START, "prepare"), ("prepare", "collect_evidence"),
                        ("collect_evidence", "analyze"),
                        ("validate_and_publish", END)):
        builder.add_edge(left, right)
    if dependencies.enable_investigation:
        builder.add_node("investigate", investigate)
        builder.add_node("execute_checks", execute_checks)
        builder.add_edge("analyze", "investigate")
        check_nodes = ["execute_checks"]
        if feedback_enabled:
            # Direct free-variable references keep actual nested graphs
            # discoverable for checkpoint inspection and streaming.
            sales_graph = dependencies.feedback_neighbors.sales
            supply_graph = dependencies.feedback_neighbors.supply
            async def check_sales(state: MarketingState, runtime: Runtime[CommerceRuntimeContext]):
                return await execute_checks(state, runtime, neighbor_graph=sales_graph)
            async def check_supply(state: MarketingState, runtime: Runtime[CommerceRuntimeContext]):
                return await execute_checks(state, runtime, neighbor_graph=supply_graph)
            builder.add_node("check_sales", check_sales)
            builder.add_node("check_supply", check_supply)
            check_nodes += ["check_sales", "check_supply"]
            if dependencies.enable_price_feedback:
                price_graph = dependencies.feedback_neighbors.price
                async def check_price(state: MarketingState, runtime: Runtime[CommerceRuntimeContext]):
                    return await execute_checks(state, runtime, neighbor_graph=price_graph)
                builder.add_node("check_price", check_price)
                check_nodes.append("check_price")
            for name in check_nodes:
                builder.add_conditional_edges(name, route_check, [*check_nodes, "investigate", "validate_and_publish"])
        else:
            builder.add_edge("execute_checks", "investigate")
        builder.add_conditional_edges("investigate", route_investigation, [*check_nodes, "investigate", "validate_and_publish"])
    else:
        builder.add_edge("analyze", "validate_and_publish")
    from commerce_agent.marketing.capability_guard import guard_marketing_capabilities
    compiled = guard_marketing_capabilities(builder.compile(name="marketing_subgraph", checkpointer=checkpointer,
        interrupt_after=interrupt_after), enabled=dependencies.enable_diagnostics,
        allow_debug=dependencies.enable_diagnostic_debug,composites=dependencies.enable_composites,scoped=dependencies.enable_scoped_reports,
        scoped_debug=dependencies.enable_scoped_debug,scoped_brief=dependencies.enable_scoped_brief,overview=dependencies.enable_overview,
        investigation=dependencies.enable_investigation, feedback=feedback_enabled, details=dependencies.enable_feedback_details,
        feedback_report_style=dependencies.feedback_report_style, price_feedback=dependencies.enable_price_feedback, research_v6=dependencies.enable_research_v6, historical_sources=dependencies.enable_historical_sources)
    return guard_conversation(compiled,enabled=dependencies.enable_conversation_memory,domain="marketing",workflow=dependencies.workflow_version)
