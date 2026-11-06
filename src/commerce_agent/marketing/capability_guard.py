"""Pre-node capability check for persisted Marketing diagnostic checkpoints.

Keep the native CompiledStateGraph/Pregel interface so parent graphs still
discover a real fixed subgraph. No RunnableLambda wrapper, new state or protocol
is introduced. Checked against the package's pinned LangGraph compiler.
"""
from langgraph.graph.state import CompiledStateGraph
from langgraph._internal._constants import CONFIG_KEY_CHECKPOINTER
from langgraph._internal._config import ensure_config
import os

from commerce_agent.debug_privacy import development_debug_enabled


class MarketingCapabilityGraph(CompiledStateGraph):
    _diagnostics_enabled: bool
    _diagnostic_debug_enabled: bool

    def __init__(self, *args, _diagnostics_enabled=False, _diagnostic_debug_enabled=False,_composites_enabled=False,_scoped_reports_enabled=False,_scoped_debug_enabled=False,_scoped_brief_enabled=False,_overview_enabled=False,_investigation_enabled=False,_feedback_enabled=False,_details_enabled=False,_feedback_report_style="technical", _price_feedback_enabled=False, _research_v6_enabled=False, _historical_sources_enabled=False, **kwargs):
        if any(type(v) is not bool for v in (_composites_enabled,_scoped_reports_enabled,_scoped_debug_enabled,_scoped_brief_enabled,_overview_enabled,_investigation_enabled,_feedback_enabled,_details_enabled,_price_feedback_enabled,_research_v6_enabled,_historical_sources_enabled)):
            raise ValueError("Marketing capability flag invalid")
        super().__init__(*args, **kwargs)
        self._diagnostics_enabled = _diagnostics_enabled
        self._diagnostic_debug_enabled = _diagnostic_debug_enabled
        self._composites_enabled=_composites_enabled
        self._scoped_reports_enabled=_scoped_reports_enabled
        self._scoped_debug_enabled=_scoped_debug_enabled
        self._scoped_brief_enabled=_scoped_brief_enabled
        self._overview_enabled=_overview_enabled
        self._investigation_enabled = _investigation_enabled
        self._feedback_enabled = _feedback_enabled
        self._details_enabled = _details_enabled
        if _feedback_report_style not in ("technical", "readable", "compact"):
            raise ValueError("Marketing report style invalid")
        self._feedback_report_style = _feedback_report_style
        self._price_feedback_enabled = _price_feedback_enabled
        self._research_v6_enabled = _research_v6_enabled
        self._historical_sources_enabled = _historical_sources_enabled

    def _assert_entry(self, input):
        from commerce_agent.marketing.overview import contains_overview_source
        from commerce_agent.marketing.research import contains_research
        # Native Command/Send/list inputs are not dictionaries. Inspect their
        # updates before compiler callbacks, without parsing human prose.
        from commerce_agent.source_guard_inputs import invocation_source_values
        if self._investigation_enabled and development_debug_enabled():
            raise ValueError("Marketing investigation source debug is not accepted")
        for values in invocation_source_values(input):
            from commerce_agent.marketing.investigation import contains_investigation
            from commerce_agent.marketing.feedback import contains_feedback
            source = {k:v for k,v in values.items() if k != "messages"}
            from commerce_agent.marketing.feedback_details import contains_details
            if contains_research(source) and not self._research_v6_enabled:
                raise ValueError("Marketing research invocation reader disabled")
            if contains_details(source) and not self._details_enabled:
                raise ValueError("Marketing details invocation reader disabled")
            if contains_feedback(source) and not self._feedback_enabled:
                raise ValueError("Marketing feedback invocation reader disabled")
            if contains_investigation(source) and (not self._investigation_enabled or development_debug_enabled()):
                raise ValueError("Marketing investigation invocation reader disabled")
            if any(key.startswith("marketing_investigation") for key in values):
                raise ValueError("Marketing investigation journal is server-owned")
        if self._overview_enabled and development_debug_enabled():
            raise ValueError("Marketing overview source debug is not accepted")
        for values in invocation_source_values(input):
            source_values = {key: value for key, value in values.items() if key != "messages"}
            if contains_overview_source(source_values) and (not self._overview_enabled or development_debug_enabled()):
                raise ValueError("Marketing overview invocation reader disabled")
        if isinstance(input,dict) and "marketing_report_presentation_version" in input:
            raise ValueError("Marketing presentation version is server-owned")
        if self._scoped_debug_enabled and os.environ.get("COMMERCE_AGENT_ENVIRONMENT")!="development":
            raise ValueError("Marketing scoped debug is development only")
        if self._scoped_reports_enabled and development_debug_enabled():
            if not self._scoped_debug_enabled:raise ValueError("Marketing scoped report debug not accepted")
            if input is not None:
                if not isinstance(input,dict):raise ValueError("Marketing scoped debug input invalid")
                from commerce_agent.marketing.policy import resolve_marketing_request
                from commerce_agent.marketing.scoped_debug_export import assert_scoped_debug_request
                from commerce_agent.models import QueryPlan
                query=next((m.content for m in reversed(input.get("messages",[])) if getattr(m,"type",None)=="human"),None)
                plan=input.get("query_plan")
                constraints=QueryPlan.model_validate(plan).constraints if plan else {}
                request=resolve_marketing_request(query,constraints.get("marketing"),allow_scoped=True)
                assert_scoped_debug_request(request)
        if self._composites_enabled and development_debug_enabled():raise ValueError("Marketing composite debug not accepted")
        if self._diagnostic_debug_enabled and os.environ.get("COMMERCE_AGENT_ENVIRONMENT") != "development":
            raise ValueError("Marketing diagnostic debug is development only")
        if self._diagnostics_enabled and development_debug_enabled():
            if not self._diagnostic_debug_enabled:
                raise ValueError("Marketing diagnostic debug not accepted")
            if input is not None:
                if not isinstance(input,dict):
                    raise ValueError("Marketing diagnostic debug input invalid")
                from commerce_agent.marketing.policy import resolve_marketing_request
                from commerce_agent.marketing.debug_export import assert_diagnostic_debug_request
                from commerce_agent.models import QueryPlan
                messages = input.get("messages",[])
                query = next((m.content for m in reversed(messages) if getattr(m,"type",None)=="human"),None)
                plan = input.get("query_plan")
                constraints = QueryPlan.model_validate(plan).constraints if plan else {}
                request = resolve_marketing_request(query,constraints.get("marketing"),allow_diagnostics=True)
                if getattr(request,"capability",None)=="sku_diagnostic":
                    assert_diagnostic_debug_request(request)

    def _assert_checkpoint_capability(self, state):
        values = state.values
        source = values.get("prepared_marketing") or {}
        if any(key in source for key in ("historical_sources", "historical_sources_version", "coverage_manifest", "coverage_manifest_version")):
            if not self._historical_sources_enabled or development_debug_enabled():
                raise ValueError("Marketing historical sources checkpoint reader disabled")
            from commerce_agent.marketing.historical_sources import validate
            validate(source)
        from commerce_agent.marketing.investigation import validate_journal, contains_investigation
        if contains_investigation(values):
            if values.get("marketing_investigation_report_style", "technical") != self._feedback_report_style:
                raise ValueError("Marketing checkpoint report style mismatch")
            if not self._investigation_enabled or development_debug_enabled():
                raise ValueError("Marketing investigation checkpoint reader disabled")
            expected_version = 6 if self._research_v6_enabled else 5 if self._price_feedback_enabled else 4 if self._feedback_report_style == "compact" else 3 if self._details_enabled else 2 if self._feedback_enabled else 1
            if type(values.get("marketing_investigation_version")) is not int or values["marketing_investigation_version"] != expected_version:
                raise ValueError("Marketing investigation checkpoint version missing")
            if self._feedback_enabled:
                from commerce_agent.marketing.feedback import validate_journal
                if self._details_enabled:
                    from commerce_agent.marketing.feedback_details import api
                    validate_journal = api().validate_journal
                if self._feedback_report_style == "compact":
                    from commerce_agent.marketing.compact import validate_journal
            if values.get("prepared_marketing"):
                if ((values["prepared_marketing"].get("marketing_price_feedback_version") == 1)
                        != self._price_feedback_enabled):
                    raise ValueError("Marketing Price checkpoint capability mismatch")
                if self._research_v6_enabled:
                    from commerce_agent.marketing.research import validate_journal as validate_v6
                    validate_v6(values["prepared_marketing"], values.get("marketing_investigation_reads", []),
                        values.get("marketing_investigation_v6", {}))
                else:
                    validate_journal(values["prepared_marketing"], values.get("marketing_investigation_decisions", []),
                        values.get("marketing_investigation_reads", []))
                if self._feedback_report_style == "readable":
                    from commerce_agent.marketing.feedback_readable import validate_readable_journal
                    validate_readable_journal(values["prepared_marketing"], values.get("marketing_investigation_decisions", []),
                        values.get("marketing_investigation_reads", []))
        elif self._investigation_enabled and values and (values.get("prepared_marketing") or values.get("marketing_candidate")):
            raise ValueError("old paused Marketing run cannot silently enable investigation")
        from commerce_agent.marketing.overview import contains_overview_source, validate_prepared, MarketingOverviewRequest, CAPABILITY
        if contains_overview_source(values):
            if not self._overview_enabled or development_debug_enabled():
                raise ValueError("Marketing overview checkpoint reader disabled")
            prepared = values.get("prepared_marketing")
            request = values.get("marketing_request")
            private_source=contains_overview_source({key:values.get(key) for key in (
                "marketing_request","prepared_marketing","marketing_candidate")})
            if private_source and prepared:
                validate_prepared(prepared)
                if request != prepared["request"]:
                    raise ValueError("Marketing overview checkpoint request changed")
            elif private_source and request and request.get("capability") == CAPABILITY:
                MarketingOverviewRequest.model_validate(request)
            elif private_source:
                # Pending candidate cannot be relabelled ordinary data. A
                # previously published result may coexist with a later turn's
                # ordinary prepared; only its capability check applies here.
                raise ValueError("Marketing overview checkpoint source missing")
        presentation=values.get("marketing_report_presentation_version")
        prepared_presentation=(values.get("prepared_marketing") or {}).get("scoped_report_presentation_version")
        if presentation is not None:
            if type(presentation) is not int or presentation!=1 or not self._scoped_brief_enabled:
                raise ValueError("Marketing brief checkpoint reader disabled")
            if (values.get("marketing_request") or {}).get("capability")!="scoped_report":
                raise ValueError("Marketing brief checkpoint request changed")
        if prepared_presentation is not None or presentation is not None and values.get("prepared_marketing"):
            if (type(prepared_presentation) is not int or prepared_presentation!=1 or presentation!=prepared_presentation
                or not self._scoped_brief_enabled):
                raise ValueError("Marketing brief checkpoint markers changed")
        if self._scoped_debug_enabled and development_debug_enabled():
            from commerce_agent.marketing.scoped_debug_export import assert_scoped_debug_checkpoint
            assert_scoped_debug_checkpoint(state)
        prepared = values.get("prepared_marketing") or {}
        request = values.get("marketing_request") or {}
        scoped=(prepared.get("capability")=="scoped_report" or request.get("capability")=="scoped_report"
            or request.get("scope") in {"single_sku","sku_list"} or bool(request.get("targets"))
            or prepared.get("request",{}).get("capability")=="scoped_report" or "scoped_marketing_policy_version" in prepared
            or "scoped_read_results" in prepared or "anchor_snapshot" in prepared
            or any(f.get("metric","").startswith("scoped.") for f in prepared.get("facts",[]))
            or any(e.get("claim","").startswith("scoped_sku:") for e in prepared.get("evidence",[]))
            or any(c.get("kind")=="scoped_target" for c in prepared.get("claims",[]))
            or "portfolio_campaign_reads" in prepared
            or any(f.get("metric","").startswith("portfolio_campaign.") for f in prepared.get("facts",[]))
            or any(e.get("claim","").startswith("portfolio_campaign_native:") for e in prepared.get("evidence",[]))
            or any(c.get("kind") in {"portfolio_campaign","portfolio_proposal"} for c in prepared.get("claims",[])))
        scoped=scoped or "scoped_debug_clock" in prepared
        # An enrolled debug graph accepts only its own validated persisted
        # profile. Deleting every recognizable marker must not turn unknown
        # source prose into an ordinary portfolio checkpoint before callbacks.
        if (self._scoped_debug_enabled and development_debug_enabled()
            and (prepared or request) and not scoped):
            raise ValueError("Marketing scoped debug checkpoint source unknown")
        if scoped:
            if not self._scoped_reports_enabled or development_debug_enabled() and not self._scoped_debug_enabled:
                raise ValueError("Marketing scoped checkpoint capability disabled")
            if development_debug_enabled():
                from commerce_agent.marketing.scoped_debug_export import assert_scoped_debug_request,assert_scoped_debug_prepared
                assert_scoped_debug_prepared(prepared,require_clock=True) if prepared else assert_scoped_debug_request(request)
            if prepared:
                from commerce_agent.marketing.scoped import build_scoped_catalog,canonical
                if canonical(build_scoped_catalog(prepared))!=canonical(prepared):raise ValueError("Marketing scoped checkpoint changed")
        composite_evidence=any(":marketing.deep_diagnostic:" in e.get("evidence_ref","") for e in prepared.get("evidence",[]))
        composite_facts=any(":marketing.deep_diagnostic:" in f.get("evidence_ref","") for f in prepared.get("facts",[]))
        snapshot_context=(not scoped and ("snapshot_context" in prepared or "snapshot_context_mode" in prepared
            or any(f.get("metric","").startswith("snapshot.") for f in prepared.get("facts",[]))))
        if prepared.get("capability")=="latest_native_diagnostic" or request.get("capability")=="latest_native_diagnostic" or "composite_policy_version" in prepared or "native_envelope" in prepared or "catalog_storage_version" in prepared or composite_evidence or composite_facts or snapshot_context:
            if not self._composites_enabled or development_debug_enabled():raise ValueError("Marketing composite checkpoint capability disabled")
            if prepared:
                from commerce_agent.marketing.composite_storage import load_composite_catalog
                load_composite_catalog(prepared)
        if (prepared.get("capability") == "sku_diagnostic" or request.get("capability") == "sku_diagnostic"):
            if not self._diagnostics_enabled or development_debug_enabled() and not self._diagnostic_debug_enabled:
                raise ValueError("Marketing diagnostic checkpoint capability is not enabled")
            if development_debug_enabled():
                from commerce_agent.marketing.debug_export import assert_diagnostic_debug_request, assert_diagnostic_debug_prepared
                assert_diagnostic_debug_prepared(prepared) if prepared else assert_diagnostic_debug_request(request)

    def _has_checkpoint_boundary(self, config):
        cfg = config or {}
        if "thread_id" not in cfg.get("configurable", {}):
            return False
        # The compiler resolves an inherited parent saver through config. This
        # pinned helper avoids treating checkpointer=None as always stateless.
        return cfg.get("configurable", {}).get(CONFIG_KEY_CHECKPOINTER, self.checkpointer) not in (None, False)

    def _assert_research_owner(self, snapshot, effective, context):
        journal = snapshot.values.get("marketing_investigation_v6")
        if not journal:
            return
        from langgraph._internal._constants import CONFIG_KEY_RUNTIME
        from commerce_agent.context import CommerceRuntimeContext
        from commerce_agent.marketing.research import assert_owner
        if context is None:
            context = getattr(effective.get("configurable", {}).get(CONFIG_KEY_RUNTIME), "context", None)
        if context is None:
            raise ValueError("Marketing research owner context missing")
        assert_owner(journal, CommerceRuntimeContext.model_validate(context))

    async def astream(self, input, config=None, **kwargs):
        self._assert_entry(input)
        effective = ensure_config(self.config, config)
        if self._has_checkpoint_boundary(effective):
            snapshot = await super().aget_state(effective)
            self._assert_checkpoint_capability(snapshot)
            self._assert_research_owner(snapshot, effective, kwargs.get("context"))
        async for item in super().astream(input, config=config, **kwargs):
            yield item

    def stream(self, input, config=None, **kwargs):
        self._assert_entry(input)
        effective = ensure_config(self.config, config)
        if self._has_checkpoint_boundary(effective):
            snapshot = super().get_state(effective)
            self._assert_checkpoint_capability(snapshot)
            self._assert_research_owner(snapshot, effective, kwargs.get("context"))
        yield from super().stream(input, config=config, **kwargs)


def guard_marketing_capabilities(compiled, *, enabled, allow_debug=False,composites=False,scoped=False,scoped_debug=False,scoped_brief=False,overview=False,investigation=False,feedback=False,details=False,feedback_report_style="technical",price_feedback=False,research_v6=False, historical_sources=False):
    if type(research_v6) is not bool or research_v6 and (not details or feedback_report_style != "compact"):
        raise ValueError("Research v6 requires compact details")
    if type(price_feedback) is not bool or price_feedback and (not details or feedback_report_style != "compact"):
        raise ValueError("Price feedback requires compact details")
    if type(details) is not bool or details and not feedback:
        raise ValueError("Marketing details require feedback")
    if type(feedback) is not bool or feedback and not investigation:
        raise ValueError("Marketing feedback requires investigation")
    if type(investigation) is not bool or investigation and (not scoped or development_debug_enabled()):
        raise ValueError("Marketing investigation requires scoped mode with debug disabled")
    if type(overview) is not bool:raise ValueError("Marketing overview capability flag invalid")
    if overview and development_debug_enabled():raise ValueError("Marketing overview source debug is not accepted")
    if type(scoped_brief) is not bool or scoped_brief and not scoped:
        raise ValueError("Marketing brief presentation requires scoped capability")
    if type(scoped_debug) is not bool:raise ValueError("Marketing scoped debug flag invalid")
    if scoped_debug and (not scoped or enabled or composites or overview):raise ValueError("Marketing scoped debug requires isolated scoped capability")
    if scoped_debug and os.environ.get("COMMERCE_AGENT_ENVIRONMENT")!="development":raise ValueError("Marketing scoped debug is development only")
    if scoped and development_debug_enabled() and not scoped_debug:raise ValueError("Marketing scoped report debug not accepted")
    if composites and development_debug_enabled():raise ValueError("Marketing composite debug not accepted")
    if allow_debug and os.environ.get("COMMERCE_AGENT_ENVIRONMENT") != "development":
        raise ValueError("Marketing diagnostic debug is development only")
    if enabled and development_debug_enabled() and not allow_debug:
        raise ValueError("Marketing diagnostic debug not accepted")
    # Preserve the exact compiler-owned instance and all channels/nodes/saver;
    # change only the entry methods through a compatible native subclass.
    # Native get_graph/get_subgraphs, serialization and parent discovery remain.
    if type(compiled) is not CompiledStateGraph:
        raise TypeError("unexpected Marketing compiler type")
    compiled.__class__ = MarketingCapabilityGraph
    compiled._diagnostics_enabled = enabled
    compiled._diagnostic_debug_enabled = allow_debug
    compiled._composites_enabled=composites
    compiled._scoped_reports_enabled=scoped
    compiled._scoped_debug_enabled=scoped_debug
    compiled._scoped_brief_enabled=scoped_brief
    compiled._overview_enabled=overview
    compiled._investigation_enabled=investigation
    compiled._feedback_enabled=feedback
    compiled._details_enabled=details
    if feedback_report_style not in ("technical", "readable", "compact") or feedback_report_style in ("readable", "compact") and not details:
        raise ValueError("Marketing report style invalid")
    compiled._feedback_report_style=feedback_report_style
    compiled._price_feedback_enabled=price_feedback
    compiled._research_v6_enabled=research_v6
    compiled._historical_sources_enabled=historical_sources
    return compiled
