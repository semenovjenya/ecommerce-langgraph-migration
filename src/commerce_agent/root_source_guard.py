"""M3 source capability guard before root callbacks, including resumed results.

No protocol, worker or persistence behavior is implemented here. Native graph
interfaces/copy/configuration are preserved, as in the domain capability guards.
"""
from langgraph.graph.state import CompiledStateGraph
from langgraph._internal._config import ensure_config
from langgraph._internal._constants import CONFIG_KEY_CHECKPOINTER

from commerce_agent.conversation_guard import ConversationGraph
from commerce_agent.debug_privacy import development_debug_enabled
from commerce_agent.supply.alternatives import METRIC, TABLE_TITLE, has_alternative_source
from commerce_agent.source_guard_inputs import invocation_source_values


def _mapping(value):
    return value.model_dump(mode="python") if hasattr(value,"model_dump") else value


def contains_supply_alternatives(values):
    if not isinstance(values,dict):
        return False
    if (has_alternative_source(values) or values.get("metric") == METRIC
        or values.get("title") == TABLE_TITLE
        or isinstance(values.get("evidence_ref"), str) and "turnover-alternatives:" in values["evidence_ref"]):
        return True
    for key in ("prepared_supply","supply_candidate"):
        candidate=_mapping(values.get(key))
        if isinstance(candidate,dict) and has_alternative_source(candidate):
            return True
    analysis=_mapping(values.get("supply_analysis"))
    result=_mapping(analysis.get("result")) if isinstance(analysis,dict) else None
    if isinstance(result,dict) and has_alternative_source(result):
        return True
    for item in values.get("evidence_matrix",[]):
        item=_mapping(item)
        if isinstance(item,dict) and ":turnover-alternatives:" in item.get("evidence_ref",""):
            return True
    return False


def contains_marketing_overview(values):
    """Inspect source-owned channels, not arbitrary human message text."""
    if not isinstance(values, dict):
        return False
    from commerce_agent.marketing.overview import contains_overview_source
    if ("marketing_overview_policy_version" in values or "overview_read" in values
        or values.get("capability") == "latest_marketing_overview"):
        return True
    if contains_overview_source({key: values[key] for key in ("metric", "claim", "evidence_ref", "title") if key in values}):
        return True
    return any(contains_overview_source(values.get(key)) for key in (
        "marketing_request", "prepared_marketing", "marketing_candidate",
        "marketing_analysis", "evidence_matrix", "facts", "tables", "evidence",
        "claims", "report", "summary",
    ))


class RootSourceGuard:
    __slots__=()

    def __init__(self,*args,_supply_alternatives_enabled=False,_marketing_overview_enabled=False,_marketing_investigation_enabled=False,_marketing_feedback_enabled=False,_marketing_details_enabled=False,_marketing_compact_enabled=False,_marketing_price_enabled=False,_price_history_enabled=False,_marketing_research_v6_enabled=False,**kwargs):
        if any(type(value) is not bool for value in (_supply_alternatives_enabled,_marketing_overview_enabled,_marketing_investigation_enabled,_marketing_feedback_enabled,_marketing_details_enabled,_marketing_compact_enabled,_marketing_price_enabled,_price_history_enabled,_marketing_research_v6_enabled)):
            raise ValueError("invalid root source capability")
        self._root_source_base.__init__(self,*args,**kwargs)
        self._supply_alternatives_enabled=_supply_alternatives_enabled
        self._marketing_overview_enabled=_marketing_overview_enabled
        self._marketing_investigation_enabled=_marketing_investigation_enabled
        self._marketing_feedback_enabled=_marketing_feedback_enabled
        self._marketing_details_enabled=_marketing_details_enabled
        self._marketing_compact_enabled=_marketing_compact_enabled
        self._marketing_price_enabled=_marketing_price_enabled
        self._price_history_enabled=_price_history_enabled
        self._marketing_research_v6_enabled=_marketing_research_v6_enabled

    def _root_source_entry(self,input):
        if (self._price_history_enabled or self._marketing_price_enabled) and development_debug_enabled():
            raise ValueError("Price history source debug is not accepted")
        from commerce_agent.marketing.investigation import contains_investigation
        if self._marketing_investigation_enabled and development_debug_enabled():
            raise ValueError("Marketing investigation source debug is not accepted")
        if self._supply_alternatives_enabled and development_debug_enabled():
            raise ValueError("Supply alternatives source debug is not accepted")
        if self._marketing_overview_enabled and development_debug_enabled():
            raise ValueError("Marketing overview source debug is not accepted")
        for value in invocation_source_values(input):
            from commerce_agent.price.history_guard import contains_history
            if contains_history({key: item for key, item in value.items() if key not in {"messages", "query_plan"}}):
                raise ValueError("Price history evidence is not root invocation input")
            if contains_investigation({k:v for k,v in value.items() if k != "messages"}):
                raise ValueError("Marketing investigation source is not root invocation input")
            if contains_supply_alternatives(value):
                raise ValueError("Supply source is not root invocation input")
            if contains_marketing_overview(value):
                raise ValueError("Marketing overview source is not root invocation input")

    def _root_source_checkpoint(self,snapshot):
        from commerce_agent.marketing.research import contains_research
        if contains_research({k:v for k,v in snapshot.values.items() if k != "messages"}) and (not self._marketing_research_v6_enabled or development_debug_enabled()):
            raise ValueError("Marketing research root checkpoint capability unavailable")
        from commerce_agent.price.history_guard import contains_history
        if contains_history({key: value for key, value in snapshot.values.items() if key != "messages"}) and (not (self._price_history_enabled or self._marketing_price_enabled) or development_debug_enabled()):
            raise ValueError("Price history root checkpoint capability unavailable")
        from commerce_agent.marketing.price_feedback import contains_price_feedback
        if contains_price_feedback({key: value for key, value in snapshot.values.items() if key != "messages"}) and (not self._marketing_price_enabled or development_debug_enabled()):
            raise ValueError("Marketing Price root checkpoint capability unavailable")
        from commerce_agent.marketing.compact import contains_compact
        if contains_compact({key: value for key, value in snapshot.values.items() if key != "messages"}) and not self._marketing_compact_enabled:
            raise ValueError("Marketing compact root checkpoint capability unavailable")
        from commerce_agent.marketing.feedback_details import contains_details
        if contains_details({k:v for k,v in snapshot.values.items() if k != "messages"}):
            if not self._marketing_details_enabled or development_debug_enabled():
                raise ValueError("Marketing details root checkpoint capability unavailable")
        from commerce_agent.marketing.feedback import contains_feedback
        if contains_feedback({k:v for k,v in snapshot.values.items() if k != "messages"}):
            if not self._marketing_feedback_enabled or development_debug_enabled():
                raise ValueError("Marketing feedback root checkpoint capability unavailable")
        from commerce_agent.marketing.investigation import contains_investigation
        if contains_investigation({k:v for k,v in snapshot.values.items() if k != "messages"}):
            if not self._marketing_investigation_enabled or development_debug_enabled():
                raise ValueError("Marketing investigation root checkpoint capability unavailable")
        if contains_supply_alternatives(snapshot.values):
            if not self._supply_alternatives_enabled or development_debug_enabled():
                raise ValueError("Supply alternatives root checkpoint capability unavailable")
        if contains_marketing_overview(snapshot.values):
            if not self._marketing_overview_enabled or development_debug_enabled():
                raise ValueError("Marketing overview root checkpoint capability unavailable")
        # A paused child may already hold an unreviewed source before publishing
        # any public root analysis. Inspect native nested snapshots as well.
        for task in snapshot.tasks:
            child=getattr(task,"state",None)
            if child is not None and hasattr(child,"values") and not isinstance(child,dict):
                self._root_source_checkpoint(child)

    def _root_source_has_saver(self,effective):
        cfg=effective.get("configurable",{})
        return "thread_id" in cfg and cfg.get(CONFIG_KEY_CHECKPOINTER,self.checkpointer) not in (None,False)

    def _root_research_owner(self, snapshot, effective, context):
        if not self._marketing_research_v6_enabled or not snapshot.values:
            return
        from langgraph._internal._constants import CONFIG_KEY_RUNTIME
        from commerce_agent.context import CommerceRuntimeContext
        from commerce_agent.marketing.research import contains_research
        values = snapshot.values
        if not any(key in values for key in ("tenant_id", "user_id")) and not contains_research(values):
            return  # checkpoint before trusted intake
        if context is None:
            context = getattr(effective.get("configurable", {}).get(CONFIG_KEY_RUNTIME), "context", None)
        if context is None:
            raise ValueError("Marketing research root owner context missing")
        verified = CommerceRuntimeContext.model_validate(context)
        # Intake persists these existing public fields from verified BFF context.
        # Completed roots skip intake and their children, so check before callbacks.
        if any(values.get(key) != getattr(verified, key) for key in ("tenant_id", "user_id")):
            raise ValueError("Marketing research root owner mismatch")

    async def astream(self,input,config=None,**kwargs):
        self._root_source_entry(input)
        effective=ensure_config(self.config,config)
        if self._root_source_has_saver(effective):
            snapshot = await self._root_source_base.aget_state(self,effective,subgraphs=True)
            self._root_source_checkpoint(snapshot)
            self._root_research_owner(snapshot, effective, kwargs.get("context"))
        async for value in self._root_source_base.astream(self,input,config=config,**kwargs):
            yield value

    def stream(self,input,config=None,**kwargs):
        self._root_source_entry(input)
        effective=ensure_config(self.config,config)
        if self._root_source_has_saver(effective):
            snapshot = self._root_source_base.get_state(self,effective,subgraphs=True)
            self._root_source_checkpoint(snapshot)
            self._root_research_owner(snapshot, effective, kwargs.get("context"))
        yield from self._root_source_base.stream(self,input,config=config,**kwargs)


class RootSourceGraph(CompiledStateGraph):
    _root_source_base=CompiledStateGraph
    __init__=RootSourceGuard.__init__
    _root_source_entry=RootSourceGuard._root_source_entry
    _root_source_checkpoint=RootSourceGuard._root_source_checkpoint
    _root_source_has_saver=RootSourceGuard._root_source_has_saver
    _root_research_owner=RootSourceGuard._root_research_owner
    astream=RootSourceGuard.astream
    stream=RootSourceGuard.stream


class RootConversationSourceGraph(ConversationGraph):
    _root_source_base=ConversationGraph
    __init__=RootSourceGuard.__init__
    _root_source_entry=RootSourceGuard._root_source_entry
    _root_source_checkpoint=RootSourceGuard._root_source_checkpoint
    _root_source_has_saver=RootSourceGuard._root_source_has_saver
    _root_research_owner=RootSourceGuard._root_research_owner
    astream=RootSourceGuard.astream
    stream=RootSourceGuard.stream


def guard_root_sources(compiled,*,supply_alternatives_enabled=False,marketing_overview_enabled=False,marketing_investigation_enabled=False,marketing_feedback_enabled=False,marketing_details_enabled=False,marketing_compact_enabled=False,marketing_price_enabled=False,price_history_enabled=False,marketing_research_v6_enabled=False):
    if any(type(value) is not bool for value in (supply_alternatives_enabled,marketing_overview_enabled,marketing_investigation_enabled,marketing_feedback_enabled,marketing_details_enabled,marketing_research_v6_enabled)):
        raise ValueError("invalid root source capability")
    classes={CompiledStateGraph:RootSourceGraph,ConversationGraph:RootConversationSourceGraph}
    if type(compiled) not in classes:
        raise TypeError("unexpected root source compiler type")
    compiled.__class__=classes[type(compiled)]
    compiled._supply_alternatives_enabled=supply_alternatives_enabled
    compiled._marketing_overview_enabled=marketing_overview_enabled
    compiled._marketing_investigation_enabled=marketing_investigation_enabled
    compiled._marketing_feedback_enabled=marketing_feedback_enabled
    compiled._marketing_details_enabled=marketing_details_enabled
    compiled._marketing_compact_enabled=marketing_compact_enabled
    compiled._marketing_price_enabled=marketing_price_enabled
    compiled._price_history_enabled=price_history_enabled
    compiled._marketing_research_v6_enabled=marketing_research_v6_enabled
    compiled._root_source_entry(None)
    return compiled
