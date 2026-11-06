"""Sales entry guards run before callbacks or loading a target checkpoint.

Preserve the pinned native CompiledStateGraph interface, including parent
subgraph discovery and inherited savers. No protocol/M1.5 change is involved.
"""
from langgraph.graph.state import CompiledStateGraph
from langgraph._internal._constants import CONFIG_KEY_CHECKPOINTER
from langgraph._internal._config import ensure_config

from commerce_agent.debug_privacy import development_debug_enabled


class SalesCapabilityGraph(CompiledStateGraph):
    _targeted_enabled: bool

    def __init__(self,*args,_targeted_enabled=False,_daily_enabled=False,_historical_funnel_enabled=False,**kwargs):
        # Pinned native copy()/with_config() reconstruct from __dict__.
        # Consume the trusted capability instead of letting base **kwargs
        # discard it. This also preserves the false (disabled) case.
        if type(_targeted_enabled) is not bool or type(_daily_enabled) is not bool or type(_historical_funnel_enabled) is not bool:
            raise ValueError("invalid Sales targeted capability")
        super().__init__(*args,**kwargs)
        self._targeted_enabled = _targeted_enabled
        self._daily_enabled = _daily_enabled
        self._historical_funnel_enabled = _historical_funnel_enabled

    def _assert_entry(self):
        if self._targeted_enabled and development_debug_enabled():
            raise ValueError("Sales targeted sources require separate debug acceptance")
        if (self._daily_enabled or self._historical_funnel_enabled) and development_debug_enabled():
            raise ValueError("Sales daily sources require separate debug acceptance")

    def _assert_checkpoint(self, state):
        prepared = state.values.get("prepared_sales") or {}
        if "historical_funnel_version" in prepared or "historical_funnel" in prepared or any(f.get("metric", "").startswith("historical_funnel.") for f in prepared.get("facts", [])):
            if not self._historical_funnel_enabled or development_debug_enabled():
                raise ValueError("Sales historical funnel checkpoint reader disabled")
            from commerce_agent.sales.historical_funnel import native
            native(prepared)
        if "targeted_sales_policy_version" in prepared or "targeted_sales_scopes" in prepared:
            if not self._targeted_enabled or development_debug_enabled():
                raise ValueError("Sales targeted checkpoint capability is not enabled")
            from commerce_agent.sales.targeted import targeted_scopes
            targeted_scopes(prepared)
        if ("scoped_sales_policy_version" in prepared or "scoped_sales_scopes" in prepared
            or prepared.get("request", {}).get("scope") in {"single_sku", "sku_list"}
            and any(f.get("target_type") == "sku" for f in prepared.get("facts", []))):
            if not self._targeted_enabled or development_debug_enabled():
                raise ValueError("Sales scoped checkpoint capability is not enabled")
            from commerce_agent.sales.scoped import scoped_scopes
            scoped_scopes(prepared)
        if ("daily_sales_policy_version" in prepared or "daily_sales_scopes" in prepared
            or any(f.get("target_type")=="sku_day" or f.get("metric","").startswith("daily.") for f in prepared.get("facts",[]))):
            if not self._daily_enabled or development_debug_enabled():
                raise ValueError("Sales daily checkpoint capability is not enabled")
            from commerce_agent.sales.daily import daily_scopes
            daily_scopes(prepared)

    def _has_checkpoint(self, config):
        cfg = config.get("configurable", {})
        return ("thread_id" in cfg and cfg.get(CONFIG_KEY_CHECKPOINTER,self.checkpointer) not in (None,False))

    async def astream(self,input,config=None,**kwargs):
        self._assert_entry()
        effective = ensure_config(self.config,config)
        if self._has_checkpoint(effective):
            self._assert_checkpoint(await super().aget_state(effective))
        async for item in super().astream(input,config=config,**kwargs):
            yield item

    def stream(self,input,config=None,**kwargs):
        self._assert_entry()
        effective = ensure_config(self.config,config)
        if self._has_checkpoint(effective):
            self._assert_checkpoint(super().get_state(effective))
        yield from super().stream(input,config=config,**kwargs)


def guard_sales_capabilities(compiled,*,enabled,daily_enabled=False,historical_funnel_enabled=False):
    if type(compiled) is not CompiledStateGraph:
        raise TypeError("unexpected Sales compiler type")
    compiled.__class__ = SalesCapabilityGraph
    compiled._targeted_enabled = enabled
    compiled._daily_enabled = daily_enabled
    compiled._historical_funnel_enabled = historical_funnel_enabled
    compiled._assert_entry()
    return compiled
