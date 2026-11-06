"""Opt-in Supply source guard before callbacks/checkpoint resume."""
from langgraph.graph.state import CompiledStateGraph
from langgraph._internal._constants import CONFIG_KEY_CHECKPOINTER
from langgraph._internal._config import ensure_config

from commerce_agent.debug_privacy import development_debug_enabled
from commerce_agent.supply.alternatives import METRIC, TABLE_TITLE, has_alternative_source, validate_alternative_prepared
from commerce_agent.source_guard_inputs import invocation_source_values


class SupplyCapabilityGraph(CompiledStateGraph):
    def __init__(self, *args, _alternatives_enabled=False, **kwargs):
        if type(_alternatives_enabled) is not bool:
            raise ValueError("invalid Supply alternative capability")
        super().__init__(*args, **kwargs)
        self._alternatives_enabled = _alternatives_enabled

    def _assert_entry(self):
        if self._alternatives_enabled and development_debug_enabled():
            raise ValueError("Supply alternative sources require separate debug acceptance")

    def _assert_input(self, value):
        for item in invocation_source_values(value):
            self._assert_input_values(item)

    def _assert_input_values(self, value):
        if not isinstance(value, dict):
            return
        if any(key.startswith("alternative_methods_") or key in {
                "prepared_supply", "supply_request", "supply_candidate"} for key in value):
            raise ValueError("Supply private source input is not accepted")
        if (has_alternative_source(value) or value.get("metric") == METRIC or value.get("title") == TABLE_TITLE
            or isinstance(value.get("evidence_ref"),str) and "turnover-alternatives:" in value["evidence_ref"]):
            if not self._alternatives_enabled or development_debug_enabled():
                raise ValueError("Supply published alternative source capability is not enabled")
        analysis = value.get("supply_analysis")
        raw = analysis.model_dump(mode="json") if hasattr(analysis, "model_dump") else analysis
        result = raw.get("result") if isinstance(raw, dict) else None
        if isinstance(result, dict) and (any(f.get("metric") == METRIC for f in result.get("facts", []))
            or any(t.get("title") == TABLE_TITLE for t in result.get("tables", []))):
            if not self._alternatives_enabled or development_debug_enabled():
                raise ValueError("Supply published alternative source capability is not enabled")

    def _assert_checkpoint(self, state):
        prepared = state.values.get("prepared_supply") or {}
        if has_alternative_source(prepared):
            if not self._alternatives_enabled or development_debug_enabled():
                raise ValueError("Supply alternative checkpoint capability is not enabled")
            validate_alternative_prepared(prepared)

    def _has_checkpoint(self, config):
        cfg = config.get("configurable", {})
        return "thread_id" in cfg and cfg.get(CONFIG_KEY_CHECKPOINTER, self.checkpointer) not in (None, False)

    async def astream(self, input, config=None, **kwargs):
        self._assert_entry()
        self._assert_input(input)
        effective = ensure_config(self.config, config)
        if self._has_checkpoint(effective):
            self._assert_checkpoint(await super().aget_state(effective))
        async for item in super().astream(input, config=config, **kwargs):
            yield item

    def stream(self, input, config=None, **kwargs):
        self._assert_entry()
        self._assert_input(input)
        effective = ensure_config(self.config, config)
        if self._has_checkpoint(effective):
            self._assert_checkpoint(super().get_state(effective))
        yield from super().stream(input, config=config, **kwargs)


def guard_supply_capabilities(compiled, *, enabled):
    if type(compiled) is not CompiledStateGraph or type(enabled) is not bool:
        raise ValueError("unexpected Supply compiler/capability")
    compiled.__class__ = SupplyCapabilityGraph
    compiled._alternatives_enabled = enabled
    compiled._assert_entry()
    return compiled
