"""Fail before callbacks when historical Price checkpoints meet an incompatible reader."""
from langgraph._internal._config import ensure_config
from langgraph._internal._constants import CONFIG_KEY_CHECKPOINTER
from commerce_agent.conversation_guard import ConversationGraph
from commerce_agent.debug_privacy import development_debug_enabled
from commerce_agent.source_guard_inputs import invocation_source_values


def contains_history(value):
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="python")
    if isinstance(value, dict):
        return (value.get("history_version") is not None or value.get("operation") == "history"
            or any(contains_history(item) for item in value.values()))
    if isinstance(value, (list, tuple)):
        return any(contains_history(item) for item in value)
    return isinstance(value, str) and value.startswith("price-history-v1:")


class PriceHistoryGraph(ConversationGraph):
    def __init__(self, *args, _price_history_enabled=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._price_history_enabled = _price_history_enabled

    def _history_entry(self, input):
        if self._price_history_enabled and development_debug_enabled():
            raise ValueError("Price history debug is not accepted")
        for values in invocation_source_values(input):
            source = {key: item for key, item in values.items() if key not in {"messages", "query_plan"}}
            if contains_history(source):
                raise ValueError("Price history evidence is server-owned")

    def _history_checkpoint(self, snapshot):
        values = snapshot.values
        if not contains_history(values):
            return
        if not self._price_history_enabled or development_debug_enabled():
            raise ValueError("Price history checkpoint reader disabled")
        if values.get("prepared_price"):
            from commerce_agent.price.history import verify_history
            verify_history(values["prepared_price"])
            if values["prepared_price"]["request"] != values.get("price_request"):
                raise ValueError("Price history checkpoint request changed")

    def _history_has_saver(self, config):
        return ("thread_id" in config.get("configurable", {})
            and config.get("configurable", {}).get(CONFIG_KEY_CHECKPOINTER, self.checkpointer) not in (None, False))

    async def astream(self, input, config=None, **kwargs):
        self._history_entry(input)
        effective = ensure_config(self.config, config)
        if self._history_has_saver(effective):
            self._history_checkpoint(await super().aget_state(effective))
        async for item in super().astream(input, config=config, **kwargs):
            yield item

    def stream(self, input, config=None, **kwargs):
        self._history_entry(input)
        effective = ensure_config(self.config, config)
        if self._history_has_saver(effective):
            self._history_checkpoint(super().get_state(effective))
        yield from super().stream(input, config=config, **kwargs)


def guard_history(compiled, enabled):
    if type(compiled) is not ConversationGraph:
        raise TypeError("Unexpected Price graph type")
    compiled.__class__ = PriceHistoryGraph
    compiled._price_history_enabled = enabled
    return compiled
