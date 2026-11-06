"""New native Marketing source is closed before root callbacks and events."""
import asyncio
import inspect
from uuid import uuid4

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from commerce_agent.root_source_guard import contains_marketing_overview, guard_root_sources
from commerce_agent.serialization import checkpoint_serializer


class GuardState(TypedDict, total=False):
    marketing_analysis: dict
    evidence_matrix: list


def published():
    return {"marketing_analysis": {"report": "# Marketing — обзор кампаний и бюджета\nNative only"}}


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    monkeypatch.setenv("COMMERCE_AGENT_ENVIRONMENT", "development")
    for key in ("LANGSMITH_TRACING", "COMMERCE_AGENT_LANGSMITH_TRACING",
                "COMMERCE_AGENT_LANGSMITH_DEBUG", "LANGFUSE_ENABLED"):
        monkeypatch.setenv(key, "false")


class Callback(BaseCallbackHandler):
    def __init__(self):
        self.seen = []

    def on_chain_start(self, *args, **kwargs):
        self.seen.append(True)


def compile_graph(saver, enabled):
    builder = StateGraph(GuardState)
    builder.add_node("publish", lambda _: published())
    builder.add_edge(START, "publish")
    builder.add_edge("publish", END)
    return guard_root_sources(builder.compile(checkpointer=saver), marketing_overview_enabled=enabled)


@pytest.mark.parametrize("state", [
    {"marketing_request": {"capability": "latest_marketing_overview"}},
    {"prepared_marketing": {"marketing_overview_policy_version": 1}},
    {"prepared_marketing": {"overview_read": {}}},
    {"marketing_candidate": {"facts": [{"metric": "overview.budget.total_budget"}]}},
    {"marketing_analysis": {"result": {"facts": [{"metric": "overview.totals.avg_ctr"}]}}},
    {"marketing_analysis": {"result": {"facts": [{"evidence_ref": "marketing-overview:request:native"}]}}},
    {"evidence_matrix": [{"claim": "native_marketing_overview"}]},
    published(),
])
def test_root_detects_private_and_marker_stripped_public_overview(state):
    assert contains_marketing_overview(state)
    assert not contains_marketing_overview({"messages": ["overview.budget.total_budget"]})
    callback = Callback()
    with pytest.raises(ValueError, match="invocation input"):
        asyncio.run(compile_graph(None, False).ainvoke(state, {"callbacks": [callback]}))
    assert callback.seen == []


@pytest.mark.parametrize("sync", [False, True])
@pytest.mark.parametrize("copied", [False, True])
def test_disabled_root_reader_stops_before_callbacks(sync, copied):
    saver = InMemorySaver(serde=checkpoint_serializer())
    cfg = {"configurable": {"thread_id": str(uuid4())}}
    writer = compile_graph(saver, True)
    if sync:
        writer.invoke({}, cfg)
    else:
        asyncio.run(writer.ainvoke({}, cfg))
    callback = Callback()
    reader = compile_graph(saver, False).with_config(callbacks=[callback])
    if copied:
        reader = reader.copy()
    with pytest.raises(ValueError, match="checkpoint capability"):
        if sync:
            reader.invoke(None, cfg)
        else:
            asyncio.run(reader.ainvoke(None, cfg))
    assert callback.seen == []


@pytest.mark.parametrize("copied", [False, True])
def test_debug_toggle_rejected_before_checkpoint_callbacks(monkeypatch, copied):
    graph = compile_graph(InMemorySaver(serde=checkpoint_serializer()), True)
    callback = Callback()
    graph = graph.with_config(callbacks=[callback])
    if copied:
        graph = graph.copy()
    monkeypatch.setenv("COMMERCE_AGENT_LANGSMITH_DEBUG", "true")
    with pytest.raises(ValueError, match="source debug"):
        asyncio.run(graph.ainvoke({}, {"configurable": {"thread_id": str(uuid4())}}))
    assert callback.seen == []


@pytest.mark.parametrize("sync", [False, True])
def test_root_inspects_paused_private_child_before_callbacks(sync):
    class ChildState(GuardState, total=False):
        prepared_marketing: dict

    def compile_parent(saver, enabled):
        child = StateGraph(ChildState)
        child.add_node("collect", lambda _: {"prepared_marketing": {"overview_read": {}}})
        child.add_node("publish", lambda _: published())
        child.add_edge(START, "collect")
        child.add_edge("collect", "publish")
        child.add_edge("publish", END)
        parent = StateGraph(GuardState)
        parent.add_node("marketing", child.compile(interrupt_after=["collect"]))
        parent.add_edge(START, "marketing")
        parent.add_edge("marketing", END)
        return guard_root_sources(parent.compile(checkpointer=saver), marketing_overview_enabled=enabled)

    saver = InMemorySaver(serde=checkpoint_serializer())
    cfg = {"configurable": {"thread_id": str(uuid4())}}
    writer = compile_parent(saver, True)
    if sync:
        writer.invoke({}, cfg)
        state = writer.get_state(cfg, subgraphs=True)
    else:
        async def start():
            await writer.ainvoke({}, cfg)
            return await writer.aget_state(cfg, subgraphs=True)
        state = asyncio.run(start())
    assert not contains_marketing_overview(state.values)
    assert contains_marketing_overview(state.tasks[0].state.values)
    callback = Callback()
    reader = compile_parent(saver, False).with_config(callbacks=[callback]).copy()
    with pytest.raises(ValueError, match="checkpoint capability"):
        if sync:
            reader.invoke(None, cfg)
        else:
            asyncio.run(reader.ainvoke(None, cfg))
    assert callback.seen == []


@pytest.mark.parametrize("version", ["v2", "v3"])
def test_event_stream_emits_nothing_before_source_rejection(version):
    async def exercise():
        saver = InMemorySaver(serde=checkpoint_serializer())
        cfg = {"configurable": {"thread_id": str(uuid4())}}
        await compile_graph(saver, True).ainvoke({}, cfg)
        callback, events = Callback(), []
        reader = compile_graph(saver, False).with_config(callbacks=[callback])
        with pytest.raises(ValueError, match="checkpoint capability"):
            stream = reader.astream_events(None, cfg, version=version)
            if inspect.isawaitable(stream):
                stream = await stream
            async for event in stream:
                events.append(event)
        assert callback.seen == events == []
    asyncio.run(exercise())


def test_conversation_and_both_source_flags_survive_copy():
    from commerce_agent.conversation_guard import guard_conversation
    builder = StateGraph(GuardState)
    builder.add_node("publish", lambda _: published())
    builder.add_edge(START, "publish")
    builder.add_edge("publish", END)
    compiled = guard_conversation(builder.compile(), enabled=False, domain="root")
    graph = guard_root_sources(compiled, supply_alternatives_enabled=True,
        marketing_overview_enabled=True).with_config({}).copy()
    assert graph._conversation_enabled is False
    assert graph._supply_alternatives_enabled is True
    assert graph._marketing_overview_enabled is True
    assert asyncio.run(graph.ainvoke({})) == published()


def test_default_graph_remains_source_disabled():
    from commerce_agent.graph import build_graph
    graph = build_graph()
    assert graph._supply_alternatives_enabled is False
    assert graph._marketing_overview_enabled is False


@pytest.mark.parametrize("control", ["update", "tuples", "batch", "resume", "resume_map", "send"])
@pytest.mark.parametrize("domain", ["marketing", "supply"])
def test_native_control_input_cannot_inject_source_before_callbacks(control, domain):
    from langgraph.types import Command, Send
    from commerce_agent.supply.alternatives import METRIC
    source = published() if domain == "marketing" else {
        "supply_analysis": {"result": {"facts": [{"metric": METRIC}]}}}
    if control == "update":
        value = Command(update=source)
    elif control == "tuples":
        key = next(iter(source))
        value = Command(update=[(key, source[key]), (key, {})])
    elif control == "batch":
        value = [Command(update={}), Command(update=source)]
    elif control == "resume":
        value = Command(resume=source)
    elif control == "resume_map":
        value = Command(resume={"interrupt-id": source})
    else:
        value = Command(goto=Send("publish", source))
    callback = Callback()
    graph = compile_graph(None, False).with_config(callbacks=[callback]).copy()
    with pytest.raises(ValueError, match="invocation input"):
        asyncio.run(graph.ainvoke(value))
    assert callback.seen == []


@pytest.mark.parametrize("domain", ["marketing", "supply"])
@pytest.mark.parametrize("shape", ["facts", "fact", "table", "evidence"])
def test_interrupt_value_cannot_hide_marker_stripped_native_source(domain, shape):
    from langgraph.types import Command
    from commerce_agent.supply.alternatives import METRIC, TABLE_TITLE
    fact = {"metric": "overview.totals.avg_ctr" if domain == "marketing" else METRIC}
    table = {"title": "Marketing overview — totals; портфель" if domain == "marketing" else TABLE_TITLE}
    source = {"facts": [fact]} if shape == "facts" else fact if shape == "fact" else table if shape == "table" else {
        "evidence_ref": "marketing-overview:request:native" if domain == "marketing" else "supply:id:turnover-alternatives:native"}
    callback = Callback()
    with pytest.raises(ValueError, match="invocation input"):
        asyncio.run(compile_graph(None, False).ainvoke(Command(resume={"interrupt-id": source}),
            {"callbacks": [callback]}))
    assert callback.seen == []
