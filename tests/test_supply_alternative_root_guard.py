"""M3 new source safety before root callbacks; no network/provider or protocol."""
import asyncio
from copy import deepcopy
from uuid import uuid4
from typing_extensions import TypedDict

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.graph import START,END,StateGraph
from langgraph.checkpoint.memory import InMemorySaver

from commerce_agent.root_source_guard import guard_root_sources,contains_supply_alternatives
from commerce_agent.supply.alternatives import METRIC,TABLE_TITLE
from commerce_agent.serialization import checkpoint_serializer


class GuardState(TypedDict,total=False):
    supply_analysis:dict
    evidence_matrix:list


def published():
    return {"supply_analysis":{"result":{"facts":[],"tables":[{"title":TABLE_TITLE,"columns":[],"rows":[]}]}}}


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    monkeypatch.setenv("COMMERCE_AGENT_ENVIRONMENT","development")
    for key in ("LANGSMITH_TRACING","COMMERCE_AGENT_LANGSMITH_TRACING","COMMERCE_AGENT_LANGSMITH_DEBUG","LANGFUSE_ENABLED"):
        monkeypatch.setenv(key,"false")


class Callback(BaseCallbackHandler):
    def __init__(self):self.seen=[]
    def on_chain_start(self,*args,**kwargs):self.seen.append(True)


def compile_graph(saver,enabled):
    builder=StateGraph(GuardState)
    builder.add_node("publish",lambda state:published())
    builder.add_edge(START,"publish");builder.add_edge("publish",END)
    return guard_root_sources(builder.compile(checkpointer=saver),supply_alternatives_enabled=enabled)


@pytest.mark.parametrize("signature",["table","fact","evidence","private"])
def test_public_or_private_source_signature_detected(signature):
    state=published()
    if signature=="fact":state={"supply_analysis":{"result":{"facts":[{"metric":METRIC}],"tables":[]}}}
    if signature=="evidence":state={"evidence_matrix":[{"evidence_ref":"supply:id:turnover-alternatives:native"}]}
    if signature=="private":state={"prepared_supply":{"alternative_methods_policy_version":1}}
    assert contains_supply_alternatives(state)
    assert not contains_supply_alternatives({"messages":["ordinary question"]})


@pytest.mark.parametrize("sync",[False,True])
@pytest.mark.parametrize("copied",[False,True])
def test_disabled_root_reader_rejects_published_checkpoint_before_callbacks(sync,copied):
    saver=InMemorySaver(serde=checkpoint_serializer());cfg={"configurable":{"thread_id":str(uuid4())}}
    writer=compile_graph(saver,True)
    if sync:writer.invoke({},cfg)
    else:asyncio.run(writer.ainvoke({},cfg))
    reader=compile_graph(saver,False);callback=Callback()
    reader=reader.with_config(callbacks=[callback])
    if copied:reader=reader.copy()
    with pytest.raises(ValueError,match="checkpoint capability"):
        if sync:reader.invoke(None,cfg)
        else:asyncio.run(reader.ainvoke(None,cfg))
    assert callback.seen == []


@pytest.mark.parametrize("copied",[False,True])
def test_debug_toggle_rejected_before_checkpoint_read_or_callbacks(monkeypatch,copied):
    graph=compile_graph(InMemorySaver(serde=checkpoint_serializer()),True)
    callback=Callback();graph=graph.with_config(callbacks=[callback])
    if copied:graph=graph.copy()
    monkeypatch.setenv("COMMERCE_AGENT_LANGSMITH_DEBUG","true")
    with pytest.raises(ValueError,match="source debug"):
        asyncio.run(graph.ainvoke({}, {"configurable":{"thread_id":str(uuid4())}}))
    assert callback.seen == []


def test_source_cannot_be_injected_as_invocation_input():
    graph=compile_graph(None,False);callback=Callback()
    with pytest.raises(ValueError,match="invocation input"):
        asyncio.run(graph.ainvoke(published(),{"callbacks":[callback]}))
    assert callback.seen == []


def test_root_class_with_conversation_keeps_both_guards_on_copy():
    from commerce_agent.conversation_guard import guard_conversation
    builder=StateGraph(GuardState);builder.add_node("publish",lambda _:published())
    builder.add_edge(START,"publish");builder.add_edge("publish",END)
    compiled=guard_conversation(builder.compile(),enabled=False,domain="root")
    root=guard_root_sources(compiled,supply_alternatives_enabled=True).with_config({}).copy()
    assert root._conversation_enabled is False and root._supply_alternatives_enabled is True
    assert asyncio.run(root.ainvoke({}))==published()


@pytest.mark.parametrize("sync",[False,True])
def test_private_paused_child_source_rejected_before_root_callbacks(sync):
    class ChildState(GuardState,total=False):
        prepared_supply:dict
    saver=InMemorySaver(serde=checkpoint_serializer());cfg={"configurable":{"thread_id":str(uuid4())}}
    def compile_parent(enabled):
        child=StateGraph(ChildState)
        child.add_node("collect",lambda _: {"prepared_supply":{"alternative_methods_policy_version":1}})
        child.add_node("publish",lambda _: published())
        child.add_edge(START,"collect");child.add_edge("collect","publish");child.add_edge("publish",END)
        parent=StateGraph(GuardState)
        parent.add_node("supply",child.compile(interrupt_after=["collect"]))
        parent.add_edge(START,"supply");parent.add_edge("supply",END)
        return guard_root_sources(parent.compile(checkpointer=saver),supply_alternatives_enabled=enabled)
    writer=compile_parent(True)
    if sync:
        writer.invoke({},cfg);state=writer.get_state(cfg,subgraphs=True)
    else:
        async def start():
            await writer.ainvoke({},cfg)
            return await writer.aget_state(cfg,subgraphs=True)
        state=asyncio.run(start())
    assert not contains_supply_alternatives(state.values)
    assert contains_supply_alternatives(state.tasks[0].state.values)
    callback=Callback();reader=compile_parent(False).with_config(callbacks=[callback]).copy()
    with pytest.raises(ValueError,match="checkpoint capability"):
        if sync:reader.invoke(None,cfg)
        else:asyncio.run(reader.ainvoke(None,cfg))
    assert callback.seen==[]


@pytest.mark.parametrize("version",["v2","v3"])
def test_event_stream_rejects_unapproved_published_source_before_events(version):
    import inspect
    async def exercise():
        saver=InMemorySaver(serde=checkpoint_serializer());cfg={"configurable":{"thread_id":str(uuid4())}}
        await compile_graph(saver,True).ainvoke({},cfg)
        callback=Callback();events=[]
        reader=compile_graph(saver,False).with_config(callbacks=[callback])
        with pytest.raises(ValueError,match="checkpoint capability"):
            stream=reader.astream_events(None,cfg,version=version)
            if inspect.isawaitable(stream):stream=await stream
            async for event in stream:events.append(event)
        assert callback.seen==events==[]
    asyncio.run(exercise())


@pytest.mark.parametrize("control",["dict","update","tuples","batch","resume","resume_map","send"])
def test_direct_supply_private_control_injection_stops_before_callbacks(control):
    from langgraph.types import Command, Send
    from commerce_agent.supply.capability_guard import guard_supply_capabilities
    class ChildState(GuardState,total=False):
        prepared_supply:dict
    value={"prepared_supply":{"alternative_methods_policy_version":1}}
    if control=="update":value=Command(update=value)
    elif control=="tuples":value=Command(update=list(value.items()))
    elif control=="batch":value=[Command(update={}),Command(update=value)]
    elif control=="resume":value=Command(resume=value)
    elif control=="resume_map":value=Command(resume={"interrupt-id":value})
    elif control=="send":value=Command(goto=Send("publish",value))
    builder=StateGraph(ChildState)
    builder.add_node("publish",lambda _:published())
    builder.add_edge(START,"publish");builder.add_edge("publish",END)
    graph=guard_supply_capabilities(builder.compile(),enabled=False)
    callback=Callback()
    with pytest.raises(ValueError,match="private source input"):
        asyncio.run(graph.with_config(callbacks=[callback]).copy().ainvoke(value))
    assert callback.seen==[]


@pytest.mark.parametrize("shape",["facts","fact","table","evidence"])
def test_direct_supply_raw_resume_source_stops_before_callbacks(shape):
    from langgraph.types import Command
    from commerce_agent.supply.capability_guard import guard_supply_capabilities
    fact={"metric":METRIC}
    value={"facts":[fact]} if shape=="facts" else fact if shape=="fact" else {
        "title":TABLE_TITLE} if shape=="table" else {"evidence_ref":"supply:id:turnover-alternatives:native"}
    builder=StateGraph(GuardState)
    builder.add_node("publish",lambda _:published())
    builder.add_edge(START,"publish");builder.add_edge("publish",END)
    graph=guard_supply_capabilities(builder.compile(),enabled=False)
    callback=Callback()
    with pytest.raises(ValueError,match="published alternative source"):
        asyncio.run(graph.ainvoke(Command(resume={"interrupt-id":value}),{"callbacks":[callback]}))
    assert callback.seen==[]
