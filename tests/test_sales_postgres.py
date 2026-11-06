"""Actual local PostgreSQL Sales checkpoints; no service/worker M1.5 changes."""
import asyncio
import json
import os
from uuid import uuid4

import pytest

from commerce_agent.persistence import PersistenceSettings, open_postgres_checkpointer
from commerce_agent.sales.runtime import build_sales_subgraph
from test_sales_runtime import FixtureModel, dependencies, invocation
from test_sales_evidence import CONTEXT


LOCAL_FIXTURE = "postgresql://commerce_agent:commerce-agent-local-only@localhost:55432/commerce_agent"


@pytest.mark.skipif(os.environ.get("COMMERCE_AGENT_DATABASE_URI") != LOCAL_FIXTURE,
                   reason="requires explicitly configured local-only Compose fixture")
@pytest.mark.parametrize("large_report", [False, True])
def test_sales_result_restores_across_pools_without_recollecting_completed_node(large_report):
    async def exercise():
        thread_id = "m3-sales-test-" + str(uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        settings = PersistenceSettings.from_environment()
        first_calls, second_calls = [], []
        from test_sales_evidence import respond
        from test_sales_artifacts import ArtifactTransport, assert_no_raw_checkpoint
        artifacts = ArtifactTransport() if large_report else None
        def initial(req):
            if artifacts is not None:
                first_calls.append(req)
                return artifacts(req)
            return respond(req, seen=first_calls)
        def forbidden(req):
            second_calls.append(req)
            raise AssertionError("completed collector must not be repeated")
        first_model = FixtureModel()
        try:
            async with open_postgres_checkpointer(settings) as saver:
                graph = build_sales_subgraph(dependencies(first_model, initial), checkpointer=saver,
                                             interrupt_after=["collect_evidence"])
                await graph.ainvoke(invocation(), config, context=CONTEXT, durability="sync")
                state = await graph.aget_state(config)
                assert state.next == ("analyze",)
                assert state.values["prepared_sales"]["facts"][1]["value"] == 0
                assert not first_model.calls
            # New pool + graph object reads only durable state, not Python locals.
            async with open_postgres_checkpointer(settings) as saver:
                graph = build_sales_subgraph(dependencies(handler=forbidden), checkpointer=saver)
                result = await graph.ainvoke(None, config, context=CONTEXT, durability="sync")
                assert result["sales_analysis"].result.facts[1].value == 0
                assert result["agent_statuses"]["sales"].status == "completed"
                assert not second_calls and len(first_calls) == (5 if large_report else 4)
            async with open_postgres_checkpointer(settings) as saver:
                graph = build_sales_subgraph(dependencies(handler=forbidden), checkpointer=saver)
                final = await graph.aget_state(config)
                assert final.next == ()
                assert final.values["sales_analysis"] == result["sales_analysis"]
                encoded = json.dumps(final.values["prepared_sales"], ensure_ascii=False)
                assert "raw markdown" not in encoded and "calculations" not in encoded
                assert "synthetic-tools-token" not in encoded
                if artifacts is not None:
                    assert len(artifacts.gets) == 1
                    async for state in graph.aget_state_history(config):
                        assert_no_raw_checkpoint(state.values, artifacts)
        finally:
            async with open_postgres_checkpointer(settings) as saver:
                await saver.adelete_thread(thread_id)  # only this test-owned synthetic thread
    asyncio.run(exercise())


@pytest.mark.skipif(os.environ.get("COMMERCE_AGENT_DATABASE_URI") != LOCAL_FIXTURE,
                   reason="requires explicitly configured local-only Compose fixture")
def test_targeted_sales_restores_native_scopes_across_pools_and_capability_checks():
    from test_sales_targeted_publication import QUERY,native_target,TargetedFixtureModel
    from test_sales_evidence import respond
    from commerce_agent.sales.report import render_sales_report
    from langchain_core.callbacks import BaseCallbackHandler
    class Counter(BaseCallbackHandler):
        def __init__(self): self.calls = 0
        def on_chain_start(self,*args,**kwargs): self.calls += 1
    async def exercise():
        thread_id = "m3-sales-targeted-test-"+str(uuid4())
        config = {"configurable":{"thread_id":thread_id}}
        settings = PersistenceSettings.from_environment()
        seen,model = [],TargetedFixtureModel()
        def initial(req): return respond(req,change=native_target,seen=seen)
        def forbidden(req): raise AssertionError("checkpoint must not recollect")
        try:
            async with open_postgres_checkpointer(settings) as saver:
                graph = build_sales_subgraph(dependencies(model,initial,enable_targeted_reports=True),
                    checkpointer=saver,interrupt_after=["collect_evidence"])
                await graph.ainvoke(invocation(QUERY),config,context=CONTEXT,durability="sync")
                first = await graph.aget_state(config)
                assert first.next == ("analyze",)
                prepared = first.values["prepared_sales"]
                assert prepared["targeted_sales_scopes"][0]["facts"][0]["value"] == 333.33
                assert not model.calls
                read_count = len(seen)
            async with open_postgres_checkpointer(settings) as saver:
                disabled = build_sales_subgraph(dependencies(model,forbidden),checkpointer=saver)
                counter = Counter()
                with pytest.raises(ValueError,match="checkpoint capability"):
                    await disabled.ainvoke(None,{**config,"callbacks":[counter]},context=CONTEXT)
                assert counter.calls == 0 and not model.calls
                enabled = build_sales_subgraph(dependencies(model,forbidden,enable_targeted_reports=True),checkpointer=saver)
                result = await enabled.ainvoke(None,config,context=CONTEXT,durability="sync")
                assert result["errors"] == [] and len(model.calls) == 1 and len(seen) == read_count
            async with open_postgres_checkpointer(settings) as saver:
                final = await build_sales_subgraph(dependencies(model,forbidden,enable_targeted_reports=True),checkpointer=saver).aget_state(config)
                assert final.next == () and final.values["prepared_sales"] == prepared
                analysis = final.values["sales_analysis"]
                assert analysis.summary == render_sales_report(analysis.result,prepared)
                assert len(analysis.result.facts) == 8
        finally:
            async with open_postgres_checkpointer(settings) as saver:
                await saver.adelete_thread(thread_id)  # exact synthetic test-owned target
    asyncio.run(exercise())
