import asyncio
from copy import deepcopy
import json
from uuid import uuid4

import httpx
import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import NodeCancelledError

from commerce_agent.graph import build_graph
from commerce_agent.models import AgentStatus, DomainAnalysis, DomainName, QueryPlan
from commerce_agent.sales.contracts import SalesDiagnosticResult
from commerce_agent.sales.publication import validate_sales_publication
from commerce_agent.sales.runtime import SalesDependencies, build_sales_subgraph
from commerce_agent.serialization import checkpoint_serializer
from commerce_agent.adapters.business_tools import BusinessToolClient
from test_sales_evidence import SETTINGS, CONTEXT, respond, collect
from commerce_agent.sales.policy import SalesRequest


class FixtureModel:
    """Deterministic provider substitute for wiring, NOT a real-model acceptance."""
    def __init__(self, change=None):
        self.change = change
        self.calls = []

    async def analyze(self, *, instructions, prepared):
        self.calls.append((instructions, deepcopy(prepared)))
        candidate = {"agent_name": "Sales Agent", "summary": "Продажи проверены по данным источника.",
                     "facts": deepcopy(prepared["facts"]), **deepcopy(prepared["tables"]),
                     "findings": [], "recommendations": [], "data_gaps": [], "cross_agent_requests": []}
        if self.change:
            self.change(candidate, prepared)
        return SalesDiagnosticResult.model_validate(candidate)


def dependencies(model=None, handler=respond, **kwargs):
    return SalesDependencies(client_factory=lambda: BusinessToolClient(SETTINGS, transport=httpx.MockTransport(handler)),
                             model=model or FixtureModel(), **kwargs)


def invocation(query="Отчет продаж", plan=None):
    return {"messages": [HumanMessage(content=query)], "query_plan": plan or QueryPlan(required_domains=(DomainName.SALES,))}


def test_four_nodes_use_m2_adapters_and_publish_owned_result_only():
    model = FixtureModel()
    graph = build_sales_subgraph(dependencies(model))
    state = {**invocation(), "agent_statuses": {"finance": AgentStatus(domain=DomainName.FINANCE, attempt=99)},
             "tenant_id": "not-returned", "finance_analysis": "not-returned"}
    result = asyncio.run(graph.ainvoke(state, context=CONTEXT))
    assert set(result) <= {"sales_analysis", "agent_statuses", "evidence_matrix", "data_gaps", "errors"}
    assert set(result["agent_statuses"]) == {"sales"}
    assert result["agent_statuses"]["sales"].status == "completed"
    assert result["sales_analysis"].result.facts[1].value == 0
    assert result["sales_analysis"].status == "partial"  # source completeness explicitly unverified
    assert len(model.calls) == 1 and "M3 Sales execution boundary" in model.calls[0][0]


def test_graph_composition_runs_sales_and_keeps_other_domains_stubbed():
    async def exercise():
        graph = build_graph(checkpointer=InMemorySaver(serde=checkpoint_serializer()), sales_dependencies=dependencies())
        config = {"configurable": {"thread_id": str(uuid4())}}
        result = await graph.ainvoke({"schema_version": 1, "messages": [HumanMessage(content="Отчет продаж")]},
                                    config, context=CONTEXT)
        assert result["sales_analysis"].result is not None
        assert result["finance_analysis"].status == "not_implemented"
        assert result["execution_receipts"] == [] and result["action_tasks"] == []
        readback = await graph.aget_state(config)
        assert readback.values["sales_analysis"] == result["sales_analysis"]
    asyncio.run(exercise())


def test_next_request_in_same_thread_increments_sales_revision_and_status():
    async def exercise():
        model = FixtureModel()
        graph = build_graph(checkpointer=InMemorySaver(serde=checkpoint_serializer()), sales_dependencies=dependencies(model))
        config = {"configurable": {"thread_id": str(uuid4())}}
        for index in (1, 2):
            ctx = CONTEXT.model_copy(update={"request_id": f"sales-turn-{index}"})
            result = await graph.ainvoke({"schema_version": 1, "messages": [HumanMessage(content="Отчет продаж")]},
                                        config, context=ctx)
            assert result["sales_analysis"].revision == index
            assert result["agent_statuses"]["sales"].attempt == index * 4
        assert len(model.calls) == 2
        assert len(result["evidence_matrix"]) == 6  # immutable per-run provenance
    asyncio.run(exercise())


@pytest.mark.parametrize("mode", ["table", "fact", "unknown_ref", "secret", "pii", "mutate_prepared", "narrative_number", "foreign_cause",
                                  "date_as_revenue", "revenue_as_profit", "foreign_summary", "unrelated_hypothesis",
                                  "word_quantity", "english_cause"])
def test_model_hallucination_or_sensitive_output_never_published(mode):
    def change(candidate, prepared):
        if mode == "table":
            candidate["sales_overview"]["rows"][0][1] = 99999
        elif mode == "fact":
            candidate["facts"][0]["value"] = 99999
        elif mode == "unknown_ref":
            candidate["facts"][0]["evidence_ref"] = "invented"
        elif mode == "mutate_prepared":
            candidate["facts"][0]["value"] = 99999
            prepared["facts"][0]["value"] = 99999
        elif mode == "narrative_number":
            candidate["summary"] = "Выручка подтверждённо составила 999999999 рублей."
            candidate["recommendations"] = ["Закупите 999999 единиц для роста на 99 процентов."]
        elif mode == "foreign_cause":
            candidate["findings"] = [{"factor": "Реклама", "observation": "Падение продаж",
                "impact": "Изменение выручки", "cause": "Реклама доказанно является основной причиной",
                "action": "Проверить", "confidence": "high", "driver_class": "primary_driver",
                "evidence": [prepared["evidence"][0]["evidence_ref"]]}]
        elif mode == "date_as_revenue":
            candidate["summary"] = "Продажи составили 2026 рублей."
        elif mode == "word_quantity":
            candidate["summary"] = "Выручка выросла вдвое."
        elif mode == "english_cause":
            candidate["summary"] = "Advertising caused the observed sales change."
        elif mode == "revenue_as_profit":
            candidate["summary"] = "Прибыль подтверждённо составила 100.25 рублей."
        elif mode == "foreign_summary":
            candidate["summary"] = "Реклама доказанно является единственной причиной роста продаж."
        elif mode == "unrelated_hypothesis":
            candidate["findings"] = [{"factor": "Реклама", "observation": "Падение продаж",
                "impact": "Изменение выручки", "cause": "Реклама доказанно является причиной; другая гипотеза требует проверки",
                "action": "Проверить", "confidence": "medium", "driver_class": "primary_driver",
                "evidence": [prepared["evidence"][0]["evidence_ref"]]}]
        else:
            candidate["summary"] = "lsv2_synthetic_output_canary_123456" if mode == "secret" else "client@example.com"
    async def exercise():
        saver = InMemorySaver(serde=checkpoint_serializer())
        graph = build_graph(checkpointer=saver, sales_dependencies=dependencies(FixtureModel(change)))
        config = {"configurable": {"thread_id": str(uuid4())}}
        result = await graph.ainvoke({"schema_version": 1, "messages": [HumanMessage(content="Отчет продаж")]},
                                    config, context=CONTEXT)
        assert result["sales_analysis"].status == "failed"
        assert result["sales_analysis"].result is None
        for entry in saver.storage[config["configurable"]["thread_id"]].values():
            assert "lsv2_synthetic_output" not in repr(entry) and "client@example.com" not in repr(entry)
        assert "SALES_ANALYSIS_REJECTED" in {item.code for item in result["errors"]}
    asyncio.run(exercise())


def test_missing_data_does_not_call_model_and_produces_explicit_gap():
    model = FixtureModel()
    result = asyncio.run(build_sales_subgraph(dependencies(model, handler=lambda req: httpx.Response(503))).ainvoke(
        invocation(), context=CONTEXT))
    assert not model.calls and result["sales_analysis"].status == "failed"
    assert result["data_gaps"][0].reason == "needs_data"
    assert result.get("evidence_matrix", []) == []


def test_skipped_sales_does_not_call_tools_or_model():
    def forbidden(req):
        raise AssertionError("unexpected tool")
    model = FixtureModel()
    result = asyncio.run(build_sales_subgraph(dependencies(model, forbidden)).ainvoke(
        invocation(plan=QueryPlan(required_domains=(DomainName.FINANCE,))), context=CONTEXT))
    assert result["agent_statuses"]["sales"].status == "skipped" and not model.calls


def test_cancellation_propagates_no_automatic_retry_or_result():
    async def cancelled(req):
        raise asyncio.CancelledError
    with pytest.raises(NodeCancelledError) as exc:
        asyncio.run(build_sales_subgraph(dependencies(handler=cancelled)).ainvoke(invocation(), context=CONTEXT))
    assert isinstance(exc.value.__cause__, asyncio.CancelledError)


def test_model_deadline_failure_has_safe_error_and_no_result():
    class SlowModel:
        async def analyze(self, **kwargs):
            await asyncio.sleep(1)
    result = asyncio.run(build_sales_subgraph(dependencies(SlowModel(), model_timeout=.01)).ainvoke(invocation(), context=CONTEXT))
    assert result["sales_analysis"].status == "failed"
    assert result["errors"][0].code == "SALES_ANALYSIS_REJECTED"


def test_publication_returns_independent_nested_models():
    prepared, _ = collect(SalesRequest(user_query="Отчет продаж"))
    candidate = asyncio.run(FixtureModel().analyze(instructions="fixture", prepared=prepared))
    published = validate_sales_publication(candidate, prepared)
    candidate.sales_overview.rows[0][1] = 99999
    assert published.sales_overview.rows[0][1] == 1
