"""Live-derived semantic failures on redacted synthetic native evidence."""
import asyncio
from copy import deepcopy

import pytest
import httpx

from commerce_agent.sales.claims import SalesClaimSelection, render_sales_claim
from commerce_agent.sales.contracts import DiagnosticFinding
from commerce_agent.sales.policy import SalesRequest
from commerce_agent.sales.publication import validate_sales_publication
from commerce_agent.sales.semantics import SalesRequestedCheck, finding_presentation, render_requested_check
from test_sales_evidence import collect
from test_sales_runtime import FixtureModel
from test_sales_model import response


def candidate_with_finding():
    prepared, _ = collect(SalesRequest(user_query="Отчет продаж"))
    candidate = asyncio.run(FixtureModel().analyze(instructions="fixture", prepared=prepared))
    observation = render_sales_claim(SalesClaimSelection(kind="amount", fact_index=2), prepared)
    candidate.findings = [DiagnosticFinding(factor="Продажи", observation=observation,
        impact="Нужна дополнительная проверка.", cause="Не устанавливается этой проверкой.",
        action=render_requested_check(SalesRequestedCheck(kind="source_completeness", claim_index=2), prepared),
        evidence=[prepared["facts"][2]["evidence_ref"]], confidence="medium", driver_class="secondary_driver")]
    return prepared, candidate


@pytest.mark.parametrize("change", [
    {"factor": "Выручка"}, {"factor": "Динамика продаж"},
    {"driver_class": "primary_driver"}, {"driver_class": "amplifier"},
    {"driver_class": "not_a_driver"}, {"confidence": "high"},
    {"action": "Проверить SKU с заметной просадкой продаж."},
])
def test_valid_amount_does_not_prove_wrong_role_or_asserted_action(change):
    prepared, candidate = candidate_with_finding()
    candidate.findings[0] = DiagnosticFinding.model_validate({**candidate.findings[0].model_dump(mode="json"), **change})
    with pytest.raises(ValueError):
        validate_sales_publication(candidate, prepared)


def test_native_labels_are_service_owned_and_neutral_checks_are_valid():
    prepared, candidate = candidate_with_finding()
    assert finding_presentation(candidate.findings[0].observation, prepared) == ("Продажи", "secondary_driver", "medium")
    candidate.recommendations = [render_requested_check(SalesRequestedCheck(kind="marketing", claim_index=2), prepared)]
    assert validate_sales_publication(candidate, prepared) == candidate


def test_finding_check_cannot_select_another_sku_or_period():
    prepared, candidate = candidate_with_finding()
    candidate.findings[0].action = render_requested_check(SalesRequestedCheck(kind="funnel", claim_index=3), prepared)
    with pytest.raises(ValueError):
        validate_sales_publication(candidate, prepared)
    changed = deepcopy(prepared)
    changed["facts"][0]["sku"] = "SKU-OTHER"
    candidate.findings[0].action = render_requested_check(SalesRequestedCheck(kind="funnel", claim_index=0), changed)
    with pytest.raises(ValueError):
        validate_sales_publication(candidate, prepared)


@pytest.mark.parametrize("index,label", [(0, "Выручка"), (2, "Продажи"), (4, "Динамика выручки"), (5, "Динамика продаж")])
def test_labels_match_native_metric_family_and_comparison_kind(index, label):
    from commerce_agent.sales.claims import native_claim_catalog
    prepared, _ = collect(SalesRequest(user_query="Отчет продаж"))
    observation = render_sales_claim(native_claim_catalog(prepared)[index], prepared)
    assert finding_presentation(observation, prepared)[0] == label


@pytest.mark.parametrize("extra", ["factor", "driver_class", "confidence", "action", "cause", "impact"])
def test_provider_cannot_choose_service_owned_presentation(extra):
    from commerce_agent.sales.model import LangChainSalesModel, SalesModelSettings
    prepared, _ = collect(SalesRequest(user_query="Отчет продаж"))
    finding = {"observation": {"kind": "native", "claims": [{"claim_index": 2}]},
               "requested_check": {"kind": "source_completeness", "claim_index": 2}, "evidence_indices": [],
               extra: "provider-owned-control"}
    narrative = {"summary": "", "grounded_claims": [{"claim_index": 2}], "findings": [finding],
                 "recommendations": [], "data_gaps": [], "cross_agent_requests": []}
    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200, json=response(narrative)))) as http:
            model = LangChainSalesModel(SalesModelSettings(api_key="synthetic-model-key"), http_async_client=http)
            with pytest.raises(ValueError):
                await model.analyze(instructions="fixture", prepared=prepared)
    asyncio.run(exercise())


def test_wrong_check_scope_is_rejected_before_checkpoint_and_valid_check_survives():
    from uuid import uuid4
    from langgraph.checkpoint.memory import InMemorySaver
    from commerce_agent.serialization import checkpoint_serializer
    from commerce_agent.sales.runtime import build_sales_subgraph
    from test_sales_runtime import dependencies, invocation
    from test_sales_evidence import CONTEXT
    prepared, candidate = candidate_with_finding()
    async def exercise():
        wrong = candidate.model_dump(mode="json")
        wrong["findings"][0]["action"] = render_requested_check(SalesRequestedCheck(kind="funnel", claim_index=3), prepared)
        graph = build_sales_subgraph(dependencies(FixtureModel(lambda data, source: data.update(findings=wrong["findings"]))),
            checkpointer=InMemorySaver(serde=checkpoint_serializer()))
        config = {"configurable": {"thread_id": str(uuid4())}}
        result = await graph.ainvoke(invocation(), config, context=CONTEXT)
        assert result["sales_analysis"].status == "failed"
        async for snapshot in graph.aget_state_history(config):
            assert snapshot.values.get("sales_candidate") is None
        valid = build_sales_subgraph(dependencies(FixtureModel(lambda data, source: data.update(
            findings=candidate.model_dump(mode="json")["findings"]))))
        result = await valid.ainvoke(invocation(), context=CONTEXT)
        assert result["sales_analysis"].result.findings[0] == candidate.findings[0]
    asyncio.run(exercise())


@pytest.mark.parametrize("role,percentage", [("new_in_current", None), ("lost_in_current", -100),
                                           ("decline_driver", -25), ("growth_driver", 12.5), ("stable", 0)])
def test_native_sql_role_and_percentage_are_preserved_not_promoted_to_causal_attribution(role, percentage):
    def change(path, payload):
        if path == "sku-revenue-change":
            payload["data"]["rows"][0].update(driver_role=role, delta_revenue_pct=percentage)
    prepared, _ = collect(SalesRequest(user_query="Отчет продаж"), change)
    table = prepared["tables"]["sales_decomposition"]
    assert table["rows"][0][table["columns"].index("driver_role")] == role
    assert table["rows"][0][table["columns"].index("delta_revenue_pct")] == percentage
    candidate = asyncio.run(FixtureModel().analyze(instructions="fixture", prepared=prepared))
    assert validate_sales_publication(candidate, prepared).sales_decomposition.rows[0][-2:] == [percentage, role]
    observation = render_sales_claim(SalesClaimSelection(kind="amount", fact_index=0), prepared)
    assert finding_presentation(observation, prepared)[1] == "secondary_driver"


@pytest.mark.parametrize("indices,label", [
    ([4, 3], "Динамика выручки; Продажи"),
    ([1, 5], "Выручка; Динамика продаж"),
])
def test_comparison_does_not_lend_dynamics_to_an_uncompared_metric(indices, label):
    from commerce_agent.sales.claims import native_claim_catalog
    prepared, _ = collect(SalesRequest(user_query="Отчет продаж"))
    catalog = native_claim_catalog(prepared)
    observation = "\n".join(render_sales_claim(catalog[index], prepared) for index in indices)
    assert finding_presentation(observation, prepared) == (label, "secondary_driver", "medium")


@pytest.mark.parametrize("text", ["Продажи отсутствуют.", "Продаж нет."])
def test_free_qualitative_assertion_is_rejected_on_provider_wire(text):
    from commerce_agent.sales.model import LangChainSalesModel, SalesModelSettings
    prepared, _ = collect(SalesRequest(user_query="Отчет продаж"))
    assert prepared["facts"][2]["value"] == 1
    narrative = {"summary": "", "grounded_claims": [], "findings": [
        {"observation": {"kind": "qualitative", "text": text},
         "requested_check": {"kind": "source_completeness", "claim_index": None},
         "evidence_indices": []}], "recommendations": [], "data_gaps": [], "cross_agent_requests": []}
    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req: httpx.Response(200, json=response(narrative)))) as http:
            model = LangChainSalesModel(SalesModelSettings(api_key="synthetic-model-key"), http_async_client=http)
            with pytest.raises(ValueError):
                await model.analyze(instructions="fixture", prepared=prepared)
    asyncio.run(exercise())


@pytest.mark.parametrize("index", [True, "0", -1, 2])
def test_sourced_signal_index_is_strict_and_invocation_bounded(index):
    from commerce_agent.sales.model import narrative_schema
    prepared, _ = collect(SalesRequest(user_query="Отчет продаж"))
    with pytest.raises(ValueError):
        narrative_schema(prepared).model_validate({"summary": "", "findings": [
            {"observation": {"kind": "qualitative", "signal_index": index},
             "requested_check": {"kind": "source_completeness", "claim_index": None},
             "evidence_indices": []}], "recommendations": [], "data_gaps": [], "cross_agent_requests": []})


def test_sourced_selection_signal_is_bound_and_impact_is_service_owned():
    from commerce_agent.sales.model import LangChainSalesModel, SalesModelSettings
    from commerce_agent.sales.semantics import signal_catalog
    prepared, _ = collect(SalesRequest(user_query="Отчет продаж"))
    narrative = {"summary": "", "grounded_claims": [], "findings": [
        {"observation": {"kind": "qualitative", "signal_index": 0},
         "requested_check": {"kind": "source_completeness", "claim_index": None},
         "evidence_indices": []}], "recommendations": [], "data_gaps": [], "cross_agent_requests": []}
    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req: httpx.Response(200, json=response(narrative)))) as http:
            model = LangChainSalesModel(SalesModelSettings(api_key="synthetic-model-key"), http_async_client=http)
            return validate_sales_publication(await model.analyze(instructions="fixture", prepared=prepared), prepared)
    result = asyncio.run(exercise())
    finding = result.findings[0]
    assert finding.observation == signal_catalog(prepared)[0]["statement"]
    assert set(signal_catalog(prepared)[0]["evidence_refs"]) <= set(finding.evidence)
    assert finding.impact == "Нужна дополнительная проверка."
    assert (finding.factor, finding.driver_class.value, finding.confidence) == ("Наблюдение", "secondary_driver", "low")
    finding.observation = "Продажи отсутствуют."
    with pytest.raises(ValueError, match="sourced signal"):
        validate_sales_publication(result, prepared)


def test_same_selection_signal_retains_provenance_from_each_scope():
    from commerce_agent.sales.semantics import signal_catalog
    prepared, candidate = candidate_with_finding()
    first = prepared["overview_signals"][0]
    second = deepcopy(first)
    second["evidence_refs"] = [prepared["facts"][0]["evidence_ref"]]
    prepared["overview_signals"].append(second)
    signal = signal_catalog(prepared)[0]
    candidate.findings = [DiagnosticFinding(factor="Наблюдение", observation=signal["statement"],
        impact="Нужна дополнительная проверка.", cause="Не устанавливается этой проверкой.",
        action="Проверить полноту источника.", evidence=signal["evidence_refs"],
        confidence="low", driver_class="secondary_driver")]
    assert validate_sales_publication(candidate, prepared).findings == candidate.findings


@pytest.mark.parametrize("role", ["summary", "gap_metric", "request_reason", "request_question", "request_check"])
def test_unsupported_absence_cannot_use_another_provider_prose_channel(role):
    from commerce_agent.sales.model import LangChainSalesModel, SalesModelSettings
    prepared, _ = collect(SalesRequest(user_query="Отчет продаж"))
    assert prepared["facts"][2]["value"] == 1
    narrative = {"summary": "", "grounded_claims": [], "findings": [],
                 "recommendations": [], "data_gaps": [], "cross_agent_requests": []}
    false = "Продажи отсутствуют."
    if role == "summary":
        narrative["summary"] = false
    elif role == "gap_metric":
        narrative["data_gaps"] = [{"domain": false}]
    else:
        field = {"request_reason": "reason", "request_question": "question", "request_check": "required_checks"}[role]
        narrative["cross_agent_requests"] = [{"kind": "marketing", "claim_index": None,
            field: [false] if role == "request_check" else false}]
    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req: httpx.Response(200, json=response(narrative)))) as http:
            model = LangChainSalesModel(SalesModelSettings(api_key="synthetic-model-key"), http_async_client=http)
            with pytest.raises(ValueError):
                await model.analyze(instructions="fixture", prepared=prepared)
    asyncio.run(exercise())


@pytest.mark.parametrize("kind", ["marketing", "price", "supply", "finance"])
@pytest.mark.parametrize("claim_index", [None, 0, 3])
def test_delegated_requests_bind_native_scope_and_service_owns_all_prose(kind, claim_index):
    from commerce_agent.sales.model import LangChainSalesModel, SalesModelSettings
    from commerce_agent.sales.semantics import render_delegated_check, render_domain_gap
    prepared, _ = collect(SalesRequest(user_query="Отчет продаж"))
    narrative = {"summary": "", "grounded_claims": [], "findings": [],
                 "recommendations": [], "data_gaps": [{"domain": kind}],
                 "cross_agent_requests": [{"kind": kind, "claim_index": claim_index}]}
    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req: httpx.Response(200, json=response(narrative)))) as http:
            model = LangChainSalesModel(SalesModelSettings(api_key="synthetic-model-key"), http_async_client=http)
            return validate_sales_publication(await model.analyze(instructions="fixture", prepared=prepared), prepared)
    if claim_index is None:
        with pytest.raises(ValueError):
            asyncio.run(exercise())
        return
    result = asyncio.run(exercise())
    assert result.data_gaps[0].model_dump(mode="json") == render_domain_gap(kind)
    request = result.cross_agent_requests[0]
    assert request.model_dump(mode="json") == render_delegated_check(
        SalesRequestedCheck(kind=kind, claim_index=claim_index), prepared)
    request.question = "Продажи отсутствуют."
    with pytest.raises(ValueError):
        validate_sales_publication(result, prepared)


@pytest.mark.parametrize("role", ["summary", "gap_dimension"])
def test_publication_also_rejects_non_schema_provider_prose(role):
    from commerce_agent.sales.contracts import DataGap
    prepared, candidate = candidate_with_finding()
    if role == "summary":
        candidate.summary = "Продажи отсутствуют."
    else:
        candidate.data_gaps = [DataGap(metric="Продажи отсутствуют.",
            reason="Метрика не предоставлена источником.", impact="Нужна дополнительная проверка.")]
    with pytest.raises(ValueError):
        validate_sales_publication(candidate, prepared)
