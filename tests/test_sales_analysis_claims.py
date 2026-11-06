"""Substantive Sales observations select native facts, never generate numbers."""
import asyncio
from copy import deepcopy
import json

import httpx
import pytest

from commerce_agent.sales.claims import SalesClaimSelection, render_sales_claim
from commerce_agent.sales.contracts import DiagnosticFinding, EvidenceFact
from commerce_agent.sales.model import LangChainSalesModel, SalesModelSettings
from commerce_agent.sales.policy import SalesRequest
from commerce_agent.sales.publication import validate_sales_publication
from test_sales_evidence import collect
from test_sales_model import response
from test_sales_runtime import FixtureModel


def test_native_sales_units_are_available_without_recalculating_or_losing_zero():
    prepared, _ = collect(SalesRequest(user_query="Отчет продаж"))
    facts = {fact["metric"]: fact for fact in prepared["facts"]}
    assert facts["current_sales"]["value"] == 1
    assert facts["previous_sales"]["value"] == 0
    assert facts["current_sales"]["unit"] == "units"
    assert render_sales_claim(SalesClaimSelection(kind="comparison", fact_index=2,
        previous_fact_index=3), prepared).startswith("Текущие продажи SKU SKU-A")
    assert "выше предыдущих" in render_sales_claim(SalesClaimSelection(
        kind="comparison", fact_index=2, previous_fact_index=3), prepared)


@pytest.mark.parametrize("current,previous,expected", [(0, 0, "равны"), (0, 2, "ниже"), (2, 0, "выше")])
def test_native_unit_comparison_preserves_zero_direction_and_dimensions(current, previous, expected):
    prepared, _ = collect(SalesRequest(user_query="Отчет продаж"))
    prepared["facts"][2]["value"], prepared["facts"][3]["value"] = current, previous
    text = render_sales_claim(SalesClaimSelection(kind="comparison", fact_index=2, previous_fact_index=3), prepared)
    assert expected in text
    with pytest.raises(ValueError):
        render_sales_claim(SalesClaimSelection(kind="comparison", fact_index=2, previous_fact_index=1), prepared)
    prepared["facts"][2]["unit"] = "RUB"
    with pytest.raises(ValueError):
        render_sales_claim(SalesClaimSelection(kind="amount", fact_index=2), prepared)


def test_100_native_revenue_and_unit_pairs_can_be_published_without_truncation():
    def change(path, payload):
        if path == "sku-revenue-change":
            row = payload["data"]["rows"][0]
            payload["data"]["rows"] = [{**row, "nm_id": index + 1, "vendor_code": f"SKU-{index}"}
                                       for index in range(100)]
            payload["data"]["row_count"] = 100
    prepared, _ = collect(SalesRequest(user_query="Отчет продаж"), change)
    candidate = asyncio.run(FixtureModel().analyze(instructions="fixture", prepared=prepared))
    published = validate_sales_publication(candidate, prepared)
    assert len(published.facts) == 400
    assert len(published.sales_decomposition.rows) == 100
    assert all(fact.status == "confirmed" for fact in published.facts)
    assert len(json.dumps(prepared, ensure_ascii=False).encode()) <= 131_072
    from commerce_agent.sales.claims import native_claim_catalog
    from commerce_agent.sales.semantics import signal_catalog
    model_input = {**prepared, "claim_catalog": [[claim.kind, claim.fact_index, claim.previous_fact_index]
                    for claim in native_claim_catalog(prepared)],
                   "signal_catalog": [signal["statement"] for signal in signal_catalog(prepared)]}
    assert len(json.dumps(model_input, ensure_ascii=False, allow_nan=False).encode()) <= 131_072
    assert len(json.dumps(published.model_dump(mode="json"), ensure_ascii=False).encode()) <= 262_144
    from commerce_agent.models import DomainAnalysis, DomainName
    from scripts.check_sales_development import _validate_analysis_budget
    _validate_analysis_budget(DomainAnalysis(domain=DomainName.SALES, revision=1,
        status="partial", summary=published.summary, result=published))


@pytest.mark.parametrize("role", ["metric", "reason", "impact"])
def test_gap_prose_cannot_persist_an_invented_amount(role):
    from commerce_agent.sales.contracts import DataGap
    prepared, _ = collect(SalesRequest(user_query="Отчет продаж"))
    candidate = asyncio.run(FixtureModel().analyze(instructions="fixture", prepared=prepared))
    fields = {"metric": "Продажи", "reason": "Нет полного источника.", "impact": "Требуется проверка полноты."}
    fields[role] = "Выручка SKU SKU-A за текущий период равна 999999 RUB."
    candidate.data_gaps = [DataGap(**fields)]
    with pytest.raises(ValueError, match="quantitative prose"):
        validate_sales_publication(candidate, prepared)


@pytest.mark.parametrize("role", ["reason", "question", "required_checks"])
def test_delegated_prose_is_not_an_alternative_quantity_channel(role):
    from commerce_agent.sales.contracts import CrossAgentRequest
    prepared, _ = collect(SalesRequest(user_query="Отчет продаж"))
    candidate = asyncio.run(FixtureModel().analyze(instructions="fixture", prepared=prepared))
    fields = {"target_agent": "marketing", "reason": "Нужна дополнительная проверка.",
              "question": "Проверить рекламу", "required_checks": []}
    false = "Выручка равна 999999 RUB."
    fields[role] = [false] if role == "required_checks" else false
    candidate.cross_agent_requests = [CrossAgentRequest(**fields)]
    with pytest.raises(ValueError, match="quantitative prose"):
        validate_sales_publication(candidate, prepared)


def test_invented_gap_never_reaches_graph_result_or_checkpoint():
    from uuid import uuid4
    from langgraph.checkpoint.memory import InMemorySaver
    from commerce_agent.sales.runtime import build_sales_subgraph
    from commerce_agent.serialization import checkpoint_serializer
    from test_sales_runtime import dependencies, invocation
    from test_sales_evidence import CONTEXT
    model = FixtureModel(change=lambda candidate, prepared: candidate.update(data_gaps=[
        {"metric": "Продажи", "reason": "Выручка равна 999999 RUB.", "impact": "Проверить полноту."}]))
    async def exercise():
        graph = build_sales_subgraph(dependencies(model), checkpointer=InMemorySaver(serde=checkpoint_serializer()))
        config = {"configurable": {"thread_id": str(uuid4())}}
        result = await graph.ainvoke(invocation(), config, context=CONTEXT)
        assert result["sales_analysis"].status == "failed" and result["sales_analysis"].result is None
        async for snapshot in graph.aget_state_history(config):
            assert "999999" not in str(snapshot.values)
    asyncio.run(exercise())


@pytest.mark.parametrize("reason", ["Реклама вызвала провал продаж.",
    "Причина провала продаж — рекламный бюджет.", "Реклама — основной драйвер выручки.",
    "Нет данных о рекламе. Это основная причина провала продаж.",
    "Данные о рекламе недоступны. Этот фактор вызвал провал продаж.",
    "Нет данных о рекламе, но рекламный бюджет объясняет провал продаж."])
def test_gap_is_not_a_channel_for_unverified_cross_domain_causes(reason):
    from commerce_agent.sales.contracts import DataGap
    prepared, _ = collect(SalesRequest(user_query="Отчет продаж"))
    candidate = asyncio.run(FixtureModel().analyze(instructions="fixture", prepared=prepared))
    candidate.data_gaps = [DataGap(metric="Реклама", reason=reason, impact="Нужна дополнительная проверка.")]
    with pytest.raises(ValueError, match="foreign-domain"):
        validate_sales_publication(candidate, prepared)


def test_genuine_absence_of_foreign_evidence_remains_a_gap_not_a_cause():
    from commerce_agent.sales.contracts import DataGap
    prepared, _ = collect(SalesRequest(user_query="Отчет продаж"))
    candidate = asyncio.run(FixtureModel().analyze(instructions="fixture", prepared=prepared))
    candidate.data_gaps = [DataGap(metric="Реклама", reason="Нет данных о рекламе.", impact="Нужна дополнительная проверка.")]
    assert validate_sales_publication(candidate, prepared).data_gaps == candidate.data_gaps


def test_provider_wire_indices_are_bounded_by_this_invocation():
    prepared, _ = collect(SalesRequest(user_query="Отчет продаж"))
    def transport(req):
        schema = json.loads(req.content)["text"]["format"]["schema"]
        defs = schema["$defs"]
        assert defs["BoundedSalesFinding"]["properties"]["evidence_indices"]["items"]["maximum"] == len(prepared["evidence"]) - 1
        from commerce_agent.sales.claims import native_claim_catalog
        assert defs["BoundedSalesClaim"]["properties"]["claim_index"]["maximum"] == len(native_claim_catalog(prepared)) - 1
        return httpx.Response(200, json=response({"summary": "", "grounded_claims": [],
            "findings": [], "recommendations": [], "data_gaps": [], "cross_agent_requests": []}))
    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as http:
            model = LangChainSalesModel(SalesModelSettings(api_key="synthetic-model-key"), http_async_client=http)
            await model.analyze(instructions="fixture", prepared=prepared)
    asyncio.run(exercise())


def test_provider_catalog_contains_only_valid_bound_native_pairs():
    from commerce_agent.sales.claims import native_claim_catalog
    prepared, _ = collect(SalesRequest(user_query="Отчет продаж"))
    catalog = native_claim_catalog(prepared)
    assert len(catalog) == 8
    assert {(claim.fact_index, claim.previous_fact_index) for claim in catalog if claim.kind == "comparison"} == {(0, 1), (2, 3)}
    prepared["facts"][1]["sku"] = "SKU-OTHER"
    changed = native_claim_catalog(prepared)
    assert not any(claim.kind == "comparison" and claim.fact_index == 0 for claim in changed)
    assert any(claim.kind == "comparison" and claim.fact_index == 2 for claim in changed)
    for claim in changed:
        render_sales_claim(claim, prepared)


@pytest.mark.parametrize("index", [True, "0", -1, 999])
def test_model_cannot_invent_or_coerce_finding_provenance(index):
    prepared, _ = collect(SalesRequest(user_query="Отчет продаж"))
    narrative = {"summary": "", "grounded_claims": [],
        "findings": [{"observation": {"kind": "qualitative", "signal_index": 0},
            "requested_check": {"kind": "source_completeness", "claim_index": None},
            "evidence_indices": [index]}], "recommendations": [], "data_gaps": [], "cross_agent_requests": []}
    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req: httpx.Response(200, json=response(narrative)))) as http:
            model = LangChainSalesModel(SalesModelSettings(api_key="synthetic-model-key"), http_async_client=http)
            with pytest.raises(ValueError):
                await model.analyze(instructions="fixture", prepared=prepared)
    asyncio.run(exercise())


def test_schema_uses_closed_sourced_signals_not_qualitative_prose():
    from commerce_agent.sales.model import SalesNarrative
    schema = SalesNarrative.model_json_schema()
    native = schema["$defs"]["SalesNativeObservation"]
    qualitative = schema["$defs"]["SalesQualitativeObservation"]
    assert native["additionalProperties"] is False and qualitative["additionalProperties"] is False
    assert set(native["properties"]) == {"kind", "claims"}
    assert set(qualitative["properties"]) == {"kind", "signal_index"}
    assert qualitative["properties"]["signal_index"]["minimum"] == 0
    assert "evidence_indices" in schema["$defs"]["SalesNarrativeFinding"]["properties"]
    assert "evidence" not in schema["$defs"]["SalesNarrativeFinding"]["properties"]


def test_compact_native_facts_restore_exact_legacy_defaults_at_publication():
    prepared, _ = collect(SalesRequest(user_query="Отчет продаж"))
    assert "period_start" not in prepared["facts"][0]
    candidate = asyncio.run(FixtureModel().analyze(instructions="fixture", prepared=prepared))
    published = validate_sales_publication(candidate, prepared)
    assert [fact.model_dump(mode="json") for fact in published.facts] == [
        EvidenceFact.model_validate(fact).model_dump(mode="json") for fact in prepared["facts"]]
    assert published.facts[1].value == 0


@pytest.mark.parametrize("bad_context", [False, True])
def test_model_renders_grounded_finding_observation_and_binds_its_evidence(bad_context):
    prepared, _ = collect(SalesRequest(user_query="Отчет продаж"))
    baseline = deepcopy(prepared)
    narrative = {"summary": "", "grounded_claims": [
        {"claim_index": 4}],
        "findings": [{"observation": {"kind": "native",
            "claims": [{"claim_index": 0}],
            **({"text": "Неверно:"} if bad_context else {})},
            "requested_check": {"kind": "source_completeness", "claim_index": 0}, "evidence_indices": []}],
        "recommendations": [], "data_gaps": [], "cross_agent_requests": []}
    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req: httpx.Response(200, json=response(narrative)))) as http:
            model = LangChainSalesModel(SalesModelSettings(api_key="synthetic-model-key"), http_async_client=http)
            if bad_context:
                with pytest.raises(ValueError):
                    await model.analyze(instructions="fixture", prepared=prepared)
            else:
                result = validate_sales_publication(await model.analyze(instructions="fixture", prepared=prepared), prepared)
                finding = result.findings[0]
                assert "100.25 RUB." in finding.observation
                assert prepared["facts"][0]["evidence_ref"] in finding.evidence
                assert (finding.factor, finding.driver_class.value, finding.confidence) == ("Выручка", "secondary_driver", "medium")
                assert "observation_claims" not in finding.model_dump()
    asyncio.run(exercise())
    assert prepared == baseline


def test_grounded_observation_cannot_cite_an_unrelated_existing_evidence_ref():
    prepared, _ = collect(SalesRequest(user_query="Отчет продаж"))
    candidate = asyncio.run(FixtureModel().analyze(instructions="fixture", prepared=prepared))
    candidate.findings = [DiagnosticFinding(factor="Выручка",
        observation=render_sales_claim(SalesClaimSelection(kind="amount", fact_index=0), prepared),
        impact="Требуется проверка.", cause="Первопричина не установлена.", action="Проверить источник.",
        evidence=[prepared["evidence"][0]["evidence_ref"]], confidence="medium", driver_class="not_a_driver")]
    assert candidate.findings[0].evidence[0] != prepared["facts"][0]["evidence_ref"]
    with pytest.raises(ValueError, match="proposition evidence"):
        validate_sales_publication(candidate, prepared)


def test_already_collected_raw_sales_and_funnel_are_not_discarded():
    def change(path, payload):
        if path == "overview":
            payload["data"]["sales_raw_rows"] = [{"nm_id": 1, "vendor_code": "SKU-A",
                "sales_date": "2026-10-08", "brand": None, "subject": None, "sales_count": 0, "revenue": 0}]
            payload["data"]["conversion_watchlist"] = [{"nm_id": 1, "vendor_code": "SKU-A",
                "report_date": "2026-10-07", "brand_name": None, "subject_name": None,
                "open_count": 12, "cart_count": 3, "order_count": 1, "add_to_cart_conversion": 0.25,
                "cart_to_order_conversion": None, "buyout_percent": None}]
    prepared, _ = collect(SalesRequest(user_query="Отчет продаж"), change)
    signals = prepared["overview_signals"][0]
    assert signals["sales_raw_rows"][0]["sales_count"] == 0
    assert signals["sales_raw_rows"][0]["sales_date"] == "2026-10-08"
    assert signals["conversion_watchlist"][0]["open_count"] == 12
    assert signals["conversion_watchlist"][0]["report_date"] == "2026-10-07"
    assert signals["conversion_watchlist"][0]["cart_to_order_conversion"] is None
    assert signals["causal_valid"] is False
    assert signals["evidence_refs"]


def test_daily_projection_preserves_native_orders_funnel_and_completeness():
    def change(path, payload):
        if path == "daily-price-sales-comparison":
            from commerce_agent.adapters.contracts import DailyPriceSalesRow
            fields = {name: None for name in DailyPriceSalesRow.model_fields}
            fields.update(nm_id=1, vendor_code="SKU-A", report_date="2026-10-09",
                price_basis="missing", non_cancelled_order_units=3, realized_sales_units=1,
                open_count=12, cart_count=3, funnel_order_count=2, add_to_cart_conversion=0.25,
                business_day_complete=True, sales_day_complete=False, marketing_day_complete=None)
            payload["data"]["rows"] = [fields]
            payload["data"]["row_count"] = 1
    prepared, _ = collect(SalesRequest(user_query="Продажи SKU SKU-A", scope="single_sku", targets=("SKU-A",)), change)
    daily = next(entry for entry in prepared["deep_dives"] if "daily_rows" in entry)
    assert daily["causal_valid"] is False
    row = daily["daily_rows"][0]
    assert (row["non_cancelled_order_units"], row["funnel_order_count"]) == (3, 2)
    assert row["business_day_complete"] is True and row["sales_day_complete"] is False
    assert row["cart_to_order_conversion"] is None
    assert "cancel_count" not in row
