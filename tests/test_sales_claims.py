"""Grounded native propositions, scope/dimension binding and model selection."""
import asyncio
from copy import deepcopy

import httpx
import pytest

from commerce_agent.sales.claims import SalesClaimSelection, render_sales_claim, native_claim_sentences
from commerce_agent.sales.contracts import DiagnosticFinding
from commerce_agent.sales.model import LangChainSalesModel, SalesModelSettings
from commerce_agent.sales.publication import validate_sales_publication
from commerce_agent.sales.policy import SalesRequest
from test_sales_evidence import collect
from test_sales_model import response
from test_sales_runtime import FixtureModel


@pytest.fixture
def prepared():
    return collect(SalesRequest(user_query="Отчет продаж SKU SKU-A", scope="single_sku", targets=("SKU-A",)))[0]


def test_native_amount_and_comparison_keep_zero_unit_scope_and_period(prepared):
    amount = render_sales_claim(SalesClaimSelection(kind="amount", fact_index=1), prepared)
    compare = render_sales_claim(SalesClaimSelection(kind="comparison", fact_index=0, previous_fact_index=1), prepared)
    assert amount == "Предыдущая выручка SKU SKU-A за период 2026-09-26/2026-10-02: 0 RUB."
    assert compare == "Текущая выручка SKU SKU-A за период 2026-10-03/2026-10-09 выше предыдущей за период 2026-09-26/2026-10-02."
    assert {amount, compare} <= native_claim_sentences(prepared)


@pytest.mark.parametrize("current,previous,direction", [(0, 100.25, "ниже"), (0, 0, "равна"),
                                                       (100.25, 100.25, "равна"), (-10, -20, "выше")])
def test_comparison_does_not_divide_by_zero_or_invent_rates(prepared, current, previous, direction):
    prepared["facts"][0]["value"], prepared["facts"][1]["value"] = current, previous
    text = render_sales_claim(SalesClaimSelection(kind="comparison", fact_index=0, previous_fact_index=1), prepared)
    assert f" {direction} " in text and "%" not in text


@pytest.mark.parametrize("mode", ["missing_fact", "mixed_sku", "mixed_unit", "mixed_source", "mixed_ref",
                                  "nonadjacent", "unequal_period", "wrong_metric", "unconfirmed",
                                  "missing_period", "unknown_ref", "boolean_value"])
def test_claim_cannot_mix_scope_dimension_period_or_unverified_source(prepared, mode):
    selection = SalesClaimSelection(kind="comparison", fact_index=0, previous_fact_index=1)
    other = prepared["facts"][1]
    if mode == "missing_fact": selection = selection.model_copy(update={"previous_fact_index": 999})
    elif mode == "mixed_sku": other["sku"] = "SKU-B"
    elif mode == "mixed_unit": other["unit"] = "units"
    elif mode == "mixed_source": other["source_table"] = "other.source"
    elif mode == "mixed_ref": other["evidence_ref"] = prepared["evidence"][0]["evidence_ref"]
    elif mode == "nonadjacent": other["period"] = "2026-09-01/2026-09-07"
    elif mode == "unequal_period": other["period"] = "2026-10-01/2026-10-02"
    elif mode == "wrong_metric": other["metric"] = "current_revenue"
    elif mode == "unconfirmed": other["status"] = "unconfirmed"
    elif mode == "missing_period": other["period"] = None
    elif mode == "unknown_ref": other["evidence_ref"] = "invented"
    elif mode == "boolean_value": other["value"] = True
    with pytest.raises(ValueError):
        render_sales_claim(selection, prepared)


@pytest.mark.parametrize("mutation", ["value", "unit", "period", "metric", "sku", "negation", "prefix", "suffix"])
def test_true_proposition_cannot_launder_changed_or_embedded_claim(prepared, mutation):
    text = render_sales_claim(SalesClaimSelection(kind="amount", fact_index=0), prepared)
    if mutation == "value": text = text.replace("100.25", "999999")
    elif mutation == "unit": text = text.replace("RUB", "units")
    elif mutation == "period": text = text.replace("2026-10-03", "2026-09-03")
    elif mutation == "metric": text = text.replace("выручка", "прибыль")
    elif mutation == "sku": text = text.replace("SKU-A", "SKU-B")
    elif mutation == "negation": text = "Не " + text
    elif mutation == "prefix": text = "Неподтверждённый факт: " + text
    elif mutation == "suffix": text += " Это прибыль."
    candidate = asyncio.run(FixtureModel().analyze(instructions="fixture", prepared=prepared))
    candidate.summary = text
    with pytest.raises(ValueError):
        validate_sales_publication(candidate, prepared)


def test_ambiguous_duplicate_periods_do_not_create_implicit_comparison(prepared):
    prepared["facts"].extend(deepcopy(prepared["facts"]))
    texts = native_claim_sentences(prepared)
    assert not any("выше" in text or "сопоставимые" in text for text in texts)


def test_explicit_unconfirmed_sales_hypothesis_remains_a_hypothesis(prepared):
    candidate = asyncio.run(FixtureModel().analyze(instructions="fixture", prepared=prepared))
    candidate.summary = "Гипотеза (не подтверждена): изменение выручки вызвано изменением конверсии."
    assert validate_sales_publication(candidate, prepared).summary == candidate.summary
    candidate.summary += " Следовательно, причина доказана."
    with pytest.raises(ValueError):
        validate_sales_publication(candidate, prepared)


@pytest.mark.parametrize("bad_index", [False, True])
def test_actual_model_selects_native_claims_without_generating_values(prepared, bad_index):
    baseline = deepcopy(prepared)
    narrative = {"summary": "", "findings": [], "recommendations": [],
                 "data_gaps": [], "cross_agent_requests": [], "grounded_claims": [
                     {"claim_index": 999 if bad_index else 0}, {"claim_index": 4}]}
    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200, json=response(narrative)))) as http:
            model = LangChainSalesModel(SalesModelSettings(api_key="synthetic-model-key"), http_async_client=http)
            if bad_index:
                with pytest.raises(ValueError):
                    await model.analyze(instructions="fixture", prepared=prepared)
            else:
                candidate = await model.analyze(instructions="fixture", prepared=prepared)
                published = validate_sales_publication(candidate, prepared)
                assert "100.25 RUB." in published.summary
                assert "выше предыдущей" in published.summary
                assert "grounded_claims" not in published.model_dump(mode="json")
                assert [fact.model_dump(mode="json", exclude_none=True, exclude_defaults=True)
                        for fact in published.facts] == baseline["facts"]
    asyncio.run(exercise())
    assert prepared == baseline


@pytest.mark.parametrize("context", ["Неверно:", "Не", "Утверждение опровергнуто:",
                                    "Показатели для артикула SKU-B:"])
def test_multiline_context_cannot_negate_or_relabel_native_claim(prepared, context):
    candidate = asyncio.run(FixtureModel().analyze(instructions="fixture", prepared=prepared))
    text = render_sales_claim(SalesClaimSelection(kind="amount", fact_index=0), prepared)
    candidate.summary = context + "\n" + text
    with pytest.raises(ValueError, match="cannot mix"):
        validate_sales_publication(candidate, prepared)


def test_model_cannot_combine_a_grounded_summary_with_provider_context(prepared):
    narrative = {"summary": "Неверно:", "findings": [], "recommendations": [],
                 "data_gaps": [], "cross_agent_requests": [], "grounded_claims": [
                     {"claim_index": 0}]}
    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200, json=response(narrative)))) as http:
            model = LangChainSalesModel(SalesModelSettings(api_key="synthetic-model-key"), http_async_client=http)
            with pytest.raises(ValueError):
                await model.analyze(instructions="fixture", prepared=prepared)
    asyncio.run(exercise())


@pytest.mark.parametrize("role", ["factor", "impact", "cause", "action", "recommendation"])
def test_native_fact_does_not_prove_a_causal_action_or_impact_role(prepared, role):
    candidate = asyncio.run(FixtureModel().analyze(instructions="fixture", prepared=prepared))
    text = render_sales_claim(SalesClaimSelection(kind="amount", fact_index=0), prepared)
    finding = {"factor": "Продажи", "observation": "Данные доступны.", "impact": "Требуется проверка.",
               "cause": "Причина не установлена.", "action": "Проверить полноту данных.",
               "confidence": "medium", "driver_class": "not_a_driver",
               "evidence": [prepared["facts"][0]["evidence_ref"]]}
    if role == "recommendation":
        candidate.recommendations = [text]
    else:
        finding[role] = text
        candidate.findings = [DiagnosticFinding.model_validate(finding)]
    with pytest.raises(ValueError):
        validate_sales_publication(candidate, prepared)
