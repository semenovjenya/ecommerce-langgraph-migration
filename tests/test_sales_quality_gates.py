"""Focused analytical-quality regressions, not complete business acceptance.

Synthetic evidence, no provider/data/account access. Native propositions must
be accepted without allowing mismatched quantities or invented causal claims.
"""
import asyncio

import pytest

from commerce_agent.sales.policy import SalesRequest
from commerce_agent.sales.publication import validate_sales_publication
from commerce_agent.sales.validation import validate_sales_diagnostic_result
from test_sales_evidence import collect
from test_sales_runtime import FixtureModel


@pytest.fixture
def evidence_and_candidate():
    prepared, _ = collect(SalesRequest(
        user_query="Отчет продаж SKU SKU-A", scope="single_sku", targets=("SKU-A",)))
    candidate = asyncio.run(FixtureModel().analyze(instructions="fixture", prepared=prepared))
    assert [(fact["metric"], fact["value"]) for fact in prepared["facts"]] == [
        ("current_revenue", 100.25), ("previous_revenue", 0),
        ("current_sales", 1), ("previous_sales", 0)]
    assert prepared["deep_dives"]
    assert all(item["causal_valid"] is False for item in prepared["deep_dives"])
    assert all(item["daily_rows"] == [] for item in prepared["deep_dives"])
    assert all(fact["period"] and fact["unit"] for fact in prepared["facts"])
    return prepared, candidate


def test_grounded_qualitative_comparison_is_accepted(evidence_and_candidate):
    prepared, candidate = evidence_and_candidate
    candidate.summary = "Текущая выручка SKU SKU-A выше предыдущей за сопоставимые периоды."
    assert not validate_sales_diagnostic_result(candidate, prepared["request"]["user_query"])
    assert validate_sales_publication(candidate, prepared).summary == candidate.summary


def test_source_bound_native_amount_can_be_published(evidence_and_candidate):
    prepared, candidate = evidence_and_candidate
    fact = prepared["facts"][0]
    candidate.summary = (f"Текущая выручка SKU {fact['sku']} за период "
                         f"{fact['period']}: "
                         f"{fact['value']} {fact['unit']}.")
    # The copied legacy guardrail accepts this same contract; the stronger
    # M3 publication guard rejects it. Schema/AST parity is not quality parity.
    assert not validate_sales_diagnostic_result(candidate, prepared["request"]["user_query"])
    assert validate_sales_publication(candidate, prepared).summary == candidate.summary


def test_reversed_native_comparison_is_rejected(evidence_and_candidate):
    prepared, candidate = evidence_and_candidate
    candidate.summary = "Текущая выручка SKU SKU-A ниже предыдущей за сопоставимые периоды."
    with pytest.raises(ValueError):
        validate_sales_publication(candidate, prepared)


def test_unverified_sales_cause_is_rejected(evidence_and_candidate):
    prepared, candidate = evidence_and_candidate
    candidate.summary = "Рост выручки вызван улучшением конверсии."
    with pytest.raises(ValueError):
        validate_sales_publication(candidate, prepared)
