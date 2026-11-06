"""Call-local indexing must preserve proof semantics, bytes and fresh guards."""
from copy import deepcopy
from hashlib import sha256
import json

import pytest

from commerce_agent.sales import claims
from commerce_agent.sales.publication import validate_sales_publication
from commerce_agent.sales.report import render_sales_report
from commerce_agent.sales.semantics import check_proofs
from test_sales_daily_publication import fixture as daily, candidate as daily_candidate
from test_sales_report import fixture as portfolio
from test_sales_scoped_closeout import prepared_for


def inputs(case):
    if case == "portfolio":
        raw, prepared = portfolio()
    elif case == "daily":
        prepared, _ = daily()
        raw = daily_candidate(prepared)
    else:
        prepared, raw, _ = prepared_for(("SKU-A",) if case == "scoped_1" else ("SKU-A", "SKU-B"))
    return prepared, raw


def digest(value):
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                             allow_nan=False, separators=(",", ":")).encode()).hexdigest()


# Captured on the unmodified 642-file/13cf29... revision BEFORE optimization.
# Order: result, Markdown bytes, ordered catalog/text, proof evidence, checks.
BASELINE = {
    "portfolio": (
        "82a275f768302ff1b58965c4cbde5f9cc9d285b74ba56d54a4571e2c166089d3",
        "3a2e82382f819f286784d7274c749ea45b0e703be6060ee5977c76f0df12f46a",
        "9d7930d312220a4c84fd8de10302f4ebda5e77ce52167449397d949f92559fcc",
        "807b308542d89078b780b2b24927c8917a0c571572bc9c07f5f3abadcc7f2ed7",
        "928269f325b28edc9f85c7032236373f444b821adcd7c56169b468834e0c7816",
    ),
    "daily": (
        "6ae2957120da0409a90b176af39dcaabf3d4b8a1b98894ff0a9f82ee35fb1c6a",
        "80fb28f255faebe0a1090391a3cef3ab170beb8e9f9b75c2d541ea33f686ef66",
        "411283b15fd8feb2e656181bc6484c50e3cc3f735ad9fa44eb673ec163961651",
        "9843ea36a3d8d17a942bb378bedd2fc807df60e235aa7aa14753e2b949cf9510",
        "7a53dd0ed71972fe4e929e46297487d7832a056f65785d65f742e57aad1c2dab",
    ),
    "scoped_1": (
        "08ebb4af9ec9dd19024bc83c8e3093ebff799a14351ec82346bb4e31efef02f8",
        "2e87347e05cebb016fb80e48c2802b8f3dd97b4ee8093f19cb83d1d9951ef02b",
        "64740a3343fee16db434fb2ceb2112e57049e62e083d95f6738a8507c9d7c57e",
        "7915d5f2d9efca365ca97cddd37828aff47f4aaff7bf1ffa1e61a6ad2212c1f2",
        "928269f325b28edc9f85c7032236373f444b821adcd7c56169b468834e0c7816",
    ),
    "scoped_2": (
        "edf073f41285c67b788d5dcd6dd3f934eeee8f99c882aadc1ba7dd79d58780db",
        "e5c274f7aee246f26371a8fcb1fbd6079c269de32a5ca9eafca7950441816a83",
        "e4694dc286c51f69f2ac36ce47ae3d2ac87fb53caed59a0e4dc46959a32e82d8",
        "c6f1466028728ef9fc85acea0c36f5e099d45450a0666d94e0352817109e9769",
        "69e6b31d3b8948e05ba31e52367e99aa01b421cc95fa13877b8dcb5e2fb09ef9",
    ),
}


@pytest.mark.parametrize("case", BASELINE)
def test_original_result_report_catalog_aliases_and_checks_are_exact(case):
    prepared, raw = inputs(case)
    before = deepcopy(prepared)
    result = validate_sales_publication(raw, prepared)
    catalog = claims.native_claim_catalog(prepared)
    actual = (
        digest(result.model_dump(mode="json")),
        sha256(render_sales_report(result, prepared).encode()).hexdigest(),
        digest([(c.model_dump(mode="json"), claims.render_sales_claim(c, prepared)) for c in catalog]),
        digest({key: sorted(refs) for key, refs in claims.native_claim_proofs(prepared).items()}),
        digest(check_proofs(prepared)),
    )
    assert actual == BASELINE[case]
    assert prepared == before


def test_one_analytical_build_per_boundary_not_per_claim(monkeypatch):
    prepared, raw = inputs("daily")
    original = claims.analytical_claims
    calls = []
    def counted(value):
        calls.append(value)
        return original(value)
    monkeypatch.setattr(claims, "analytical_claims", counted)
    result = validate_sales_publication(raw, prepared)
    assert len(calls) == 1
    render_sales_report(result, prepared)
    assert len(calls) == 2  # the public renderer still performs its own guard
    validate_sales_publication(raw, prepared)
    assert len(calls) == 3  # never reuse an index across calls/checkpoints
    assert not any("index" in key or "cache" in key for key in prepared)


@pytest.mark.parametrize("mutation", ["fact", "period", "evidence", "source"])
def test_same_prepared_object_mutation_is_revalidated(mutation):
    prepared, raw = inputs("daily")
    result = validate_sales_publication(raw, prepared)
    scope = prepared["daily_sales_scopes"][0]
    if mutation == "fact":
        prepared["facts"][-1]["value"] = 999999
    elif mutation == "period":
        scope["native_daily"]["rows"][0]["report_date"] = "2026-10-08"
    elif mutation == "evidence":
        prepared["evidence"].clear()
    else:
        scope["native_daily"]["source_tables"][0] = "sales.untrusted"
    with pytest.raises(ValueError):
        validate_sales_publication(raw, prepared)
    with pytest.raises(ValueError):
        render_sales_report(result, prepared)


def test_a_private_index_cannot_be_supplied_to_public_validator():
    prepared, raw = inputs("daily")
    index = claims._SalesClaimIndex(prepared)
    with pytest.raises(TypeError):
        validate_sales_publication(raw, prepared, index=index)
