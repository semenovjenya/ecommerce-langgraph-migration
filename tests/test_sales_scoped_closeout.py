"""S02 pure-scope policy; synthetic fixtures are NOT actual scoped evidence."""
import asyncio
from copy import deepcopy
from uuid import uuid4

import httpx
import pytest
from langgraph.checkpoint.memory import InMemorySaver

from commerce_agent.sales.analytics import analytical_claims
from commerce_agent.sales.claims import native_claim_catalog, render_sales_claim
from commerce_agent.sales.contracts import SalesDiagnosticResult
from commerce_agent.sales.policy import SalesRequest
from commerce_agent.sales.publication import validate_sales_publication
from commerce_agent.sales.report import render_sales_report
from commerce_agent.sales.runtime import build_sales_subgraph
from commerce_agent.sales.scoped import scoped_scopes, required_scoped_claims
from commerce_agent.sales.semantics import SalesRequestedCheck, finding_presentation, render_requested_check, render_delegated_check
from commerce_agent.serialization import checkpoint_serializer
from test_sales_evidence import collect, respond, CONTEXT
from test_sales_runtime import dependencies, invocation


def request(targets=("SKU-A",), scope=None):
    return SalesRequest(user_query="Продажи SKU " + ", ".join(targets) + " за 2026-10-03 — 2026-10-09",
        scope=scope or ("single_sku" if len(targets) == 1 else "sku_list"), targets=targets,
        date_from="2026-10-03", date_to="2026-10-09")


def candidate_for(prepared):
    claims = required_scoped_claims(prepared)
    catalog = native_claim_catalog(prepared)
    findings, requests = [], []
    for scope in scoped_scopes(prepared):
        selected = [c for c in claims if c["requested_target"] == scope["target"]]
        text = "\n".join(c["text"] for c in selected)
        index = next(i for i, c in enumerate(catalog) if render_sales_claim(c, prepared) == selected[0]["text"])
        factor, driver, confidence = finding_presentation(text, prepared)
        check = SalesRequestedCheck(kind="marketing", claim_index=index)
        findings.append({"factor": factor, "driver_class": driver, "confidence": confidence, "observation": text,
            "evidence": selected[0]["evidence_refs"], "cause": "Не устанавливается этой проверкой.",
            "impact": "Нужна дополнительная проверка.", "action": render_requested_check(check, prepared)})
        requests.append(render_delegated_check(check, prepared))
    return {"agent_name": "Sales Agent", "summary": "\n".join(c["text"] for c in claims) or "Проверены доступные данные.",
        "facts": deepcopy(prepared["facts"]), **deepcopy(prepared["tables"]), "findings": findings,
        "recommendations": [], "data_gaps": [], "cross_agent_requests": requests}


def prepared_for(targets=("SKU-A",), change=None, **kwargs):
    prepared, seen = collect(request(targets, **kwargs), change, publish_targeted=True)
    return prepared, candidate_for(prepared), seen


@pytest.mark.parametrize("targets", [("SKU-A",), ("SKU-A", "SKU-B")])
@pytest.mark.parametrize("omitted", ["summary", "findings", "cross_agent_requests"])
def test_empty_or_amount_only_scoped_analysis_is_rejected(targets, omitted):
    prepared, candidate, _ = prepared_for(targets)
    if omitted == "summary":
        amount = next(c for c in native_claim_catalog(prepared) if c.kind == "amount")
        candidate[omitted] = render_sales_claim(amount, prepared)
    else:
        candidate[omitted] = []
    with pytest.raises(ValueError, match="explicitly requested"):
        validate_sales_publication(candidate, prepared)


@pytest.mark.parametrize("targets", [("SKU-A",), ("SKU-A", "SKU-B")])
def test_scoped_dynamics_checks_and_rows_stay_exact_per_target(targets):
    prepared, candidate, seen = prepared_for(targets)
    output = validate_sales_publication(candidate, prepared)
    assert prepared["scoped_sales_policy_version"] == 3
    assert [args["product_keyword"] for path, args in seen if path == "overview"] == list(targets)
    assert len(output.findings) == len(targets) and len(output.cross_agent_requests) == len(targets)
    assert len(output.facts) == 4 * len(targets)
    assert {fact.target_id for fact in output.facts} == set(targets)
    assert all("2026-10-03/2026-10-09" in req.question for req in output.cross_agent_requests)
    claims = required_scoped_claims(prepared)
    assert len(claims) == 2 * len(targets)
    assert all(c["factor"] in {"Динамика выручки", "Динамика продаж"} for c in claims)
    assert all(c["text"] in output.summary for c in claims)
    # One native scope gives two required families, without amount/delta math.
    assert any("100.25 / 0.0 RUB" in c["text"] or "100.25 / 0 RUB" in c["text"] for c in claims)
    assert all(c["sku"] is not None for c in analytical_claims(prepared))


def missing_second(path, payload):
    if path == "overview" and payload["data"]["focus_keyword"] == "SKU-B":
        payload["data"]["totals"] = None


def test_missing_list_target_is_explicit_and_does_not_erase_valid_scope():
    prepared, candidate, seen = prepared_for(("SKU-A", "SKU-B"), missing_second)
    result = validate_sales_publication(candidate, prepared)
    assert {fact.sku for fact in result.facts} == {"SKU-A"}
    assert {scope["target"] for scope in scoped_scopes(prepared)} == {"SKU-A"}
    assert "needs_data:target_unavailable:SKU-B" in prepared["warnings"]
    report = render_sales_report(result, prepared)
    assert "SKU SKU-B: данные этой области недоступны; это не означает нулевые продажи." in report
    assert "SKU-B" not in result.summary and len(result.cross_agent_requests) == 1
    assert all(args.get("product_keyword") != "SKU-A, SKU-B" for _, args in seen)


@pytest.mark.parametrize("mutation", ["version", "scopes", "scope", "fact", "native", "native_bool", "period", "rows", "columns", "missing_gap"])
def test_scope_receipt_mutations_fail_closed(mutation):
    prepared, candidate, _ = prepared_for(("SKU-A", "SKU-B"), missing_second)
    source = prepared["scoped_sales_scopes"][0]
    if mutation == "version": prepared.pop("scoped_sales_policy_version")
    if mutation == "scopes": prepared.pop("scoped_sales_scopes")
    if mutation == "scope": source["target"] = "SKU-X"
    if mutation == "fact": source["facts"][0]["value"] = 900
    if mutation == "native": source["native_change"]["rows"][0]["delta_revenue"] = 900
    if mutation == "native_bool": source["native_change"]["rows"][0]["previous_sales"] = False
    if mutation == "period": source["period"]["date_from"] = "2026-10-04"
    if mutation == "rows": prepared["tables"]["sales_decomposition"]["rows"][0][1] = "SKU-X"
    if mutation == "columns": prepared["tables"]["sales_decomposition"]["columns"][2] = "profit"
    if mutation == "missing_gap": prepared["warnings"] = [w for w in prepared["warnings"] if not w.startswith("needs_data:")]
    with pytest.raises((ValueError, KeyError)):
        validate_sales_publication(candidate, prepared)


def test_unmarked_historical_scoped_prepared_keeps_old_reader():
    prepared, _ = collect(request())  # deliberately old capability off
    candidate = {"agent_name": "Sales Agent", "summary": "Проверены доступные данные.",
        "facts": deepcopy(prepared["facts"]), **deepcopy(prepared["tables"]), "findings": [],
        "recommendations": [], "data_gaps": [], "cross_agent_requests": []}
    assert not scoped_scopes(prepared)
    assert validate_sales_publication(candidate, prepared).findings == []
    assert "scoped_sales_policy_version" not in prepared


class ScopedFixtureModel:
    def __init__(self): self.calls = []
    async def analyze(self, *, instructions, prepared):
        self.calls.append(deepcopy(prepared))
        return SalesDiagnosticResult.model_validate(candidate_for(prepared))


def test_pure_list_full_subgraph_missing_target_preserves_owned_result():
    model = ScopedFixtureModel()
    output = asyncio.run(build_sales_subgraph(dependencies(model,
        lambda req: respond(req, change=missing_second), enable_targeted_reports=True)).ainvoke(
            invocation(request(("SKU-A", "SKU-B")).user_query), context=CONTEXT))
    assert output["errors"] == [] and len(model.calls) == 1
    assert output["sales_analysis"].status == "partial"
    assert {f.sku for f in output["sales_analysis"].result.facts} == {"SKU-A"}
    assert any(gap.reason == "needs_data:target_unavailable:SKU-B" for gap in output["data_gaps"])


@pytest.mark.parametrize("disabled", [False, True])
def test_new_scoped_checkpoint_reader_and_no_repeat_collection(disabled):
    async def exercise():
        model, seen = ScopedFixtureModel(), []
        dep = dependencies(model, lambda req: respond(req, seen=seen), enable_targeted_reports=True)
        saver = InMemorySaver(serde=checkpoint_serializer())
        config = {"configurable": {"thread_id": str(uuid4())}}
        first = build_sales_subgraph(dep, checkpointer=saver, interrupt_after=["collect_evidence"])
        await first.ainvoke(invocation(request().user_query), config, context=CONTEXT)
        calls = deepcopy(seen)
        from dataclasses import replace
        second = build_sales_subgraph(replace(dep, enable_targeted_reports=not disabled), checkpointer=saver)
        if disabled:
            with pytest.raises(ValueError, match="scoped checkpoint capability"):
                await second.ainvoke(None, config, context=CONTEXT)
            assert model.calls == []
        else:
            result = await second.ainvoke(None, config, context=CONTEXT)
            assert result["errors"] == [] and len(model.calls) == 1
        assert seen == calls
    asyncio.run(exercise())
