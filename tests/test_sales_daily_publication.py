"""Daily native observations remain non-causal and preserve source grain."""
from copy import deepcopy
import asyncio
import json
from uuid import uuid4

import httpx
import pytest
from langgraph.checkpoint.memory import InMemorySaver

from commerce_agent.adapters.contracts import DailyPriceSalesRow
from commerce_agent.sales.policy import SalesRequest
from commerce_agent.sales.analytics import analytical_claims
from commerce_agent.sales.daily import daily_scopes,required_daily_claims,FIELDS
from commerce_agent.sales.claims import native_claim_catalog,render_sales_claim
from commerce_agent.sales.contracts import SalesDiagnosticResult
from commerce_agent.sales.publication import validate_sales_publication
from commerce_agent.sales.report import render_sales_report
from commerce_agent.sales.runtime import build_sales_subgraph
from commerce_agent.sales.semantics import SalesRequestedCheck,render_requested_check,render_delegated_check,finding_presentation
from commerce_agent.serialization import checkpoint_serializer
from test_sales_evidence import collect
from test_sales_evidence import respond,CONTEXT
from test_sales_runtime import dependencies,invocation

QUERY="Продажи SKU SKU-A за последние 7 дней"


def native_daily(path,payload):
    if path!="daily-price-sales-comparison":return
    row={field:None for field in DailyPriceSalesRow.model_fields}
    row.update(nm_id=1,vendor_code="SKU-A",report_date="2026-10-09",price_basis="seller_snapshot_exact_day",
        base_price_rub=123.45,seller_discount_pct=10.0,seller_effective_price_rub=111.105,
        observed_transaction_seller_price_avg_rub=109.25,observed_buyer_price_avg_rub=99.5,observed_spp_pct_avg=8.5,
        non_cancelled_order_units=3,realized_sales_units=1,realized_for_pay_rub=100.25,
        open_count=12,cart_count=3,funnel_order_count=2,add_to_cart_conversion=0.25,
        cart_to_order_conversion=None,buyout_percent=None,business_day_complete=True,
        sales_day_complete=False,marketing_day_complete=None)
    payload["data"].update(source_tables=["agent_marts.v_agent_price_sku_daily","sales.wb_sales_funnel_products_daily"],
        rows=[row],row_count=1)
    period={"date_from":"2026-10-09","date_to":"2026-10-09"}
    payload["source"]="agent_marts.v_agent_price_sku_daily"
    payload["period"]=period
    payload["freshness"].update(as_of="2026-10-09",status="available")
    payload["evidence"]=[{"id":"sales.daily_price_sales_comparison:"+format(i,"064x"),"source":source,"grain":"product_filter","period":period}
        for i,source in enumerate(payload["data"]["source_tables"])]
    payload["warnings"]=["causality:not_established","provenance_only:source_completeness_unverified","needs_data:incomplete_business_days"]


def request(**changes):
    return SalesRequest(**{"user_query":QUERY,"scope":"single_sku","targets":("SKU-A",),**changes})


def test_characterization_daily_data_is_collected_but_not_published_in_old_policy():
    prepared,seen=collect(request(),native_daily)
    assert any(path=="daily-price-sales-comparison" for path,_ in seen)
    daily=next(row for row in prepared["deep_dives"] if "daily_rows" in row)
    assert daily["daily_rows"][0]["sales_units"]==1
    assert "seller_effective_price_rub" not in daily["daily_rows"][0]
    assert len(prepared["facts"])==4 and not any(f["metric"].startswith("daily.") for f in prepared["facts"])
    assert not any(c["factor"].startswith("Дневн") for c in analytical_claims(prepared))


def fixture(change=native_daily,**changes):
    prepared,seen=collect(request(**changes),change,publish_daily=True)
    return prepared,seen


def candidate(prepared):
    claims=required_daily_claims(prepared)
    catalog=native_claim_catalog(prepared)
    findings=[];requests=[]
    for claim in claims:
        index=next(i for i,c in enumerate(catalog) if render_sales_claim(c,prepared)==claim["text"])
        factor,driver,confidence=finding_presentation(claim["text"],prepared)
        findings.append({"factor":factor,"driver_class":driver,"confidence":confidence,"observation":claim["text"],
            "cause":"Не устанавливается этой проверкой.","impact":"Нужна дополнительная проверка.",
            "action":render_requested_check(SalesRequestedCheck(kind="source_completeness",claim_index=index),prepared),
            "evidence":claim["evidence_refs"]})
        proposed=render_delegated_check(SalesRequestedCheck(kind="price" if claim["family"]=="price" else "marketing",claim_index=index),prepared)
        if proposed not in requests:requests.append(proposed)
    return {"agent_name":"Sales Agent","summary":"\n".join(c["text"] for c in claims) or "Проверены доступные данные.",
        "facts":deepcopy(prepared["facts"]),**deepcopy(prepared["tables"]),"findings":findings,"cross_agent_requests":requests,
        "recommendations":[],"data_gaps":[]}


def test_daily_source_values_families_report_and_public_fact_provenance():
    prepared,seen=fixture()
    assert len(seen)==5 and len(daily_scopes(prepared))==1
    scoped=[f for f in prepared["facts"] if f.get("target_type")=="sku_day"]
    assert len(scoped)==13
    values={f["metric"]:f["value"] for f in scoped}
    assert values["daily.seller_effective_price_rub"]==111.105
    assert values["daily.non_cancelled_order_units"]==3 and values["daily.funnel_order_count"]==2
    assert "daily.cart_to_order_conversion" not in values
    assert all(f.get("freshness_status","unknown")=="unknown" for f in scoped)
    result=validate_sales_publication(candidate(prepared),prepared)
    assert len(result.facts)==17 and len(result.findings)==3
    assert all(f.confidence=="low" for f in result.findings)
    assert all(r.period=="2026-10-09/2026-10-09" and r.vendor_codes==["SKU-A"] for r in result.cross_agent_requests)
    report=render_sales_report(result,prepared)
    assert "## Дневные наблюдения" in report and "111.105" in report and "нет данных" in report
    assert "False" in report and "не доказывают влияние" in report


@pytest.mark.parametrize("field",["summary","findings","cross_agent_requests"])
def test_available_daily_families_and_checks_cannot_be_silently_omitted(field):
    prepared,_=fixture();raw=candidate(prepared)
    raw[field]="Проверены доступные данные." if field=="summary" else []
    with pytest.raises(ValueError):validate_sales_publication(raw,prepared)


@pytest.mark.parametrize("mutation",["native","bool","period","source","scope","row_count","fact","ref","version","parent","request_bool","namespace","price_basis"])
def test_daily_native_provenance_and_raw_types_cannot_drift(mutation):
    prepared,_=fixture();raw=candidate(prepared);scope=prepared["daily_sales_scopes"][0]
    if mutation=="native":scope["native_daily"]["rows"][0]["realized_sales_units"]=987
    if mutation=="bool":scope["native_daily"]["rows"][0]["realized_sales_units"]=True
    if mutation=="period":scope["native_daily"]["rows"][0]["report_date"]="2026-10-08"
    if mutation=="source":scope["native_daily"]["source_tables"][0]="finance.fake"
    if mutation=="scope":scope["target"]="SKU-X"
    if mutation=="row_count":scope["native_daily"]["row_count"]=2
    if mutation=="fact":prepared["facts"][-1]["value"]=987
    if mutation=="ref":scope["source_refs"][0]["evidence_ref"]="fake"
    if mutation=="version":prepared.pop("daily_sales_policy_version")
    if mutation=="parent":prepared["evidence"]=[e for e in prepared["evidence"] if e["source"]!="agent_marts.v_agent_price_sku_daily"]
    if mutation=="request_bool":prepared["request"]["days"]=True
    if mutation=="namespace":scope["source_refs"][0]["evidence_ref"]=scope["source_refs"][0]["evidence_ref"].replace("sales.daily_price_sales_comparison:","finance.fake:")
    if mutation=="price_basis":scope["native_daily"]["rows"][0]["seller_effective_price_rub"]=None
    with pytest.raises(ValueError):validate_sales_publication(raw,prepared)


@pytest.mark.parametrize("changes",[{"days":32},{"date_from":"2026-10-03","date_to":"2026-10-09"}])
def test_historical_or_long_requests_never_receive_latest_daily_substitution(changes):
    prepared,seen=fixture(**changes)
    assert all(path!="daily-price-sales-comparison" for path,_ in seen)
    assert daily_scopes(prepared)==[]
    assert "scope:daily_signal_skipped_no_matching_period_capability" in prepared["warnings"]
    result=validate_sales_publication(candidate(prepared),prepared)
    assert "Дневные данные недоступны" in render_sales_report(result,prepared)


class DailyModel:
    def __init__(self):self.calls=[]
    async def analyze(self,*,instructions,prepared):
        self.calls.append(deepcopy(prepared))
        return SalesDiagnosticResult.model_validate(candidate(prepared))


def test_daily_checkpoint_resume_without_collection_and_disabled_capability_guard():
    async def exercise():
        seen=[];model=DailyModel();saver=InMemorySaver(serde=checkpoint_serializer());config={"configurable":{"thread_id":str(uuid4())}}
        def handler(req):return respond(req,change=native_daily,seen=seen)
        deps=dependencies(model,handler,enable_daily_reports=True)
        first=build_sales_subgraph(deps,checkpointer=saver,interrupt_after=["collect_evidence"])
        await first.ainvoke(invocation(QUERY),config,context=CONTEXT)
        count=len(seen)
        disabled=build_sales_subgraph(dependencies(model,handler),checkpointer=saver)
        with pytest.raises(ValueError,match="daily checkpoint capability"):await disabled.ainvoke(None,config,context=CONTEXT)
        output=await build_sales_subgraph(deps,checkpointer=saver).ainvoke(None,config,context=CONTEXT)
        assert len(seen)==count and len(model.calls)==1 and output["errors"]==[]
        assert "## Дневные наблюдения" in output["sales_analysis"].summary
    asyncio.run(exercise())


def test_daily_capability_clone_and_debug_suppression(monkeypatch):
    graph=build_sales_subgraph(dependencies(enable_daily_reports=True))
    assert graph.copy()._daily_enabled and graph.with_config({"recursion_limit":40})._daily_enabled
    monkeypatch.setenv("COMMERCE_AGENT_ENVIRONMENT","development")
    monkeypatch.setenv("COMMERCE_AGENT_LANGSMITH_DEBUG","true")
    with pytest.raises(ValueError,match="daily sources require separate debug"):
        build_sales_subgraph(dependencies(enable_daily_reports=True))
    with pytest.raises(ValueError,match="daily sources require separate debug"):
        asyncio.run(graph.ainvoke(invocation(QUERY),context=CONTEXT))


def test_multiple_native_nm_id_rows_per_day_are_not_aggregated():
    def change(path,payload):
        native_daily(path,payload)
        if path=="daily-price-sales-comparison":
            row=payload["data"]["rows"][0]
            payload["data"]["rows"].append({**row,"nm_id":2,"realized_sales_units":8})
            payload["data"]["row_count"]=2
    prepared,_=fixture(change)
    facts=[f for f in prepared["facts"] if f["metric"]=="daily.realized_sales_units"]
    assert [f["value"] for f in facts]==[1,8] and len({f["target_id"] for f in facts})==2
    assert len(required_daily_claims(prepared))==3
    report=render_sales_report(validate_sales_publication(candidate(prepared),prepared),prepared)
    assert "| 2026-10-09 | SKU-A | 1 |" in report and "| 2026-10-09 | SKU-A | 2 |" in report
    assert "Всего за день" not in report


def test_absent_price_and_funnel_families_are_not_zero_imputed():
    def change(path,payload):
        native_daily(path,payload)
        if path=="daily-price-sales-comparison":
            row=payload["data"]["rows"][0]
            for key,_,_,_,family in FIELDS:
                if family!="sales":row[key]=None
            row["price_basis"]="missing"
            payload["evidence"]=payload["evidence"][:1]
    prepared,_=fixture(change)
    assert [c["family"] for c in required_daily_claims(prepared)]==["sales"]
    facts=[f for f in prepared["facts"] if f.get("target_type")=="sku_day"]
    assert len(facts)==3 and all(f["value"] is not None for f in facts)
    assert "нет данных" in render_sales_report(validate_sales_publication(candidate(prepared),prepared),prepared)


def test_supported_request_cannot_hide_daily_loss_with_historical_skip_marker():
    prepared,_=fixture()
    prepared["daily_sales_scopes"]=[]
    prepared["facts"]=[f for f in prepared["facts"] if f.get("target_type")!="sku_day"]
    prepared["warnings"].append("scope:daily_signal_skipped_no_matching_period_capability")
    with pytest.raises(ValueError,match="missing daily scope"):
        daily_scopes(prepared)


def test_daily_fact_markers_cannot_downgrade_checkpoint_policy():
    from types import SimpleNamespace
    prepared,_=fixture();raw=candidate(prepared)
    prepared.pop("daily_sales_policy_version");prepared.pop("daily_sales_scopes")
    with pytest.raises(ValueError,match="cannot lose policy"):
        validate_sales_publication(raw,prepared)
    disabled=build_sales_subgraph(dependencies())
    with pytest.raises(ValueError,match="daily checkpoint capability"):
        disabled._assert_checkpoint(SimpleNamespace(values={"prepared_sales":prepared}))
    enabled=build_sales_subgraph(dependencies(enable_daily_reports=True))
    with pytest.raises(ValueError,match="cannot lose policy"):
        enabled._assert_checkpoint(SimpleNamespace(values={"prepared_sales":prepared}))


def test_failed_daily_target_keeps_successful_target_and_weekly_results():
    prepared,seen=fixture(scope="sku_list",targets=("SKU-A","SKU-B"))
    assert len(seen)==9
    assert [s["target"] for s in daily_scopes(prepared)]==["SKU-A"]
    assert "needs_data:daily_signal_unavailable:SKU-B" in prepared["warnings"]
    assert len([f for f in prepared["facts"] if f.get("target_type")!="sku_day"])==8
    result=validate_sales_publication(candidate(prepared),prepared)
    assert result.facts and "SKU-B" in render_sales_report(result,prepared)


def test_daily_price_and_weekly_observations_cannot_be_mixed_into_a_finding():
    prepared,_=fixture();raw=candidate(prepared)
    catalog=native_claim_catalog(prepared)
    from commerce_agent.sales.claims import claim_evidence
    weekly=next(c for c in catalog if c.kind=="analytic" and not render_sales_claim(c,prepared).startswith("Дневн"))
    finding=next(f for f in raw["findings"] if f["factor"]=="Дневной ценовой контекст")
    finding["observation"]+="\n"+render_sales_claim(weekly,prepared)
    finding["evidence"]+=claim_evidence(weekly,prepared)
    with pytest.raises(ValueError,match="foreign-domain"):
        validate_sales_publication(raw,prepared)


def test_daily_responses_wire_schema_price_gap_and_closed_service_binding():
    from commerce_agent.sales.model import LangChainSalesModel,SalesModelSettings
    from commerce_agent.sales.semantics import DAILY_PRICE_GAP
    from test_sales_model import response
    prepared,_=fixture();catalog=native_claim_catalog(prepared)
    indices=[next(i for i,c in enumerate(catalog) if render_sales_claim(c,prepared)==claim["text"])
        for claim in required_daily_claims(prepared)]
    seen=[]
    def transport(req):
        body=json.loads(req.content);seen.append(body)
        assert not body.get("tools") and body["store"] is False
        defs=body["text"]["format"]["schema"]["$defs"]
        assert set(defs["SalesFindingCheckKind"]["properties"])=={"kind"}
        assert "Keep daily findings separate" in json.dumps(body)
        return httpx.Response(200,json=response({"summary":"","grounded_claims":[{"claim_index":i} for i in indices],
            "findings":[{"observation":{"kind":"native","claims":[{"claim_index":i}]},
                "requested_check":{"kind":"source_completeness"},"evidence_indices":[]} for i in indices],
            "recommendations":[],"data_gaps":[{"domain":"price"}],
            "cross_agent_requests":[{"kind":"price","claim_index":indices[1]}]}))
    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as http:
            result=await LangChainSalesModel(SalesModelSettings(api_key="synthetic-model-key"),http_async_client=http).analyze(
                instructions="Fixture instructions",prepared=prepared)
        return validate_sales_publication(result,prepared)
    result=asyncio.run(exercise())
    assert len(seen)==1 and len(result.findings)==3
    assert any(g.reason==DAILY_PRICE_GAP for g in result.data_gaps)
    assert all("2026-10-09" in f.action for f in result.findings)


def test_old_checkpoint_under_enabled_builder_keeps_recorded_policy():
    async def exercise():
        seen=[];saver=InMemorySaver(serde=checkpoint_serializer());config={"configurable":{"thread_id":str(uuid4())}}
        def handler(req):return respond(req,change=native_daily,seen=seen)
        class OldModel:
            async def analyze(self,*,instructions,prepared):
                assert "daily_sales_policy_version" not in prepared
                return SalesDiagnosticResult.model_validate({"agent_name":"Sales Agent","summary":"Проверены доступные данные.",
                    "facts":prepared["facts"],**prepared["tables"]})
        model=OldModel()
        await build_sales_subgraph(dependencies(model,handler),checkpointer=saver,interrupt_after=["collect_evidence"]).ainvoke(
            invocation(QUERY),config,context=CONTEXT)
        count=len(seen)
        result=await build_sales_subgraph(dependencies(model,handler,enable_daily_reports=True),checkpointer=saver).ainvoke(
            None,config,context=CONTEXT)
        assert len(seen)==count and result["errors"]==[]
        assert "## Дневные наблюдения" not in result["sales_analysis"].summary
    asyncio.run(exercise())
