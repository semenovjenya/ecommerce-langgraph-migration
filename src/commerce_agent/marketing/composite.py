"""Latest native Marketing diagnostic: independent windows, read-only proposals.

No financial formulas, proposal-rule reimplementation, fuzzy cross-source
join, invented aggregate, write tools or automatic source-debug approval.
"""
from datetime import date,timedelta
from hashlib import sha256
import json
from typing import Literal

from pydantic import BaseModel,ConfigDict,Field,StrictStr,model_validator

from commerce_agent.adapters.contracts import DatabaseTime,MarketingDeepDiagnostic,MarketingSnapshot,ToolResult,ToolArtifactReference
from commerce_agent.marketing.evidence import _hydrate
from commerce_agent.marketing.policy import assert_bounded_safe


class MarketingCompositeDiagnosticRequest(BaseModel):
    model_config=ConfigDict(extra="forbid",frozen=True)
    capability:Literal["latest_native_diagnostic"]="latest_native_diagnostic"
    user_query:StrictStr=Field(min_length=1,max_length=8000)
    product_keyword:StrictStr|None=Field(default=None,min_length=1,max_length=128)
    days:int=Field(default=7,ge=1,le=30,strict=True)
    limit:int=Field(default=20,ge=1,le=50,strict=True)

    @model_validator(mode="after")
    def bounded(self):
        if self.product_keyword is not None and self.product_keyword!=self.product_keyword.strip():
            raise ValueError("Marketing composite filter must be literal")
        assert_bounded_safe(self.model_dump(mode="json"))
        return self


MARTS={
    "campaign_mart":"marketing.mart_marketing_campaign_daily",
    "query_bands_mart":"marketing.mart_marketing_query_bands",
    "forecast_ready_mart":"marketing.mart_marketing_forecast_ready",
    "query_demand_mart":"marketing.mart_marketing_query_demand_daily",
    "query_share_mart":"marketing.mart_marketing_query_share_daily",
    "query_forecast_mart":"marketing.mart_marketing_query_forecast_ready",
    "campaign_demand_mart":"marketing.mart_marketing_campaign_demand_forecast",
}
SOURCES=set(MARTS.values())|{"marketing.wb_marketing_campaign_stats_daily","marketing.wb_search_query_orders_text_daily",
    "marketing.wb_search_query_details","marketing.wb_search_query_report_overview","marketing.wb_marketing_campaigns",
    "marketing.wb_marketing_campaign_nms","marketing.wb_marketing_budget","sales.wb_supplier_sales",
    "sales.wb_sales_funnel_products_daily"}


def canonical(value):return json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(",",":")).encode()


def _window(anchor,days):
    return f"{anchor-timedelta(days=days-1)}/{anchor}" if anchor else None


def validate_native(envelope,request,today):
    """Structural source binding, not business freshness/action-rule computation."""
    result=ToolResult[MarketingDeepDiagnostic].model_validate(envelope)
    if canonical(result.model_dump(mode="json",exclude_unset=True))!=canonical(envelope):
        raise ValueError("Marketing composite native types changed")
    data=result.data
    keyword=request.product_keyword;days=request.days;limit=request.limit
    if data.focus_keyword!=keyword or data.period_days!=days or result.period is not None or any(item.period for item in result.evidence):
        raise ValueError("Marketing composite scope/shared period changed")
    if result.source not in SOURCES or any(item.source not in SOURCES for item in result.evidence):
        raise ValueError("Marketing composite evidence source changed")
    portfolio="marketing.wb_search_query_report_overview"
    if any(item.grain!=("product_filter" if keyword and item.source!=portfolio else "portfolio") for item in result.evidence):
        raise ValueError("Marketing composite provenance grain changed")
    for name in ("campaign_efficiency","query_performance","bid_pressure","visibility","search_demand","action_matrix"):
        part=getattr(data,name)
        if part.focus_keyword!=keyword or name!="visibility" and part.period_days!=days:
            raise ValueError("Marketing composite nested scope changed")
    def rows(main,*subsets):
        if len(main)>limit:raise ValueError("Marketing composite source exceeded requested limit")
        known={canonical(r.model_dump(mode="json")) for r in main}
        if any(canonical(r.model_dump(mode="json")) not in known for subset in subsets for r in subset):
            raise ValueError("Marketing composite native subset changed")
    c,q,b,v,s,m=(getattr(data,n) for n in ("campaign_efficiency","query_performance","bid_pressure","visibility","search_demand","action_matrix"))
    rows(c.campaign_rows,c.waste_rows,c.scale_rows)
    rows(q.top_query_rows,q.query_waste_rows,q.query_scale_rows,q.query_position_risk_rows)
    rows(b.bid_rows,b.overbid_rows,b.underbid_rows,b.bid_efficiency_rows)
    rows(v.visibility_rows,v.position_loss_rows,v.open_card_gap_rows,v.post_click_conversion_rows)
    rows(s.search_text_rows,s.growing_query_rows,s.high_frequency_low_order_rows,s.order_driving_query_rows)
    if any(len(family)>limit for family in (m.campaign_actions,m.query_actions,m.sku_actions)):
        raise ValueError("Marketing composite proposal output exceeded native limit")
    # Native counts precede output limits. Do NOT equate them to list lengths.
    fresh=m.source_freshness
    states={name:getattr(fresh.sources,name).status for name in MARTS}
    if fresh.stale_sources!=[name for name,status in states.items() if status=="stale"] or fresh.invalid_sources!=[name for name,status in states.items() if status!="fresh"]:
        raise ValueError("Marketing composite native freshness diagnostics changed")
    if b.bid_rows or b.overbid_rows or b.underbid_rows or b.bid_efficiency_rows:
        raise ValueError("Marketing composite bid capability is not implemented")
    for row in q.top_query_rows:
        if row.spend!=0 or row.cpc is not None:
            raise ValueError("Marketing composite search proxy became paid query data")
    for row in s.search_text_rows:
        if row.report_date!=s.report_date:raise ValueError("Marketing composite demand row date changed")
    for row in q.top_query_rows:
        if row.report_date!=q.report_date:raise ValueError("Marketing composite query row date changed")
    if (s.search_text_rows and s.report_date is None) or (q.top_query_rows and q.report_date is None):
        raise ValueError("Marketing composite query/demand source date absent")
    # Relative native freshness is preserved even if old/future in wall-clock
    # terms; a warning, never a recomputed native status or approved action.
    if type(today) is not date:raise ValueError("Marketing composite database date required")
    return result


async def collect_marketing_composite(client,context,request,*,policy_version=3):
    from commerce_agent.debug_privacy import development_debug_enabled
    if development_debug_enabled():raise ValueError("Marketing composite debug requires separate source acceptance")
    clock=await _hydrate(client,context,await client.database_time(context),DatabaseTime)
    result=await client.marketing_deep_diagnostic(context,product_keyword=request.product_keyword,days=request.days,limit=request.limit)
    mode="hydrated" if isinstance(result,ToolArtifactReference) else "inline"
    if isinstance(result,ToolArtifactReference):
        result=await client.hydrate(context,result,result_type=MarketingDeepDiagnostic)
    # Preserve native sparse proposal fields; the existing generic helper's
    # full DTO roundtrip would materialize absent optional nulls.
    envelope=result.model_dump(mode="json",exclude_unset=True)
    snapshot=None;snapshot_mode=None
    if policy_version>=3:
        snapshot=await client.marketing_report_snapshot(context,product_keyword=request.product_keyword)
        snapshot_mode="hydrated" if isinstance(snapshot,ToolArtifactReference) else "inline"
        if isinstance(snapshot,ToolArtifactReference):snapshot=await client.hydrate(context,snapshot,result_type=MarketingSnapshot)
        snapshot=snapshot.model_dump(mode="json",exclude_unset=True)
    return prepare_native(envelope,request,clock.data.db_current_date,context.request_id,mode,policy_version=policy_version,
        snapshot_envelope=snapshot,snapshot_mode=snapshot_mode)


def prepare_native(envelope,request,today,request_id,mode,*,policy_version=3,snapshot_envelope=None,snapshot_mode=None):
    if type(policy_version) is not int or policy_version not in (1,2,3,4):
        raise ValueError("Marketing composite policy version unsupported")
    if mode not in {"inline","hydrated"}:raise ValueError("Marketing composite source mode unknown")
    result=validate_native(envelope,request,today)
    # Re-establish schema field order after JSON/checkpoint roundtrips; object
    # key order is not business data and must not reorder report columns.
    envelope=result.model_dump(mode="json",exclude_unset=True)
    data=result.data;days=request.days
    c,q,v,s,m=data.campaign_efficiency,data.query_performance,data.visibility,data.search_demand,data.action_matrix
    windows={"campaign_totals":_window(c.totals.report_date,days),
        "campaign_rows":[{"advert_id":r.advert_id,"period":_window(r.report_date,days)} for r in c.campaign_rows],
        "queries":_window(q.report_date,days),"demand_current":_window(s.report_date,days),
        "demand_previous":_window(s.report_date-timedelta(days=days),days) if s.report_date else None,
        "visibility_rows":[{"nm_id":r.nm_id,"vendor_code":r.vendor_code,"snapshot_date":str(r.snapshot_date) if r.snapshot_date else None} for r in v.visibility_rows],
        "visibility_overview_snapshot":str(v.overview.snapshot_date) if v.overview and v.overview.snapshot_date else None,
        "actions_shared_period":None,"action_reference_date":str(m.source_freshness.reference_date) if m.source_freshness.reference_date else None}
    signature=sha256(canonical(envelope["data"])).hexdigest()[:16]
    evidence=[{"domain":"marketing","claim":"native composite diagnostic; proposals are not decisions",
        "source":item.source,"period":None,"freshness":"unknown",
        "evidence_ref":f"marketing:{request_id}:{item.id}:{signature}"} for item in result.evidence]
    warnings=[*result.warnings,"scope:keyword_candidates_not_exact_sku_or_historical_links",
        "quality:native_proposals_not_approved_actions_no_execution","freshness:native_relative_status_not_wall_clock_coverage",
        "scope:hidden_overview_not_published_evidence"]
    dates=[data.report_date,c.totals.report_date,q.report_date,s.report_date,v.report_date,m.report_date,
        *(r.report_date for r in c.campaign_rows),*(r.snapshot_date for r in v.visibility_rows),
        *(getattr(m.source_freshness.sources,n).report_date for n in MARTS)]
    if any(d and d>today for d in dates):warnings.append("needs_data:composite_future_source_date")
    if any(r.prior_orders_count is None for r in s.search_text_rows):warnings.append("needs_data:demand_prior_missing_delta_not_confirmed_growth")
    if policy_version>=2 and v.visibility_rows:
        warnings.append("needs_data:visibility_window_end_not_observation_timestamp_aggregation_window_unknown")
    prepared={"request":request.model_dump(mode="json"),"capability":"latest_native_diagnostic","composite_policy_version":policy_version,
        "today":str(today),"source_mode":mode,"native_envelope":envelope,
        "source_windows":windows,"evidence":evidence,"warnings":sorted(set(warnings))}
    if policy_version>=3:
        from commerce_agent.marketing.composite_snapshot import prepare_snapshot
        snapshot,extra_evidence,extra_warnings=prepare_snapshot(snapshot_envelope,request,today,request_id,snapshot_mode)
        prepared["snapshot_context"]=snapshot;prepared["snapshot_context_mode"]=snapshot_mode
        prepared["evidence"].extend(extra_evidence);prepared["warnings"]=sorted(set([*prepared["warnings"],*extra_warnings]))
    assert_bounded_safe(prepared,max_bytes=262144)
    return prepared
