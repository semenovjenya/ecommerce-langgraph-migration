"""Read-only diagnostic collection with independent source/window contracts.

Opt-in Marketing capability. Exact SKU comparisons are distinct
from keyword-selected campaigns and snapshot visibility. No derived metrics,
write proposals, or substituted zeroes are produced here.
"""
from datetime import date, timedelta
from hashlib import sha256
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, model_validator

from commerce_agent.adapters.contracts import (
    DatabaseTime, MarketingSkuComparison, MarketingEfficiency,
    MarketingQueryPerformance, MarketingBidPressure, MarketingVisibility,
    ToolArtifactReference,
)
from commerce_agent.marketing.evidence import _hydrate
from commerce_agent.marketing.policy import assert_bounded_safe


class MarketingDiagnosticRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    capability: Literal["sku_diagnostic"] = "sku_diagnostic"
    user_query: StrictStr = Field(min_length=1, max_length=8000)
    # Literal vendor code, NOT nm_id or fuzzy keyword chosen by the model.
    sku: StrictStr = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    date_from: date
    date_to: date

    @model_validator(mode="after")
    def bounded(self):
        if not 0 <= (self.date_to - self.date_from).days < 30:
            raise ValueError("Marketing diagnostic period must be 1-30 days")
        assert_bounded_safe(self.model_dump(mode="json"))
        return self


SOURCE_ALLOWLIST = {
    "sku": {"marketing.wb_marketing_campaign_stats_daily", "sales.wb_supplier_sales"},
    "campaigns": {"marketing.wb_marketing_campaign_stats_daily"},
    "queries": {"marketing.wb_search_query_orders_text_daily"},
    "bids": set(),
    "visibility": {"marketing.wb_search_query_details", "marketing.wb_search_query_report_overview"},
}


def _check_window(value, start, end):
    for period in (value.period, *(item.period for item in value.evidence)):
        if period is not None and (period.date_from != start or period.date_to != end):
            raise ValueError("Marketing diagnostic provenance period mismatch")


def _period(end, days):
    return f"{end - timedelta(days=days - 1)}/{end}" if end else None


async def collect_marketing_diagnostic(client, context, request: MarketingDiagnosticRequest, *, allow_debug=False):
    """Collect six typed reads. Cancellation propagates; adapter owns read retries.

    BusinessToolClient applies credentials/actor checks privately. Failures are
    raised to the future subgraph's static error boundary, never persisted with
    raw transport exceptions. Hydration uses existing owned bounded retrieval.
    """
    from commerce_agent.debug_privacy import development_debug_enabled
    if development_debug_enabled():
        if allow_debug is not True:
            raise ValueError("Marketing diagnostic debug not accepted")
        from commerce_agent.marketing.debug_export import assert_diagnostic_debug_request
        assert_diagnostic_debug_request(request)
    source_mode = "inline"
    async def hydrate(value, model):
        nonlocal source_mode
        if isinstance(value,ToolArtifactReference):
            source_mode = "hydrated"
            if development_debug_enabled():
                raise ValueError("Marketing diagnostic artifact debug not accepted")
        return await _hydrate(client,context,value,model)
    clock = await hydrate(await client.database_time(context), DatabaseTime)
    today = clock.data.db_current_date
    if request.date_to > today:
        raise ValueError("Marketing diagnostic requested period is in the future")
    start, end = request.date_from, request.date_to
    days = (end - start).days + 1
    sku = await hydrate(await client.marketing_sku_period_comparison(context,
        date_from=str(start), date_to=str(end), sku_list=request.sku, product_keyword=None,
        period_days=days, limit=100), MarketingSkuComparison)
    data = sku.data
    previous_end, previous_start = start - timedelta(days=1), start - timedelta(days=days)
    if (data.date_from != start or data.date_to != end or data.period_days != days
        or data.previous_date_from != previous_start or data.previous_date_to != previous_end
        or data.focus_keyword is not None or data.focus_sku_list != [request.sku]
        or data.row_count != len(data.rows) or len(data.rows) != 1):
        raise ValueError("Marketing diagnostic exact SKU coverage mismatch")
    row = data.rows[0]
    if (row.vendor_code != request.sku or row.nm_id is None or row.nm_id <= 0
        or row.current_date_from != start or row.current_date_to != end
        or row.previous_date_from != previous_start or row.previous_date_to != previous_end):
        raise ValueError("Marketing diagnostic identity/window mismatch")
    _check_window(sku, start, end)
    values = {"sku": sku}
    for label, method, model in (
        ("campaigns", client.marketing_campaign_efficiency, MarketingEfficiency),
        ("queries", client.marketing_query_performance, MarketingQueryPerformance),
        ("bids", client.marketing_bid_pressure, MarketingBidPressure),
    ):
        values[label] = await hydrate(await method(context, product_keyword=request.sku, days=days,
                         limit=20 if label == "campaigns" else 30), model)
        if label=="queries" and development_debug_enabled() and any(getattr(values[label].data,key) for key in
            ("top_query_rows","query_waste_rows","query_scale_rows","query_position_risk_rows")):
            raise ValueError("Marketing populated diagnostic query debug not accepted")
    values["visibility"] = await hydrate(await client.marketing_visibility(context, product_keyword=request.sku, limit=30), MarketingVisibility)

    warnings, evidence, windows = [], [], {"sku": f"{start}/{end}"}
    for label, value in values.items():
        native = value.data
        if label != "sku":
            if native.focus_keyword != request.sku or (label != "visibility" and native.period_days != days):
                raise ValueError("Marketing diagnostic returned scope mismatch")
            if native.report_date is not None and native.report_date > today:
                raise ValueError("Marketing diagnostic future source date")
        actual_sources = {item.source for item in value.evidence}
        if not actual_sources <= SOURCE_ALLOWLIST[label]:
            raise ValueError("Marketing diagnostic unknown evidence source")
        if any(item.grain != ("portfolio" if item.source == "marketing.wb_search_query_report_overview"
                              else "product_filter") for item in value.evidence):
            raise ValueError("Marketing diagnostic evidence grain mismatch")
        digest = sha256(json.dumps(native.model_dump(mode="json"), ensure_ascii=False,
            sort_keys=True, allow_nan=False).encode()).hexdigest()[:16]
        for item in value.evidence:
            evidence.append({"domain": "marketing", "claim": label, "source": item.source,
                "evidence_ref": f"marketing:{context.request_id}:{item.id}:{digest}",
                "period": f"{item.period.date_from}/{item.period.date_to}" if item.period else None,
                "freshness": "unknown"})
        warnings.extend(value.warnings)

    campaigns, queries, bids, visibility = (values[key].data for key in ("campaigns", "queries", "bids", "visibility"))
    sku_sources = {item.source for item in sku.evidence}
    if "marketing.wb_marketing_campaign_stats_daily" not in sku_sources:
        raise ValueError("Marketing diagnostic SKU advertising provenance missing")
    if "sales.wb_supplier_sales" not in sku_sources:
        # Native zero sales may be a coalesced missing source. Preserve the
        # payload, never elevate absence of source evidence into a proved zero.
        warnings.append("needs_data:realized_sales_coverage_unconfirmed")
    if any(item.grain != "product_filter" for item in sku.evidence):
        raise ValueError("Marketing diagnostic SKU evidence grain mismatch")
    if campaigns.campaign_rows and not values["campaigns"].evidence:
        raise ValueError("Marketing diagnostic campaign provenance missing")
    for item in [campaigns.totals, *campaigns.campaign_rows, *campaigns.waste_rows, *campaigns.scale_rows]:
        if item.report_date and item.report_date > today:
            raise ValueError("Marketing diagnostic future campaign date")
    if any(item.advert_id is None or item.advert_id <= 0 for item in campaigns.campaign_rows):
        raise ValueError("Marketing diagnostic campaign identity missing")
    for subset in (campaigns.waste_rows, campaigns.scale_rows):
        if any(item not in campaigns.campaign_rows for item in subset):
            raise ValueError("Marketing diagnostic campaign subset mismatch")
    # Independently anchored totals/rows: shared report_date is not a join key.
    anchors = [campaigns.totals.report_date, *(item.report_date for item in campaigns.campaign_rows)]
    shared_campaign_anchor = campaigns.report_date if all(d == campaigns.report_date for d in anchors) else None
    windows["campaigns"] = _period(shared_campaign_anchor, days)
    warnings.extend(("scope:campaign_keyword_candidates_not_verified_historical_sku_links",
        "scope:campaign_native_cpc_not_sku_cpc", "freshness:source_completeness_unverified"))
    for item in campaigns.campaign_rows:
        if item.report_date != campaigns.report_date:
            warnings.append("scope:campaign_rows_have_independent_windows")
    if campaigns.totals.report_date != campaigns.report_date:
        warnings.append("scope:campaign_totals_have_independent_window")
    if shared_campaign_anchor is None:
        warnings.append("needs_data:shared_campaign_window_unconfirmed")
    # A non-null envelope campaign window is permitted only if ALL anchors agree.
    campaign_periods = [values["campaigns"].period, *(item.period for item in values["campaigns"].evidence)]
    if any(period is not None for period in campaign_periods):
        if shared_campaign_anchor is None:
            raise ValueError("Marketing diagnostic false shared campaign window")
        _check_window(values["campaigns"], campaigns.report_date - timedelta(days=days-1), campaigns.report_date)

    def check_sku_rows(rows):
        for item in rows:
            if item.vendor_code != request.sku or item.nm_id != row.nm_id:
                raise ValueError("Marketing diagnostic keyword collision/identity mismatch")
    for subset in (queries.query_waste_rows, queries.query_scale_rows, queries.query_position_risk_rows):
        if any(item not in queries.top_query_rows for item in subset):
            raise ValueError("Marketing diagnostic query subset mismatch")
    check_sku_rows(queries.top_query_rows)
    if queries.top_query_rows:
        if queries.report_date is None or any(item.report_date != queries.report_date for item in queries.top_query_rows):
            raise ValueError("Marketing diagnostic query date mismatch")
        if {item.source for item in values["queries"].evidence} != SOURCE_ALLOWLIST["queries"]:
            raise ValueError("Marketing diagnostic query provenance missing")
    elif values["queries"].evidence:
        raise ValueError("Marketing diagnostic empty query has evidence")
    windows["queries"] = _period(queries.report_date, days)
    if queries.report_date:
        _check_window(values["queries"], queries.report_date - timedelta(days=days-1), queries.report_date)
    elif values["queries"].period:
        raise ValueError("Marketing diagnostic query window without source date")
    # Native spend=0/cpc=null belong to a search-frequency proxy, NOT paid terms.
    warnings.extend(("semantics:query_views_clicks_search_frequency_proxy_not_paid_clicks",
        "needs_data:query_ad_spend_and_cpc_not_available", "needs_data:bid_capability_not_implemented"))
    if values["bids"].evidence or values["bids"].period:
        raise ValueError("Marketing diagnostic bid source is not implemented")
    windows["bids"] = None

    check_sku_rows(visibility.visibility_rows)
    for subset in (visibility.position_loss_rows, visibility.open_card_gap_rows, visibility.post_click_conversion_rows):
        if any(item not in visibility.visibility_rows for item in subset):
            raise ValueError("Marketing diagnostic visibility subset mismatch")
    visibility_dates = [item.snapshot_date for item in visibility.visibility_rows]
    if visibility.overview:
        visibility_dates.append(visibility.overview.snapshot_date)
    if any(d is not None and d > today for d in visibility_dates):
        raise ValueError("Marketing diagnostic future visibility date")
    if any(d is None for d in visibility_dates):
        warnings.append("freshness:visibility_source_date_unconfirmed")
    if visibility.visibility_rows and "marketing.wb_search_query_details" not in {
        item.source for item in values["visibility"].evidence}:
        raise ValueError("Marketing diagnostic visibility provenance missing")
    overview_present = visibility.overview and any(value is not None for key, value in
        visibility.overview.model_dump().items() if key != "snapshot_date")
    overview_evidence = [item for item in values["visibility"].evidence
        if item.source == "marketing.wb_search_query_report_overview"]
    if overview_present and (not overview_evidence or any(item.grain != "portfolio" for item in overview_evidence)):
        raise ValueError("Marketing diagnostic portfolio visibility provenance missing")
    if not overview_present and overview_evidence:
        raise ValueError("Marketing diagnostic empty portfolio visibility has evidence")
    if values["visibility"].period is not None or any(item.period for item in values["visibility"].evidence):
        raise ValueError("Marketing diagnostic visibility is not requested-period data")
    windows["visibility"] = None
    warnings.extend(("scope:visibility_snapshot_not_requested_period",
        "scope:visibility_overview_is_portfolio_not_sku"))
    for label in ("campaigns", "queries"):
        if windows[label] != windows["sku"]:
            warnings.append(f"scope:{label}_window_differs_from_requested_sku_period")
    if not campaigns.campaign_rows:
        warnings.append("needs_data:keyword_campaigns_missing")
    if not queries.top_query_rows:
        warnings.append("needs_data:search_query_rows_missing_not_zero_paid_spend")
    if not visibility.visibility_rows:
        warnings.append("needs_data:sku_visibility_missing")
    for label, rows, limit in (("campaigns", campaigns.campaign_rows, 20),
                              ("queries", queries.top_query_rows, 30),
                              ("visibility", visibility.visibility_rows, 30)):
        if len(rows) > limit:
            raise ValueError("Marketing diagnostic source exceeded requested limit")
        if len(rows) == limit:
            warnings.append(f"scope:{label}_limit_may_hide_tail")
    prepared = {"request": request.model_dump(mode="json"), "today": str(today),
        "sku": request.sku, "nm_id": row.nm_id, "requested_period": f"{start}/{end}",
        "source_windows": windows, "evidence": evidence, "warnings": sorted(set(warnings)),
        "diagnostic_source_mode": source_mode,
        "campaign_row_windows": [{"advert_id": item.advert_id,
            "period": _period(item.report_date, days)} for item in campaigns.campaign_rows],
        "campaign_totals_window": _period(campaigns.totals.report_date, days),
        "native_data": {key: value.data.model_dump(mode="json") for key, value in values.items()}}
    assert_bounded_safe(prepared, max_bytes=262144)
    return prepared
