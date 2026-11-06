"""Native Marketing metrics only: no aggregation, allocation or financial math."""
from datetime import timedelta
from hashlib import sha256
import json

from commerce_agent.adapters.contracts import (DatabaseTime, MarketingSnapshot,
    MarketingCampaignPeriodReport, MarketingSkuComparison, ToolResult, ToolArtifactReference)
from commerce_agent.sales.contracts import EvidenceFact, ReportTable
from commerce_agent.marketing.policy import assert_bounded_safe


async def _hydrate(client, context, value, model):
    if isinstance(value, ToolArtifactReference):
        return await client.hydrate(context, value, result_type=model)
    return ToolResult[model].model_validate(value.model_dump(mode="json"))


SOURCES = {
    "snapshot": {"marketing.wb_marketing_campaign_stats_daily", "marketing.wb_marketing_campaign_nms",
                 "sales.wb_supplier_sales", "sales.wb_sales_funnel_products_daily"},
    "sku": {"marketing.wb_marketing_campaign_stats_daily", "sales.wb_supplier_sales"},
    "campaigns": {"marketing.wb_marketing_campaign_stats_daily"},
}


def _provenance(value, label, context):
    if not value.evidence or {item.source for item in value.evidence} != SOURCES[label]:
        raise ValueError("Marketing source provenance missing")
    digest = sha256(json.dumps(value.data.model_dump(mode="json"), ensure_ascii=False,
                               sort_keys=True).encode()).hexdigest()[:16]
    return [{"domain": "marketing", "claim": label, "source": item.source,
        "evidence_ref": f"marketing:{context.request_id}:{item.id}:{digest}",
        "period": f"{item.period.date_from}/{item.period.date_to}" if item.period else None,
        "freshness": "unknown"} for item in value.evidence]


def _table(title, rows):
    if not rows:
        return ReportTable(title=title, columns=["Данные"], rows=[]).model_dump(mode="json")
    keys = list(rows[0])
    return ReportTable(title=title, columns=keys, rows=[[row[key] for key in keys] for row in rows]).model_dump(mode="json")


async def collect_marketing_evidence(client, context, request):
    clock = await _hydrate(client, context, await client.database_time(context), DatabaseTime)
    snapshot = await _hydrate(client, context, await client.marketing_report_snapshot(context), MarketingSnapshot)
    today, anchor = clock.data.db_current_date, snapshot.data.report_date
    if anchor is None or anchor > today or len(snapshot.data.rows) != 3 or snapshot.data.focus_keyword is not None:
        raise ValueError("Marketing snapshot is unavailable")
    # Report anchor is service-owned min(complete Marketing, realized Sales).
    # Never substitute latest Marketing day or current calendar yesterday.
    end = request.date_to or anchor
    start = request.date_from or end - timedelta(days=request.days - 1)
    if end > anchor:
        raise ValueError("Marketing requested period exceeds complete source day")
    days = (end - start).days + 1
    sku = await _hydrate(client, context, await client.marketing_sku_period_comparison(context,
        date_from=str(start), date_to=str(end), period_days=days, limit=100), MarketingSkuComparison)
    campaigns = await _hydrate(client, context, await client.marketing_campaign_period_report(context,
        date_from=str(start), date_to=str(end), limit=200), MarketingCampaignPeriodReport)
    previous_end, previous_start = start - timedelta(days=1), start - timedelta(days=days)
    if (sku.data.date_from != start or sku.data.date_to != end or sku.data.period_days != days
        or sku.data.previous_date_from != previous_start or sku.data.previous_date_to != previous_end
        or sku.data.focus_keyword is not None or sku.data.focus_sku_list
        or campaigns.data.date_from != start or campaigns.data.date_to != end
        or sku.data.row_count != len(sku.data.rows) or campaigns.data.row_count != len(campaigns.data.rows)
        or not sku.data.rows or not campaigns.data.rows):
        raise ValueError("Marketing period/scope/coverage mismatch")
    for row in sku.data.rows:
        if (row.current_date_from != start or row.current_date_to != end
            or row.previous_date_from != previous_start or row.previous_date_to != previous_end
            or not row.vendor_code):
            raise ValueError("Marketing SKU identity or period missing")
    for days_count, row in zip((1, 7, 30), snapshot.data.rows, strict=True):
        current, before = row.current_period, row.previous_period
        if (current.date_to != anchor or current.date_from != anchor - timedelta(days=days_count - 1)
            or before.date_to != current.date_from - timedelta(days=1)
            or before.date_from != current.date_from - timedelta(days=days_count)):
            raise ValueError("Marketing native comparison windows mismatch")
    if campaigns.period is None or campaigns.period.date_from != start or campaigns.period.date_to != end or any(
        item.period is None or item.period.date_from != start or item.period.date_to != end
        for item in campaigns.evidence):
        raise ValueError("Marketing campaign provenance period mismatch")
    # SKU M2 provenance currently has no period: preserve unknown, but reject
    # contradictory non-null provenance instead of relabeling it as requested.
    for period in [sku.period, *(item.period for item in sku.evidence)]:
        if period is not None and (period.date_from != start or period.date_to != end):
            raise ValueError("Marketing SKU provenance period mismatch")
    evidence, warnings, refs = [], [], {}
    for key, value in (("snapshot", snapshot), ("sku", sku), ("campaigns", campaigns)):
        native_refs = _provenance(value, key, context)
        refs[key] = [item["evidence_ref"] for item in native_refs]
        evidence.extend(native_refs)
        warnings.extend(value.warnings)
    warnings.extend(("freshness:source_completeness_unverified", "semantics:drr_is_not_ad_revenue_or_profit",
                     "scope:snapshot_latest_windows_separate_from_requested_week"))
    if len(sku.data.rows) == 100 or len(campaigns.data.rows) == 200:
        warnings.append("coverage:row_limit_reached")
    current, previous = f"{start}/{end}", f"{previous_start}/{previous_end}"
    def metric_refs(family, metric):
        allowed = {"sales.wb_supplier_sales"} if metric == "promoted_sku_revenue_rub" else (
            {"marketing.wb_marketing_campaign_stats_daily", "sales.wb_supplier_sales"}
            if metric in {"drr_pct", "estimated_drr_pct"} else {"marketing.wb_marketing_campaign_stats_daily"})
        native = [item for item in evidence if item["evidence_ref"] in refs[family] and item["source"] in allowed]
        if {item["source"] for item in native} != allowed:
            raise ValueError("Marketing metric-specific provenance missing")
        return [item["evidence_ref"] for item in native]
    sku_rows = []
    for row in sku.data.rows:
        # Do not publish product names/raw documents/credentials. Keep exact
        # native numeric identities internally for scoped downstream checks.
        sku_rows.append({key: value for key, value in row.model_dump(mode="json").items()
                         if key not in {"subject_name", "nm_id"}})
    campaign_rows = [{key: getattr(row, key) for key in ("advert_id", "impressions", "clicks", "ctr_pct",
        "cpc_rub", "cr_pct", "add_to_cart", "ordered_items", "spend_rub", "spend_share_pct")}
        for row in campaigns.data.rows]
    tables = [snapshot.data.report_table.model_dump(mode="json"),
        _table(f"Кампании за {current}; текущие названия/статусы не доказывают историю", campaign_rows),
        _table(f"Динамика SKU за {current} / {previous}; атрибуция указана в каждой строке", sku_rows)]
    claims, facts = [], []
    def add_claim(kind, metric, label, a, b, unit, period, prior_period, delta, delta_unit, references,
                  *, sku_code=None, nm_id=None, campaign_id=None, attribution=None):
        if a is None and b is None:
            return
        absent = lambda value: "нет данных" if value is None else str(value)
        scope = f"SKU {sku_code}" if sku_code else f"кампания {campaign_id}" if campaign_id is not None else "портфель"
        text = f"{label}, {scope}: {absent(a)} {unit} за {period}"
        if prior_period:
            text += f" против {absent(b)} {unit} за {prior_period}"
        if delta is not None:
            text += f"; изменение по данным сервиса {delta} {delta_unit}"
        if attribution:
            text += ("; оценочное равное распределение статистики кампании, не точная SKU-атрибуция"
                     if attribution == "estimated_allocated_from_campaign" else
                     "; campaign_single_sku, без доказательства причинного рекламного эффекта")
        text += "."
        claims.append({"kind": kind, "metric": metric, "text": text, "period": period,
            "previous_period": prior_period, "sku": sku_code, "nm_id": nm_id, "campaign_id": campaign_id,
            "attribution": attribution, "evidence_refs": references})
        for prefix, value, window in (("current_", a, period), ("previous_", b, prior_period)):
            if window:
                facts.append(EvidenceFact(metric=prefix + metric, value=value, unit=unit, period=window,
                    sku=sku_code, target_type="campaign" if campaign_id is not None else "sku" if sku_code else "portfolio",
                    target_id=str(campaign_id) if campaign_id is not None else sku_code,
                    source_table=evidence[next(index for index, item in enumerate(evidence)
                        if item["evidence_ref"] == references[0])]["source"], evidence_ref=references[0],
                    source_agent="marketing", metric_definition=attribution or label).model_dump(mode="json"))
    for row in snapshot.data.rows:
        period = f"{row.current_period.date_from}/{row.current_period.date_to}"
        before = f"{row.previous_period.date_from}/{row.previous_period.date_to}"
        for metric, label, unit, delta, delta_unit in (
            ("spend_rub", "Расходы рекламы", "RUB", "spend_delta_rub", "RUB"),
            ("clicks", "Клики", "clicks", "clicks_delta", "clicks"),
            ("ad_orders", "Заказы в статистике рекламы", "orders", "ad_orders_delta", "orders"),
            ("promoted_sku_revenue_rub", "Вся реализованная выручка продвигаемых SKU, не рекламная атрибуция",
             "RUB", "promoted_sku_revenue_delta_rub", "RUB"),
            ("drr_pct", "ДРР относительно всей выручки продвигаемых SKU, не ROI", "%", "drr_delta_pp", "п.п."),
            ("cpc_rub", "CPC", "RUB/click", None, None)):
            add_claim("portfolio", metric, label, getattr(row, "current_" + metric),
                getattr(row, "previous_" + metric), unit, period, before,
                getattr(row, delta) if delta else None, delta_unit, metric_refs("snapshot", metric))
    for row in sku.data.rows:
        for metric, label, unit, delta in (("spend_rub", "Расходы рекламы", "RUB", "spend_delta_rub"),
            ("clicks", "Клики", "clicks", "clicks_delta"),
            ("ad_orders", "Заказы в статистике рекламы", "orders", "ad_orders_delta"),
            ("estimated_drr_pct", "Оценочный ДРР ко всей реализованной выручке SKU, не ROI", "%", None)):
            add_claim("sku", metric, label, getattr(row, "current_" + metric), getattr(row, "previous_" + metric),
                unit, current, previous, getattr(row, delta) if delta else None, unit, metric_refs("sku", metric),
                sku_code=row.vendor_code, nm_id=row.nm_id, attribution=row.attribution_quality)
    for row in campaigns.data.rows:
        if row.advert_id is None:
            raise ValueError("Marketing campaign identity missing")
        add_claim("campaign", "spend_rub", "Расходы рекламы кампании", row.spend_rub, None,
            "RUB", current, None, None, None, refs["campaigns"], campaign_id=row.advert_id)
    prepared = {"request": request.model_dump(mode="json"), "today": str(today), "report_date": str(anchor),
        "period": current, "previous_period": previous, "tables": tables, "facts": facts,
        "claims": claims, "evidence": evidence, "warnings": sorted(set(warnings))}
    assert_bounded_safe(prepared, max_bytes=262144)
    return prepared
