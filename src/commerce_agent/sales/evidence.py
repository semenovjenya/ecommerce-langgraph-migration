"""M2 tools → bounded Sales projections; native envelopes remain node-local."""
from __future__ import annotations

from datetime import timedelta
from hashlib import sha256
import json

from commerce_agent.adapters.contracts import (
    DatabaseTime, SalesSnapshot, SalesOverview, SalesRevenueChange,
    DailyPriceSalesComparison, SalesVendorCodeMap, ToolArtifactReference, ToolResult,
)
from commerce_agent.adapters.business_tools import ToolAdapterError
from commerce_agent.sales.contracts import ReportTable, EvidenceFact
from commerce_agent.sales.policy import assert_bounded_safe


async def _hydrate(client, context, result, model):
    if isinstance(result, ToolArtifactReference):
        return await client.hydrate(context, result, result_type=model)
    return ToolResult[model].model_validate(result.model_dump(mode="json"))


def _provenance(result, label, context):
    signature = sha256(json.dumps(result.data.model_dump(mode="json"), sort_keys=True,
                                  ensure_ascii=False).encode()).hexdigest()[:16]
    return [{"evidence_ref": f"sales:{context.request_id}:{item.id}:{signature}",
             "domain": "sales", "claim": label, "source": item.source,
             "period": f"{item.period.date_from}/{item.period.date_to}" if item.period else None,
             "freshness": "unknown"} for item in result.evidence]


def _check_scope(target, rows):
    # Legacy keyword SQL is fuzzy; do not mislabel a substring/brand match as
    # an exact SKU total. No post-hoc filtering of its scalar calculations.
    if target and any(str(row.nm_id) != target and
                      (row.vendor_code or "").casefold() != target.casefold() for row in rows):
        raise ValueError("Sales SKU filter is ambiguous")


def _table(title, columns, rows):
    return ReportTable(title=title, columns=columns, rows=rows).model_dump(mode="json")


async def _scope(client, context, request, target, today, *, retain_native=False):
    snapshot = await _hydrate(client, context,
        await client.sales_report_snapshot(context, product_keyword=target), SalesSnapshot)
    overview = await _hydrate(client, context,
        await client.sales_overview(context, product_keyword=target, limit=100), SalesOverview)
    anchor = snapshot.data.report_date
    totals = overview.data.totals
    if anchor is None or totals is None or totals.sales_date is None:
        raise ValueError("Sales source is missing")
    if anchor > today or totals.sales_date > today:
        raise ValueError("Sales source date is in the future")
    end = request.date_to or anchor
    start = request.date_from or end - timedelta(days=request.days - 1)
    if end > anchor:
        raise ValueError("Sales period exceeds complete source day")
    change = await _hydrate(client, context,
        await client.sales_revenue_change(context, date_from=str(start), date_to=str(end),
                                         product_keyword=target, limit=100), SalesRevenueChange)
    data = change.data
    previous_end = start - timedelta(days=1)
    previous_start = start - timedelta(days=(end - start).days + 1)
    if data.previous_period_start != previous_start or data.previous_period_end != previous_end:
        raise ValueError("Sales previous comparison period mismatch")
    if data.date_from != start or data.date_to != end or data.period_days != (end - start).days + 1:
        raise ValueError("Sales declared period mismatch")
    if data.current_period_start != start or data.current_period_end != end or any(
        row.current_period_start != start or row.current_period_end != end
        or row.previous_period_start != data.previous_period_start
        or row.previous_period_end != data.previous_period_end for row in data.rows):
        raise ValueError("Sales tool period mismatch")
    if change.period is None or change.period.date_from != start or change.period.date_to != end or any(
        item.period is None or item.period.date_from != start or item.period.date_to != end for item in change.evidence):
        raise ValueError("Sales provenance period mismatch")
    _check_scope(target, [*overview.data.top_revenue_rows, *overview.data.sales_raw_rows,
                          *overview.data.conversion_watchlist, *data.rows])
    if not data.rows:
        raise ValueError("Sales evidence is insufficient")
    evidence, warnings = [], []
    for label, result in (("Sales snapshot", snapshot), ("Sales overview", overview), ("Sales decomposition", change)):
        if not result.evidence:
            raise ValueError("Sales provenance is missing")
        evidence.extend(_provenance(result, label, context))
        warnings.extend(result.warnings)
    if request.date_from:
        warnings.append("scope:snapshot_and_overview_are_latest_not_requested_history")
    if not target:
        warnings.append("coverage:top_rows_are_not_a_complete_portfolio_matrix")
    if any(len(rows) == 100 for rows in (overview.data.top_revenue_rows, data.rows)):
        warnings.append("coverage:row_limit_reached")
    missing_ids = sorted({row.nm_id for row in [*overview.data.top_revenue_rows, *data.rows]
                          if not row.vendor_code and row.nm_id is not None})
    identities = {}
    for offset in range(0, len(missing_ids), 100):
        mapping = await _hydrate(client, context,
            await client.sales_vendor_code_map(context, nm_ids=missing_ids[offset:offset + 100]), SalesVendorCodeMap)
        identities.update(mapping.data.root)
    if retain_native and any(not (row.vendor_code or identities.get(str(row.nm_id))) for row in data.rows):
        # Reject this target inside the per-target failure boundary, not later
        # at whole-portfolio publication. Never invent a vendor identity.
        raise ValueError("Sales targeted source lacks canonical vendor code")
    def sku(row):
        value = row.vendor_code or identities.get(str(row.nm_id))
        if not value:
            warnings.append("needs_data:vendor_code_unmapped")
        return value or "Артикул не сопоставлен"
    current, previous = f"{start}/{end}", f"{data.previous_period_start}/{data.previous_period_end}"
    tables = {
        "sales_report_snapshot": snapshot.data.report_table.model_dump(mode="json"),
        "sales_overview": _table("Обзор продаж по последним данным", ["Метрика", "Значение", "Период"], [
            ["Продажи за 1 день, шт", totals.sales_1d_units, str(totals.sales_date)],
            ["Продажи за 7 дней, шт", totals.sales_7d_units, f"7 дней до {totals.sales_date}"],
            ["Выручка за 1 день, ₽", totals.revenue_1d, str(totals.sales_date)],
            ["Выручка за 7 дней, ₽", totals.revenue_7d, f"7 дней до {totals.sales_date}"],
            ["SKU с продажами за 7 дней", totals.sold_sku_7d, f"7 дней до {totals.sales_date}"],
        ]),
        "sales_dynamics": _table("Продажи и выручка по SKU за 7 дней", ["Артикул", "Продажи, шт", "Выручка, ₽", "Дата"],
            [[sku(row), row.sales_7d_units, row.revenue_7d, str(row.report_date)] for row in overview.data.top_revenue_rows]),
        "sales_decomposition": _table("Вклад SKU в изменение выручки и продаж", [
            "Артикул", "current_revenue", "previous_revenue", "delta_revenue",
            "current_sales", "previous_sales", "delta_sales", "Текущий период", "Предыдущий период",
            "delta_revenue_pct", "driver_role"],
            [[sku(row), row.current_revenue, row.previous_revenue, row.delta_revenue,
              row.current_sales, row.previous_sales, row.delta_sales, current, previous,
              row.delta_revenue_pct, row.driver_role] for row in data.rows]),
    }
    ref = _provenance(change, "Sales decomposition", context)[0]
    facts = [EvidenceFact(metric=metric, value=getattr(row, metric),
             unit="RUB" if "revenue" in metric else "units",
             period=previous if metric.startswith("previous") else current, sku=sku(row),
             source_table=ref["source"], evidence_ref=ref["evidence_ref"], source_agent="sales").model_dump(
                 mode="json", exclude_none=True, exclude_defaults=True)
             for row in data.rows for metric in ("current_revenue", "previous_revenue", "current_sales", "previous_sales")]
    # Row families have independent source dates and selection semantics.
    # Keep their native values, including zero/None, without making a causal
    # join or implying a complete funnel/portfolio universe.
    overview_signals = {"target": target, "causal_valid": False,
        "evidence_refs": [item["evidence_ref"] for item in _provenance(overview, "Sales overview", context)],
        "snapshot_comparisons": [row.model_dump(mode="json") for row in snapshot.data.rows],
        "snapshot_evidence_refs": [item["evidence_ref"] for item in _provenance(snapshot, "Sales snapshot", context)
                                   if item["source"] == "sales.wb_supplier_sales"],
        "funnel_evidence_refs": [item["evidence_ref"] for item in _provenance(overview, "Sales overview", context)
                                if item["source"] == "sales.wb_sales_funnel_products_daily"],
        "sales_raw_rows": [{"sku": sku(row), "sales_date": str(row.sales_date) if row.sales_date else None,
            "sales_count": row.sales_count, "revenue": row.revenue} for row in overview.data.sales_raw_rows],
        "conversion_watchlist": [{"sku": sku(row), "report_date": str(row.report_date),
            **{key: getattr(row, key) for key in ("open_count", "cart_count", "order_count",
                "add_to_cart_conversion", "cart_to_order_conversion", "buyout_percent")}}
            for row in overview.data.conversion_watchlist]}
    scope = {"target": target, "period": {"date_from": str(start), "date_to": str(end)},
            "tables": tables, "facts": facts, "evidence": evidence, "warnings": warnings,
            "overview_signals": overview_signals}
    if retain_native:
        scope.update(native_change=data.model_dump(mode="json"),native_identities=identities)
    return scope


async def collect_sales_evidence(client, context, request, *, publish_targeted=False,publish_daily=False,scoped_policy_version=3):
    """Multiple SKU totals stay separate; no joined keyword or new aggregate math."""
    clock = await _hydrate(client, context, await client.database_time(context), DatabaseTime)
    targets = (None,) if request.scope == "portfolio" else request.targets
    scopes, scope_warnings = [], []
    for target in targets:
        try:
            scope = await _scope(client, context, request, target, clock.data.db_current_date,
                                 retain_native=publish_targeted and request.scope != "portfolio")
            if publish_targeted and request.scope != "portfolio":
                for fact in scope["facts"]:
                    fact.update(target_type="sku", target_id=target)
            scopes.append(scope)
        except (ValueError, ToolAdapterError):
            # Keep valid independent SKU results, but never reinterpret a
            # failed target as zero or imply complete coverage.
            if request.scope != "sku_list":
                raise
            scope_warnings.append(f"needs_data:target_unavailable:{target}")
    if not scopes:
        raise ValueError("Sales evidence is insufficient")
    evidence = [item for scope in scopes for item in scope["evidence"]]
    warnings = [*scope_warnings, *(item for scope in scopes for item in scope["warnings"])]
    tables = scopes[0]["tables"]
    if request.scope == "sku_list":
        tables = {}
        for key, table in scopes[0]["tables"].items():
            if any(scope["tables"][key]["columns"] != table["columns"] for scope in scopes):
                raise ValueError("Sales table schema mismatch")
            tables[key] = _table(table["title"], ["Запрошенный артикул", *table["columns"]], [
                [scope["target"], *row] for scope in scopes for row in scope["tables"][key]["rows"]])
    deep_dives, targeted, daily_scopes = [], [], []
    if request.scope == "portfolio":
        for target in request.targets:
            try:
                scoped = await _scope(client, context, request, target, clock.data.db_current_date,retain_native=publish_targeted)
            except (ValueError, ToolAdapterError):
                warnings.append(f"needs_data:target_unavailable:{target}")
                continue
            if publish_targeted:
                for fact in scoped["facts"]:
                    fact.update(target_type="sku",target_id=target)
                targeted.append({key:scoped[key] for key in ("target","period","tables","facts","overview_signals","native_change","native_identities")})
            else:
                deep_dives.append({"target": target, "period": scoped["period"], "tables": scoped["tables"],
                                   "overview_signals": scoped["overview_signals"]})
            evidence.extend(scoped["evidence"])
            warnings.extend(scoped["warnings"])
    # Daily reader has no historical bounds; don't silently clip >31 days or
    # claim latest signals cover a historical request. Explicitly non-causal.
    if request.targets and request.date_from is None and request.days <= 31:
        for target in request.targets:
            try:
                daily = await _hydrate(client, context,
                    await client.daily_price_sales_comparison(context, product_keyword=target, days=request.days),
                    DailyPriceSalesComparison)
                _check_scope(target, daily.data.rows)
                if publish_daily:
                    from commerce_agent.sales.daily import source_facts
                    daily_scope={"target":target,"native_daily":daily.data.model_dump(mode="json"),
                        "source_refs":_provenance(daily,"Daily Sales signals (non-causal)",context)}
                    daily_scope["facts"]=source_facts(daily_scope,request.model_dump(mode="json"),str(clock.data.db_current_date))
            except (ValueError, ToolAdapterError):
                warnings.append(f"needs_data:daily_signal_unavailable:{target}")
                continue
            evidence.extend(_provenance(daily, "Daily Sales signals (non-causal)", context))
            warnings.extend(daily.warnings)
            if publish_daily:
                daily_scopes.append(daily_scope)
                warnings.append("scope:daily_price_anchor_independent_of_weekly_sales_window")
                warnings.append("causality:daily_price_sales_observation_not_a_price_effect")
            deep_dives.append({"target": target, "causal_valid": False, "daily_rows": [
                {"date": str(row.report_date), "sales_units": row.realized_sales_units,
                 "revenue": row.realized_for_pay_rub, "price_basis": row.price_basis,
                 **{key: getattr(row, key) for key in ("non_cancelled_order_units", "open_count", "cart_count",
                     "funnel_order_count", "add_to_cart_conversion", "cart_to_order_conversion", "buyout_percent",
                     "business_day_complete", "marketing_day_complete", "sales_day_complete")}}
                 for row in daily.data.rows]})
    elif request.targets:
        warnings.append("scope:daily_signal_skipped_no_matching_period_capability")
    prepared = {"request": request.model_dump(mode="json"), "today": str(clock.data.db_current_date),
                "periods": [{"target": scope["target"], **scope["period"]} for scope in scopes],
                "tables": tables, "facts": [fact for scope in scopes for fact in scope["facts"]],
                "evidence": evidence, "warnings": sorted(set(warnings)), "deep_dives": deep_dives,
                "overview_signals": [scope["overview_signals"] for scope in scopes]}
    if publish_targeted and request.scope == "portfolio" and request.targets:
        prepared.update(targeted_sales_policy_version=2,targeted_sales_scopes=targeted)
        prepared["facts"].extend(fact for scope in targeted for fact in scope["facts"])
        from commerce_agent.sales.targeted import targeted_scopes
        targeted_scopes(prepared)
    if publish_targeted and request.scope != "portfolio":
        if type(scoped_policy_version) is not int or scoped_policy_version not in (1, 2, 3):
            raise ValueError("Sales scoped writer policy unsupported")
        prepared.update(scoped_sales_policy_version=scoped_policy_version, scoped_sales_scopes=[
            {key: scope[key] for key in ("target", "period", "tables", "facts", "overview_signals", "native_change", "native_identities")}
            for scope in scopes])
        from commerce_agent.sales.scoped import scoped_scopes
        scoped_scopes(prepared)
    if publish_daily and request.targets:
        prepared.update(daily_sales_policy_version=1,daily_sales_scopes=daily_scopes)
        prepared["facts"].extend(fact for scope in daily_scopes for fact in scope["facts"])
        from commerce_agent.sales.daily import daily_scopes as validate_daily
        validate_daily(prepared)
    assert_bounded_safe(prepared)
    return prepared
