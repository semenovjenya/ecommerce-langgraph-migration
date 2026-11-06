"""Prepare native Supply data without formulas, SQL or inferred stock totals."""
from datetime import date, timedelta
from hashlib import sha256
import json

from commerce_agent.adapters.contracts import (
    DatabaseTime, InventoryTurnover, InventoryTurnoverAlternatives, InventoryValue, SupplyOverview, ToolArtifactReference, ToolResult,
)
from commerce_agent.sales.contracts import EvidenceFact, ReportTable
from commerce_agent.supply.policy import assert_bounded_safe

STOCK = "supply.wb_supplier_stocks"
SALES = "sales.wb_supplier_sales"
COST = "finance.wb_product_coast"
SOURCES = {"overview": {STOCK, SALES, "supply.wb_supplies", "supply.wb_supplies_goods"},
           "turnover": {STOCK, SALES}, "value": {STOCK, COST}}
METRIC_LABELS = {"stock_qty": "доступный остаток", "stock_qty_full": "полный остаток",
    "available_stock_qty": "доступный остаток", "full_stock_qty": "полный остаток",
    "unavailable_stock_qty": "недоступный остаток", "days_of_stock": "покрытие текущим запасом",
    "reorder_qty_recommended": "рекомендация дозаказа", "sales_units": "продажи",
    "sales_revenue": "выручка продаж", "avg_stock_qty": "средний доступный остаток",
    "avg_stock_qty_full": "средний полный остаток", "turnover_ratio": "оборачиваемость",
    "days_on_hand": "дни запаса за период", "unit_cost_rub": "себестоимость единицы",
    "current_stock_value_rub": "стоимость доступного запаса"}
UNIT_LABELS = {"units": "шт", "RUB": "₽", "RUB/unit": "₽/шт", "turns": "оборота", "days": "дней"}


async def _hydrate(client, context, value, model):
    if isinstance(value, ToolArtifactReference):
        return await client.hydrate(context, value, result_type=model)
    return ToolResult[model].model_validate(value.model_dump(mode="json"))


def _table(title, rows, keys):
    return ReportTable(title=title, columns=list(keys),
        rows=[[row[key] for key in keys] for row in rows]).model_dump(mode="json")


async def collect_supply_evidence(client, context, request, *, enable_alternative_methods=False,
                                  alternative_boundary_tolerance_days=0):
    if type(enable_alternative_methods) is not bool or type(alternative_boundary_tolerance_days) is not int or not 0 <= alternative_boundary_tolerance_days <= 31:
        raise ValueError("invalid Supply alternative capability")
    if enable_alternative_methods:
        from commerce_agent.debug_privacy import development_debug_enabled
        if development_debug_enabled():
            raise ValueError("Supply alternative sources require separate debug acceptance")
    clock = await _hydrate(client, context, await client.database_time(context), DatabaseTime)
    keyword = None if request.scope == "portfolio" else ",".join(request.targets)
    overview = await _hydrate(client, context, await client.supply_overview(context,
        product_keyword=keyword, limit=100), SupplyOverview)
    today = clock.data.db_current_date
    anchor = overview.data.simulation.source_snapshot_date
    if overview.data.focus_keyword != keyword or (anchor is not None and anchor > today):
        raise ValueError("Supply overview source/scope mismatch")
    if (overview.data.simulation.enabled != (overview.data.data_mode == "simulation_latest_known")
        or overview.data.simulation.effective_status != ("simulated_current" if overview.data.simulation.enabled else "observed_latest_known")
        or overview.data.sku_stock_lookup.data_mode != overview.data.data_mode
        or overview.data.sku_stock_lookup.simulation != overview.data.simulation):
        raise ValueError("Supply simulation provenance mismatch")
    envelopes = {"overview": overview}
    chosen_period = str(anchor) if anchor else "unknown"
    if request.operation in {"turnover", "matrix"}:
        if request.date_to is None and anchor is None:
            raise ValueError("Supply turnover source anchor unavailable")
        end = request.date_to or min(anchor, today - timedelta(days=1))
        start = request.date_from or end - timedelta(days=request.days - 1)
        if end >= today:
            raise ValueError("Supply turnover requires a past period")
        turnover = await _hydrate(client, context, await client.inventory_turnover(context,
            date_from=str(start), date_to=str(end), vendor_codes=keyword, limit=100), InventoryTurnover)
        data = turnover.data
        if (data.date_from != start or data.date_to != end
            or set(data.vendor_codes) != set(request.targets if keyword else ())
            or data.exact_available != (data.calculation_status == "exact")
            or (data.totals and (data.totals.date_from != start or data.totals.date_to != end))
            or len(data.rows) != len(data.sku_metric_matrix)
            or [row.vendor_code for row in data.rows] != [row.sku for row in data.sku_metric_matrix]):
            raise ValueError("Supply turnover period/matrix mismatch")
        for period in [turnover.period, *(item.period for item in turnover.evidence)]:
            if period is None or period.date_from != start or period.date_to != end:
                raise ValueError("Supply turnover period provenance mismatch")
        # Matrix is the public native calculator output, not a second
        # independently computed result. Verify its duplicate channels agree.
        for row, matrix in zip(data.rows, data.sku_metric_matrix, strict=True):
            if row.sku_conclusion != matrix.conclusion:
                raise ValueError("Supply duplicate matrix conclusion mismatch")
            for key in set(type(matrix).model_fields) & set(type(row).model_fields):
                if getattr(matrix, key) != getattr(row, key):
                    raise ValueError("Supply duplicate matrix metric mismatch")
        envelopes["turnover"] = turnover
        chosen_period = f"{start}/{end}"
    valuation_envelopes = []
    if request.operation in {"value", "matrix"}:
        # Native valuation has a keyword filter, not a comma-list parser.
        # Bound one read to each requested target; never sum their totals.
        value_targets = request.targets if request.scope == "sku_list" else (keyword,)
        for index, target in enumerate(value_targets):
            value = await _hydrate(client, context, await client.inventory_value(context,
                product_keyword=target, limit=100), InventoryValue)
            if value.data.focus_keyword != target or (value.data.report_date and value.data.report_date > today):
                raise ValueError("Supply value source/scope mismatch")
            if target and any(row.vendor_code != target and str(row.nm_id) != target for row in value.data.rows):
                raise ValueError("Supply value returned a foreign target")
            key = f"value:{index}" if request.scope == "sku_list" else "value"
            envelopes[key] = value
            valuation_envelopes.append((key, target, value))
        if request.operation == "value":
            dates = {str(v.data.report_date) if v.data.report_date else "unknown" for _, _, v in valuation_envelopes}
            chosen_period = next(iter(dates)) if len(dates) == 1 else "multiple_source_dates"

    evidence, warnings, by_source = [], [], {}
    for operation, envelope in envelopes.items():
        family = operation.split(":", 1)[0]
        if any(item.source not in SOURCES[family] for item in envelope.evidence):
            raise ValueError("Supply foreign source provenance")
        digest = sha256(json.dumps(envelope.data.model_dump(mode="json"), sort_keys=True,
            ensure_ascii=False).encode()).hexdigest()[:16]
        by_source[operation] = {}
        for item in envelope.evidence:
            expected_grain = "product_filter" if keyword and item.source not in {"supply.wb_supplies", "supply.wb_supplies_goods"} else "portfolio"
            if item.grain != expected_grain:
                raise ValueError("Supply evidence scope provenance mismatch")
            ref = f"supply:{context.request_id}:{operation}:{item.id}:{digest}"
            by_source[operation][item.source] = ref
            evidence.append({"domain": "supply", "claim": family, "source": item.source,
                "evidence_ref": ref, "period": f"{item.period.date_from}/{item.period.date_to}" if item.period else None,
                "freshness": "unknown"})
        warnings.extend(envelope.warnings)
    if request.operation in {"turnover", "matrix"} and request.date_from is None:
        warnings.append("scope:default_last_known_30_days" if request.days == 30 else "scope:default_source_anchored_period")
    # This is a bounded API result, never proof of all portfolio SKUs.
    if any(len(envelope.data.rows) == 100 for envelope in envelopes.values()):
        warnings.append("scope:row_limit_may_hide_additional_skus")
    warnings.append("provenance_only:source_completeness_unverified")
    facts, claims, tables = [], [], []
    required_context = []
    required_finance_checks = []

    def claim(operation, row, keys, units, period, *, sku=None, nm_id=None, sources=(STOCK,), note=""):
        refs = [by_source[operation][source] for source in sources if source in by_source[operation]]
        if not refs or len(refs) != len(sources):
            warnings.append(f"needs_data:{operation}_metric_provenance_missing")
            return
        text = f"{'SKU ' + sku if sku else 'Портфель'}, {period}: "
        text += "; ".join(f"{METRIC_LABELS[key]} — {('нет данных' if row[key] is None else row[key])} {UNIT_LABELS[units[key]]}" for key in keys)
        if note: text += "; " + note
        claims.append({"operation": operation.split(":", 1)[0], "text": text + ".", "sku": sku, "nm_id": nm_id, "period": period, "evidence_refs": refs})
        for key in keys:
            metric_source = SALES if key in {"sales_units", "sales_revenue"} else COST if key in {"unit_cost_rub", "current_stock_value_rub"} else STOCK
            ref = by_source[operation].get(metric_source)
            # A null cost is evidence of missing valuation, not a confirmed
            # fact from a cost source that was absent from this response.
            if ref is None and row[key] is not None:
                raise ValueError("Supply metric has no native source reference")
            historical = "/" in period
            boundaries = period.split("/") if historical else [period, period]
            begin, end = (date.fromisoformat(value) for value in boundaries)
            definition = key
            if key in {"turnover_ratio", "days_on_hand"}:
                definition = envelopes[operation].data.method + "; sources=" + ",".join(sources)
            elif key in {"days_of_stock", "reorder_qty_recommended", "current_stock_value_rub"}:
                definition = key + "; native service result; sources=" + ",".join(sources)
            facts.append(EvidenceFact(metric=key, value=row[key], unit=units[key], period=period,
                period_start=begin, period_end=end,
                sku=sku, target_type="sku" if sku else "portfolio", target_id=sku,
                source_table=metric_source if ref else None, evidence_ref=ref or refs[0], source_agent="supply",
                # Envelope freshness.as_of describes CURRENT stock, not the
                # historical average or the cost/sales source snapshot.
                source_snapshot_at=begin if not historical and metric_source == STOCK else None,
                retrieved_at=envelopes[operation].freshness.observed_at,
                status="confirmed" if row[key] is not None and ref else "unconfirmed",
                metric_definition=definition).model_dump(mode="json"))

    lookup = overview.data.sku_stock_lookup
    if keyword:
        rows = [match.model_dump(mode="json") for match in lookup.matches]
        if lookup.match_count != len(rows):
            raise ValueError("Supply stock match count mismatch")
        if lookup.status == "found":
            if len(rows) != 1 or lookup.available_stock_qty != rows[0]["available_stock_qty"]:
                raise ValueError("Supply stock lookup authoritative total mismatch")
        if request.scope == "sku_list":
            if lookup.query != keyword or lookup.requested_skus != list(request.targets):
                raise ValueError("Supply list lookup request mismatch")
            matched = [target for target in request.targets if any(
                row["vendor_code"] == target or str(row["nm_id"]) == target for row in rows)]
            missing = [target for target in request.targets if target not in matched]
            status = "found_multiple" if matched and not missing else "partial" if matched else "no_data"
            if (lookup.matched_queries != matched or lookup.missing_queries != missing or lookup.status != status
                or any(sum(row["vendor_code"] == target or str(row["nm_id"]) == target for row in rows) > 1 for target in matched)
                or len({(row["vendor_code"], row["nm_id"]) for row in rows}) != len(rows)):
                raise ValueError("Supply list lookup membership ambiguous")
            if missing: warnings.append("needs_data:some_requested_skus_missing")
        if lookup.status in {"no_data", "multiple_matches"}:
            warnings.append("needs_data:stock_lookup_not_confirmed")
            rows = []
        # Never fall back to warehouse rows or risk marts for a direct lookup.
        tables.append(_table("Подтверждённый текущий остаток SKU (сумма складов)", rows,
            ("vendor_code", "nm_id", "snapshot_date", "available_stock_qty", "full_stock_qty", "unavailable_stock_qty")))
        for row in rows:
            if not row["vendor_code"] or row["snapshot_date"] is None:
                raise ValueError("Supply stock identity/date missing")
            if row["snapshot_date"] > str(today):
                raise ValueError("Supply future stock observation")
            if request.scope != "portfolio" and not (row["vendor_code"] in request.targets or str(row["nm_id"]) in request.targets):
                raise ValueError("Supply lookup returned a foreign SKU")
            claim("overview", row, ("available_stock_qty", "full_stock_qty", "unavailable_stock_qty"),
                dict.fromkeys(("available_stock_qty", "full_stock_qty", "unavailable_stock_qty"), "units"),
                row["snapshot_date"], sku=row["vendor_code"], nm_id=row["nm_id"])
    else:
        rows = [row.model_dump(mode="json") for row in overview.data.rows]
        keys = ("vendor_code", "nm_id", "report_date", "stock_qty", "stock_qty_full", "days_of_stock",
                "reorder_flag", "reorder_qty_recommended", "supplier_inbound_qty", "availability_status")
        tables.append(_table("Остатки и риски: доступный запас, не полный запас для аудита", rows, keys))
        for row in rows:
            if not row["vendor_code"] or row["report_date"] is None:
                raise ValueError("Supply risk identity/date missing")
            if row["report_date"] > str(today):
                raise ValueError("Supply future risk observation")
            claim("overview", row, ("stock_qty", "stock_qty_full", "days_of_stock", "reorder_qty_recommended"),
                {"stock_qty": "units", "stock_qty_full": "units", "days_of_stock": "days", "reorder_qty_recommended": "units"},
                row["report_date"], sku=row["vendor_code"], nm_id=row["nm_id"], sources=(STOCK, SALES),
                note=f"сервисный статус {row['reorder_flag']}; {row['coverage_band']}; {row['availability_status']}; рекомендация дозаказа не выполнена")
    pipeline = [row.model_dump(mode="json") for row in overview.data.pipeline_rows]
    tables.append(_table("Поставки: портфель, НЕ фильтр выбранного SKU", pipeline,
        ("supply_id", "snapshot_date", "quantity", "ready_for_sale_quantity", "accepted_quantity", "unloading_quantity")))
    if "turnover" in envelopes:
        data = envelopes["turnover"].data
        rows = [row.model_dump(mode="json") for row in data.sku_metric_matrix]
        tables.append(_table(f"Оборачиваемость {chosen_period}; текущий остаток имеет отдельную дату", rows,
            tuple(type(data.sku_metric_matrix[0]).model_fields) if rows else ("sku",)))
        if data.calculation_status != "exact":
            warnings.append("needs_data:alternative_methods_not_executed")
        for native, row in zip(data.rows, rows, strict=True):
            if not native.vendor_code or (keyword and native.vendor_code not in request.targets and str(native.nm_id) not in request.targets):
                raise ValueError("Supply turnover returned foreign/missing SKU")
            keys = ("sales_units", "avg_stock_qty", "avg_stock_qty_full", "turnover_ratio", "days_on_hand")
            claim("turnover", row, keys, {"sales_units": "units", "avg_stock_qty": "units", "avg_stock_qty_full": "units",
                "turnover_ratio": "turns", "days_on_hand": "days"}, chosen_period, sku=native.vendor_code,
                nm_id=native.nm_id, sources=(STOCK, SALES), note=f"calculation_status={data.calculation_status}; "
                f"{native.turnover_band}; {native.availability_status}; вывод сервиса: {native.sku_conclusion}")
        methods = {"method": data.method, "coverage_formula": data.coverage_formula,
            "calculation_status": data.calculation_status, "source_coverage": data.source_coverage.model_dump(),
            "limitation": data.limitation, "alternative_methods": [v.model_dump() for v in data.alternative_methods],
            "current_stock_dates": [str(row.latest_snapshot_date) if row.latest_snapshot_date else None for row in data.rows]}
        if data.totals is not None and keyword is None:
            # Use the business calculator's portfolio aggregate, NEVER sum
            # the bounded returned SKU rows or average their ratios.
            methods["turnover_totals"] = data.totals.model_dump(mode="json")
            totals = methods["turnover_totals"]
            keys = ("sales_units", "sales_revenue", "avg_stock_qty", "avg_stock_qty_full", "turnover_ratio", "days_on_hand")
            index = len(claims)
            claim("turnover", totals, keys, {"sales_units": "units", "sales_revenue": "RUB", "avg_stock_qty": "units",
                "avg_stock_qty_full": "units", "turnover_ratio": "turns", "days_on_hand": "days"}, chosen_period,
                sources=(STOCK, SALES), note=f"агрегат бизнес-сервиса; {data.totals.turnover_band}; "
                "не сумма ограниченной SKU-таблицы и не прогноз будущего спроса")
            if len(claims) > index:
                required_context.append(index)
        # Current 7-day coverage and historical period coverage answer
        # different questions. Retain both service values and explicitly
        # require a contrast for observed active critical/warning SKU.
        native_by_sku = {row.vendor_code: row for row in data.rows}
        refs = [by_source["overview"].get(STOCK), by_source["overview"].get(SALES),
            by_source["turnover"].get(STOCK), by_source["turnover"].get(SALES)]
        contrasts = [row for row in overview.data.rows if row.vendor_code in native_by_sku
            and row.reorder_flag in {"critical", "warning"} and row.sales_7d_units is not None
            and row.sales_7d_units > 0 and row.days_of_stock is not None
            and native_by_sku[row.vendor_code].days_on_hand is not None]
        contrasts.sort(key=lambda row: (0 if row.reorder_flag == "critical" else 1, row.vendor_code))
        if contrasts and all(refs):
            for row in contrasts[:4]:
                historical = native_by_sku[row.vendor_code]
                if row.nm_id != historical.nm_id:
                    raise ValueError("Supply time-axis SKU identity mismatch")
                if row.report_date is None or row.report_date > today:
                    raise ValueError("Supply contrast current observation date missing/future")
                required_context.append(len(claims))
                claims.append({"operation": "turnover", "sku": row.vendor_code, "nm_id": row.nm_id,
                    "period": chosen_period, "evidence_refs": list(dict.fromkeys(refs)),
                    "text": f"SKU {row.vendor_code}: за {chosen_period} средний доступный запас покрывал "
                    f"{historical.days_on_hand} дней ({historical.turnover_band}); текущий доступный запас на "
                    f"{row.report_date} покрывает {row.days_of_stock} дней по последним 7 полным дням продаж "
                    f"({row.coverage_band}, дозаказ: {row.reorder_flag}). Рекомендация сервиса — "
                    f"{row.reorder_qty_recommended} шт, не выполнена. Это разные временные оси и базы спроса; "
                    "изменение спроса или причина дефицита этой парой не доказаны."})
    else:
        methods = dict(overview.data.calculation_methods)
    if valuation_envelopes:
        rows = [row.model_dump(mode="json") for _, _, value in valuation_envelopes for row in value.data.rows]
        if request.scope == "sku_list" and len({(r["vendor_code"], r["nm_id"], r["report_date"]) for r in rows}) != len(rows):
            raise ValueError("Supply value target aliases overlap")
        tables.append(_table("Стоимость доступных запасов; только возвращённые SKU", rows,
            ("vendor_code", "nm_id", "report_date", "available_stock_qty", "unit_cost_rub", "current_stock_value_rub", "valuation_status")))
        bound_rows = [(key, target, row.model_dump(mode="json")) for key, target, value in valuation_envelopes for row in value.data.rows]
        for operation, target, row in bound_rows:
            if not row["vendor_code"] or row["report_date"] is None:
                raise ValueError("Supply value identity/date missing")
            if row["report_date"] > str(today):
                raise ValueError("Supply future value observation")
            if keyword and row["vendor_code"] not in request.targets and str(row["nm_id"]) not in request.targets:
                raise ValueError("Supply value returned a foreign SKU")
            if request.scope == "sku_list":
                matches = [m for m in lookup.matches if m.vendor_code == target or str(m.nm_id) == target]
                if (len(matches) != 1 or matches[0].vendor_code != row["vendor_code"] or matches[0].nm_id != row["nm_id"]
                    or str(matches[0].snapshot_date) != row["report_date"] or matches[0].available_stock_qty != row["available_stock_qty"]):
                    raise ValueError("Supply value/lookup identity or stock mismatch")
            if row["valuation_status"] == "missing_cogs" and (row["unit_cost_rub"] is not None or row["current_stock_value_rub"] is not None):
                raise ValueError("Supply missing COGS has a spurious value")
            if row["valuation_status"] == "confirmed" and (row["unit_cost_rub"] is None or row["current_stock_value_rub"] is None):
                raise ValueError("Supply confirmed COGS has no valuation")
            keys = ("available_stock_qty", "unit_cost_rub", "current_stock_value_rub")
            index = len(claims)
            claim(operation, row, keys, {"available_stock_qty": "units", "unit_cost_rub": "RUB/unit", "current_stock_value_rub": "RUB"},
                row["report_date"], sku=row["vendor_code"], nm_id=row["nm_id"],
                sources=(STOCK, COST) if row["valuation_status"] == "confirmed" else (STOCK,),
                note=f"valuation_status={row['valuation_status']}")
            # A one-SKU value question must answer valuation in "Главное",
            # even when the model selects only an overview summary. Older
            # checkpointed prepared objects keep their recorded context list.
            if request.operation in {"value", "matrix"} and request.scope in {"single_sku", "sku_list"} and len(claims) > index:
                required_context.append(index)
                if row["valuation_status"] == "missing_cogs":
                    required_finance_checks.append(index)
        if request.scope == "sku_list":
            methods["valuations_by_target"] = [{"target": target,
                "snapshot_date": str(value.data.report_date) if value.data.report_date else None,
                "valuation_method": value.data.calculation_method,
                "totals": value.data.totals.model_dump(mode="json")} for _, target, value in valuation_envelopes]
        else:
            data = valuation_envelopes[0][2].data
            methods["valuation_method"] = data.calculation_method
            methods["value_snapshot_date"] = str(data.report_date) if data.report_date else None
            methods["totals"] = data.totals.model_dump(mode="json")
    if overview.data.simulation.enabled:
        methods["simulation"] = overview.data.simulation.model_dump(mode="json")
    # Prioritize native reorder classes, not a newly calculated turnover or
    # invented causal score. All other SKU remain visible in full tables.
    risk_order = {"critical": 0, "warning": 1, "overstock": 2, "ok": 3, None: 4}
    flags = {row.vendor_code: row.reorder_flag for row in overview.data.rows}
    active = {row.vendor_code for row in overview.data.rows if row.sales_7d_units is not None and row.sales_7d_units > 0}
    required_operations = ["overview", "turnover", "value"] if request.operation == "matrix" else [request.operation]
    required = []
    for operation in required_operations:
        indices = [i for i, c in enumerate(claims) if c["operation"] == operation and c["sku"] and i not in required_context]
        indices.sort(key=lambda i: (risk_order[flags.get(claims[i]["sku"])], claims[i]["sku"] not in active, claims[i]["sku"] or ""))
        required.extend(indices[:4])
    prepared = {"request": request.model_dump(mode="json"), "today": str(today), "report_date": str(anchor) if anchor else None,
        "period": chosen_period, "tables": tables, "facts": facts, "claims": claims, "evidence": evidence,
        "methods": methods, "required_operations": required_operations, "required_claim_indices": required,
        "required_context_claim_indices": required_context, "warnings": sorted(set(warnings))}
    if required_finance_checks:
        prepared["required_finance_check_indices"] = required_finance_checks[:8]
        if len(required_finance_checks) > 8:
            prepared["warnings"] = sorted(set(prepared["warnings"] + ["needs_data:valuation_check_queue_capped_at_8"]))
    if enable_alternative_methods and "turnover" in envelopes and envelopes["turnover"].data.calculation_status != "exact":
        from commerce_agent.supply.alternatives import append_alternatives, canonical_vendor_scope
        vendor_codes = canonical_vendor_scope(request, lookup)
        start, end = chosen_period.split("/")
        alternate = await _hydrate(client, context, await client.inventory_turnover_alternatives(context,
            date_from=start, date_to=end, vendor_codes=",".join(vendor_codes) if vendor_codes else None,
            limit=100, boundary_tolerance_days=alternative_boundary_tolerance_days), InventoryTurnoverAlternatives)
        append_alternatives(prepared, alternate, request_id=context.request_id,
            vendor_codes=vendor_codes, tolerance=alternative_boundary_tolerance_days)
    assert_bounded_safe(prepared, max_bytes=524288)
    return prepared
