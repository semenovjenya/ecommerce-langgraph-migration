"""Project the narrow business calculator; no operational math or SQL here."""
from copy import deepcopy
from hashlib import sha256
import json
import re
from uuid import UUID

from commerce_agent.adapters.contracts import InventoryTurnoverAlternatives, ToolResult
from commerce_agent.sales.contracts import EvidenceFact, ReportTable
from commerce_agent.supply.policy import SupplyRequest

STOCK = "supply.wb_supplier_stocks"
SALES = "sales.wb_supplier_sales"
METRIC = "observed_stock_coverage_days_estimate"
TABLE_TITLE = "Альтернативные методы: наблюдаемый запас — оценка, не точная оборачиваемость"
MARKER_KEYS = {"alternative_methods_policy_version", "alternative_methods_source",
    "alternative_methods_vendor_codes", "alternative_methods_boundary_tolerance_days",
    "alternative_methods_request_id"}
CORE_WARNINGS = {"source:raw_stock_snapshot_completeness_unverified", "source:sales_period_completeness_unverified",
    "estimate:not_exact_turnover_or_total_stock", "method:movement_journal_not_registered"}
METADATA_WARNINGS = {"quality:observed_stock_estimate_not_exact_turnover_or_total_stock",
    "quality:snapshot_completeness_unverified", "needs_data:complete_movement_journal_not_registered",
    "scope:exact_requested_sales_period_latest_stock_independently_dated", "scope:returned_rows_only",
    "freshness:observed_source_dates_not_current_store_freshness"}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def canonical_vendor_scope(request, lookup):
    if request.scope == "portfolio":
        return None
    result = []
    for target in request.targets:
        matches = [row for row in lookup.matches if row.vendor_code == target or str(row.nm_id) == target]
        if len(matches) != 1 or not matches[0].vendor_code or matches[0].nm_id is None:
            raise ValueError("alternative method requires a canonical Supply identity")
        if matches[0].vendor_code not in result:
            result.append(matches[0].vendor_code)
    return result


def project_alternatives(source, *, request, today, period, request_id, vendor_codes, tolerance):
    source = ToolResult[InventoryTurnoverAlternatives].model_validate(source)
    data = source.data
    start, end = period.split("/")
    if (type(tolerance) is not int or not 0 <= tolerance <= 31
        or data.boundary_tolerance_days != tolerance or str(data.database_date) != today
        or (str(data.date_from), str(data.date_to)) != (start, end)
        or data.vendor_codes != (vendor_codes or []) or source.period is None
        or (str(source.period.date_from), str(source.period.date_to)) != (start, end)):
        raise ValueError("Supply alternative source/scope/clock policy mismatch")
    # Canonical scope is recorded explicitly, never inferred from a model.
    if request.scope == "portfolio" and vendor_codes is not None:
        raise ValueError("Supply alternative portfolio was filtered")
    if request.scope != "portfolio" and (not vendor_codes or len(vendor_codes) > len(request.targets)):
        raise ValueError("Supply alternative canonical scope unavailable")
    expected_grain = "portfolio" if vendor_codes is None else "product_filter"
    stock_observed = any(observation.source_rows > 0 for row in data.rows
        for observation in (row.boundary_start, row.boundary_end, row.latest_stock) if observation)
    sales_observed = any(row.sales_source_rows > 0 for row in data.rows)
    expected_sources = {name for name, present in ((STOCK, stock_observed), (SALES, sales_observed)) if present}
    observed_dates = [observation.snapshot_date for row in data.rows
        for observation in (row.boundary_start, row.boundary_end, row.latest_stock)
        if observation and observation.source_rows > 0]
    observed_dates.extend(row.sales_observed_to for row in data.rows if row.sales_observed_to)
    expected_as_of = min(observed_dates) if observed_dates else None
    required_warnings = set(METADATA_WARNINGS) | set(data.warnings)
    if data.row_limit_reached:
        required_warnings.add("scope:row_limit_hides_additional_skus")
    if data.unreturned_vendor_codes:
        required_warnings.add("needs_data:requested_skus_unreturned")
    if data.calculation_status == "unavailable":
        required_warnings.add("needs_data:no_usable_alternative_estimate")
    if (source.source != STOCK or source.freshness.as_of != expected_as_of
        or source.freshness.status != ("available" if expected_as_of else "missing")
        or source.freshness.observed_at.tzinfo is None or source.freshness.observed_at.utcoffset() is None
        or not CORE_WARNINGS <= set(data.warnings) or not required_warnings <= set(source.warnings)
        or {item.source for item in source.evidence} != expected_sources
        or len(source.evidence) != len(expected_sources) or any(item.grain != expected_grain or item.period is None
            or not re.fullmatch(r"supply\.inventory_turnover_alternatives:[a-f0-9]{64}", item.id)
            or (str(item.period.date_from), str(item.period.date_to)) != (start, end) for item in source.evidence)
        or len({item.id for item in source.evidence}) != len(source.evidence)):
        raise ValueError("Supply alternative source provenance mismatch")
    digest = sha256(canonical(data.model_dump(mode="json")).encode()).hexdigest()[:16]
    refs, evidence = {}, []
    for item in source.evidence:
        ref = f"supply:{request_id}:turnover-alternatives:{item.id}:{digest}"
        refs[item.source] = ref
        evidence.append({"domain": "supply", "claim": "turnover_alternatives", "source": item.source,
            "evidence_ref": ref, "period": period, "freshness": "unknown"})
    facts, claims, table_rows, checks = [], [], [], []
    for row in data.rows:
        latest = row.latest_stock
        for attempt in row.method_attempts:
            table_rows.append([row.vendor_code, row.nm_id, start, end, row.sales_units,
                str(latest.snapshot_date) if latest else None,
                latest.observed_sellable_stock_qty if latest else None,
                ", ".join(latest.covered_warehouses) if latest else None,
                attempt.method, attempt.status, attempt.reason,
                attempt.observed_stock_coverage_days_estimate,
                ", ".join(str(value) for value in attempt.used_stock_dates), "; ".join(attempt.caveats)])
        proxy = row.method_attempts[-1]
        text = (f"SKU {row.vendor_code}, продажи за {period}: "
            + ("источник продаж отсутствует" if row.sales_units is None else f"наблюдаемые реализации — {row.sales_units} шт")
            + "; ")
        if proxy.status == "estimate_available":
            text += (f"оценка покрытия наблюдаемыми строками доступного запаса на {latest.snapshot_date} — "
                f"{proxy.observed_stock_coverage_days_estimate} дней. ")
            facts.append(EvidenceFact(metric=METRIC, value=proxy.observed_stock_coverage_days_estimate,
                unit="days", period=period, period_start=data.date_from, period_end=data.date_to,
                sku=row.vendor_code, target_type="sku", target_id=row.vendor_code,
                source_table=STOCK, evidence_ref=refs[STOCK], source_agent="supply",
                source_snapshot_at=latest.snapshot_date, retrieved_at=source.freshness.observed_at,
                status="unconfirmed", confidence="low", metric_definition=(
                    "Native observed-stock velocity estimate; sources=" + STOCK + "," + SALES
                    + "; stock_snapshot=" + str(latest.snapshot_date)
                    + "; exact sales period=" + period + "; snapshot completeness unverified; not total stock or period average"
                )).model_dump(mode="json"))
        else:
            text += f"оценка покрытия недоступна ({proxy.reason}). "
        text += ("Это не полный остаток, не исторический средний запас и не точная оборачиваемость; "
            "полнота warehouse snapshot и периода продаж не подтверждена. "
            "Boundary average: " + row.method_attempts[0].reason + "; movement reconstruction: "
            + row.method_attempts[1].reason + ". Не зарегистрированный журнал движения не запрашивался.")
        if refs:
            claims.append({"operation": "turnover", "text": text, "sku": row.vendor_code, "nm_id": row.nm_id,
                "period": period, "evidence_refs": [refs[name] for name in (STOCK, SALES) if name in refs],
                "alternative_methods_claim": True, "confidence": "low"})
        checks.append({"sku": row.vendor_code, "nm_id": row.nm_id,
            "boundary_start": row.boundary_start.model_dump(mode="json") if row.boundary_start else None,
            "boundary_end": row.boundary_end.model_dump(mode="json") if row.boundary_end else None,
            "latest_stock_date": str(latest.snapshot_date) if latest else None,
            "method_attempts": [value.model_dump(mode="json") for value in row.method_attempts]})
    table = ReportTable(title=TABLE_TITLE, columns=["vendor_code", "nm_id", "sales_from", "sales_to",
        "observed_realized_sales_units", "observed_stock_date", "observed_sellable_stock_qty",
        "covered_warehouses", "alternative_method", "method_status", "method_reason",
        METRIC, "used_stock_dates", "caveats"], rows=table_rows).model_dump(mode="json")
    warning = ("estimate:alternative_methods_checked_with_observed_stock_proxy"
        if data.calculation_status == "estimates_available" else "needs_data:alternative_methods_unavailable")
    return {"facts": facts, "claims": claims, "tables": [table], "evidence": evidence,
        "checks": checks, "warnings": sorted(set([*source.warnings, *data.warnings,
            "needs_data:exact_turnover_unavailable", warning]))}


def append_alternatives(prepared, source, *, request_id, vendor_codes, tolerance):
    projection = project_alternatives(source, request=SupplyRequest.model_validate(prepared["request"]),
        today=prepared["today"], period=prepared["period"], request_id=request_id,
        vendor_codes=vendor_codes, tolerance=tolerance)
    base = len(prepared["claims"])
    for key in ("facts", "claims", "tables", "evidence"):
        prepared[key].extend(projection[key])
    prepared["required_context_claim_indices"].extend(range(base, base + len(projection["claims"])))
    prepared["warnings"] = sorted(set([warning for warning in prepared["warnings"]
        if warning != "needs_data:alternative_methods_not_executed"] + projection["warnings"]))
    prepared["methods"]["alternative_method_checks"] = projection["checks"]
    prepared.update(alternative_methods_policy_version=1,
        alternative_methods_source=source.model_dump(mode="json"),
        alternative_methods_vendor_codes=deepcopy(vendor_codes),
        alternative_methods_boundary_tolerance_days=tolerance,
        alternative_methods_request_id=str(UUID(request_id)))


def has_alternative_source(prepared):
    return bool(MARKER_KEYS & prepared.keys() or "alternative_method_checks" in prepared.get("methods", {})
        or any(fact.get("metric") == METRIC for fact in prepared.get("facts", []))
        or any(table.get("title") == TABLE_TITLE for table in prepared.get("tables", []))
        or any(claim.get("alternative_methods_claim") for claim in prepared.get("claims", []))
        or any("turnover-alternatives:" in item.get("evidence_ref", "") for item in prepared.get("evidence", [])))


def validate_alternative_prepared(prepared):
    if not has_alternative_source(prepared):
        return
    if not MARKER_KEYS <= prepared.keys() or type(prepared["alternative_methods_policy_version"]) is not int or prepared["alternative_methods_policy_version"] != 1:
        raise ValueError("Supply alternative source marker missing or invalid")
    request_id = str(UUID(prepared["alternative_methods_request_id"]))
    projection = project_alternatives(prepared["alternative_methods_source"],
        request=SupplyRequest.model_validate(prepared["request"]), today=prepared["today"], period=prepared["period"],
        request_id=request_id, vendor_codes=prepared["alternative_methods_vendor_codes"],
        tolerance=prepared["alternative_methods_boundary_tolerance_days"])
    actual = {"facts": [fact for fact in prepared["facts"] if fact["metric"] == METRIC],
        "claims": [claim for claim in prepared["claims"] if claim.get("alternative_methods_claim")],
        "tables": [table for table in prepared["tables"] if table["title"] == TABLE_TITLE],
        "evidence": [item for item in prepared["evidence"] if "turnover-alternatives:" in item["evidence_ref"]],
        "checks": prepared["methods"].get("alternative_method_checks")}
    if any(canonical(actual[key]) != canonical(projection[key]) for key in actual):
        raise ValueError("Supply alternative source projection changed")
    if not set(projection["warnings"]) <= set(prepared["warnings"]) or "needs_data:alternative_methods_not_executed" in prepared["warnings"]:
        raise ValueError("Supply alternative source limitations changed")
    if any(index not in prepared["required_context_claim_indices"] for index, claim in enumerate(prepared["claims"])
           if claim.get("alternative_methods_claim")):
        raise ValueError("Supply alternative observations omitted from required context")
