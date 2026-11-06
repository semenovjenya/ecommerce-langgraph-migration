"""Collect native Price envelopes through the typed business API only.

No financial calculations or model calls. This prepared format preserves all
returned rows and independent source dates; a later publication guard owns the
legacy result/Markdown projection. Raw competitor rows are never strict proof.
"""
from datetime import date, timedelta
from hashlib import sha256
import json

from commerce_agent.adapters.contracts import (
    DatabaseTime, PriceContext, PriceDecisionOverview, PriceCompetitorOverview,
    PriceCompetitorDetails, DailyPriceSalesComparison, ToolArtifactReference, ToolResult,
)
from commerce_agent.price.policy import PriceRequest, assert_bounded_safe

MAX_PREPARED_BYTES = 4_000_000
CONTEXT = "analytics_ai.v_agent_price_context_latest"
DECISION = "pricing.mart_price_decision_latest"
RAW = "analytics_ai.wb_price_ai_competitors_raw"
LOOKUP = "analytics_ai.wb_sku_ai_analysis"
DAILY = "agent_marts.v_agent_price_sku_daily"
FUNNEL = "sales.wb_sales_funnel_products_daily"
SOURCES = {"context": {CONTEXT, "analytics_ai.wb_serp_runs"}, "decision": {DECISION},
           "competitor_overview": {RAW, LOOKUP}, "competitor_details": {RAW, LOOKUP}, "daily": {DAILY, FUNNEL}}
MODELS = {"context": PriceContext, "decision": PriceDecisionOverview,
          "competitor_overview": PriceCompetitorOverview, "competitor_details": PriceCompetitorDetails,
          "daily": DailyPriceSalesComparison}


def _matches_identity(row, target, namespace):
    if namespace not in {"nm_id", "vendor_code"}:
        raise ValueError("Price invalid resolved identity namespace")
    value = getattr(row, namespace)
    return value is not None and (str(value) if namespace == "nm_id" else value) == target


async def _hydrate(client, context, response, model):
    if isinstance(response, ToolArtifactReference):
        response = await client.hydrate(context, response, result_type=model)
    # Omitted native fields stay omitted; serialize optional defaults only in
    # typed access, never silently turn the source payload into different data.
    return ToolResult[model].model_validate(response.model_dump(mode="json", exclude_unset=True))


def validate_envelope(operation, envelope, args, today, *, identity_namespace=None):
    data = envelope.data
    primary = {"context": CONTEXT, "decision": DECISION, "competitor_overview": RAW,
               "competitor_details": RAW, "daily": DAILY}[operation]
    if (envelope.source != primary
            or any(item.source not in SOURCES[operation] for item in envelope.evidence)):
        raise ValueError("Price foreign source provenance")
    if data.row_count != len(data.rows) or len(data.rows) > (3100 if operation == "daily" else args["limit"]):
        raise ValueError("Price row count or limit mismatch")
    dates, required_warnings = [], set()
    if operation == "context":
        required_warnings.update({"scope:price_layers_have_independent_dates",
            "provenance_only:generated_refs_may_reference_missing_sources"})
        if data.source_view != CONTEXT or data.grain != "one latest row per nm_id":
            raise ValueError("Price canonical source or grain mismatch")
        ids = [row.nm_id for row in data.rows]
        if len(ids) != len(set(ids)):
            raise ValueError("Price duplicate canonical identity")
        for row in data.rows:
            if args.get("vendor_code") is not None and row.vendor_code != args["vendor_code"]:
                raise ValueError("Price exact vendor mismatch")
            if args.get("nm_id") is not None and row.nm_id != int(args["nm_id"]):
                raise ValueError("Price exact nm_id mismatch")
            dates.extend(day for day in (row.seller_snapshot_date, row.analysis_run_date, row.context_snapshot_date) if day)
        if any(item.source_nm_id not in ids for item in data.evidence) or not args["include_evidence"] and data.evidence:
            raise ValueError("Price SERP evidence outside returned identity")
        expected_sources = ([CONTEXT] + (["analytics_ai.wb_serp_runs"] if data.evidence else [])) if data.rows else []
        if not data.rows:
            required_warnings.add("needs_data:price_context_missing")
        if any(row.freshness_status != "fresh" for row in data.rows):
            required_warnings.add("needs_data:price_context_stale_or_incomplete")
        if any(row.competitor_quality != "trusted" or row.analysis_status != "success"
                or (row.strict_competitor_count or 0) < 5 or row.strict_competitor_price_median is None for row in data.rows):
            required_warnings.add("needs_data:strict_competitor_quality_unconfirmed")
        if len(data.rows) == args["limit"]:
            required_warnings.add("scope:row_limit_may_hide_additional_skus")
        if data.rows and CONTEXT not in {item.source for item in envelope.evidence}:
            raise ValueError("Price canonical evidence missing")
    elif operation == "daily":
        required_warnings.update({"causality:not_established", "provenance_only:source_completeness_unverified"})
        target = args["product_keyword"]
        if data.source_tables != [DAILY, FUNNEL] or data.focus_keyword != target or data.days != args["days"]:
            raise ValueError("Price daily scope/source mismatch")
        dates = [row.report_date for row in data.rows]
        identities = [(row.nm_id, row.report_date) for row in data.rows]
        if len(identities) != len(set(identities)):
            raise ValueError("Price daily duplicate SKU/business day")
        anchor = max(dates, default=None)
        for row in data.rows:
            exact = (_matches_identity(row, target, identity_namespace) if identity_namespace
                else target in {str(row.nm_id), row.vendor_code})
            if not exact or row.report_date < anchor - timedelta(days=args["days"] - 1):
                raise ValueError("Price daily rows outside exact target/window")
        period = {"date_from": str(min(dates)), "date_to": str(max(dates))} if dates else None
        if (envelope.period.model_dump(mode="json") if envelope.period else None) != period:
            raise ValueError("Price daily native period mismatch")
        if data.rows and DAILY not in {item.source for item in envelope.evidence}:
            raise ValueError("Price daily evidence missing")
        funnel_present = any(any(getattr(row, key) is not None for key in (
            "open_count", "cart_count", "funnel_order_count", "add_to_cart_conversion", "cart_to_order_conversion", "buyout_percent")) for row in data.rows)
        expected_sources = ([DAILY] + ([FUNNEL] if funnel_present else [])) if dates else []
        if not dates:
            required_warnings.add("needs_data:price_sales_source_missing")
        elif any(row.business_day_complete is not True or row.sales_day_complete is not True or row.marketing_day_complete is not True for row in data.rows):
            required_warnings.add("needs_data:incomplete_business_days")
        if dates and not funnel_present:
            required_warnings.add("needs_data:funnel_source_missing")
    else:
        if data.source_table != (DECISION if operation == "decision" else RAW):
            raise ValueError("Price source table mismatch")
        if data.focus_keyword != args.get("product_keyword"):
            raise ValueError("Price request filter mismatch")
        if operation == "decision":
            if identity_namespace and any(not _matches_identity(row, args["product_keyword"], identity_namespace) for row in data.rows):
                raise ValueError("Price scoped decision identity namespace mismatch")
            required_warnings.update({"scope:portfolio_context_reference_not_sku_specific",
                "freshness:relative_context_lag_not_wall_clock", "quality:scenario_not_final_manager_decision",
                "semantics:legacy_price_aliases_not_canonical_layers", "scope:returned_rows_only"})
            dates = [row.snapshot_date for row in data.rows if row.snapshot_date]
            expected_sources = [DECISION] if data.rows else []
            if not data.rows:
                required_warnings.add("needs_data:price_decision_missing")
            if any(row.decision_freshness != "fresh" or row.decision_lag_days is None or row.decision_lag_days < 0 for row in data.rows):
                required_warnings.add("needs_data:decision_freshness_unconfirmed")
            if len(data.rows) == args["limit"]:
                required_warnings.add("scope:row_limit_may_hide_additional_skus")
        else:
            required_warnings.update({"quality:legacy_candidates_not_trusted_strict_analogs",
                "provenance_only:sku_lookup_completeness_unverified", "scope:returned_rows_only"})
            expected_sources = [RAW] if data.rows else []
            if not data.rows:
                required_warnings.add("needs_data:competitor_rows_missing")
            if len(data.rows) == args["limit"]:
                required_warnings.add("scope:row_limit_may_hide_additional_competitors")
            if data.sku_lookup_table != LOOKUP:
                raise ValueError("Price competitor lookup mismatch")
            dates = [data.run_date] if data.run_date else []
            if operation == "competitor_details":
                expected_nm = int(args["nm_id"]) if args.get("nm_id") else None
                if data.nm_id != expected_nm or data.vendor_code != args.get("vendor_code"):
                    raise ValueError("Price raw details header mismatch")
                for row in data.rows:
                    if ((expected_nm is not None and row.source_nm_id != expected_nm)
                            or (args.get("vendor_code") is not None and row.vendor_code != args["vendor_code"])):
                        raise ValueError("Price raw details exact source mismatch")
            if any(row.run_date != data.run_date for row in data.rows):
                raise ValueError("Price competitor run-date mismatch")
        if data.rows and data.source_table not in {item.source for item in envelope.evidence}:
            raise ValueError("Price row evidence missing")
    if [item.source for item in envelope.evidence] != expected_sources or not required_warnings <= set(envelope.warnings):
        raise ValueError("Price native evidence/warnings incomplete")
    expected_as_of = (max(dates) if operation == "daily" else min(dates)) if dates else None
    if envelope.freshness.as_of != expected_as_of or envelope.freshness.status != ("available" if dates else "missing"):
        raise ValueError("Price native source date mismatch")
    if operation != "daily" and envelope.period is not None:
        raise ValueError("latest Price snapshot is not a historical period")
    grain = "product_filter" if args.get("nm_id") or args.get("vendor_code") or args.get("product_keyword") else "portfolio"
    for item in envelope.evidence:
        prefix = {"context": "price.context:", "decision": "price.decision_overview:",
            "competitor_overview": "price.competitor_overview:", "competitor_details": "price.competitor_details:",
            "daily": "sales.daily_price_sales_comparison:"}[operation]
        if item.grain != grain or item.period != envelope.period or not item.id.startswith(prefix):
            raise ValueError("Price evidence scope/period mismatch")
    # Future native dates are retained as warnings later, never labelled fresh.
    # This parameter intentionally does not clamp them to the clock date.
    assert_bounded_safe(envelope.model_dump(mode="json", exclude_unset=True), max_bytes=MAX_PREPARED_BYTES)


def _identity_arguments(target, rows, namespace="auto"):
    # Numeric vendor codes and nm_id can be ambiguous. Prefer a resolved native
    # vendor identity only when no different nm_id row matches that token.
    if namespace in {"nm_id", "vendor_code"}:
        return {namespace: str(int(target)) if namespace == "nm_id" else target}
    if namespace != "auto":
        raise ValueError("Price invalid identity namespace")
    vendor = [row for row in rows if row["vendor_code"] == target]
    nm = [row for row in rows if str(row["nm_id"]) == target]
    if vendor and nm and {row["nm_id"] for row in vendor} != {row["nm_id"] for row in nm}:
        raise ValueError("Price target ambiguous between vendor and nm_id")
    if vendor:
        return {"vendor_code": target}
    if nm:
        return {"nm_id": target}
    # An unseen number can be a vendor code or an nm_id outside a capped read.
    # Do not claim absence after guessing its namespace. Explicit trusted plan
    # namespaces (or a literal nm_id question) can resolve outside this matrix.
    if target.isascii() and target.isdigit():
        raise ValueError("Price target identity namespace unresolved")
    return {"vendor_code": target}


async def collect_price_evidence(client, context, request):
    request = PriceRequest.model_validate(request)
    clock = await _hydrate(client, context, await client.database_time(context), DatabaseTime)
    today = clock.data.db_current_date
    reads, evidence = [], []

    async def read(operation, method, args, scope, *, identity_namespace=None):
        envelope = await _hydrate(client, context, await getattr(client, method)(context, **args), MODELS[operation])
        validate_envelope(operation, envelope, args, today, identity_namespace=identity_namespace)
        warnings = native_read_warnings(operation, envelope, today)
        raw = envelope.model_dump(mode="json", exclude_unset=True)
        # Wire sources and independent dates are authoritative; no recalculation.
        digest = sha256(json.dumps(raw, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()[:16]
        refs = []
        for item in envelope.evidence:
            ref = f"price:{context.request_id}:{operation}:{item.id}:{digest}"
            refs.append(ref)
            evidence.append({"domain": "price", "source": item.source, "claim": operation,
                "evidence_ref": ref, "period": f"{item.period.date_from}/{item.period.date_to}" if item.period else None,
                "freshness": "unknown"})
        reads.append({"operation": operation, "arguments": args, "scope": scope,
                      "identity_namespace": identity_namespace,
                      "envelope": raw, "evidence_refs": refs, "warnings": list(dict.fromkeys(warnings))})
        return raw["data"]["rows"]

    # Auto namespace resolution needs the canonical portfolio matrix to reject
    # vendor/nm_id collisions. An explicit trusted namespace resolves without
    # that supporting global read; exact native context below still proves the
    # requested identity. Portfolio behavior and ambiguity guards stay intact.
    native_rows = []
    if request.scope == "portfolio" or request.target_namespace == "auto":
        native_rows = await read("context", "price_context", {"limit": 100, "include_evidence": True}, None)
    if request.scope != "portfolio" and not request.targets:
        raise ValueError("Price exact scope requires targets")
    if request.operation == "report" and request.scope == "portfolio":
        await read("decision", "price_decision_overview", {"limit": 100, "product_keyword": None}, None)
    if request.operation in {"report", "competitors"} and request.scope == "portfolio":
        await read("competitor_overview", "price_competitor_overview", {"limit": 100, "product_keyword": None}, None)
        await read("competitor_details", "price_competitor_details", {"limit": 100}, None)
    for target in request.targets:
        args = _identity_arguments(target, native_rows, request.target_namespace)
        namespace = next(iter(args))
        exact_token = args[namespace]
        # Exact target uses the native default bound; portfolio uses 100. Any
        # capped exact result retains the native row-limit warning, not a claim
        # that the keyword/native identity family was exhaustively enumerated.
        selected = await read("context", "price_context", {**args, "limit": 20, "include_evidence": True}, target)
        if not selected:
            reads[-1]["warnings"].append(f"needs_data:price_target_missing:{target}")
        if request.operation == "report":
            # Native keyword readers can match more than an exact identifier;
            # validate every returned row instead of silently filtering it.
            await read("decision", "price_decision_overview",
                {"product_keyword": exact_token, "limit": 100}, target, identity_namespace=namespace)
        if request.operation in {"report", "competitors"}:
            await read("competitor_details", "price_competitor_details", {**args, "limit": 100}, target)
        if request.operation == "daily":
            await read("daily", "daily_price_sales_comparison", {"product_keyword": exact_token, "days": request.days}, target,
                identity_namespace=namespace)
    # The portfolio identity read is supporting provenance for exact scope, not
    # an excuse to publish unrelated items as if requested.
    if request.scope != "portfolio":
        reads = [item for item in reads if item["scope"] is not None]
        used = {ref for item in reads for ref in item["evidence_refs"]}
        evidence = [item for item in evidence if item["evidence_ref"] in used]
    prepared = {"version": 1, "request": request.model_dump(mode="json"), "today": str(today),
        "reads": reads, "evidence": evidence,
        "warnings": list(dict.fromkeys(warning for item in reads for warning in item["warnings"])),
        "clock": clock.data.model_dump(mode="json"),
        "capability": "native Price read collection only; model/publication/runtime not yet implemented"}
    assert_bounded_safe(prepared, max_bytes=MAX_PREPARED_BYTES)
    return prepared


def native_read_warnings(operation, envelope, today):
    """Source limitations reused by collection and independent publication."""
    warnings = list(envelope.warnings)
    if envelope.freshness.as_of and envelope.freshness.as_of > today:
        warnings.append(f"needs_data:future_price_source:{operation}")
    if operation == "context":
        for row in envelope.data.rows:
            if any(day and day > today for day in (row.seller_snapshot_date, row.analysis_run_date, row.source_snapshot_date,
                    row.context_snapshot_date, row.context_as_of_date)):
                warnings.append("needs_data:future_price_source:context")
            if row.customer_price_basis in {"wb_wallet", "site_display"}:
                warnings.append("provenance:customer_layer_exact_source_date_unconfirmed_coalesced_sources")
    if operation == "decision":
        if any(row.snapshot_date is None or row.snapshot_date < today for row in envelope.data.rows):
            warnings.append("scope:decision_is_dated_scenario_not_current_execution_authority")
        if any(row.cost_price_rub is None for row in envelope.data.rows):
            warnings.append("needs_data:scenario_cost_unconfirmed")
        if any(day and day > today for row in envelope.data.rows for day in (row.snapshot_date, row.price_ai_run_date, row.price_context_as_of_date)):
            warnings.append("needs_data:future_price_source:decision")
    return list(dict.fromkeys(warnings))
