"""V6 source pointers: native values remain in the lossless evidence catalog."""
from commerce_agent.marketing import compact
from commerce_agent.marketing.receipt_coverage import receipt_coverage


def observation_index(prepared, reads):
    output = {}
    def add(identity, receipt, metric, path, *, layer="none", previous_path=None, comparison=None, relation="not_comparable", usable=False, native_scope=None):
        output[identity] = dict(receipt=receipt, metric=metric, path=path, previous_path=previous_path,
            layer=layer, native_comparison_ref=comparison, relation=relation, usable=usable,
            request_scope=dict(sku=prepared["request"]["targets"][0], current_period=prepared["period"],
                previous_period=prepared["previous_period"]), native_scope=native_scope)
    for i, fact in enumerate(prepared["facts"]):
        add(f"b{i}", "baseline", fact["metric"], f"/facts/{i}", usable=False, native_scope={k: fact.get(k) for k in ("sku", "period", "source_table")})
    from commerce_agent.marketing.historical_sources import validate as historical_sources
    historical = historical_sources(prepared)
    if historical is not None:
        for name in ("funnel", "funnel_periods", "search_details", "search_texts", "serp", "limitations"):
            add("context_"+name, "context", name, "/historical_sources/data/"+name,
                usable=False, native_scope={"sku":historical.vendor_code,"measurement_period":"each_row_has_its_own_period"})
    for i, read in enumerate(reads):
        ref = f"r{i}"
        if read["status"] != "ok" or not read.get("envelope"):
            add(ref+"_status", ref, "read_status", "/status")
            continue
        data = read["envelope"]["data"]
        if read["operation"] == "price":
            for j, item in enumerate(data["history"]["history_envelope"]["data"]["comparisons"]):
                layer = "market_summary" if item["metric"] == "market_price_median" else item["layer"]
                path = f"/envelope/data/history/history_envelope/data/comparisons/{j}"
                add(f"{ref}_p{j}", ref, item["metric"], path, layer=layer, comparison=ref+path,
                    relation=item["direction"] if item["direction"] != "insufficient_data" else "not_comparable",
                    usable=item["direction"] != "insufficient_data")
            for layer in ("seller", "customer", "competitors", "market_summary"):
                key = "customer" if layer == "market_summary" else layer
                add(ref+"_coverage_"+layer, ref, "coverage", "/envelope/data/history/history_envelope/data/"+key, layer=layer)
        elif read["operation"] in {"stock-history", "ad-history"}:
            metrics = ["stock_qty", "warehouse_count"] if read["operation"] == "stock-history" else ["views", "clicks", "ad_orders", "ad_spend_rub"]
            for metric in metrics:
                rows = data["rows"]
                usable = any(row.get(metric) is not None and (read["operation"] == "stock-history"
                    or row["ad_metrics_complete"]) for row in rows)
                add(ref+"_"+metric, ref, metric, "/envelope/data/rows", usable=usable)
        elif read["operation"] == "exact-search":
            for metric in ("frequency", "orders"):
                rows = data["current_days"]+data["previous_days"]
                usable = (data["coverage"] == "available" and bool(rows)
                    and all(row.get(metric) is not None and not row["repeated_query_rows"] for row in rows))
                add(ref+"_"+metric, ref, metric, "/envelope/data/current_days",
                    previous_path="/envelope/data/previous_days", usable=usable)
        else:
            for j, fact in enumerate((data["analysis"].get("result") or {}).get("facts", [])):
                add(f"{ref}_f{j}", ref, fact["metric"], f"/envelope/data/analysis/result/facts/{j}",
                    usable=(fact.get("status") == "confirmed" and fact.get("value") is not None
                        and fact.get("sku") == prepared["request"]["targets"][0]
                        and fact.get("period") in {prepared["period"], prepared["previous_period"],
                            prepared["period"]+" vs "+prepared["previous_period"]}),
                    native_scope={k: fact.get(k) for k in ("sku", "period", "source_table")})
            if read["operation"] == "sales" and "historical_funnel" in data:
                from commerce_agent.adapters.historical_sources import DELIVERY_FIELDS
                from commerce_agent.sales.historical_funnel import validate_delivery_receipt
                history = validate_delivery_receipt(data["historical_funnel"], vendor_code=read["target"],
                    date_from=read["arguments"]["date_from"], date_to=read["arguments"]["date_to"],
                    evidence=data["evidence"])
                for metric in DELIVERY_FIELDS:
                    # These are source-period observations, not weekly comparisons.
                    # Replay/unknown origin and all-zero duration cannot support a real delivery claim.
                    def unusable_reason(row):
                        if row.observation_origin != "observed":
                            return "origin_" + row.observation_origin
                        if getattr(row, metric) is None:
                            return "field_is_null"
                        if row.selected_period_start != row.day or row.selected_period_end != row.day:
                            return "selected_window_is_not_exact_day"
                        if metric.startswith(("past_", "comparison_")) and row.past_period_start is None:
                            return "past_window_unknown"
                        if metric == "comparison_localization_percent_dynamic" and (
                                row.localization_percent is None or row.past_localization_percent is None):
                            return "localization_pair_is_null"
                        if "localization" in metric:
                            values = [row.localization_percent, row.past_localization_percent] if metric.startswith("comparison_") else [getattr(row, metric)]
                            if any(v < 0 or v > 100 for v in values):
                                return "localization_out_of_range"
                        if "time_to_ready" in metric:
                            prefixes = ("time_to_ready_", "past_time_to_ready_") if metric.startswith("comparison_") else (
                                "past_time_to_ready_" if metric.startswith("past_") else "time_to_ready_",)
                            for prefix in prefixes:
                                values = [getattr(row, prefix + suffix) for suffix in ("days", "hours", "mins")]
                                if any(v is None for v in values):
                                    return "duration_components_missing"
                                if any(v < 0 for v in values):
                                    return "negative_duration_invalid"
                                if not any(values):
                                    return "all_zero_duration_unknown"
                        return None
                    add(ref+"_delivery_"+metric, ref, metric, "/envelope/data/historical_funnel/data/funnel",
                        usable=any(unusable_reason(row) is None for row in history.funnel),
                        native_scope={"sku":history.vendor_code,"source_table":history.sources[0],
                            "measurement_period":"each_row_selected_and_past_periods",
                            "coverage":"see_funnel_periods_and_nullable_fields", "weekly_aggregation":"unavailable_no_weights",
                            "eligible_days":[str(row.day) for row in history.funnel if unusable_reason(row) is None],
                            "excluded_days":{str(row.day):unusable_reason(row) for row in history.funnel if unusable_reason(row) is not None},
                            "limitations":history.limitations})
        add(ref+"_coverage", ref, "coverage", "/envelope/data", usable=False)
    return output


def model_evidence(prepared, reads):
    catalog = compact.evidence_catalog(prepared, reads)
    packed = compact.pack(catalog)
    if compact.canonical(compact.unpack(packed)) != compact.canonical(catalog):
        raise ValueError("research_lossless_mismatch")
    return dict(native=packed, observations=observation_index(prepared, reads), coverage=receipt_coverage(reads))
