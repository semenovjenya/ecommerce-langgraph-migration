"""Model-facing source index: completeness metadata, never business conclusions.

Full native receipts remain in evidence. This projection neither changes their
values nor determines a hypothesis, causal direction, priority or assessment.
"""
from copy import deepcopy
from datetime import date, timedelta


def _window(period, rows, day_key):
    start, end = (date.fromisoformat(period[key]) for key in ("date_from", "date_to"))
    expected = {(start + timedelta(days=i)).isoformat() for i in range((end-start).days+1)}
    selected = [row for row in rows if row[day_key] in expected]
    observed = {row[day_key] for row in selected}
    return dict(status="window_absent" if not observed else "partial" if expected-observed else "all_days_returned",
        missing_days=sorted(expected-observed), observed_days=sorted(observed)), selected


def receipt_coverage(reads):
    output = {}
    for index, read in enumerate(reads):
        item = {"operation": read["operation"], "read_status": read["status"]}
        output[f"r{index}"] = item
        if read["status"] != "ok" or not read.get("envelope"):
            item["source_result"] = "check_not_completed"
            continue
        envelope = read["envelope"]
        data = envelope["data"]
        item["source"] = envelope["source"]
        if read["operation"] == "price":
            from commerce_agent.price.history import build_catalog
            item["layers"] = deepcopy(build_catalog(data["history"])["coverage"])
            item["source_result"] = "independent_price_layers; market_summary_is_not_individual_competitors"
            continue
        if read["operation"] in {"sales", "supply"}:
            item["analysis_status"] = data["analysis"]["status"]
            item["source_result"] = ("all_sales_not_ad_attributed" if read["operation"] == "sales"
                else "aggregate_turnover_and_inventory_not_historical_stockout_proof")
            continue
        item["identity_status"] = data["identity_status"]
        if read["operation"] == "exact-search":
            item["source_coverage"] = data["coverage"]
            item["source_result"] = "search_frequency_not_paid_clicks; query_composition_unproven"
            for side in ("current", "previous"):
                period = data["requested_period" if side == "current" else "previous_period"]
                status, rows = _window(period, data[side+"_days"], "day")
                status["null_frequency_days"] = sorted(row["day"] for row in rows if row["frequency"] is None)
                status["repeated_snapshot_days"] = sorted(row["day"] for row in rows if row["repeated_query_rows"])
                item[side] = status
            continue
        rows = data["rows"]
        item["source_result"] = "no_rows_returned" if not rows else "rows_returned_not_proof_of_complete_source"
        for side, period_key in (("current", "requested_period"), ("previous", "previous_period")):
            status, window_rows = _window(data[period_key], rows, "day")
            if read["operation"] == "stock-history":
                status["null_stock_days"] = sorted(row["day"] for row in window_rows if row["stock_qty"] is None)
                status["nonpositive_stock_days"] = sorted(row["day"] for row in window_rows
                    if row["stock_qty"] is not None and row["stock_qty"] <= 0)
                status["intraday_regional_coverage"] = "unknown"
            else:
                metrics = ("views", "clicks", "ad_orders", "ad_spend_rub")
                values = [row[key] for row in window_rows for key in metrics]
                status["all_metrics_unknown"] = bool(values) and all(value is None for value in values)
                status["all_rows_complete"] = bool(window_rows) and all(row["ad_metrics_complete"] for row in window_rows)
                status["all_returned_metrics_zero"] = bool(values) and all(value == 0 for value in values)
                status["all_metrics_observed_zero"] = status["all_rows_complete"] and status["all_returned_metrics_zero"]
                status["campaign_coverage"] = {
                    str(campaign): _window(data[period_key], [row for row in rows if row["advert_id"] == campaign], "day")[0]
                    for campaign in sorted({row["advert_id"] for row in rows})}
                status["all_campaigns_known"] = False
            item[side] = status
    return output
