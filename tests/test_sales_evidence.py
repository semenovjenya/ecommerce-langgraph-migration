"""Actual M2 HTTP adapters, synthetic native payloads, no provider/SQL access."""
import asyncio
from copy import deepcopy
from datetime import date, timedelta
import json

import httpx
import pytest

from commerce_agent.adapters.business_tools import BusinessToolClient, BusinessToolsSettings, ToolAdapterError
from commerce_agent.context import CommerceRuntimeContext
from commerce_agent.sales.evidence import collect_sales_evidence
from commerce_agent.sales.policy import SalesRequest, resolve_sales_request


CONTEXT = CommerceRuntimeContext(tenant_id="fixture", user_id="47", roles=("viewer",), request_id="m3-sales-fixture")
SETTINGS = BusinessToolsSettings(api_url="http://fixture.internal", service_token="fixture-token-" + "x" * 40)
DAY = date(2026, 10, 9)


def envelope(data, source="sales.wb_supplier_sales", period=None):
    return {"schema_version": 1, "data": data,
            "evidence": [{"id": "native-fixture", "source": source, "grain": "portfolio", "period": period}],
            "freshness": {"as_of": str(DAY), "observed_at": "2026-10-10T10:00:00Z", "status": "available"},
            "source": source, "period": period, "warnings": ["provenance_only:source_completeness_unverified"]}


def respond(request, *, change=None, seen=None):
    args = json.loads(request.content)
    path = request.url.path.rsplit("/", 1)[-1]
    if seen is not None:
        seen.append((path, args))
    target = args.get("product_keyword") or "SKU-A"
    identity = {"nm_id": 1, "vendor_code": target}
    start = args.get("date_from", "2026-10-03")
    end = args.get("date_to", str(DAY))
    period = {"date_from": start, "date_to": end}
    if path == "database-time":
        data = {"db_current_date": "2026-10-10", "db_now": "2026-10-10T10:00:00Z"}
    elif path == "report-snapshot":
        data = {"report_date": str(DAY), "focus_keyword": args.get("product_keyword"),
                "source_table": "sales.wb_supplier_sales", "required_fields": [], "calculations": {}, "rows": [],
                "markdown_table": "raw markdown intentionally not checkpointed",
                "report_table": {"title": "Сводка продаж", "columns": ["Период", "Продажи, шт", "Выручка, ₽"],
                    "rows": [["Последний день в данных", 1, 100.25], ["Последние 7 дней", 7, 701.75],
                             ["Последние 30 дней", 30, 3007.5]]}}
    elif path == "overview":
        data = {"focus_keyword": args.get("product_keyword"),
                "totals": {"sales_date": str(DAY), "sales_1d_units": 1, "sales_7d_units": 7,
                    "sold_sku_7d": 1, "revenue_1d": 100.25, "revenue_7d": 701.75, "avg_price_with_disc_7d": None},
                "top_revenue_rows": [{**identity, "report_date": str(DAY), "brand_name": None, "subject_name": None,
                    "sales_7d_units": 7, "revenue_7d": 701.75, "avg_price_with_disc_7d": None}],
                "conversion_watchlist": [], "sales_raw_rows": []}
    elif path == "sku-revenue-change":
        days = (date.fromisoformat(end) - date.fromisoformat(start)).days + 1
        previous_end = str(date.fromisoformat(start) - timedelta(days=1))
        previous_start = str(date.fromisoformat(start) - timedelta(days=days))
        bounds = {"current_period_start": start, "current_period_end": end,
                  "previous_period_start": previous_start, "previous_period_end": previous_end}
        data = {"date_from": start, "date_to": end, **bounds, "period_days": days,
                "focus_keyword": args.get("product_keyword"), "source_table": "sales.wb_supplier_sales", "row_count": 1,
                "rows": [{**bounds, **identity, "brand_name": None, "subject_name": None,
                    "current_revenue": 100.25, "previous_revenue": 0, "delta_revenue": 100.25,
                    "delta_revenue_pct": None, "current_sales": 1, "previous_sales": 0, "delta_sales": 1,
                    "driver_role": "new_in_current"}]}
    elif path == "daily-price-sales-comparison":
        data = {"focus_keyword": target, "days": args["days"], "source_tables": ["sales.wb_supplier_sales"],
                "row_count": 0, "rows": [], "causal_valid": False}
    else:
        raise AssertionError(path)
    payload = envelope(data, period=period if path == "sku-revenue-change" else None)
    if change:
        change(path, payload)
    return httpx.Response(200, json=payload)


def collect(request, change=None, **kwargs):
    seen = []
    async def exercise():
        async with BusinessToolClient(SETTINGS, transport=httpx.MockTransport(
            lambda req: respond(req, change=change, seen=seen))) as client:
            return await collect_sales_evidence(client, CONTEXT, request,**kwargs)
    return asyncio.run(exercise()), seen


@pytest.mark.parametrize("query,scope,targets,days", [
    ("Отчет продаж", "portfolio", (), 7),
    ("Продажи SKU SKU-A за последние 14 дней", "single_sku", ("SKU-A",), 14),
    ("Продажи артикулам SKU-A, SKU-B за 2 недели", "sku_list", ("SKU-A", "SKU-B"), 14),
    ("Весь портфель, отдельно SKU SKU-A", "portfolio", ("SKU-A",), 7),
    ("Продажи SKU-A", "single_sku", ("SKU-A",), 7),
    ("Продажи артикул 1", "single_sku", ("1",), 7),
    ("Продажи SKU Майка-XL", "single_sku", ("Майка-XL",), 7),
    ("Продажи SKU Мартин", "single_sku", ("Мартин",), 7),
])
def test_request_scope(query, scope, targets, days):
    actual = resolve_sales_request(query)
    assert (actual.scope, actual.targets, actual.days) == (scope, targets, days)


@pytest.mark.parametrize("query", ["Продажи вчера", "Продажи за квартал", "Продажи за 2 месяца",
                                      "Продажи 09.10.2026", "Продажи с 2026-10-01",
                                      "Продажи за сентябрь 2026", "Продажи за прошлую неделю",
                                      "Продажи за текущую неделю", "Продажи SKU по месяцам",
                                      "Покажи продажи по неделям с 2026-10-01 по 2026-10-07"])
def test_ambiguous_calendar_requests_fail_closed(query):
    with pytest.raises(ValueError):
        resolve_sales_request(query)


def test_portfolio_and_target_deep_dive_do_not_filter_portfolio():
    prepared, seen = collect(SalesRequest(user_query="Все товары, SKU SKU-A", targets=("SKU-A",)))
    snapshots = [args["product_keyword"] for path, args in seen if path == "report-snapshot"]
    assert snapshots == [None, "SKU-A"]
    assert prepared["tables"]["sales_decomposition"]["rows"][0][2] == 0
    assert prepared["facts"][1]["value"] == 0
    serialized = json.dumps(prepared)
    assert "raw markdown" not in serialized and "calculations" not in serialized and "observed_at" not in serialized
    assert "coverage:top_rows_are_not_a_complete_portfolio_matrix" in prepared["warnings"]


def test_multi_sku_uses_independent_calls_and_never_sums_totals():
    prepared, seen = collect(SalesRequest(user_query="Продажи SKU-A и SKU-B", scope="sku_list", targets=("SKU-A", "SKU-B")))
    assert [args["product_keyword"] for path, args in seen if path == "overview"] == ["SKU-A", "SKU-B"]
    assert len(prepared["tables"]["sales_overview"]["rows"]) == 10
    assert len(prepared["facts"]) == 8


@pytest.mark.parametrize("mode", ["missing", "future", "period", "scope", "provenance", "secret", "pii",
                                 "previous_period", "evidence_period", "period_days"])
def test_invalid_tool_evidence_rejected(mode):
    def change(path, result):
        if mode == "missing" and path == "overview":
            result["data"]["totals"] = None
        if mode == "future" and path == "report-snapshot":
            result["data"]["report_date"] = "2026-10-11"
        if mode == "period" and path == "sku-revenue-change":
            result["data"]["current_period_end"] = "2026-10-08"
        if mode == "scope" and path == "overview":
            result["data"]["top_revenue_rows"][0]["vendor_code"] = "SKU-A-extra"
        if mode == "provenance" and path == "sku-revenue-change":
            result["evidence"] = []
        if mode == "previous_period" and path == "sku-revenue-change":
            for value in (result["data"], result["data"]["rows"][0]):
                value["previous_period_start"] = "1990-01-01"
                value["previous_period_end"] = "1990-01-02"
        if mode == "evidence_period" and path == "sku-revenue-change":
            result["evidence"][0]["period"] = {"date_from": "2000-01-01", "date_to": "2000-01-02"}
        if mode == "period_days" and path == "sku-revenue-change":
            result["data"]["period_days"] = 1
        if mode in {"secret", "pii"} and path == "report-snapshot":
            result["data"]["markdown_table"] = ("lsv2_synthetic_test_canary_123456" if mode == "secret" else "client@example.com")
    with pytest.raises((ValueError, ToolAdapterError)):
        collect(SalesRequest(user_query="Продажи SKU-A", scope="single_sku", targets=("SKU-A",)), change)


@pytest.mark.parametrize("historical", [False, True])
def test_daily_signal_is_not_silently_clipped_or_used_for_history(historical):
    request = SalesRequest(user_query="Проверка", scope="single_sku", targets=("SKU-A",), days=60,
        date_from="2026-08-01" if historical else None, date_to="2026-08-07" if historical else None)
    prepared, seen = collect(request)
    assert all(path != "daily-price-sales-comparison" for path, _ in seen)
    assert "scope:daily_signal_skipped_no_matching_period_capability" in prepared["warnings"]
    if historical:
        assert "scope:snapshot_and_overview_are_latest_not_requested_history" in prepared["warnings"]


def test_partial_multi_sku_failure_preserves_success_without_zero_imputation():
    def change(path, payload):
        if path == "overview" and payload["data"]["focus_keyword"] == "SKU-B":
            payload["data"]["totals"] = None
    prepared, _ = collect(SalesRequest(user_query="Проверка", scope="sku_list", targets=("SKU-A", "SKU-B")), change)
    assert {row[0] for row in prepared["tables"]["sales_overview"]["rows"]} == {"SKU-A"}
    assert "needs_data:target_unavailable:SKU-B" in prepared["warnings"]


def test_100_native_rows_fit_projection_budget_without_silent_truncation():
    def change(path, payload):
        if path == "sku-revenue-change":
            row = payload["data"]["rows"][0]
            payload["data"]["rows"] = [{**row, "nm_id": index + 1, "vendor_code": f"SKU-{index}"} for index in range(100)]
            payload["data"]["row_count"] = 100
    prepared, _ = collect(SalesRequest(user_query="Отчет продаж"), change)
    assert len(prepared["tables"]["sales_decomposition"]["rows"]) == 100
    assert len(prepared["facts"]) == 400
    assert "coverage:row_limit_reached" in prepared["warnings"]
