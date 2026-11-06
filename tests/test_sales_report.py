import asyncio
from copy import deepcopy

import pytest

from commerce_agent.sales.policy import SalesRequest
from commerce_agent.sales.report import render_sales_report
from test_sales_evidence import collect, CONTEXT
from test_sales_runtime import FixtureModel, dependencies, invocation
from commerce_agent.sales.runtime import build_sales_subgraph


def fixture(request=None, change=None):
    prepared, _ = collect(request or SalesRequest(user_query="Отчет продаж"), change)
    candidate = asyncio.run(FixtureModel().analyze(instructions="", prepared=prepared))
    return candidate, prepared


def test_report_is_published_by_sales_and_preserves_native_result():
    output = asyncio.run(build_sales_subgraph(dependencies()).ainvoke(invocation(), context=CONTEXT))
    analysis = output["sales_analysis"]
    assert analysis.summary.startswith("# Отчёт Sales")
    assert "701.75" in analysis.summary and "3 007.5" in analysis.summary
    assert "SKU-A | 100.25 | 0 | 100.25 | нет данных | 1 | 0 | 1" in analysis.summary
    assert analysis.result.summary == "Продажи проверены по данным источника."
    assert "Причины изменений не установлены" in analysis.summary
    assert analysis.status == "partial"


def test_native_deltas_are_not_recomputed_and_null_is_not_zero():
    def change(path, payload):
        if path == "sku-revenue-change":
            row = payload["data"]["rows"][0]
            row.update(previous_revenue=110, delta_revenue=-9.75, delta_revenue_pct=-8.86,
                       previous_sales=2, delta_sales=-1, driver_role="decline_driver")
    candidate, prepared = fixture(change=change)
    before = deepcopy(prepared)
    report = render_sales_report(candidate, prepared)
    assert "100.25 | 110 | -9.75 | -8.86 | 1 | 2 | -1" in report
    assert "Наибольшее снижение" in report
    assert prepared == before


def test_multiple_requested_skus_are_not_aggregated():
    candidate, prepared = fixture(SalesRequest(user_query="Отчет продаж", scope="sku_list", targets=("SKU-A", "SKU-B")))
    report = render_sales_report(candidate, prepared)
    assert "Область: SKU-A" in report and "Область: SKU-B" in report
    assert "пересекающиеся области" in report


def test_single_sku_scope_is_not_mislabeled_as_portfolio():
    candidate, prepared = fixture(SalesRequest(user_query="Продажи SKU-A", scope="single_sku", targets=("SKU-A",)))
    report = render_sales_report(candidate, prepared)
    assert "Область: SKU-A" in report
    assert "Область: портфель" not in report


@pytest.mark.parametrize("mutate", [
    lambda c: setattr(c, "summary", "Падение произошло из-за рекламы."),
    lambda c: c.sales_decomposition.rows[0].__setitem__(1, 999999),
    lambda c: setattr(c, "summary", "token=sk-" + "x" * 40),
    lambda c: setattr(c, "summary", "contact person@example.com"),
])
def test_renderer_cannot_bypass_publication_guard(mutate):
    candidate, prepared = fixture()
    mutate(candidate)
    with pytest.raises(ValueError):
        render_sales_report(candidate, prepared)


def test_unknown_snapshot_layout_is_not_interpreted_as_known_metrics():
    def change(path, payload):
        if path == "report-snapshot":
            payload["data"]["report_table"]["columns"][0] = "Неизвестный период"
    candidate, prepared = fixture(change=change)
    report = render_sales_report(candidate, prepared)
    assert "недоступна в поддерживаемом формате" in report
    assert "Неизвестный период" not in report


def test_native_production_snapshot_comparisons_render_verbatim():
    def change(path, payload):
        if path == "report-snapshot":
            payload["data"]["report_table"].update(
                columns=["Сравнение", "Периоды", "Продажи тек./пред.", "Δ продаж", "Выручка тек./пред.", "Δ выручки"],
                rows=[[label, "24.09.2026 - 30.09.2026 к 17.09.2026 - 23.09.2026", "163 / 175", "-12 (-6.9%)", "380 563 ₽ / 414 068 ₽", "-33 505.65 ₽ (-8.1%)"] for label in (
                    "Последний день в данных к предыдущему", "Последние 7 дней к предыдущим 7", "Последние 30 дней к предыдущим 30")])
    candidate, prepared = fixture(change=change)
    report = render_sales_report(candidate, prepared)
    assert "163 / 175" in report and "-33 505.65 ₽ (-8.1%)" in report
    assert "24.09.2026 - 30.09.2026" in report


def test_empty_comparison_is_missing_not_zero():
    def change(path, payload):
        if path == "sku-revenue-change":
            payload["data"].update(rows=[], row_count=0)
    from test_sales_evidence import respond
    output = asyncio.run(build_sales_subgraph(dependencies(handler=lambda r: respond(r, change=change))).ainvoke(invocation(), context=CONTEXT))
    assert output["sales_analysis"].status == "failed"
    assert output["sales_analysis"].result is None


def test_markup_is_escaped():
    from commerce_agent.sales.report import _cell
    assert _cell("[link](url)|<script>\ntext") == "\\[link\\](url)\\|&lt;script&gt; text"
    assert _cell(None) == "нет данных" and _cell(0) == "0"


def test_rankings_are_bounded_and_keep_native_rounded_percent():
    def change(path, payload):
        if path == "sku-revenue-change":
            template = payload["data"]["rows"][0]
            rows = []
            for i in range(1, 8):
                rows.append({**template, "vendor_code": f"DROP-{i}", "current_revenue": 100 - i,
                             "previous_revenue": 100, "delta_revenue": -i, "delta_revenue_pct": -i,
                             "driver_role": "decline_driver"})
            rows.append({**template, "vendor_code": "TINY", "current_revenue": 99.999,
                         "previous_revenue": 100, "delta_revenue": -0.001, "delta_revenue_pct": 0.0,
                         "driver_role": "decline_driver"})
            payload["data"].update(rows=rows, row_count=len(rows))
    candidate, prepared = fixture(change=change)
    report = render_sales_report(candidate, prepared)
    assert report.index("DROP-7") < report.index("DROP-6") < report.index("DROP-5")
    assert "DROP-1" not in report and "DROP-2" not in report and "TINY" not in report
    # A tiny absolute decline must not become "stable" because the native % rounds to zero.
    prepared["tables"]["sales_decomposition"]["rows"] = [prepared["tables"]["sales_decomposition"]["rows"][-1]]
    candidate.sales_decomposition.rows = deepcopy(prepared["tables"]["sales_decomposition"]["rows"])
    report = render_sales_report(candidate, prepared)
    assert "TINY | 99.999 | 100 | -0.001 | 0.0" in report
