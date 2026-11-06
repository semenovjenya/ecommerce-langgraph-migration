"""Closed native propositions: source values only, no business arithmetic.

Portfolio comparisons use typed service rows; SKU changes use the unchanged
decomposition table. Selection is bounded, never presented as full coverage.
"""
from datetime import date, timedelta
import math

from commerce_agent.adapters.contracts import SalesComparison


def _window(current, previous=None):
    start, end = (date.fromisoformat(value) for value in current.split("/"))
    if end < start:
        raise ValueError("invalid analytic period")
    if previous:
        before, last = (date.fromisoformat(value) for value in previous.split("/"))
        if last + timedelta(days=1) != start or end - start != last - before:
            raise ValueError("incompatible analytic comparison")
    return start, end


def _number(value):
    return type(value) in {int, float} and math.isfinite(value)


def _change(label, scope, current, previous, now, before, delta, rate, unit):
    if not all(_number(value) for value in (now, before, delta)) or rate is not None and not _number(rate):
        raise ValueError("invalid analytic native value")
    direction = "рост" if delta > 0 else "снижение" if delta < 0 else "без изменения"
    percentage = "процент недоступен" if rate is None else f"{rate}%"
    return (f"{label}, {scope}: {current} к {previous}: {now} / {before} {unit}; "
            f"{direction}, изменение {delta} {unit} ({percentage}).")


def analytical_claims(prepared):
    claims = []
    known_refs = {item["evidence_ref"] for item in prepared["evidence"]}
    def add(text, sku, period, refs, factor):
        _window(period)
        if not refs or not set(refs) <= known_refs or sku == "Артикул не сопоставлен":
            return
        claims.append({"text": text, "sku": sku, "period": period, "evidence_refs": list(refs),
                       "factor": factor, "confidence": "medium"})

    for scope in prepared.get("overview_signals", []):
        target = scope["target"]
        label = f"SKU {target}" if target else "портфель"
        for raw in scope.get("snapshot_comparisons", []):
            row = SalesComparison.model_validate(raw)
            current = f"{row.current_period.date_from}/{row.current_period.date_to}"
            previous = f"{row.previous_period.date_from}/{row.previous_period.date_to}"
            _window(current, previous)
            if prepared["request"].get("date_from") and (
                current != f"{prepared['request']['date_from']}/{prepared['request']['date_to']}"):
                continue
            for family, now, before, delta, rate, unit in (
                ("продаж", row.current_sales_units, row.previous_sales_units, row.sales_units_delta, row.sales_units_delta_pct, "units"),
                ("выручки", row.current_revenue_rub, row.previous_revenue_rub, row.revenue_delta_rub, row.revenue_delta_pct, "RUB")):
                text = _change(f"Динамика {family}", label, current, previous, now, before, delta, rate, unit)
                add(text, target, current, scope["snapshot_evidence_refs"], f"Динамика {family}")

    table = prepared.get("tables", {}).get("sales_decomposition", {"columns": [], "rows": []})
    groups = {}
    for cells in table["rows"]:
        row = dict(zip(table["columns"], cells))
        required = {"Артикул", "Текущий период", "Предыдущий период", "delta_revenue_pct",
                    "current_revenue", "previous_revenue", "delta_revenue", "current_sales", "previous_sales", "delta_sales"}
        if not required <= set(row):
            continue
        key = (row.get("Запрошенный артикул"), row["Текущий период"], row["Предыдущий период"])
        groups.setdefault(key, []).append(row)
    for (_, current, previous), rows in groups.items():
        _window(current, previous)
        # Native SQL already ranks abs(delta). Preserve its first five rows;
        # all rows/facts still remain in the authoritative public result.
        for row in rows[:5]:
            sku = row["Артикул"]
            # All four native facts must agree on source and values. Ambiguous
            # duplicates never attach a last-wins provenance to another row.
            bound = [fact for fact in prepared["facts"] if fact.get("sku") == sku
                     and fact.get("period") in {current, previous}]
            metrics = ("current_revenue", "previous_revenue", "current_sales", "previous_sales")
            if any(len([f for f in bound if f["metric"] == metric]) != 1 for metric in metrics):
                continue
            if any(f["value"] != row[f["metric"]] or f["period"] != (
                previous if f["metric"].startswith("previous") else current) for f in bound):
                continue
            refs = {f["evidence_ref"] for f in bound}
            if len(refs) != 1 or {f["source_table"] for f in bound} != {"sales.wb_supplier_sales"}:
                continue
            ref = next(iter(refs))
            for family, metric, rate, unit in (("выручки", "revenue", row["delta_revenue_pct"], "RUB"),
                                               ("продаж", "sales", None, "units")):
                text = _change(f"Динамика {family}", f"SKU {sku}", current, previous,
                    row[f"current_{metric}"], row[f"previous_{metric}"], row[f"delta_{metric}"], rate, unit)
                add(text, sku, current, [ref] if ref else [], f"Динамика {family}")

    for scope in prepared.get("overview_signals", []):
        for row in scope["conversion_watchlist"][:5]:
            if not row["report_date"]:
                continue
            fields = [(name, row[key]) for key, name in (("open_count", "просмотры"), ("cart_count", "корзины"),
                       ("order_count", "заказы"), ("add_to_cart_conversion", "open-to-cart, %"),
                       ("cart_to_order_conversion", "cart-to-order, %"), ("buyout_percent", "выкуп, %"))]
            values = "; ".join(f"{name}: {value if value is not None else 'нет данных'}" for name, value in fields)
            if any(value is not None and not _number(value) for _, value in fields):
                raise ValueError("invalid native funnel value")
            period = f"{row['report_date']}/{row['report_date']}"
            if prepared["request"].get("date_from") and not (
                prepared["request"]["date_from"] <= row["report_date"] <= prepared["request"]["date_to"]):
                continue
            add(f"Воронка SKU {row['sku']} на {row['report_date']}: {values}. Наблюдение не доказывает причину изменения продаж.",
                row["sku"], period, scope["funnel_evidence_refs"], "Воронка")
    return claims
