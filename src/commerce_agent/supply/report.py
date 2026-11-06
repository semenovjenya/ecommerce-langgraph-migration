"""Readable Supply publication, using the same validated domain result."""
from commerce_agent.supply.publication import validate_supply_publication

LABELS = {"vendor_code": "Артикул", "nm_id": "WB nm_id", "sku": "Артикул", "report_date": "Дата",
    "snapshot_date": "Дата снимка", "sales_units": "Продажи, шт", "sales_7d_units": "Продажи за 7 дней, шт",
    "stock_qty": "Доступный остаток, шт", "stock_qty_full": "Полный остаток, шт",
    "available_stock_qty": "Доступный остаток, шт", "full_stock_qty": "Полный остаток, шт",
    "unavailable_stock_qty": "Недоступный остаток, шт", "avg_stock_qty": "Средний доступный остаток, шт",
    "avg_stock_qty_full": "Средний полный остаток, шт", "current_stock_qty": "Текущий доступный остаток, шт",
    "current_stock_qty_full": "Текущий полный остаток, шт", "days_of_stock": "Покрытие доступным запасом, дней",
    "turnover_ratio": "Оборачиваемость", "days_on_hand": "Дни запаса за период",
    "turnover_band": "Класс покрытия", "availability_status": "Доступность", "conclusion": "Вывод сервиса",
    "reorder_flag": "Статус дозаказа", "reorder_qty_recommended": "Рекомендация дозаказа, шт",
    "supplier_inbound_qty": "Входящая поставка, шт", "unit_cost_rub": "Себестоимость единицы, RUB",
    "current_stock_value_rub": "Стоимость запаса, RUB", "valuation_status": "Статус оценки"}


def render_supply_report(result, prepared):
    result = validate_supply_publication(result, prepared)
    methods = prepared["methods"]
    parts = ["# Supply — остатки и поставки", f"Исходный снимок остатков: {prepared['report_date']}.",
        f"Период/снимок запрошенного анализа: {prepared['period']}.", "## Главное", result.summary,
        "## Метод и границы расчёта"]
    if "method" in methods: parts.append(methods["method"])
    if "valuation_method" in methods:
        parts.append(f"Стоимость запасов на {methods['value_snapshot_date']}: {methods['valuation_method']}")
    if "coverage_formula" in methods: parts.append(methods["coverage_formula"])
    if "calculation_status" in methods:
        parts.append("Расчёт сервиса: " + ("точный (exact)" if methods["calculation_status"] == "exact" else
            "точная историческая оборачиваемость недоступна; статусы альтернативных методов показаны отдельно"
            if "alternative_method_checks" in methods else
            "нужны альтернативные источники; альтернативные расчёты ещё не выполнялись"))
    if methods.get("limitation"): parts.append(methods["limitation"])
    if "source_coverage" in methods:
        coverage = methods["source_coverage"]
        parts.append(f"Покрытие источников: {coverage['stock_rows']} строк остатков, {coverage['sales_rows']} строк продаж; это не доказательство полноты каждого дня.")
    if "current_stock_dates" in methods:
        parts.append("Отдельные даты текущего остатка: " + ", ".join(dict.fromkeys(date or "нет данных" for date in methods["current_stock_dates"])))
    if "totals" in methods:
        totals = methods["totals"]
        parts.append(f"Подтверждённая стоимость возвращённых запасов: {totals['confirmed_stock_value_rub']} RUB; SKU с оценкой: {totals['valued_sku_count']}; без себестоимости: {totals['missing_cogs_sku_count']}.")
    if "valuations_by_target" in methods:
        parts.append("Оценки получены отдельными точными чтениями каждого SKU; общий денежный итог списка не рассчитан.")
        for valuation in methods["valuations_by_target"]:
            totals = valuation["totals"]
            parts.append(f"SKU {valuation['target']}, снимок {valuation['snapshot_date'] or 'неизвестен'}: "
                f"native confirmed subtotal — {totals['confirmed_stock_value_rub']} RUB; "
                f"строк с оценкой {totals['valued_sku_count']}, без себестоимости {totals['missing_cogs_sku_count']}. "
                "Это сумма только подтверждённых строк, не оценка строк без COGS. Метод: " + valuation["valuation_method"])
    if "simulation" in methods:
        parts.append(f"Эмуляция: последний известный снимок {methods['simulation']['source_snapshot_date']} используется как текущий; это не наблюдение на сегодняшнюю дату.")
    parts.append("## Наблюдения")
    parts.extend(finding.observation for finding in result.findings)
    parts.extend(["Влияние на продажи, рекламу, цену и прибыль пока не подтверждено.", "## Проверки соседними агентами (ещё не выполнены)"])
    parts.extend(f"- {check.target_agent.value}: {check.question}" for check in result.cross_agent_requests)
    parts.append("## Ограничения")
    parts.extend(f"- {gap.reason}" for gap in result.data_gaps)
    def cell(value):
        return ("нет данных" if value is None else str(value)).replace("|", "\\|").replace("\n", " ")
    for table in result.tables:
        parts.append(f"## {table.title}")
        lines = ["| " + " | ".join(cell(LABELS.get(v, v)) for v in table.columns) + " |",
            "| " + " | ".join("---" for _ in table.columns) + " |"]
        lines.extend("| " + " | ".join(cell(v) for v in row) + " |" for row in table.rows)
        parts.append("\n".join(lines))
    return "\n\n".join(parts)
