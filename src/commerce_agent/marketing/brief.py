"""Versioned readable scoped projection; native evidence stays in result.

No new ratios, totals, causal claims, model prose or source joins. Call only
after scoped publication validation; this module formats the native cells.
"""


def cell(value):
    return ("нет данных" if value is None else str(value)).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def table(columns, rows):
    return "\n".join(["| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
        *("| " + " | ".join(cell(value) for value in row) + " |" for row in rows)])


def render_brief_scoped_report(result, prepared):
    """Readable projection of an already validated scoped result, not evidence."""
    request = prepared["request"]
    def fact_value(metric,period,*,sku=None,target=None):
        # A raw native table cell is not necessarily source-qualified evidence.
        # Use only the exact facts accepted by publication, including its gaps.
        matches=[fact for fact in result.facts if fact.metric==metric and fact.period==period
            and fact.sku==sku and (target is None or f"; target {target};" in (fact.metric_definition or ""))]
        return matches[0].value if len(matches)==1 and matches[0].status=="confirmed" else None
    parts = ["# Marketing — результат", "## Вопрос", request["user_query"],
        f"Период: {prepared['period']}; сравнение: {prepared['previous_period']}."]
    if request["scope"] == "portfolio":
        rows = prepared["anchor_snapshot"]["data"]["rows"]
        selected = next((row for row in rows if
            f"{row['current_period']['date_from']}/{row['current_period']['date_to']}" == prepared["period"]
            and f"{row['previous_period']['date_from']}/{row['previous_period']['date_to']}" == prepared["previous_period"]), None)
        if selected is None:
            parts += ["## Портфель", "Нет native портфельного сравнения именно за запрошенные окна. Latest snapshot остаётся отдельным справочным контекстом в result."]
        else:
            specs = [("Расходы рекламы, ₽", "spend_rub", "spend_delta_pct", "%"),
                ("Клики", "clicks", "clicks_delta_pct", "%"),
                ("Заказы рекламной статистики", "ad_orders", "ad_orders_delta_pct", "%"),
                ("Вся выручка продвигаемых SKU, ₽", "promoted_sku_revenue_rub", "promoted_sku_revenue_delta_pct", "%"),
                ("ДРР к этой выручке, %", "drr_pct", "drr_delta_pp", "п.п."),
                ("CPC, ₽/клик", "cpc_rub", None, None)]
            parts += ["## Что установлено", table(["Показатель", "Сейчас", "Ранее", "Native изменение"],
                [[label, fact_value("snapshot.current_"+metric,prepared["period"]),
                    fact_value("snapshot.previous_"+metric,prepared["previous_period"]),
                    "не подтверждено источником" if delta is None or fact_value("snapshot."+delta,prepared["period"]+" vs "+prepared["previous_period"]) is None
                    else f"{fact_value('snapshot.'+delta,prepared['period']+' vs '+prepared['previous_period'])} {unit}"]
                 for label, metric, delta, unit in specs])]
    target_rows = []
    for record in prepared["scoped_read_results"]:
        target = record["target"]
        if record["mode"] == "failed":
            target_rows.append([target, "чтение недоступно", "нет данных", "нет данных", "нет данных"])
            continue
        rows = record["envelope"]["data"]["rows"]
        if not rows:
            target_rows.append([target, "нет подтверждённых строк", "нет данных", "нет данных", "нет данных"])
        for row in rows:
            def pair(metric):
                a=fact_value("scoped.current_"+metric,prepared["period"],sku=row["vendor_code"],target=target)
                b=fact_value("scoped.previous_"+metric,prepared["previous_period"],sku=row["vendor_code"],target=target)
                return ("нет подтверждённых данных" if a is None else str(a)) + " / " + ("нет подтверждённых данных" if b is None else str(b))
            target_rows.append([f"{target} → {row['vendor_code']} / nm_id {row['nm_id']}", "native comparison",
                pair("spend_rub"), pair("ad_orders"), pair("estimated_drr_pct")])
    parts += ["## Запрошенные SKU", table(["Target → native identity", "Данные", "Расход, ₽: сейчас / ранее",
        "Ad orders: сейчас / ранее", "Оценочный ДРР, %: сейчас / ранее"], target_rows),
        "## Вывод и границы", "Показатели выше — наблюдения, не доказательство причины изменения или прибыльности. Выручка SKU не равна ad-attributed выручке; ДРР не ROI. Отсутствующие строки не означают нулевые продажи/расходы.",
        "## Что проверить дальше"]
    parts += ["- " + check.target_agent.value + ": " + check.question for check in result.cross_agent_requests]
    if not result.cross_agent_requests:
        parts += ["- Адресные проверки не сформированы: подтверждённого scope недостаточно."]
    parts += ["Проверки только предложены: соседние агенты ещё не вызваны, внешние действия не выполнялись.",
        "## Полные данные и ограничения", "Полные проверенные facts, tables, findings, data_gaps и cross_agent_requests сохранены отдельно в marketing_analysis.result; evidence_refs — в marketing_analysis. Этот короткий текст не заменяет их и не подтверждает актуальность магазина сегодня.",
        f"Последний общий день источников: {prepared['report_date']}. Независимые окна/freshness и ограничения каждой строки смотрите в result.data_gaps; точная рекламная атрибуция и causal lift не подтверждены."]
    return "\n\n".join(parts)
