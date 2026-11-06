"""Readable validated Marketing report; never an independent model output."""
from commerce_agent.marketing.publication import validate_marketing_publication


def render_marketing_report(result, prepared):
    result = validate_marketing_publication(result, prepared)
    parts = ["# Marketing — анализ рекламы", f"Последний общий день данных: {prepared['report_date']}.",
        f"Кампании и SKU: {prepared['period']} против {prepared['previous_period']}.",
        "ДРР — расходы / вся выручка продвигаемых SKU, не ROI и не доказанный рекламный эффект.",
        "## Главное", result.summary, "## Наблюдения"]
    for finding in result.findings:
        parts.extend([finding.observation, finding.cause])
    parts.append("## Что проверить у соседних агентов")
    parts.extend(f"- {check.target_agent.value}: {check.question}" for check in result.cross_agent_requests)
    parts.append("## Ограничения")
    parts.extend(f"- {gap.reason}" for gap in result.data_gaps)
    parts.append("## Проверенные таблицы")
    def cell(value):
        return ("нет данных" if value is None else str(value)).replace("|", "\\|").replace("\n", " ")
    for table in result.tables:
        parts.extend([f"### {table.title}", "| " + " | ".join(cell(v) for v in table.columns) + " |",
                      "| " + " | ".join("---" for _ in table.columns) + " |"])
        parts.extend("| " + " | ".join(cell(v) for v in row) + " |" for row in table.rows)
    return "\n\n".join(parts)
