"""Readable Price report; native tables, no arithmetic or causal prose."""
from commerce_agent.price.publication import validate_price_publication
from commerce_agent.price.analysis import shown


def render_price_report(result, prepared, *, table_format="gfm"):
    if table_format not in {"gfm", "legacy_spaced"}:
        raise ValueError("Price unknown table format")
    result = validate_price_publication(result, prepared)
    parts = ["# Price — цены, качество рынка и сценарии", f"Дата проверки: {prepared['today']}.",
        ("Точные исторические даты; пропуски не заполнены последним снимком. Ценовые слои разделены."
            if prepared.get("history_version") else
            "Latest snapshot не является историческим окном метрики. Цена сайта, кошелька и эффективная цена не пересчитываются."),
        "## Главное", result.summary,
        "## Аналитические наблюдения — не установленные причины"]
    headings = {"layers": "Слои цен и нативные отклонения среди полученных SKU", "quality": "Актуальность и качество аналогов",
        "scenarios": "Сценарии переоценки — не текущие факты и не approval", "raw_market": "Широкий рынок и raw выборка",
        "daily": "Дневные наблюдения цены и продаж — не причинность"}
    for finding in result.findings:
        lines = set(finding.observation.split("\n"))
        for group, indices in prepared["analytical_groups"].items():
            if indices and lines <= {prepared["claims"][index]["text"] for index in indices}:
                parts.append("### " + headings[group])
                break
        parts.append(finding.observation)
    parts.extend(["## Проверки соседними агентами — ещё не выполнены"])
    parts.extend(f"- {check.target_agent.value}: {check.question}" for check in result.cross_agent_requests)
    parts.append("## Ограничения данных")
    parts.extend(f"- {gap.reason}" for gap in result.data_gaps)
    parts.append("Нет изменения цены, выполненного действия или подтверждённого финансового эффекта.")
    def cell(value):
        return shown(value).replace("|", "\\|").replace("\n", " ")
    for table in result.tables:
        if not table.columns:
            parts.extend([f"## {table.title}", "Источник вернул 0 строк; данные недоступны."])
            continue
        parts.append(f"## {table.title}")
        lines = ["| " + " | ".join(cell(value) for value in table.columns) + " |",
            "| " + " | ".join("---" for _ in table.columns) + " |"]
        lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in table.rows)
        if table_format == "legacy_spaced":
            parts.extend(lines)  # Explicit read-only reconstruction, never default.
        else:
            parts.append("\n".join(lines))
    return "\n\n".join(parts)
