"""Human-readable Finance report from the validated result, without arithmetic."""
from commerce_agent.finance.publication import validate_finance_publication
from commerce_agent.finance.evidence import analysis_policy_version


def render_finance_report(result, prepared, *, table_format="gfm"):
    if table_format not in {"gfm", "legacy_spaced"}:
        raise ValueError("Finance unknown table rendering format")
    result = validate_finance_publication(result, prepared)
    window = (["Latest overview: окно агрегата не подтверждено; дата среза не является периодом метрики."]
              if prepared["request"]["operation"] == "overview" else
              [f"Период запроса: {prepared['period']}.", f"Предыдущий период для сравнения: {prepared['previous_period']}."])
    parts = ["# Finance — экономика маркетплейса", *window,
        "Latest overview приведён отдельно; его дата не заменяет исторический период отчёта и не подтверждает однодневное окно.",
        "## Главное", result.summary, "## Аналитические наблюдения"]
    headings = {"expenses": "Структура расходов маркетплейса и отдельная себестоимость",
                "declines": "SKU со снижением маржи",
                "growth": "SKU с ростом маржи — наблюдение, не установленная причина"}
    for finding in result.findings:
        if analysis_policy_version(prepared) == 2:
            lines = set(finding.observation.split("\n"))
            for group, indices in prepared["analytical_groups"].items():
                if indices and lines <= {prepared["claims"][index]["text"] for index in indices}:
                    parts.append("### " + headings[group])
                    break
        parts.append(finding.observation)
    parts.extend(["Маржа калькулятора — не чистая прибыль. Выплаты — не выручка. Выводы о ликвидности не сделаны.",
                  "## Проверки соседними агентами — ещё не выполнены"])
    parts.extend(f"- {check.target_agent.value}: {check.question}" for check in result.cross_agent_requests)
    parts.append("## Ограничения данных")
    parts.extend(f"- {gap.reason}" for gap in result.data_gaps)
    def cell(value):
        return ("нет данных" if value is None else str(value)).replace("|", "\\|").replace("\n", " ")
    for table in result.tables:
        parts.append(f"## {table.title}")
        lines = ["| " + " | ".join(cell(value) for value in table.columns) + " |",
                 "| " + " | ".join("---" for _ in table.columns) + " |"]
        lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in table.rows)
        if table_format == "legacy_spaced":
            # Read-only replay of historical reports; new runtime uses GFM.
            parts.extend(lines)
        else:
            parts.append("\n".join(lines))
    return "\n\n".join(parts)
