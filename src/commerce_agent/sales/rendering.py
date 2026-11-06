"""Pure Sales rendering, copied verbatim and parity-tested."""
from __future__ import annotations
from commerce_agent.sales.contracts import SalesDiagnosticResult, ReportTable

def _is_delta_column(column: str) -> bool:
    normalized = column.lower()
    return (
        "изменен" in normalized
        or "изменён" in normalized
        or "дельт" in normalized
        or "delta" in normalized
        or "Δ" in column
    )


def _format_signed_value(value: str | int | float | None) -> str | int | float | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return f"+{value}" if value > 0 else value
    if not isinstance(value, str):
        return value

    stripped = value.strip()
    if not stripped or stripped.startswith(("+", "-")):
        return value

    compact = stripped.replace(" ", "").replace(",", ".")
    try:
        numeric = float(compact)
    except ValueError:
        return value
    if numeric > 0:
        return f"+{stripped}"
    return value


def _normalize_cell(value: str | int | float | None, max_len: int = 120, signed: bool = False) -> str:
    if value is None or value == "":
        return "—"

    if signed:
        value = _format_signed_value(value)

    text = " ".join(str(value).split()).replace("|", "/")
    if len(text) > max_len:
        return text[: max_len - 1].rstrip() + "…"
    return text


def _render_table(table: ReportTable) -> list[str]:
    columns = [_normalize_cell(column, max_len=40) for column in table.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| "
        + " | ".join(
            _normalize_cell(
                cell,
                signed=index < len(columns) and _is_delta_column(columns[index]),
            )
            for index, cell in enumerate(row)
        )
        + " |"
        for row in (table.rows or [["Нет данных"] + [None] * (len(columns) - 1)])
    ]
    return [f"**{table.title}**", "", header, separator, *rows]


def render_report_table(table: ReportTable) -> str:
    return "\n".join(_render_table(table))


def render_sales_diagnostic(result: SalesDiagnosticResult) -> str:
    lines: list[str] = []

    if result.sales_report_snapshot:
        lines.extend(_render_table(result.sales_report_snapshot))
        lines.append("")

    lines.extend(
        [
            f"# {result.agent_name}",
            "",
            result.summary,
            "",
        ]
    )

    for index, table in enumerate(result.as_tables()):
        if index:
            lines.append("")
        lines.extend(_render_table(table))
        lines.append("")

    if result.findings:
        lines.append("Выводы:")
        for finding in result.findings:
            lines.append(
                "- "
                f"{_normalize_cell(finding.factor, max_len=80)}: "
                f"{_normalize_cell(finding.observation, max_len=180)} "
                f"Влияние: {_normalize_cell(finding.impact, max_len=140)}"
            )
        lines.append("")

    if result.recommendations:
        lines.append("Рекомендации:")
        for recommendation in result.recommendations:
            lines.append(f"- {_normalize_cell(recommendation, max_len=180)}")
        lines.append("")

    if result.data_gaps:
        lines.append("Ограничения данных:")
        for gap in result.data_gaps:
            lines.append(f"- {gap.metric}: {_normalize_cell(gap.impact, max_len=180)}")

    return "\n".join(line for line in lines if line is not None).strip()
