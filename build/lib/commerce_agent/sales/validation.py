"""Pure legacy Sales guardrails; no Agents SDK or hidden DB access."""
from __future__ import annotations
import re
from typing import Any
from commerce_agent.sales.contracts import DiagnosticFinding, ReportTable, SalesDiagnosticResult

FORBIDDEN_METRIC_PATTERNS: tuple[tuple[str, str], ...] = (
    ("cac", r"\bcac\b"),
    ("romi", r"\bromi\b"),
    ("mer", r"\bmer\b"),
    ("retention", r"\bretention\b"),
    ("new-vs-repeat", r"\bnew\s*-\s*vs\s*-\s*repeat\b"),
    ("cash-flow", r"\bcash[\s-]*flow\b"),
    ("working capital", r"\bworking\s+capital\b"),
    ("sla", r"\bsla\b"),
    ("contribution after cogs", r"\bcontribution\s+after\s+cogs\b"),
    ("profitability after procurement cost", r"\bprofitability\s+after\s+procurement\s+cost\b"),
)


MISSING_METRIC_CONTEXT_MARKERS: tuple[str, ...] = (
    "known_data_gap",
    "data_gap",
    "no ",
    "without ",
    "cannot ",
    "can't ",
    "нет ",
    "без ",
    "отсутств",
    "недоступ",
    "не подтверж",
    "не рассчит",
)


CONTEXTUAL_NM_ID_PATTERN = re.compile(
    r"(?:(?:sku|артикул|nm[_\s-]?id)\s*[:#№-]?\s*)?(\b\d{7,12}\b)(?=\s*:)|"
    r"(?:sku|артикул|nm[_\s-]?id)\s*[:#№-]?\s*(\b\d{7,12}\b)",
    re.IGNORECASE,
)


MECHANICAL_CAUSE_PATTERN = re.compile(
    r"механик|механическ|снижен\w*\s+(?:продаж|реализац|выручк)|"
    r"паден\w*\s+(?:продаж|реализац|выручк)|спад\w*\s+(?:фактическ\w*\s+)?(?:продаж|реализац|выручк)|"
    r"сжати\w*\s+продаж|"
    r"рост\w*\s+внутри\s+негативн\w*\s+тренд|рост\w*\s+отдельн\w*\s+sku|"
    r"краткосрочн\w*\s+отскок|компенсационн\w*\s+эффект|локальн\w*\s+улучшени",
    re.IGNORECASE,
)


def _table_violations(table: ReportTable, label: str) -> list[str]:
    violations: list[str] = []
    if not table.title.strip():
        violations.append(f"{label} has empty title.")
    if not table.columns:
        violations.append(f"{label} has no columns.")
    for index, row in enumerate(table.rows, start=1):
        if len(row) != len(table.columns):
            violations.append(
                f"{label} row {index} length {len(row)} does not match columns {len(table.columns)}."
            )
    return violations


def _normalized_header(value: str) -> str:
    return value.strip().lower().replace("ё", "е")


def _is_empty_decomposition_value(value: str | int | float | None) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() in {"", "-", "—"}
    return False


def _find_column_index(columns: list[str], *patterns: tuple[str, ...]) -> int | None:
    normalized_columns = [_normalized_header(column) for column in columns]
    for pattern in patterns:
        for index, column in enumerate(normalized_columns):
            if all(token in column for token in pattern):
                return index
    return None


def _sales_decomposition_violations(table: ReportTable, label: str) -> list[str]:
    required_columns = {
        "current_revenue": (("current_revenue",), ("текущ", "выруч")),
        "previous_revenue": (("previous_revenue",), ("предыдущ", "выруч"), ("прошл", "выруч")),
        "delta_revenue": (
            ("delta_revenue",),
            ("delta", "выруч"),
            ("дельт", "выруч"),
            ("измен", "выруч"),
            ("δ", "выруч"),
        ),
        "current_sales": (("current_sales",), ("текущ", "продаж"), ("текущ", "шт")),
        "previous_sales": (
            ("previous_sales",),
            ("предыдущ", "продаж"),
            ("предыдущ", "шт"),
            ("прошл", "продаж"),
            ("прошл", "шт"),
        ),
        "delta_sales": (
            ("delta_sales",),
            ("delta", "продаж"),
            ("delta", "шт"),
            ("дельт", "продаж"),
            ("дельт", "шт"),
            ("измен", "продаж"),
            ("измен", "шт"),
            ("δ", "продаж"),
        ),
    }
    violations: list[str] = []
    column_indexes: dict[str, int] = {}

    for column_name, patterns in required_columns.items():
        index = _find_column_index(table.columns, *patterns)
        if index is None:
            violations.append(f"{label} is missing required decomposition column `{column_name}`.")
            continue
        column_indexes[column_name] = index

    for row_index, row in enumerate(table.rows, start=1):
        for column_name, column_index in column_indexes.items():
            if column_index >= len(row) or _is_empty_decomposition_value(row[column_index]):
                violations.append(
                    f"{label} row {row_index} has empty `{column_name}`; use the deterministic sales decomposition tool."
                )

    return violations


def _finding_violations(findings: list[DiagnosticFinding], prefix: str) -> list[str]:
    violations: list[str] = []
    for index, finding in enumerate(findings, start=1):
        finding_fields = [
            finding.factor,
            finding.observation,
            finding.impact,
            finding.cause,
            finding.action,
            *finding.evidence,
        ]
        for label, pattern in FORBIDDEN_METRIC_PATTERNS:
            unsupported_claim = any(
                any(
                    not _unsupported_metric_mention_is_limitation(
                        str(field).lower(),
                        match,
                    )
                    for match in re.finditer(pattern, str(field).lower())
                )
                for field in finding_fields
            )
            if unsupported_claim:
                violations.append(f"{prefix} finding {index} references unsupported metric: {label}")
        if not finding.evidence:
            violations.append(f"{prefix} finding {index} has no evidence.")
        if finding.confidence.strip().lower() not in {"high", "medium", "low"}:
            violations.append(
                f"{prefix} finding {index} has invalid confidence `{finding.confidence}`."
            )
    return violations


def _unsupported_metric_mention_is_limitation(
    text: str,
    match: re.Match[str],
) -> bool:
    """Return true only when the matched metric itself is explicitly unavailable.

    Missing-data text elsewhere in the same field must not legitimise a positive
    metric claim (for example, ``working capital improved, but no source exists``).
    """

    clause_start = max(
        [
            text.rfind(separator, 0, match.start())
            for separator in (".", ";", "!", "?", "\n")
        ],
        default=-1,
    )
    before = text[clause_start + 1 : match.start()]
    before = re.split(
        r"\b(?:but|however|whereas|while|но|однако|тогда как|при этом)\b",
        before,
    )[-1]

    # A missing-data qualifier before a metric can govern a short list, e.g.
    # ``no working capital / cash-flow layer``.
    if any(marker in before[-120:] for marker in MISSING_METRIC_CONTEXT_MARKERS):
        return True
    if re.search(
        r"(?:\bmissing\b|\bunavailable\b|\bunknown\b|\black(?:s|ing)?\b)"
        r"[^.;!?\n]{0,80}$",
        before,
    ):
        return True

    after = text[match.end() :]
    after = re.split(
        r"[.;!?\n]|\b(?:but|however|whereas|while|но|однако|тогда как|при этом)\b",
        after,
        maxsplit=1,
    )[0]
    # A qualifier after the metric is accepted only when it is the immediate
    # predicate of that metric. A later ``no source`` cannot mask ``improved``.
    return bool(
        re.match(
            r"^\s*(?:(?:metric|data|layer|source|value|calculation|данн\w*|"
            r"сло\w*|источник\w*|значени\w*|расч[её]т\w*)\s+){0,3}"
            r"(?:(?:is|are|was|were|remains?|оказал\w*|являет\w*)\s+)?"
            r"(?:not\s+(?:available|known|confirmed|calculated|provided)|"
            r"unavailable|unknown|missing|cannot\s+be\s+(?:calculated|confirmed)|"
            r"can't\s+be\s+(?:calculated|confirmed)|нет\b|без\b|отсутств\w*|"
            r"недоступ\w*|не\s+(?:подтвержд\w*|рассчит\w*|извест\w*|доступ\w*))",
            after,
        )
    )


def _text_has_nm_id(value: Any) -> bool:
    return bool(CONTEXTUAL_NM_ID_PATTERN.search(str(value or "")))


def _is_user_identifier_column(column: str) -> bool:
    normalized = _normalized_header(column)
    return (
        "sku" in normalized
        or "артикул" in normalized
        or "nm_id" in normalized
        or "nm id" in normalized
    )


def _sales_user_facing_contract_violations(result: SalesDiagnosticResult) -> list[str]:
    violations: list[str] = []
    user_text_parts: list[str] = []
    for table in [
        result.sales_report_snapshot,
        result.sales_overview,
        result.sales_dynamics,
        result.sales_decomposition,
    ]:
        if table:
            for index, column in enumerate(table.columns):
                if _is_user_identifier_column(column):
                    user_text_parts.extend(
                        str(row[index])
                        for row in table.rows
                        if index < len(row)
                    )
    for finding in result.findings:
        user_text_parts.extend(
            [
                finding.factor,
                finding.observation,
                finding.impact,
                finding.cause,
                finding.action,
            ]
        )
        if MECHANICAL_CAUSE_PATTERN.search(finding.cause):
            violations.append(f"{result.agent_name} finding cause treats mechanics as cause.")
    user_text_parts.extend(result.recommendations)
    user_text = " ".join(user_text_parts).lower()
    if _text_has_nm_id(user_text):
        violations.append(f"{result.agent_name} exposes nm_id in user-facing fields.")
    for domain_name in ("price", "marketing", "supply", "finance"):
        if re.search(rf"\b{domain_name}\b", user_text, re.IGNORECASE):
            violations.append(
                f"{result.agent_name} exposes internal domain name `{domain_name}` in user-facing fields."
            )
    return violations


def _is_sales_report_request(user_query: str | None) -> bool:
    text = (user_query or "").lower().replace("ё", "е")
    if not text:
        return False
    has_report_word = bool(re.search(r"\bотчет\w*\b|\bсводк\w*\b|\breport\b|\bsnapshot\b", text))
    has_sales_word = bool(re.search(r"\bпродаж\w*\b|\bзаказ\w*\b|\bвыручк\w*\b|\bsales\b", text))
    has_sales_comparison = bool(
        re.search(
            r"вчера\s+к\s+позавчера|недел\w*\s+к\s+недел\w*|месяц\w*\s+к\s+месяц\w*|"
            r"\bweek\s+over\s+week\b|\bmonth\s+over\s+month\b|\bwow\b|\bmom\b",
            text,
        )
    )
    return has_report_word and (has_sales_word or has_sales_comparison)


def _sales_report_snapshot_violations(table: ReportTable | None) -> list[str]:
    if table is None:
        return ["Sales report request is missing `sales_report_snapshot`."]

    violations = _table_violations(table, "Sales Agent sales_report_snapshot")
    table_text = " ".join(
        [
            table.title,
            " ".join(table.columns),
            " ".join(str(cell) for row in table.rows for cell in row),
        ]
    ).lower()
    required_period_labels = (
        ("вчера к позавчера", "последний день в данных"),
        ("неделя к неделе", "последние 7 дней"),
        ("месяц к месяцу", "последние 30 дней"),
    )
    for labels in required_period_labels:
        if not any(label in table_text for label in labels):
            violations.append(
                f"Sales Agent sales_report_snapshot is missing `{' / '.join(labels)}`."
            )
    if "продаж" not in table_text:
        violations.append("Sales Agent sales_report_snapshot is missing sales units.")
    if "выруч" not in table_text:
        violations.append("Sales Agent sales_report_snapshot is missing revenue.")
    return violations


def validate_sales_diagnostic_result(
    result: SalesDiagnosticResult,
    user_query: str | None = None,
) -> list[str]:
    violations: list[str] = []
    if result.sales_report_snapshot:
        if _is_sales_report_request(user_query):
            violations.extend(_sales_report_snapshot_violations(result.sales_report_snapshot))
        else:
            violations.extend(
                _table_violations(
                    result.sales_report_snapshot,
                    "Sales Agent sales_report_snapshot",
                )
            )
    elif _is_sales_report_request(user_query):
        violations.extend(_sales_report_snapshot_violations(None))
    tables = {
        "sales_overview": result.sales_overview,
        "sales_dynamics": result.sales_dynamics,
        "sales_decomposition": result.sales_decomposition,
    }
    for label, table in tables.items():
        violations.extend(_table_violations(table, f"{result.agent_name} {label}"))
    violations.extend(
        _sales_decomposition_violations(
            result.sales_decomposition,
            f"{result.agent_name} sales_decomposition",
        )
    )
    violations.extend(_finding_violations(result.findings, result.agent_name))
    violations.extend(_sales_user_facing_contract_violations(result))
    return violations
