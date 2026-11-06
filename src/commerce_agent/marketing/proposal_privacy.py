"""Closed native proposal projection, not approval to enable scoped tracing.

Action-matrix query identities are free search text, even in ``entity_id``.
Preserve scalar business data and known service templates, never infer that
arbitrary prose is safe merely because a source/family field claims Marketing.
Export helpers do not themselves approve a new source or tracing capability.
Authoritative native records/checkpoints are never changed.
"""
from copy import deepcopy
import json
import math
import re

OMITTED = "[omitted-private]"
FAMILIES = {"campaign_actions": "кампания", "query_actions": "поисковый запрос", "sku_actions": "артикул"}
_NM_IDS = re.compile(r"[1-9][0-9]{0,18}(?:, [1-9][0-9]{0,18})*")
_NUMBER = r"(?:None|[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)"
_REASONS = {
    "campaign_actions": (
        rf"заказы={_NUMBER}, затраты={_NUMBER}, waste_share={_NUMBER}",
        rf"legacy_snapshot: заказы={_NUMBER}, затраты={_NUMBER}"),
    "query_actions": (
        rf"band=(?:waste|traffic_only|winner|), заказы={_NUMBER}, затраты={_NUMBER}, avg_pos={_NUMBER}",
        rf"legacy_snapshot: затраты={_NUMBER}, заказы={_NUMBER}"),
    "sku_actions": (rf"текущая средняя позиция={_NUMBER}, текущая видимость={_NUMBER}",),
}
_EVIDENCE = {
    "campaign_actions": {"ctr", "cpc", "cr"},
    "query_actions": {"views_7d", "clicks_7d", "effective_cpc_7d", "cpc", "avg_pos"},
    "sku_actions": {"текущие заказы"},
}
_ACTIONS = {
    "campaign_actions": {
        "Оставить под контролем.", "Сократить или остановить кампанию по waste-сигналу.",
        "Не масштабировать автоматически; сначала ручная проверка.",
        "Разрешен только тестовый шаг без auto-scale.",
        "Можно проверять осторожное расширение после demand-подтверждения.",
        "Scale-сигнал найден, но demand marts не готовы.",
        "Сократить или остановить кампанию по свежему campaign snapshot.",
        "Оставить под контролем по свежему campaign snapshot."},
    "query_actions": {
        "Оставить запрос под контролем.", "Снизить ставку, минусовать или отключить waste-запрос.",
        "Проверить посадку и конверсию, а не масштабировать.",
        "Проверить расширение winner-запроса и новые ключи.",
        "Winner-запрос найден, но demand forecast не готов.", "Legacy fallback из query snapshot."},
    "sku_actions": {"Проверить видимость, ставки и качество карточки вместе."},
}
_ENUMS = {
    "status": {"оставить", "сократить", "проверить", "тест", "проверить масштаб"},
    "action_mode": {"hold", "reduce", "needs_review", "test", "scale_review"},
    "marketing_source_status": {"ready", "partial", "invalid_freshness", "legacy_fallback", "unpopulated", "missing", "stale"},
    "demand_source_status": {"ready", "partial", "invalid_freshness", "legacy_fallback", "unpopulated", "missing", "stale"},
    "decision_hint": {"hold", "hold_or_test", "scale", "reduce"},
    "confidence": {"low", "medium", "high"},
    "query_band": {"waste", "traffic_only", "winner", ""},
}
_NUMBERS = {"safe_growth_pct", "marginal_quality_coef", "winner_query_share", "waste_query_share"}
_BOOLS = {"is_cut_candidate", "is_scale_candidate"}
_IDS = {"advert_id", "nm_id"}
_BLOCKERS = {"marketing_low_confidence", "marketing_test_only", "marketing_demand_mart_unpopulated"}
_BASE = {"entity_type", "entity_id", "entity_name", "status", "action_mode", "reason", "evidence",
         "suggested_action", "marketing_source_status", "demand_source_status", "blockers"}
_OPTIONAL = set(_ENUMS) | _NUMBERS | _BOOLS | _IDS | {"norm_query", "vendor_code"}
_PROPOSAL_SUFFIX = ". Окно/row provenance неизвестны; не численный факт или approval/исполнение."
_PROPOSAL_TEXT = re.compile(
    r"Read-only portfolio proposal (campaign_actions|query_actions|sku_actions), (.*): "
    r"mode=(hold|reduce|needs_review|test|scale_review); blockers=(\[[^\r\n]*\])"
    + re.escape(_PROPOSAL_SUFFIX))


def project_native_nm_ids(value):
    """Technical-ID syntax, not arbitrary text in a field named like an ID."""
    if value is None or value == OMITTED:
        return value
    if isinstance(value, str):
        return value if len(value) <= 4096 and _NM_IDS.fullmatch(value) else OMITTED
    if isinstance(value, (list, tuple)):
        return [v if type(v) is int and 0 < v < 10**19 or isinstance(v, str) and re.fullmatch(r"[1-9][0-9]{0,18}", v) else OMITTED for v in value]
    return OMITTED


def _text(value, allowed):
    return value if value is None or value == OMITTED or isinstance(value, str) and value in allowed else OMITTED


def _identity(value, *, sku=False):
    if value is None or value == OMITTED:
        return value
    if type(value) is int and value > 0:
        return value
    pattern = r"[1-9][0-9]{4}" if sku else r"[1-9][0-9]{0,9}"
    return value if isinstance(value, str) and re.fullmatch(pattern, value) else OMITTED


def project_native_proposal(row, *, family):
    """Copy a single known-family row; fail closed for malformed/new schemas.

    Valid numbers/nulls/booleans and sparse-field presence are unchanged.
    Free labels/prose use an explicit marker, not a made-up business value.
    Numeric literals in service templates are retained verbatim, not parsed
    into facts or recalculated. Subsequent generic secret filtering is still
    mandatory before SDK wire export.
    """
    if (family not in FAMILIES or not isinstance(row, dict) or not _BASE <= set(row) <= _BASE | _OPTIONAL
        or row["entity_type"] != FAMILIES[family] or not isinstance(row["reason"], str)
        or not isinstance(row["evidence"], list) or not all(isinstance(v, str) for v in row["evidence"])
        or not isinstance(row["blockers"], list) or not all(isinstance(v, str) for v in row["blockers"])):
        raise ValueError("unapproved Marketing proposal schema")
    result = deepcopy(row)
    for key, value in row.items():
        if key in {"entity_name", "norm_query"}:
            result[key] = None if value is None else OMITTED
        elif key == "entity_id":
            result[key] = (None if value is None else OMITTED) if family == "query_actions" else _identity(value, sku=family == "sku_actions")
        elif key == "vendor_code":
            result[key] = _identity(value, sku=True)
        elif key in _IDS:
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError("unapproved Marketing native numeric identity")
        elif key in _NUMBERS:
            if value is not None and (type(value) not in {int, float} or not math.isfinite(value)):
                raise ValueError("unapproved Marketing native scalar")
        elif key in _BOOLS:
            if value is not None and type(value) is not bool:
                raise ValueError("unapproved Marketing native boolean")
        elif key in _ENUMS:
            result[key] = _text(value, _ENUMS[key])
        elif key == "reason":
            result[key] = value if value == OMITTED or any(re.fullmatch(pattern, value) for pattern in _REASONS[family]) else OMITTED
        elif key == "suggested_action":
            result[key] = _text(value, _ACTIONS[family])
        elif key == "evidence":
            result[key] = [v if v == OMITTED or any(re.fullmatch(re.escape(metric) + "=" + _NUMBER, v) for metric in _EVIDENCE[family]) else OMITTED for v in value]
        elif key == "blockers":
            result[key] = [_text(v, _BLOCKERS) for v in value]
    return result


def project_native_proposal_table(table, *, family):
    """Copy the exact published proposal table, preserving every numeric cell.

    Published evidence/blocker lists and booleans are serialized JSON cells.
    Decode only those typed columns, then preserve the original cell byte text
    when unchanged. Unknown columns/duplicate columns/ragged rows fail closed;
    no raw Markdown/prose mirror is admitted by this table contract.
    """
    if (family not in FAMILIES or not isinstance(table, dict) or set(table) != {"title", "columns", "rows"}
        or not isinstance(table["title"], str) or not isinstance(table["columns"], list)
        or not all(isinstance(k, str) for k in table["columns"])
        or len(table["columns"]) != len(set(table["columns"]))
        or not _BASE <= set(table["columns"]) <= _BASE | _OPTIONAL
        or not isinstance(table["rows"], list) or len(table["rows"]) > 50):
        raise ValueError("unapproved Marketing proposal table schema")
    result = deepcopy(table)
    if table["title"] not in {OMITTED, f"Read-only portfolio {family}; не approved actions"}:
        result["title"] = OMITTED
    rows = []
    for cells in table["rows"]:
        if not isinstance(cells, list) or len(cells) != len(table["columns"]):
            raise ValueError("unapproved Marketing proposal table row")
        row = dict(zip(table["columns"], cells, strict=True))
        for key in {"evidence", "blockers"} | _BOOLS:
            if key in row and row[key] is not None:
                if not isinstance(row[key], str):
                    raise ValueError("unapproved Marketing proposal serialized cell")
                try:
                    row[key] = json.loads(row[key])
                except (ValueError, TypeError):
                    raise ValueError("unapproved Marketing proposal serialized cell") from None
        projected = project_native_proposal(row, family=family)
        output = []
        for key, original in zip(table["columns"], cells, strict=True):
            clean = projected[key]
            if type(clean) is type(row[key]) and clean == row[key]:
                output.append(original)
            elif key in {"evidence", "blockers"}:
                output.append(json.dumps(clean, ensure_ascii=False, allow_nan=False))
            else:
                output.append(clean)
        rows.append(output)
    result["rows"] = rows
    return result


def project_native_proposal_claim_text(text):
    """Mask the identity channel duplicated by the service's claim renderer.

    Only that exact service template is eligible, not arbitrary model prose.
    Invalid/new templates are omitted; non-query numeric identities and
    known action modes/blocker literals are preserved without new assertions.
    This is a projection helper, never a source approval or instruction.
    """
    if not isinstance(text, str) or len(text) > 8192 or "\n" in text or "\r" in text:
        return OMITTED
    if text == OMITTED:
        return text
    match = _PROPOSAL_TEXT.fullmatch(text)
    if not match:
        return OMITTED
    family, identity, mode, raw_blockers = match.groups()
    try:
        blockers = json.loads(raw_blockers)
    except (ValueError, TypeError):
        return OMITTED
    if not isinstance(blockers, list) or not all(isinstance(v, str) and v in _BLOCKERS for v in blockers):
        return OMITTED
    clean_identity = OMITTED if family == "query_actions" else _identity(identity, sku=family == "sku_actions")
    return f"Read-only portfolio proposal {family}, {clean_identity}: mode={mode}; blockers={raw_blockers}" + _PROPOSAL_SUFFIX


def proposal_table_family(table):
    """Recognize a proposal table by its native dimensions, not source claims."""
    if not isinstance(table, dict):
        return None
    title = table.get("title")
    claimed_family = next((f for f in FAMILIES if isinstance(title, str) and title.startswith("Read-only portfolio " + f)), None)
    columns = table.get("columns", table.get("headers"))
    names = set(k for k in columns if isinstance(k, str)) if isinstance(columns, list) else set()
    signature = {"entity_id", "action_mode"} <= names
    if not claimed_family and not signature:
        return None
    if "headers" in table or not isinstance(columns, list) or not {"entity_type", "entity_id", "action_mode"} <= names:
        raise ValueError("unapproved Marketing proposal table signature")
    rows = table.get("rows")
    if not isinstance(rows, list):
        raise ValueError("unapproved Marketing proposal table rows")
    index = columns.index("entity_type")
    kinds = {row[index] for row in rows if isinstance(row, list) and len(row) == len(columns) and isinstance(row[index], str)}
    if rows and (len(kinds) != 1 or len([row for row in rows if isinstance(row, list) and len(row) == len(columns)]) != len(rows)):
        raise ValueError("unapproved Marketing proposal table family")
    family = next((f for f, kind in FAMILIES.items() if kind in kinds), None)
    if not rows:
        family = next((f for f in FAMILIES if table.get("title") == f"Read-only portfolio {f}; не approved actions"), None)
    if family is None:
        raise ValueError("unapproved Marketing proposal table family")
    return family


def _markdown_cells(line):
    # Native renderer escapes backslashes before pipes. Count the preceding
    # slash run, rather than treating every preceding slash as an escape.
    cells, start, slashes = [], 0, 0
    for index, char in enumerate(line):
        if char == "|" and slashes % 2 == 0:
            cells.append(line[start:index]); start = index + 1
        slashes = slashes + 1 if char == "\\" else 0
    cells.append(line[start:])
    if len(cells) < 3 or cells[0].strip() or cells[-1].strip():
        return None
    return cells[1:-1]


def _unescape_cell(text):
    # Reject unknown escape sequences: accepting malformed mirrors could leak
    # a private label through a channel that never existed in native output.
    output, index = [], 0
    while index < len(text):
        if text[index] == "\\":
            if index + 1 >= len(text) or text[index + 1] not in {"\\", "|"}:
                raise ValueError("unapproved Marketing Markdown escape")
            index += 1
        output.append(text[index]); index += 1
    return "".join(output)


def redact_native_proposal_text(text):
    """Project duplicated service claims and standalone native GFM tables.

    Keep untouched numeric cell spelling and the rest of the business report.
    Unknown/malformed proposal mirrors fail closed; this function does not
    authorize arbitrary free prose or a source that claims an approved name.
    """
    if not isinstance(text, str):
        return OMITTED
    lines = text.split("\n")
    columns, nm_indices, nm_width = None, set(), 0
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        cells = _markdown_cells(line)
        following = next((candidate for candidate in lines[index + 1:] if candidate.strip()), "")
        separator = _markdown_cells(following)
        is_header = (cells is not None and separator is not None and len(cells) == len(separator)
            and all(re.fullmatch(r"\s*:?-+:?\s*", cell) for cell in separator))
        # Unknown/partial/no-outer-pipe mirrors are not the approved native
        # renderer. Never silently treat their private identities as prose.
        if "entity_id" in line and "action_mode" in line and not is_header:
            return OMITTED
        if "nm_ids" in line and "|" in line and not is_header:
            return OMITTED
        if line.lstrip().startswith("### Read-only portfolio "):
            expected = next((f for f in FAMILIES if line.strip().startswith("### Read-only portfolio " + f)), None)
            if expected and not (following.startswith("| entity_type |") and _markdown_cells(following)):
                return OMITTED
        if is_header:
            names = [cell.strip() for cell in cells]
            proposal = {"entity_type", "entity_id", "action_mode"} <= set(names)
            if proposal and (len(set(names)) != len(names) or not _BASE <= set(names) <= _BASE | _OPTIONAL):
                return OMITTED
            columns = names if proposal else None
            nm_indices, nm_width = {i for i, key in enumerate(names) if key == "nm_ids"}, len(names)
            continue
        if nm_indices:
            if cells is None:
                if not line.lstrip().startswith("#"):
                    return OMITTED
                nm_indices, nm_width = set(), 0
            elif not all(re.fullmatch(r"\s*:?-+:?\s*", cell) for cell in cells):
                if len(cells) != nm_width:
                    return OMITTED
                try:
                    for i in nm_indices:
                        raw_id = _unescape_cell(cells[i].strip())
                        if raw_id != "нет данных" and project_native_nm_ids(raw_id) == OMITTED:
                            cells[i] = " " + OMITTED + " "
                    lines[index] = "|" + "|".join(cells) + "|"
                except ValueError:
                    return OMITTED
        if columns is not None:
            if cells is not None and all(re.fullmatch(r"\s*:?-+:?\s*", cell) for cell in cells):
                continue
            if cells is None:
                if not line.lstrip().startswith("#"):
                    return OMITTED
                columns = None
            else:
                if len(cells) != len(columns):
                    return OMITTED
                try:
                    decoded = [_unescape_cell(cell.strip()) for cell in cells]
                    row = {}
                    for key, value in zip(columns, decoded, strict=True):
                        if value == "нет данных":
                            row[key] = None
                        elif key in _NUMBERS | _IDS:
                            row[key] = json.loads(value)
                        else:
                            row[key] = value
                    family = next((f for f, kind in FAMILIES.items() if row.get("entity_type") == kind), None)
                    if family is None:
                        return OMITTED
                    table = {"title":f"Read-only portfolio {family}; не approved actions", "columns":columns,
                        "rows":[[row[k] for k in columns]]}
                    projected = project_native_proposal_table(table, family=family)["rows"][0]
                    for i, (original, clean) in enumerate(zip([row[k] for k in columns], projected, strict=True)):
                        if type(clean) is not type(original) or clean != original:
                            cells[i] = " " + ("нет данных" if clean is None else str(clean)).replace("\\", "\\\\").replace("|", "\\|") + " "
                    lines[index] = "|" + "|".join(cells) + "|"
                except (ValueError, TypeError, OverflowError):
                    return OMITTED
                continue
        if "Read-only portfolio proposal " in line:
            # Existing summary/observed_fact contains one exact claim per line.
            # A free prefix/suffix is not a second approved prose channel.
            lines[index] = project_native_proposal_claim_text(line.strip())
            if lines[index] == OMITTED:
                return OMITTED  # Multiline/private suffix must not survive on the next line.
        elif "Current nm links=" in line:
            match = re.fullmatch(r"(.*Current nm links=)(.*)(; не historical SKU attribution, причина/прибыль не установлены\.)", line)
            if match is None:
                return OMITTED
            else:
                original = match[2]
                projected = project_native_nm_ids(None if original == "None" else original)
                lines[index] = match[1] + ("None" if projected is None else str(projected)) + match[3]
    return "\n".join(lines)
