"""Price-only approved export shapes and supplier/link projection.

Never applied to authoritative native data or prompts sent to the provider.
This is not semantic name detection of arbitrary prose/new source schemas.
"""
from datetime import date
import json
import re

OMITTED = "[omitted-private]"
OPERATIONS = frozenset({"context", "decision", "competitor_overview", "competitor_details", "daily"})
GROUPS = frozenset({"layers", "quality", "scenarios", "raw_market", "daily"})
_LINK = re.compile(r"(?:[a-z][a-z0-9+.-]{1,16}://|//(?=[a-z0-9])|www\.)(?:\\\||[^\s<>\[\]\"|])+", re.IGNORECASE)
_PIPE = re.compile(r"(?<!\\)\|")
_SEPARATOR = re.compile(r"\s*:?-+:?\s*")
_SUPPLIER_LABEL = re.compile(r"\b(?:competitor[_ -]*)?(?:supplier|seller)[_ -]*(?:name|id|account[_ -]*id)\s*[\"']?\s*[:=]", re.IGNORECASE)


def private_price_key(key):
    compact = re.sub(r"[^\w]", "", key).replace("_", "").lower()
    return (compact.endswith(("suppliername", "supplierid", "supplieraccountid", "sellername", "sellerid", "selleraccountid", "url", "uri"))
            or compact in {"supplier", "seller", "поставщик", "продавец", "имяпоставщика", "имяпродавца", "фиопоставщика", "фиопродавца"})


def is_price_prepared(value):
    keys = {"export_contract", "request", "today", "warnings", "required_claim_indices", "analytical_groups", "claim_catalog", "eligible_check_indices"}
    if not isinstance(value, dict) or set(value) != keys or value["export_contract"] != "price_prepared_v1":
        return False
    request = value["request"]
    fields = {"scope", "targets", "days", "date_from", "date_to", "operation", "target_namespace"}
    if (not isinstance(request, dict) or not fields <= set(request) <= fields | {"user_query"}
        or "user_query" in request and not isinstance(request["user_query"], str)
        or not isinstance(request["scope"], str) or request["scope"] not in {"portfolio", "single_sku", "sku_list"}
        or not isinstance(request["targets"], list) or len(request["targets"]) > 20
        or not all(isinstance(target, str) and 1 <= len(target) <= 128 and target == target.strip() for target in request["targets"])
        or len(set(request["targets"])) != len(request["targets"])
        or request["scope"] == "single_sku" and len(request["targets"]) != 1
        or request["scope"] == "sku_list" and not request["targets"]
        or type(request["days"]) is not int or not 1 <= request["days"] <= 31
        or request["date_from"] is not None or request["date_to"] is not None
        or not isinstance(request["operation"], str) or request["operation"] not in {"report", "context", "competitors", "daily"}
        or not isinstance(request["target_namespace"], str) or request["target_namespace"] not in {"auto", "vendor_code", "nm_id"}):
        return False
    try:
        if not isinstance(value["today"], str) or date.fromisoformat(value["today"]).isoformat() != value["today"]: return False
    except ValueError: return False
    claims = value["claim_catalog"]
    if (not isinstance(claims, list) or not claims
        or not all(isinstance(claim, dict) and set(claim) == {"index", "operation", "text"}
                   and type(claim["index"]) is int and claim["index"] == index
                   and isinstance(claim["operation"], str) and claim["operation"] in OPERATIONS
                   and isinstance(claim["text"], str) for index, claim in enumerate(claims))
        or not isinstance(value["warnings"], list) or not all(isinstance(warning, str) for warning in value["warnings"])):
        return False
    def indices(items):
        return (isinstance(items, list) and all(type(index) is int and 0 <= index < len(claims) for index in items)
                and len(set(items)) == len(items))
    groups = value["analytical_groups"]
    return (indices(value["required_claim_indices"]) and indices(value["eligible_check_indices"])
            and isinstance(groups, dict) and set(groups) == GROUPS and all(indices(items) for items in groups.values()))


def redact_price_text(value, *, private_key, clean_cell):
    """Sanitize standalone native Markdown, not only its sibling typed table.

    Headers identify sensitive columns; escaped pipes and native blank lines
    remain valid. A malformed sensitive table fails closed. JSON cells use
    the same recursive projection; links are always private, even in prose.
    """
    if "|" in value:
        lines = value.split("\n")
        active, width = set(), 0
        def split(line):
            cells = _PIPE.split(line)
            if len(cells) < 2: return None
            outer_start, outer_end = not cells[0].strip(), not cells[-1].strip()
            return (cells[1 if outer_start else 0:len(cells) - 1 if outer_end else len(cells)], outer_start, outer_end)
        for index, line in enumerate(lines):
            if not line.strip(): continue
            parsed = split(line)
            if parsed is None:
                if active and not line.lstrip().startswith("#"):
                    return OMITTED  # Cannot silently reset on a malformed sensitive data row.
                active, width = set(), 0
                continue
            cells, outer_start, outer_end = parsed
            private = {i for i, cell in enumerate(cells) if private_key(cell.strip())}
            following = next((candidate for candidate in lines[index + 1:] if candidate.strip()), "")
            next_row = split(following)
            separator = next_row[0] if next_row is not None else []
            is_header = len(separator) == len(cells) and all(_SEPARATOR.fullmatch(cell) for cell in separator)
            if private:
                if not is_header:
                    return OMITTED
            if is_header:
                active, width = private, len(cells)
            elif active and not all(_SEPARATOR.fullmatch(cell) for cell in cells):
                if len(cells) != width: return OMITTED
                cells = [" " + OMITTED + " " if i in active else cell for i, cell in enumerate(cells)]
            # Nested serialized envelopes may be carried by a non-private cell.
            for i, cell in enumerate(cells):
                text = cell.strip()
                if text.startswith(("{", "[")) and text != OMITTED:
                    try:
                        decoded = json.loads(text.replace("\\|", "|"))
                    except ValueError: continue
                    if isinstance(decoded, (dict, list)):
                        cells[i] = " " + json.dumps(clean_cell(decoded), ensure_ascii=False).replace("|", "\\|") + " "
            lines[index] = ("|" if outer_start else "") + "|".join(cells) + ("|" if outer_end else "")
        value = "\n".join(lines)
    if _SUPPLIER_LABEL.search(value):
        return OMITTED  # Recognized private labels in unstructured prose are not a trusted report contract.
    return _LINK.sub(OMITTED, value)
