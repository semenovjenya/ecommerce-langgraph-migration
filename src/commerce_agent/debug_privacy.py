"""Development trace projection, never a mutation of authoritative business data.

Aggregate M2 schemas exclude personal records. This additional boundary removes
direct identifiers and secret/context fields, not arbitrary semantic PII. Raw
user prose and opaque objects are deliberately not eligible for export; a
closed report-question grammar permits the exact safe original question.
"""
from collections.abc import Mapping
from datetime import date, datetime
from hashlib import sha256
import json
import math
import os
import re

from pydantic import BaseModel

from commerce_agent.adapters.privacy import redact_direct_identifiers
from commerce_agent.safety import SensitiveStateError, assert_secret_free
from commerce_agent.models import SchemaVersion
from commerce_agent.price.debug_export import is_price_prepared, private_price_key, redact_price_text
from commerce_agent.marketing.debug_export import (is_diagnostic_prepared,
    private_marketing_label, private_marketing_prose, safe_diagnostic_question)
from commerce_agent.marketing.proposal_privacy import (FAMILIES, project_native_proposal,
    project_native_proposal_table, proposal_table_family, redact_native_proposal_text, project_native_nm_ids)
from commerce_agent.marketing.scoped_debug_export import is_scoped_model_input, safe_scoped_question

OMITTED = "[omitted-private]"
MAX_EXPORT_BYTES = 8 * 1024 * 1024
_BLOCKED = frozenset({
    "tenantid", "userid", "requestid", "actorid", "username", "name",
    "firstname", "lastname", "fullname", "customer", "customers", "customerid",
    "customername", "buyer", "buyers", "recipient", "recipientname", "contact",
    "contacts", "email", "phone", "telephone", "address", "deliveryaddress",
    "birthdate", "dateofbirth", "passport", "ssn", "ip", "ipaddress", "cookies",
    "cookie", "authorization", "credentials", "password", "secret", "apikey",
    "token", "accesstoken", "refreshtoken", "servicetoken", "dsn", "headers",
    "context", "runtime", "runtimecontext", "configurable", "config", "client",
    "settings", "environment", "metadata", "responsemetadata", "additionalkwargs",
    "reasoning", "reasoningsummary", "encryptedcontent", "attachments", "stacktrace",
    "traceback", "exception", "parsingerror", "rawquery", "query", "userquery", "memory", "document",
    "documents", "customerrecords", "contenturl", "url", "uri",
    "фамилия", "имя", "отчество", "фио", "адрес", "телефон", "паспорт",
})
_OPAQUE_CREDENTIAL = re.compile(r"(?<![\w])[a-fA-F0-9]{64}(?![\w])")
_UUID_REFERENCE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
_PERSONAL_CONTEXT = re.compile(
    r"\b(?:customer\s+name|buyer\s+name|full\s+name|passport|ssn|"
    r"ФИО|паспорт|клиент|покупатель|получатель|адрес\s+доставки)\b", re.I)
_MESSAGE_ROLES = {"human": "human", "HumanMessage": "human", "user": "human",
                  "system": "system", "SystemMessage": "system", "ai": "ai", "AIMessage": "ai", "assistant": "ai",
                  "tool": "tool", "ToolMessage": "tool", "function": "function", "FunctionMessage": "function",
                  "chat": "chat", "ChatMessage": "chat", "AIMessageChunk": "ai", "HumanMessageChunk": "human"}

# Owner-approved question visibility is deliberately a closed diagnostic
# grammar, not regex-only PII detection of arbitrary names/free-form prose.
_REPORT_QUESTION = re.compile(
    r"(?:отч[её]т|сводка|анализ)\s+(?:рекламы|продаж)"
    r"(?:\.?\s+период:\s*\d{4}-\d{2}-\d{2}\s*[—–-]\s*\d{4}-\d{2}-\d{2})?\.?",
    re.IGNORECASE)
_SUPPLY_QUESTION = re.compile(
    r"оборачиваемость запасов(?: по портфелю)?"
    r"(?: за \d{4}-\d{2}-\d{2}\s*[—–-]\s*\d{4}-\d{2}-\d{2})?"
    r"(?:\. Покажи матрицу товаров и проверки смежных причин)?\.?",
    re.IGNORECASE)
_SUPPLY_OPERATIONS = frozenset({"overview", "turnover", "value"})
_FINANCE_QUESTION = re.compile(
    r"финансовый отч[её]т(?: по всему портфелю за \d{4}-\d{2}-\d{2} — \d{4}-\d{2}-\d{2}\. "
    r"Покажи выручку после возвратов, расходы маркетплейса, маржу в рублях и процентах, "
    r"матрицу SKU и изменение маржи относительно \d{4}-\d{2}-\d{2} — \d{4}-\d{2}-\d{2}\. "
    r"Отдельно укажи отсутствующую себестоимость и проверки смежных причин\.)?", re.IGNORECASE)
_FINANCE_OPERATIONS = frozenset({"overview", "marketplace", "sku", "margin_change"})
_FINANCE_CLAIM_OPERATIONS = _FINANCE_OPERATIONS | {"current_cogs", "previous_cogs"}
_PRICE_QUESTION = re.compile(
    r"отч[её]т по ценам(?: всего портфеля\. Покажи цену продавца, цену сайта, "
    r"цену кошелька и эффективную цену покупателя, даты каждого слоя, "
    r"качество строгих аналогов и широкий рынок\. Покажи готовые сценарии "
    r"переоценки отдельно от текущих фактов; укажи ограничения и нужные "
    r"проверки соседних агентов, без изменения цен\.)?", re.IGNORECASE)
_ALTERNATIVE_METHOD_NAMES = frozenset({
    "boundary_snapshot_average", "movement_reconstruction", "latest_stock_velocity_proxy"})


def _safe_original_question(value):
    if not isinstance(value, str) or len(value) > 512 or "\n" in value or "\r" in value:
        return None
    if safe_diagnostic_question(value):
        return value
    if safe_scoped_question(value):
        return value
    if _FINANCE_QUESTION.fullmatch(value.strip()) or _PRICE_QUESTION.fullmatch(value.strip()):
        return value
    return value if (len(value) <= 160 and (_REPORT_QUESTION.fullmatch(value.strip())
                     or _SUPPLY_QUESTION.fullmatch(value.strip()))) else None


def development_debug_enabled():
    return (os.environ.get("COMMERCE_AGENT_ENVIRONMENT") == "development"
            and os.environ.get("COMMERCE_AGENT_LANGSMITH_DEBUG", "").lower() in {"true", "1", "yes", "on"})


def _private_key(key):
    compact = re.sub(r"[^\w]", "", key).replace("_", "").lower()
    return (compact in _BLOCKED or private_price_key(key) or private_marketing_label(key,include_descriptors=False) or any(word in compact for word in
            ("password", "secret", "credential", "apikey", "accesstoken", "servicetoken"))
            or any(compact.endswith(suffix) for suffix in
                   ("email", "phone", "address", "firstname", "lastname", "fullname", "customerid", "userid", "tenantid",
                    "token", "cookie", "cookies", "dsn", "authorization", "authentication", "displayname", "accountname")))


def _prepared_json(value):
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError):
        return None
    # Only approved domain-model prepared evidence contracts are eligible.
    required = {"request", "today", "tables", "facts", "evidence", "warnings"}
    allowed = required | {"periods", "deep_dives", "overview_signals", "claim_catalog", "signal_catalog"}
    if isinstance(parsed, dict) and required <= set(parsed) <= allowed:
        return parsed
    marketing_keys = {"export_contract", "request", "today", "report_date", "period", "previous_period",
                      "warnings", "claim_catalog", "required_portfolio_groups", "sku_priorities"}
    if (isinstance(parsed, dict) and set(parsed) == marketing_keys
        and parsed["export_contract"] == "marketing_prepared_v2"
        and isinstance(parsed["request"], dict) and isinstance(parsed["claim_catalog"], list)
        and isinstance(parsed["sku_priorities"], list)
        and all(isinstance(claim, dict) and set(claim) == {"index", "kind", "text"}
                and type(claim["index"]) is int and claim["kind"] in {"portfolio", "sku", "campaign"}
                and isinstance(claim["text"], str) for claim in parsed["claim_catalog"])):
        return parsed
    if _is_supply_prepared(parsed):
        return parsed
    if _is_finance_prepared(parsed):
        return parsed
    if is_price_prepared(parsed):
        return parsed
    if is_diagnostic_prepared(parsed):
        return parsed
    if is_scoped_model_input(parsed):
        return parsed
    return None


def _is_supply_prepared(value):
    """Closed compact model contract; no arbitrary JSON or other domain input.

    Privacy projection still runs afterwards. Missing user_query is accepted
    only so that already-projected prompts remain idempotent.
    """
    keys = {"export_contract", "request", "today", "report_date", "period", "methods", "warnings",
            "claim_catalog", "required_claim_indices", "required_context_claim_indices", "required_operations"}
    if not isinstance(value, dict) or set(value) != keys or value["export_contract"] != "supply_prepared_v1":
        return False
    request = value["request"]
    request_keys = {"scope", "targets", "days", "date_from", "date_to", "operation"}
    if (not isinstance(request, dict) or not request_keys <= set(request) <= request_keys | {"user_query"}
        or ("user_query" in request and not isinstance(request["user_query"], str))
        or request["scope"] not in {"portfolio", "single_sku", "sku_list"}
        or not isinstance(request["targets"], list) or len(request["targets"]) > 20
        or not all(isinstance(target, str) for target in request["targets"])
        or type(request["days"]) is not int or not 1 <= request["days"] <= 366
        or request["operation"] not in _SUPPLY_OPERATIONS | {"matrix"}):
        return False
    for candidate, nullable in ((value["today"], False), (value["report_date"], True),
                                (request["date_from"], True), (request["date_to"], True)):
        if nullable and candidate is None:
            continue
        try:
            if not isinstance(candidate, str) or date.fromisoformat(candidate).isoformat() != candidate:
                return False
        except ValueError:
            return False
    claims = value["claim_catalog"]
    if (not isinstance(claims, list) or not claims
        or not all(isinstance(claim, dict) and set(claim) == {"index", "operation", "text"}
                   and type(claim["index"]) is int and claim["index"] == index
                   and claim["operation"] in _SUPPLY_OPERATIONS and isinstance(claim["text"], str)
                   for index, claim in enumerate(claims))
        or not isinstance(value["period"], str) or not isinstance(value["methods"], dict)
        or not isinstance(value["warnings"], list) or not all(isinstance(warning, str) for warning in value["warnings"])):
        return False
    for field in ("required_claim_indices", "required_context_claim_indices"):
        indices = value[field]
        if (not isinstance(indices, list) or not all(type(index) is int and 0 <= index < len(claims) for index in indices)
            or len(set(indices)) != len(indices)):
            return False
    operations = value["required_operations"]
    return (isinstance(operations, list) and bool(operations)
            and all(isinstance(operation, str) and operation in _SUPPLY_OPERATIONS for operation in operations)
            and len(set(operations)) == len(operations))


def _is_finance_prepared(value):
    """Exact compact v1/v2 contracts, followed by recursive privacy projection.

    Already-projected requests may lack user_query for idempotence. Unknown
    keys/versions, arbitrary JSON and cross-domain operations stay hidden.
    """
    base = {"export_contract", "request", "today", "period", "previous_period", "warnings",
            "claim_catalog", "required_claim_indices", "required_operations"}
    extra = {"analysis_policy_version", "analytical_groups", "summary_claim_indices", "eligible_check_indices"}
    if not isinstance(value, dict):
        return False
    version = 1 if value.get("export_contract") == "finance_prepared_v1" else 2 if value.get("export_contract") == "finance_prepared_v2" else None
    if version is None or set(value) != (base if version == 1 else base | extra):
        return False
    request = value["request"]
    keys = {"scope", "targets", "days", "date_from", "date_to", "previous_date_from", "previous_date_to", "operation"}
    if (not isinstance(request, dict) or not keys <= set(request) <= keys | {"user_query"}
        or ("user_query" in request and not isinstance(request["user_query"], str))
        or not isinstance(request["scope"], str) or request["scope"] not in {"portfolio", "single_sku", "sku_list"}
        or not isinstance(request["targets"], list) or len(request["targets"]) > 20
        or not all(isinstance(target, str) and 1 <= len(target) <= 128 for target in request["targets"])
        or len(set(request["targets"])) != len(request["targets"])
        or request["scope"] == "single_sku" and len(request["targets"]) != 1
        or request["scope"] == "sku_list" and not request["targets"]
        or type(request["days"]) is not int or not 1 <= request["days"] <= 366
        or not isinstance(request["operation"], str) or request["operation"] not in _FINANCE_OPERATIONS | {"report"}):
        return False
    def iso(candidate, nullable=False):
        if candidate is None and nullable:
            return True
        try:
            return isinstance(candidate, str) and date.fromisoformat(candidate).isoformat() == candidate
        except ValueError:
            return False
    if not iso(value["today"]) or any(not iso(request[key], True) for key in ("date_from", "date_to", "previous_date_from", "previous_date_to")):
        return False
    for start, end in (("date_from", "date_to"), ("previous_date_from", "previous_date_to")):
        if (request[start] is None) != (request[end] is None) or request[start] and request[start] > request[end]:
            return False
    for key in ("period", "previous_period"):
        period = value[key]
        if (not isinstance(period, str) or len(period.split("/")) != 2
            or not all(iso(boundary) for boundary in period.split("/"))
            or period.split("/")[0] > period.split("/")[1]):
            return False
    if value["previous_period"].split("/")[1] >= value["period"].split("/")[0]:
        return False
    claims = value["claim_catalog"]
    if (not isinstance(claims, list) or not claims
        or not all(isinstance(claim, dict) and set(claim) == {"index", "operation", "text"}
                   and type(claim["index"]) is int and claim["index"] == index
                   and isinstance(claim["operation"], str) and claim["operation"] in _FINANCE_CLAIM_OPERATIONS and isinstance(claim["text"], str)
                   for index, claim in enumerate(claims))
        or not isinstance(value["warnings"], list) or not all(isinstance(warning, str) for warning in value["warnings"])):
        return False
    def indices(items):
        return (isinstance(items, list) and all(type(index) is int and 0 <= index < len(claims) for index in items)
                and len(set(items)) == len(items))
    ops = value["required_operations"]
    if (not indices(value["required_claim_indices"]) or not isinstance(ops, list) or not ops
        or not all(isinstance(op, str) and op in _FINANCE_OPERATIONS for op in ops)
        or len(set(ops)) != len(ops)):
        return False
    if version == 2:
        groups = value["analytical_groups"]
        if (type(value["analysis_policy_version"]) is not int or value["analysis_policy_version"] != 2
            or not isinstance(groups, dict) or set(groups) != {"expenses", "declines", "growth"}
            or not all(indices(items) for items in groups.values())
            or not indices(value["summary_claim_indices"]) or not indices(value["eligible_check_indices"])):
            return False
    return True


def sanitize_debug_inputs(payload):
    # Chat tracer serialization has a different shape from graph/user inputs.
    # Only that shape can carry the already-prepared aggregate evidence message.
    chat = (isinstance(payload, Mapping) and set(payload) == {"messages"}
            and isinstance(payload["messages"], list) and bool(payload["messages"])
            and all(isinstance(batch, list) and all(isinstance(msg, Mapping)
                    and msg.get("type") == "constructor" for msg in batch) for batch in payload["messages"]))
    return sanitize_debug_payload(payload, allow_prepared_messages=chat)


def sanitize_debug_payload(payload, *, allow_prepared_messages=False):
    """Idempotent JSON-safe projection with depth/size guards; no repr fallback."""
    count = 0

    def clean(value, depth=0, provider_generation=False, marketing=False):
        nonlocal count
        count += 1
        if depth > 32 or count > 100_000:
            return "[omitted-export-budget]"
        if isinstance(value, BaseModel):
            return clean(value.model_dump(mode="python"), depth + 1, provider_generation, marketing)
        if isinstance(value, Mapping):
            # Tool outputs are traced before the collector can reject them.
            # Recognize proposal shapes independently of spoofable source/
            # domain labels; a malformed mirror must not bypass projection.
            if ("entity_type" in value and ("entity_id" in value or "action_mode" in value)
                or "entity_id" in value and "action_mode" in value):
                family = next((f for f, kind in FAMILIES.items() if value.get("entity_type") == kind), None)
                try:
                    value = project_native_proposal(dict(value), family=family)
                except (ValueError, TypeError, OverflowError):
                    return OMITTED
            try:
                family = proposal_table_family(value)
                if family:
                    value = project_native_proposal_table(dict(value), family=family)
            except (ValueError, TypeError, OverflowError):
                return OMITTED
            source = value.get("source")
            native_marketing_shape = ("promoted_sku_rows" in value
                or {"focus_sku_list", "focus_keyword", "period_days", "row_count", "metric_grain", "rows"} <= set(value)
                or {"campaign_actions", "query_actions", "sku_actions", "source_freshness"} <= set(value))
            marketing = (marketing or value.get("domain")=="marketing" or value.get("agent_name")=="Marketing Agent" or value.get("capability") in {"sku_diagnostic", "scoped_report"}
                or native_marketing_shape
                or isinstance(source,str) and source.startswith("marketing.")
                or value.get("export_contract") in {"marketing_diagnostic_prepared_v1", "marketing_scoped_prepared_v1"})
            # Fresh tool output is traced before collector policy can reject
            # it. Unknown populated frequency-query sources therefore fail
            # closed here too, not just at resume/model/publication gates.
            if any(value.get(key) for key in
                ("top_query_rows","query_waste_rows","query_scale_rows","query_position_risk_rows")):
                return {"omitted":"unapproved_marketing_query_source"}
            def private_key(key):
                return _private_key(key) or marketing and private_marketing_label(key)
            # LangChain secret placeholders and private provider reasoning.
            if value.get("type") in {"secret", "reasoning", "reasoning_summary"}:
                return OMITTED
            result = {}
            message_types = _MESSAGE_ROLES
            message_type = value.get("type", value.get("role"))
            serialized_type = value.get("id", [])
            if value.get("type") == "constructor" and isinstance(serialized_type, (list, tuple)):
                serialized_type = serialized_type[-1] if serialized_type else None
            else:
                serialized_type = None
            if not allow_prepared_messages and (message_type in message_types or serialized_type in message_types):
                role = _MESSAGE_ROLES.get(message_type, _MESSAGE_ROLES.get(serialized_type))
                nested = value.get("kwargs") or value.get("data")
                content = value.get("content")
                if content is None and isinstance(nested, Mapping):
                    content = nested.get("content")
                return {"type": role, "content": (_safe_original_question(content) or OMITTED) if role == "human" else OMITTED}
            if message_type in message_types and isinstance(value.get("data"), Mapping):
                value = {**value, "data": {**value["data"], "type": message_type}}
            columns = value.get("columns", value.get("headers"))
            nm_columns = {i for i, col in enumerate(columns) if col == "nm_ids"} if isinstance(columns, (list, tuple)) else set()
            if nm_columns and (not isinstance(value.get("rows"), (list, tuple))
                or not all(isinstance(col, str) for col in columns) or len(columns) != len(set(columns))
                or any(not isinstance(row, (list, tuple)) or len(row) != len(columns) for row in value["rows"])):
                return OMITTED
            private_columns = {i for i, col in enumerate(columns) if isinstance(col, str) and private_key(col)} \
                if isinstance(columns, (list, tuple)) else set()
            for key, item in value.items():
                if provider_generation and key == "generation_info":
                    continue
                # Only the native closed AlternativeMethod enum is a business
                # name. Arbitrary names (including supplier/customer names)
                # remain private in every other shape.
                if (key == "name" and set(value) == {"name", "meaning", "when_to_use"}
                    and isinstance(item, str) and item in _ALTERNATIVE_METHOD_NAMES
                    and isinstance(value["meaning"], str) and isinstance(value["when_to_use"], str)):
                    result[key] = item
                    continue
                if key == "user_query" and _safe_original_question(item) is not None:
                    result[key] = item
                    continue
                if not isinstance(key, str) or private_key(key):
                    continue
                if key == "nm_ids":
                    item = project_native_nm_ids(item)
                if key in FAMILIES:
                    if not isinstance(item, (list, tuple)):
                        result[key] = OMITTED
                        continue
                    family_rows = []
                    for row in item:
                        try:
                            family_rows.append(project_native_proposal(dict(row), family=key) if isinstance(row, Mapping) else OMITTED)
                        except (ValueError, TypeError, OverflowError):
                            family_rows.append(OMITTED)
                    item = family_rows
                try:
                    assert_secret_free(key)
                    safe_key, changed = redact_direct_identifiers(key)
                    if changed:
                        continue
                except (SensitiveStateError, ValueError):
                    continue
                if private_columns and key in {"markdown_table", "markdown", "text", "report_text"}:
                    result[key] = OMITTED
                    continue
                if key == "rows" and private_columns and isinstance(item, (list, tuple)):
                    item = [[OMITTED if i in private_columns else cell for i, cell in enumerate(row)]
                            if isinstance(row, (list, tuple)) else row for row in item]
                if key == "rows" and nm_columns and isinstance(item, (list, tuple)):
                    item = [[project_native_nm_ids(cell) if i in nm_columns else cell for i, cell in enumerate(row)]
                            if isinstance(row, (list, tuple)) else OMITTED for row in item]
                # Human graph messages may contain any personal prose. The LLM
                # receives only prepared JSON business evidence, which is eligible.
                if key == "content" and message_type in message_types:
                    if not allow_prepared_messages:
                        result[key] = OMITTED
                        continue
                    if message_type in {"human", "HumanMessage"}:
                        decoded = _prepared_json(item) if isinstance(item, str) else None
                        result[key] = json.dumps(clean(decoded, depth + 1, marketing=marketing), ensure_ascii=False) if decoded is not None else OMITTED
                    else:
                        result[key] = clean(item, depth + 1, marketing=marketing)
                else:
                    result[safe_key] = clean(item, depth + 1, provider_generation or key == "generations",
                        marketing or key in {"prepared_marketing","marketing_candidate","marketing_analysis"})
            # Serialized messages wrap content in kwargs, type=constructor.
            if value.get("type") == "constructor" and isinstance(value.get("id"), (list, tuple)):
                message_type = value["id"][-1] if value["id"] else ""
                kwargs = value.get("kwargs")
                if message_type in message_types and isinstance(kwargs, Mapping):
                    result["kwargs"] = clean({**kwargs, "type": message_type}, depth + 1, marketing=marketing)
            return result
        if isinstance(value, (list, tuple)):
            return [clean(item, depth + 1, provider_generation, marketing) for item in value]
        if isinstance(value, str):
            # Provider generation prose has not passed domain publication.
            # Closed parsed selections remain visible on the parser/analyze
            # spans; a duplicated raw text/logprob channel must not bypass it.
            if provider_generation:
                return OMITTED
            if len(value.encode("utf-8")) > MAX_EXPORT_BYTES:
                return "[omitted-export-budget]"
            # Embedded JSON prompts need field filtering, not just regex masking.
            if value.lstrip().startswith(("{", "[")):
                try:
                    decoded = json.loads(value)
                    if isinstance(decoded, (dict, list)):
                        return json.dumps(clean(decoded, depth + 1, marketing=marketing), ensure_ascii=False)
                except (ValueError, TypeError):
                    pass
            if "\n- user_query:" in value:
                # Use the LAST full trusted override, not an attacker-spoofable
                # partial marker. If unavailable, omit the entire suffix.
                from commerce_agent.sales.prompts import MIGRATION_OVERRIDE
                from commerce_agent.marketing.prompts import MIGRATION_OVERRIDE as MARKETING_OVERRIDE
                from commerce_agent.supply.prompts import MIGRATION_OVERRIDE as SUPPLY_OVERRIDE
                from commerce_agent.finance.prompts import MIGRATION_OVERRIDE as FINANCE_OVERRIDE
                from commerce_agent.price.prompts import MIGRATION_OVERRIDE as PRICE_OVERRIDE
                start = value.index("\n- user_query:")
                end = max(value.rfind(MIGRATION_OVERRIDE), value.rfind(MARKETING_OVERRIDE), value.rfind(SUPPLY_OVERRIDE), value.rfind(FINANCE_OVERRIDE), value.rfind(PRICE_OVERRIDE))
                question = value[start + len("\n- user_query:"):end].strip() if end > start else ""
                value = value[:start] + "\n- user_query: " + (_safe_original_question(question) or OMITTED) + ("\n\n" + value[end:] if end > start else "")
            marketing = marketing or value.startswith("# Marketing — диагностика SKU ")
            value = redact_price_text(value,
                private_key=lambda key:_private_key(key) or marketing and private_marketing_label(key),
                clean_cell=lambda item: clean(item, depth + 1, marketing=marketing))
            value = redact_native_proposal_text(value)
            if _OPAQUE_CREDENTIAL.search(value) or _PERSONAL_CONTEXT.search(value) or private_marketing_prose(value,include_descriptors=marketing):
                return OMITTED
            try:
                assert_secret_free(value)
            except SensitiveStateError:
                return OMITTED
            value = _UUID_REFERENCE.sub(lambda match: "ref-" + sha256(match.group().lower().encode()).hexdigest()[:16], value)
            return redact_direct_identifiers(value)[0]
        if type(value) is SchemaVersion:
            return int(value)
        if value is None or type(value) in {bool, int}:
            return value
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if type(value) is float and math.isfinite(value):
            return value
        return OMITTED

    try:
        projected = clean(payload)
        encoded = json.dumps(projected, ensure_ascii=False, allow_nan=False)
        if len(encoded.encode("utf-8")) > MAX_EXPORT_BYTES:
            return {"omitted": "export_budget_exceeded"}
        assert_secret_free(projected, path="langsmith.debug")
        return projected
    except (ValueError, TypeError, RecursionError):
        return {"omitted": "privacy_projection_failed"}
