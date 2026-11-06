"""Development trace projection, never a mutation of authoritative business data.

Aggregate M2 schemas exclude personal records. This additional boundary removes
direct identifiers and secret/context fields, not arbitrary semantic PII. Raw
user prose and opaque objects are deliberately not eligible for export.
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


def development_debug_enabled():
    return (os.environ.get("COMMERCE_AGENT_ENVIRONMENT") == "development"
            and os.environ.get("COMMERCE_AGENT_LANGSMITH_DEBUG", "").lower() in {"true", "1", "yes", "on"})


def _private_key(key):
    compact = re.sub(r"[^\w]", "", key).replace("_", "").lower()
    return (compact in _BLOCKED or any(word in compact for word in
            ("password", "secret", "credential", "apikey", "accesstoken", "servicetoken"))
            or any(compact.endswith(suffix) for suffix in
                   ("email", "phone", "address", "firstname", "lastname", "fullname", "customerid", "userid", "tenantid",
                    "token", "cookie", "cookies", "dsn", "authorization", "authentication", "displayname", "accountname")))


def _prepared_json(value):
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError):
        return None
    # Only the approved Sales model's prepared evidence contract is eligible.
    required = {"request", "today", "tables", "facts", "evidence", "warnings"}
    allowed = required | {"periods", "deep_dives", "overview_signals", "claim_catalog", "signal_catalog"}
    if isinstance(parsed, dict) and required <= set(parsed) <= allowed:
        return parsed
    return None


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

    def clean(value, depth=0):
        nonlocal count
        count += 1
        if depth > 32 or count > 100_000:
            return "[omitted-export-budget]"
        if isinstance(value, BaseModel):
            return clean(value.model_dump(mode="python"), depth + 1)
        if isinstance(value, Mapping):
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
                return {"type": _MESSAGE_ROLES.get(message_type, _MESSAGE_ROLES.get(serialized_type)), "content": OMITTED}
            if message_type in message_types and isinstance(value.get("data"), Mapping):
                value = {**value, "data": {**value["data"], "type": message_type}}
            columns = value.get("columns", value.get("headers"))
            private_columns = {i for i, col in enumerate(columns) if isinstance(col, str) and _private_key(col)} \
                if isinstance(columns, (list, tuple)) else set()
            for key, item in value.items():
                if not isinstance(key, str) or _private_key(key):
                    continue
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
                # Human graph messages may contain any personal prose. The LLM
                # receives only prepared JSON business evidence, which is eligible.
                if key == "content" and message_type in message_types:
                    if not allow_prepared_messages:
                        result[key] = OMITTED
                        continue
                    if message_type in {"human", "HumanMessage"}:
                        decoded = _prepared_json(item) if isinstance(item, str) else None
                        result[key] = json.dumps(clean(decoded, depth + 1), ensure_ascii=False) if decoded is not None else OMITTED
                    else:
                        result[key] = clean(item, depth + 1)
                else:
                    result[safe_key] = clean(item, depth + 1)
            # Serialized messages wrap content in kwargs, type=constructor.
            if value.get("type") == "constructor" and isinstance(value.get("id"), (list, tuple)):
                message_type = value["id"][-1] if value["id"] else ""
                kwargs = value.get("kwargs")
                if message_type in message_types and isinstance(kwargs, Mapping):
                    result["kwargs"] = clean({**kwargs, "type": message_type}, depth + 1)
            return result
        if isinstance(value, (list, tuple)):
            return [clean(item, depth + 1) for item in value]
        if isinstance(value, str):
            if len(value.encode("utf-8")) > MAX_EXPORT_BYTES:
                return "[omitted-export-budget]"
            # Embedded JSON prompts need field filtering, not just regex masking.
            if value.lstrip().startswith(("{", "[")):
                try:
                    decoded = json.loads(value)
                    if isinstance(decoded, (dict, list)):
                        return json.dumps(clean(decoded, depth + 1), ensure_ascii=False)
                except (ValueError, TypeError):
                    pass
            if "\n- user_query:" in value:
                # Use the LAST full trusted override, not an attacker-spoofable
                # partial marker. If unavailable, omit the entire suffix.
                from commerce_agent.sales.prompts import MIGRATION_OVERRIDE
                start = value.index("\n- user_query:")
                end = value.rfind(MIGRATION_OVERRIDE)
                value = value[:start] + "\n- user_query: " + OMITTED + ("\n\n" + value[end:] if end > start else "")
            if _OPAQUE_CREDENTIAL.search(value) or _PERSONAL_CONTEXT.search(value):
                return OMITTED
            try:
                assert_secret_free(value)
            except SensitiveStateError:
                return OMITTED
            value = _UUID_REFERENCE.sub(lambda match: "ref-" + sha256(match.group().lower().encode()).hexdigest()[:16], value)
            return redact_direct_identifiers(value)[0]
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
