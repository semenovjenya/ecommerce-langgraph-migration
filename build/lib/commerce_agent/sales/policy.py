"""Bounded Sales request and publication policies; no business calculations."""
from __future__ import annotations

from datetime import date
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, model_validator

from commerce_agent.adapters.privacy import assert_no_direct_identifiers
from commerce_agent.safety import assert_secret_free


MAX_PREPARED_BYTES = 131_072
MAX_RESULT_BYTES = 262_144  # full legacy metadata for 100 native revenue + unit pairs


def assert_bounded_safe(value: Any, *, max_bytes: int = MAX_PREPARED_BYTES) -> None:
    assert_secret_free(value)
    assert_no_direct_identifiers(value)
    if len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode()) > max_bytes:
        raise ValueError("Sales payload exceeds budget")


class SalesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    user_query: StrictStr = Field(min_length=1, max_length=8000)
    scope: Literal["portfolio", "single_sku", "sku_list"] = "portfolio"
    targets: tuple[StrictStr, ...] = Field(default=(), max_length=20)
    days: int = Field(default=7, ge=1, le=366, strict=True)
    date_from: date | None = None
    date_to: date | None = None

    @model_validator(mode="after")
    def validate_scope_and_period(self):
        if (self.scope == "single_sku" and len(self.targets) != 1
                or self.scope == "sku_list" and not self.targets):
            raise ValueError("Sales scope requires targets")
        if any(not target.strip() or len(target) > 128 for target in self.targets):
            raise ValueError("invalid Sales target")
        if len(set(self.targets)) != len(self.targets):
            raise ValueError("duplicate Sales target")
        if (self.date_from is None) != (self.date_to is None):
            raise ValueError("Sales explicit period requires both boundaries")
        if self.date_from and self.date_to < self.date_from:
            raise ValueError("invalid Sales period")
        if self.date_from and (self.date_to - self.date_from).days >= 366:
            raise ValueError("Sales explicit period exceeds budget")
        assert_bounded_safe(self.model_dump(mode="json"))
        return self


def resolve_sales_request(user_query: str, constraints: dict | None = None) -> SalesRequest:
    """M3 deterministic thin request; full structured query planning belongs to M4.

    Explicit trusted query-plan constraints take precedence. The thin parser
    accepts literal SKU labels, ISO ranges and common day/week/month windows;
    ambiguous periods fail closed rather than silently changing the question.
    """
    if constraints:
        return SalesRequest.model_validate({**constraints, "user_query": user_query})
    text = user_query.lower().replace("ё", "е")
    label = r"(?:\bsku\b|\bартикул\w*)\s*[:№#]?\s*"
    token = r"[a-zа-я0-9][a-zа-я0-9_.-]{0,127}"
    groups = re.findall(label + "(" + token + r"(?:\s*[,;]\s*" + token + ")*)", user_query,
                        flags=re.IGNORECASE)
    inline_skus = re.findall(r"\bsku[-_][a-zа-я0-9_.-]+", user_query, flags=re.IGNORECASE)
    targets = tuple(dict.fromkeys([*(target.strip() for group in groups for target in re.split(r"[,;]", group)),
                                  *inline_skus]))
    if any(target.casefold() in {"по", "за", "заказ", "все", "всем", "на", "и", "в"} for target in targets):
        raise ValueError("unresolved Sales SKU label")
    calendar_text = text
    for target in targets:
        calendar_text = calendar_text.replace(target.lower().replace("ё", "е"), "")
    if re.search(r"\bпо\s+(?:месяц|недел|час|дням)", calendar_text):
        raise ValueError("Sales grouped calendar grain requires a separate capability")
    portfolio = any(word in text for word in ("портфел", "все товары", "всем товарам", "вся матрица"))
    scope = "portfolio" if portfolio or not targets else ("single_sku" if len(targets) == 1 else "sku_list")
    dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text)
    if dates and len(dates) != 2:
        raise ValueError("ambiguous Sales date range")
    if not dates and re.search(r"\d{1,2}[./]\d{1,2}[./]\d{2,4}", text):
        raise ValueError("Sales date range must use ISO boundaries")
    days_match = re.search(r"(?:последни\w*|за)\s+(\d+)\s+д(?:н|ен)", text)
    weeks_match = re.search(r"(?:последни\w*|за)\s+(\d+)\s+недел", text)
    months_match = re.search(r"(?:последни\w*|за)\s+(\d+)\s+месяц", text)
    if months_match:
        raise ValueError("calendar months require explicit Sales period")
    days = (int(days_match[1]) if days_match else int(weeks_match[1]) * 7
            if weeks_match else 30 if "месяц" in text else 7)
    if "вчера" in text or "позавчера" in text:
        # Full report comparisons are anchored to the latest complete source
        # day. Calendar-specific diagnostics require explicit plan boundaries.
        if not re.search(r"отчет|сводк|report", text):
            raise ValueError("calendar day requires explicit Sales period")
    if not dates and any(word in text for word in ("год", "квартал", "час")):
        raise ValueError("unsupported implicit Sales period")
    if not dates and re.search(r"\b(?:январ|феврал|март|апрел|ма[йяе]|июн|июл|август|сентябр|октябр|ноябр|декабр)[а-я]*\b|"
                               r"(?:прошл|предыдущ|текущ|эт)\w*\s+(?:недел|месяц)", calendar_text):
        raise ValueError("calendar period requires explicit Sales boundaries")
    return SalesRequest(user_query=user_query, scope=scope, targets=targets, days=days,
                        date_from=dates[0] if dates else None, date_to=dates[1] if dates else None)
