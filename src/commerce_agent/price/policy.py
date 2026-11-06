"""Explicit Price read capabilities. Latest readers never pretend to be history."""
import re
from typing import Literal

from pydantic import Field, model_validator
from commerce_agent.sales.policy import SalesRequest, assert_bounded_safe, resolve_sales_request


class PriceRequest(SalesRequest):
    days: int = Field(default=14, ge=1, le=31, strict=True)
    operation: Literal["report", "context", "competitors", "daily", "history"] = "report"
    target_namespace: Literal["auto", "vendor_code", "nm_id"] = "auto"

    @model_validator(mode="after")
    def validate_price_capability(self):
        if self.operation == "history":
            from commerce_agent.adapters.price_history import PriceHistoryRequest
            if self.scope != "single_sku" or len(self.targets) != 1 or self.target_namespace != "vendor_code":
                raise ValueError("Price history requires one exact vendor code")
            PriceHistoryRequest(vendor_code=self.targets[0], date_from=self.date_from, date_to=self.date_to)
        elif self.date_from is not None:
            raise ValueError("Price readers have no explicit historical anchor")
        if self.operation == "daily" and not self.targets:
            raise ValueError("daily Price/Sales comparison requires exact targets")
        if any("," in target or ";" in target or target != target.strip() for target in self.targets):
            raise ValueError("Price targets must be separate exact identifiers")
        if self.target_namespace == "nm_id" and any(not target.isascii() or not target.isdigit()
                or not 0 < int(target) < 2 ** 63 for target in self.targets):
            raise ValueError("Price nm_id requires positive bigint identifiers")
        return self


def resolve_price_request(query, constraints=None):
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Price needs a textual question")
    constraints = dict(constraints or {})
    explicit = constraints.pop("operation", None)
    namespace_hint = constraints.pop("target_namespace", None)
    text = query.casefold().replace("ё", "е")
    daily = bool(re.search(r"(?:динамик|по дням|daily|влияни|связь).*(?:цен|price)|(?:цен|price).*(?:динамик|по дням|daily|влияни|связь)", text))
    operation = explicit or ("daily" if daily else "competitors" if "конкурент" in text and "отчет" not in text else "report")
    if explicit is None and not any(word in text for word in ("цен", "price", "кошел", "конкурент", "аналог")):
        raise ValueError("explicit Price capability required")
    parse_query = re.sub(r"\bSKU\b(?=\s+(?:и|за|по|для)\b)", "товаров", query, flags=re.I) if "портфел" in text else query
    if not constraints and namespace_hint is None and re.search(r"\bnm_id\b", query, flags=re.I):
        if re.search(r"\b(?:SKU|vendor_code)\b|артикул", query, flags=re.I):
            raise ValueError("mixed Price identity namespaces require an explicit plan")
        parse_query = re.sub(r"\bnm_id\b", "SKU", parse_query, flags=re.I)
        namespace = "nm_id"
    else:
        namespace = namespace_hint if namespace_hint is not None else "auto"
    base = resolve_sales_request(parse_query, constraints or None).model_dump()
    base["user_query"] = query
    if not constraints and not re.search(r"\b(?:дн\w*|день|недел\w*)\b", text):
        base["days"] = 14
    if operation not in {"daily", "history"} and ("days" in constraints or re.search(
        r"(?:за|последни\w*|прошл\w*|предыдущ\w*|текущ\w*)\s+(?:\d+\s+)?(?:дн\w*|день|недел\w*|месяц\w*)|вчера|позавчера", text)):
        raise ValueError("latest Price context does not support requested history")
    return PriceRequest.model_validate({**base, "operation": operation, "target_namespace": namespace})
