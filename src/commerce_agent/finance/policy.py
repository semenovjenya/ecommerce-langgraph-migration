"""Financial query capabilities and explicit, non-overlapping period pairs."""
from datetime import date
import re
from typing import Literal
from pydantic import Field, model_validator
from commerce_agent.sales.policy import SalesRequest, assert_bounded_safe, resolve_sales_request


class FinanceRequest(SalesRequest):
    days: int = Field(default=30, ge=1, le=366, strict=True)
    operation: Literal["report", "marketplace", "sku", "margin_change", "overview"] = "report"
    previous_date_from: date | None = None
    previous_date_to: date | None = None

    @model_validator(mode="after")
    def validate_financial_windows(self):
        if (self.previous_date_from is None) != (self.previous_date_to is None):
            raise ValueError("Finance previous window needs both boundaries")
        if self.previous_date_from:
            if self.date_from is None or self.previous_date_to < self.previous_date_from or self.previous_date_to >= self.date_from:
                raise ValueError("Finance periods reversed, overlapping or incomplete")
            if (self.previous_date_to - self.previous_date_from).days >= 366:
                raise ValueError("Finance comparison exceeds period budget")
        if self.operation == "marketplace" and self.scope != "portfolio":
            raise ValueError("native marketplace report is portfolio only")
        if self.operation == "overview" and (self.date_from or self.previous_date_from):
            raise ValueError("latest Finance overview cannot replace historical report")
        if any("," in target or ";" in target for target in self.targets):
            raise ValueError("Finance targets are separate exact identifiers")
        return self


def resolve_finance_request(query, constraints=None):
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Finance requires a textual question")
    if constraints:
        return FinanceRequest.model_validate({**constraints, "user_query": query})
    text = query.casefold().replace("ё", "е")
    if not any(word in text for word in ("финанс", "марж", "себестоим", "прибыл", "комисси", "выплат", "расход", "экономик")):
        raise ValueError("explicit Finance capability required")
    dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", query)
    if len(dates) not in {0, 2, 4}:
        raise ValueError("ambiguous Finance periods")
    change = any(word in text for word in ("изменени", "динамик", "сравн", "относительно"))
    if len(dates) == 4 and not change:
        raise ValueError("second Finance period has no comparison intent")
    portfolio = any(word in text for word in ("портфел", "все товары", "всем товарам", "вся матрица"))
    # Sales thin parser supplies common scope/date rules, not Finance arithmetic.
    # Remove the second date pair and portfolio's generic 'матрицу SKU и ...'
    # grammar before parsing exact target labels.
    parse_query = query
    if len(dates) == 4:
        parse_query = parse_query[:parse_query.index(dates[2])]
    if portfolio:
        parse_query = re.sub(r"\bSKU\b(?=\s+(?:и|за|по|для)\b)", "товаров", parse_query, flags=re.I)
    base = resolve_sales_request(parse_query).model_dump()
    base["user_query"] = query
    if portfolio:
        base["scope"] = "portfolio"
    if not dates and not re.search(r"\b(?:дн\w*|день|недел\w*)\b", text):
        base["days"] = 30
    overview = any(word in text for word in ("обзор", "overview")) and not dates and not change
    operation = "overview" if overview else "report" if "отчет" in text else "margin_change" if change else "sku" if base["scope"] != "portfolio" else "marketplace"
    return FinanceRequest.model_validate({**base, "operation": operation,
        "previous_date_from": dates[2] if len(dates) == 4 else None,
        "previous_date_to": dates[3] if len(dates) == 4 else None})
