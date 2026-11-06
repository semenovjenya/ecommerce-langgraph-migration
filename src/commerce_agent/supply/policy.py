"""Read-only Supply capabilities; unsupported methods never silently change scope."""
from typing import Literal
import re
from pydantic import Field
from commerce_agent.sales.policy import SalesRequest, resolve_sales_request, assert_bounded_safe


class SupplyRequest(SalesRequest):
    days: int = Field(default=30, ge=1, le=366, strict=True)
    operation: Literal["overview", "turnover", "value", "matrix"] = "overview"


def resolve_supply_request(query, constraints=None):
    constraints = dict(constraints or {})
    explicit_operation = constraints.pop("operation", None)
    text = query.casefold().replace("ё", "е")
    has_value = any(word in text for word in ("стоимость запас", "стоимость остат", "inventory value"))
    operation = explicit_operation or ("matrix" if has_value and "оборачиваем" in text else
        "turnover" if "оборачиваем" in text else "value" if has_value else "overview")
    if explicit_operation is None and not any(word in text for word in
        ("остат", "запас", "постав", "оборачиваем", "налич", "stock", "inventory", "дефицит", "дозаказ")):
        raise ValueError("Supply query needs an explicit supported capability")
    base = resolve_sales_request(query, constraints or None).model_dump()
    # Last-known full 30-day window is explicit in prepared data. Current
    # stock/value never masquerade as a requested historical snapshot.
    if not constraints and not any(word in text for word in ("дн", "день", "недел", "месяц")):
        base["days"] = 30
    request = SupplyRequest.model_validate({**base, "operation": operation})
    if operation not in {"turnover", "matrix"} and request.date_from is not None:
        raise ValueError("historical Supply stock/value needs a separate snapshot capability")
    if operation not in {"turnover", "matrix"} and ("days" in constraints or re.search(
        r"(?:за|последни\w*|прошл\w*|предыдущ\w*|текущ\w*)\s+(?:\d+\s+)?(?:дн\w*|день|недел\w*|месяц\w*)|вчера|позавчера", text)):
        raise ValueError("Supply current stock/value does not support historical windows")
    if any("," in target or ";" in target for target in request.targets):
        raise ValueError("ambiguous Supply target delimiter")
    return request
