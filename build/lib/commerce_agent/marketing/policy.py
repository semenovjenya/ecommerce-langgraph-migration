"""Bounded report capability; unsupported campaign/query diagnostics fail closed."""
from pydantic import Field
from commerce_agent.sales.policy import SalesRequest, resolve_sales_request, assert_bounded_safe


class MarketingRequest(SalesRequest):
    days: int = Field(default=7, ge=1, le=31, strict=True)


def resolve_marketing_request(query, constraints=None):
    request = MarketingRequest.model_validate(resolve_sales_request(query, constraints).model_dump())
    if request.scope != "portfolio" or request.targets:
        raise ValueError("scoped Marketing diagnostics are not enabled in this report slice")
    if request.date_from and (request.date_to - request.date_from).days >= 31:
        raise ValueError("Marketing period exceeds report budget")
    text = query.casefold().replace("ё", "е")
    if not any(token in text for token in ("отчет", "сводк", "report")) or any(
        token in text for token in ("запрос", "ставк", "кампани", "конкурент", "позици")):
        raise ValueError("Marketing query requires a separate diagnostic capability")
    return request
