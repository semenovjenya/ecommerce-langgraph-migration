"""Bounded report capability; unsupported campaign/query diagnostics fail closed."""
from pydantic import Field
from commerce_agent.sales.policy import SalesRequest, resolve_sales_request, assert_bounded_safe


class MarketingRequest(SalesRequest):
    days: int = Field(default=7, ge=1, le=31, strict=True)


def parse_marketing_request(value):
    if value.get("capability")=="latest_marketing_overview":
        from commerce_agent.marketing.overview import MarketingOverviewRequest
        return MarketingOverviewRequest.model_validate(value)
    if value.get("capability")=="scoped_report":
        from commerce_agent.marketing.scoped import MarketingScopedRequest
        return MarketingScopedRequest.model_validate(value)
    if value.get("capability")=="latest_native_diagnostic":
        from commerce_agent.marketing.composite import MarketingCompositeDiagnosticRequest
        return MarketingCompositeDiagnosticRequest.model_validate(value)
    if value.get("capability") == "sku_diagnostic":
        from commerce_agent.marketing.diagnostics import MarketingDiagnosticRequest
        return MarketingDiagnosticRequest.model_validate(value)
    return MarketingRequest.model_validate(value)


def resolve_marketing_request(query, constraints=None, *, allow_diagnostics=False,allow_composites=False,allow_scoped=False,allow_overview=False):
    import re
    overview=re.fullmatch(r"Обзор рекламы(?: по фильтру ([A-Za-z0-9][A-Za-z0-9_.-]{0,127}))?",query,re.IGNORECASE)
    if constraints and constraints.get("capability")=="latest_marketing_overview" or overview:
        if not allow_overview:raise ValueError("Marketing overview capability is disabled")
        from commerce_agent.marketing.overview import MarketingOverviewRequest
        return MarketingOverviewRequest.model_validate({**constraints,"user_query":query}) if constraints else MarketingOverviewRequest(
            user_query=query,product_keyword=overview[1])
    if constraints and constraints.get("capability")=="scoped_report":
        if not allow_scoped:raise ValueError("scoped Marketing report capability is disabled")
        from commerce_agent.marketing.scoped import MarketingScopedRequest
        return MarketingScopedRequest.model_validate({**constraints,"user_query":query})
    composite=re.fullmatch(r"Расширенная диагностика рекламы(?: по фильтру ([A-Za-z0-9][A-Za-z0-9_.-]{0,127}))? за последние ([0-9]+) дней",query,re.IGNORECASE)
    if constraints and constraints.get("capability")=="latest_native_diagnostic" or composite:
        if not allow_composites:raise ValueError("Marketing composite capability is disabled")
        from commerce_agent.marketing.composite import MarketingCompositeDiagnosticRequest
        return MarketingCompositeDiagnosticRequest.model_validate({**constraints,"user_query":query}) if constraints else MarketingCompositeDiagnosticRequest(
            user_query=query,product_keyword=composite[1],days=int(composite[2]))
    match = re.fullmatch(r"Диагностика рекламы SKU ([A-Za-z0-9][A-Za-z0-9_.-]{0,127}) за (\d{4}-\d{2}-\d{2}) [—–-] (\d{4}-\d{2}-\d{2})", query, re.IGNORECASE)
    diagnostic = bool(constraints and constraints.get("capability") == "sku_diagnostic") or match
    if diagnostic:
        if not allow_diagnostics:
            raise ValueError("Marketing diagnostic capability is disabled")
        from commerce_agent.marketing.diagnostics import MarketingDiagnosticRequest
        if constraints:
            return MarketingDiagnosticRequest.model_validate({**constraints, "user_query": query})
        return MarketingDiagnosticRequest(user_query=query, sku=match[1], date_from=match[2], date_to=match[3])
    request = MarketingRequest.model_validate(resolve_sales_request(query, constraints).model_dump())
    if (request.scope != "portfolio" or request.targets) and not allow_scoped:
        raise ValueError("scoped Marketing diagnostics are not enabled in this report slice")
    if request.scope=="portfolio" and request.targets and not allow_scoped:
        raise ValueError("Marketing portfolio targets require separate acceptance")
    if request.date_from and (request.date_to - request.date_from).days >= 31:
        raise ValueError("Marketing period exceeds report budget")
    text = query.casefold().replace("ё", "е")
    if not any(token in text for token in ("отчет", "сводк", "report")) or any(
        token in text for token in ("запрос", "ставк", "кампани", "конкурент", "позици")):
        raise ValueError("Marketing query requires a separate diagnostic capability")
    if request.scope!="portfolio" or request.targets:
        from commerce_agent.marketing.scoped import MarketingScopedRequest
        return MarketingScopedRequest.model_validate(request.model_dump(mode="json"))
    return request
