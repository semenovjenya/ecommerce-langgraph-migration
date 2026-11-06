"""Bounded read-only HTTP clients; credentials never enter graph context/state."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from urllib.parse import urlsplit
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from commerce_agent.adapters.contracts import (
    EmptyToolRequest, DatabaseTime, SalesVendorCodeMapRequest, SalesVendorCodeMap,
    FormulaCatalog, TableCatalog,
    InventoryTurnover, InventoryTurnoverRequest, InventoryValue, InventoryValueRequest,
    InventoryTurnoverAlternatives, InventoryTurnoverAlternativesRequest,
    MarketingSnapshot, MarketingSnapshotRequest, SalesSnapshot, SalesSnapshotRequest, ToolResult, ToolArtifactReference,
    SupplyOverview, SupplyOverviewRequest,
    SalesOverview, SalesOverviewRequest, DailyPriceSalesComparison, DailyPriceSalesRequest,
    SalesRevenueChange, SalesRevenueChangeRequest,
    PriceContext, PriceContextRequest, PriceDecisionOverview, PriceOverviewRequest,
    PriceCompetitorOverview, PriceCompetitorDetails, PriceDetailsRequest,
    FinanceOverviewRequest, FinanceOverview, FinancePeriodRequest, FinanceMarketplaceReport,
    FinanceMarketplaceText, FinanceSkuRequest, FinanceSkuTextRequest, FinanceSkuReport, FinanceSkuText,
    FinanceProfitChangeRequest, FinanceProfitChangeReport, FinanceProfitChangeText,
    MarketingOverviewRequest,
    MarketingOverview,
    MarketingCampaignPeriodRequest,
    MarketingCampaignPeriodReport,
    MarketingDrrRequest,
    MarketingDrrReport,
    MarketingSkuComparisonRequest,
    MarketingSkuComparison,
    MarketingEfficiencyRequest,
    MarketingEfficiency,
    MarketingQueryRequest,
    MarketingQueryPerformance,
    MarketingBidPressure,
    MarketingVisibilityRequest,
    MarketingVisibility,
    MarketingSearchDemand,
    MarketingCompositeRequest, MarketingActionMatrix, MarketingDeepDiagnostic,
)
from commerce_agent.context import CommerceRuntimeContext
from commerce_agent.adapters.contracts import MarketingExactSearchRequest, MarketingExactSearch
from commerce_agent.adapters.contracts import ExactDiagnosticHistory
from commerce_agent.safety import SensitiveStateError, assert_secret_free
from commerce_agent.adapters.privacy import PrivacyError, assert_no_direct_identifiers

T = TypeVar("T", bound=BaseModel)


class ToolAdapterError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class BusinessToolsSettings:
    api_url: str
    service_token: str = field(repr=False)
    environment: str = "development"
    timeout_seconds: float = 10.0

    @classmethod
    def from_environment(cls) -> BusinessToolsSettings:
        return cls(
            api_url=os.getenv("COMMERCE_AGENT_TOOLS_API_URL", ""),
            service_token=os.getenv("COMMERCE_AGENT_TOOLS_SERVICE_TOKEN", ""),
            environment=os.getenv("COMMERCE_AGENT_ENVIRONMENT", "development"),
        )

    def __post_init__(self) -> None:
        url = urlsplit(self.api_url)
        if (
            url.scheme not in {"https", "http"} or not url.hostname
            or url.username or url.password or url.query or url.fragment
            or url.path not in {"", "/"}
            or self.environment not in {"development", "test", "staging", "production"}
            or (self.environment in {"staging", "production"} and url.scheme != "https")
            or len(self.service_token.encode()) < 32
            or any(char in self.service_token for char in "\r\n")
            or not 0 < self.timeout_seconds <= 30
        ):
            raise ValueError("invalid business tools configuration")


class BusinessToolClient:
    def __init__(
        self, settings: BusinessToolsSettings, *, transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.api_url, timeout=settings.timeout_seconds,
            trust_env=False, follow_redirects=False, transport=transport,
        )

    async def __aenter__(self) -> BusinessToolClient:
        return self

    async def __aexit__(self, *_args) -> None:
        await self._client.aclose()

    async def database_time(self, context: CommerceRuntimeContext) -> ToolResult[DatabaseTime] | ToolArtifactReference:
        return await self._operation(context, {}, EmptyToolRequest, DatabaseTime,
                                     "/internal/v1/agent-tools/shared/database-time")

    async def sales_vendor_code_map(
        self, context: CommerceRuntimeContext, *, nm_ids: list[int | str],
    ) -> ToolResult[SalesVendorCodeMap] | ToolArtifactReference:
        return await self._operation(context, {"nm_ids": nm_ids}, SalesVendorCodeMapRequest, SalesVendorCodeMap,
                                     "/internal/v1/agent-tools/sales/vendor-code-map")

    async def sales_formula_catalog(self, context: CommerceRuntimeContext) -> ToolResult[FormulaCatalog] | ToolArtifactReference:
        return await self._operation(context, {}, EmptyToolRequest, FormulaCatalog,
                                     "/internal/v1/agent-tools/sales/formula-catalog")

    async def marketing_formula_catalog(self, context: CommerceRuntimeContext) -> ToolResult[FormulaCatalog] | ToolArtifactReference:
        return await self._operation(context, {}, EmptyToolRequest, FormulaCatalog,
                                     "/internal/v1/agent-tools/marketing/formula-catalog")

    async def supply_formula_catalog(self, context: CommerceRuntimeContext) -> ToolResult[FormulaCatalog] | ToolArtifactReference:
        return await self._operation(context, {}, EmptyToolRequest, FormulaCatalog,
                                     "/internal/v1/agent-tools/supply/formula-catalog")

    async def price_formula_catalog(self, context: CommerceRuntimeContext) -> ToolResult[FormulaCatalog] | ToolArtifactReference:
        return await self._operation(context, {}, EmptyToolRequest, FormulaCatalog,
                                     "/internal/v1/agent-tools/price/formula-catalog")

    async def finance_formula_catalog(self, context: CommerceRuntimeContext) -> ToolResult[FormulaCatalog] | ToolArtifactReference:
        return await self._operation(context, {}, EmptyToolRequest, FormulaCatalog,
                                     "/internal/v1/agent-tools/finance/formula-catalog")

    async def sales_table_catalog(self, context: CommerceRuntimeContext) -> ToolResult[TableCatalog] | ToolArtifactReference:
        return await self._operation(context, {}, EmptyToolRequest, TableCatalog,
                                     "/internal/v1/agent-tools/sales/table-catalog")

    async def marketing_table_catalog(self, context: CommerceRuntimeContext) -> ToolResult[TableCatalog] | ToolArtifactReference:
        return await self._operation(context, {}, EmptyToolRequest, TableCatalog,
                                     "/internal/v1/agent-tools/marketing/table-catalog")

    async def supply_table_catalog(self, context: CommerceRuntimeContext) -> ToolResult[TableCatalog] | ToolArtifactReference:
        return await self._operation(context, {}, EmptyToolRequest, TableCatalog,
                                     "/internal/v1/agent-tools/supply/table-catalog")

    async def price_table_catalog(self, context: CommerceRuntimeContext) -> ToolResult[TableCatalog] | ToolArtifactReference:
        return await self._operation(context, {}, EmptyToolRequest, TableCatalog,
                                     "/internal/v1/agent-tools/price/table-catalog")

    async def sales_report_snapshot(
        self, context: CommerceRuntimeContext, *, product_keyword: str | None = None,
    ) -> ToolResult[SalesSnapshot] | ToolArtifactReference:
        return await self._operation(context, {"product_keyword": product_keyword}, SalesSnapshotRequest, SalesSnapshot,
                                    "/internal/v1/agent-tools/sales/report-snapshot")

    async def marketing_report_snapshot(
        self, context: CommerceRuntimeContext, *, product_keyword: str | None = None,
    ) -> ToolResult[MarketingSnapshot] | ToolArtifactReference:
        return await self._operation(context, {"product_keyword": product_keyword}, MarketingSnapshotRequest, MarketingSnapshot,
                                    "/internal/v1/agent-tools/marketing/report-snapshot")

    async def supply_overview(
        self, context: CommerceRuntimeContext, *, product_keyword: str | None = None, limit: int = 100,
    ) -> ToolResult[SupplyOverview] | ToolArtifactReference:
        return await self._operation(context, {"product_keyword": product_keyword, "limit": limit},
                                     SupplyOverviewRequest, SupplyOverview,
                                     "/internal/v1/agent-tools/supply/overview")

    async def inventory_value(
        self, context: CommerceRuntimeContext, *, product_keyword: str | None = None, limit: int = 100,
    ) -> ToolResult[InventoryValue] | ToolArtifactReference:
        return await self._operation(context, {"product_keyword": product_keyword, "limit": limit},
                                     InventoryValueRequest, InventoryValue,
                                     "/internal/v1/agent-tools/supply/inventory-value")

    async def inventory_turnover(
        self, context: CommerceRuntimeContext, *, date_from: str, date_to: str | None = None,
        vendor_codes: str | None = None, limit: int = 100,
    ) -> ToolResult[InventoryTurnover] | ToolArtifactReference:
        return await self._operation(context, {
            "date_from": date_from, "date_to": date_to, "vendor_codes": vendor_codes, "limit": limit,
        }, InventoryTurnoverRequest, InventoryTurnover,
            "/internal/v1/agent-tools/supply/inventory-turnover")

    async def inventory_turnover_alternatives(
        self, context: CommerceRuntimeContext, *, date_from: str, date_to: str,
        vendor_codes: str | None = None, limit: int = 100, boundary_tolerance_days: int = 0,
    ) -> ToolResult[InventoryTurnoverAlternatives] | ToolArtifactReference:
        from commerce_agent.debug_privacy import development_debug_enabled
        if development_debug_enabled():
            # This additive source has no approved wire/privacy projection yet.
            # Reject before RunnableLambda callbacks, request or result export.
            raise ToolAdapterError("unsupported_debug_source")
        return await self._operation(context, {
            "date_from": date_from, "date_to": date_to, "vendor_codes": vendor_codes,
            "limit": limit, "boundary_tolerance_days": boundary_tolerance_days,
        }, InventoryTurnoverAlternativesRequest, InventoryTurnoverAlternatives,
            "/internal/v1/agent-tools/supply/inventory-turnover-alternatives")

    async def sales_overview(
        self, context: CommerceRuntimeContext, *, product_keyword: str | None = None, limit: int = 100,
    ) -> ToolResult[SalesOverview] | ToolArtifactReference:
        return await self._operation(context, {"product_keyword": product_keyword, "limit": limit},
                                     SalesOverviewRequest, SalesOverview, "/internal/v1/agent-tools/sales/overview")

    async def daily_price_sales_comparison(
        self, context: CommerceRuntimeContext, *, product_keyword: str, days: int = 14,
    ) -> ToolResult[DailyPriceSalesComparison] | ToolArtifactReference:
        return await self._operation(context, {"product_keyword": product_keyword, "days": days},
                                     DailyPriceSalesRequest, DailyPriceSalesComparison,
                                     "/internal/v1/agent-tools/sales/daily-price-sales-comparison")

    async def sales_revenue_change(
        self, context: CommerceRuntimeContext, *, date_from: str | None = None,
        date_to: str | None = None, product_keyword: str | None = None, limit: int = 20,
    ) -> ToolResult[SalesRevenueChange] | ToolArtifactReference:
        return await self._operation(context, {
            "date_from": date_from, "date_to": date_to, "product_keyword": product_keyword, "limit": limit,
        }, SalesRevenueChangeRequest, SalesRevenueChange, "/internal/v1/agent-tools/sales/sku-revenue-change")

    async def price_history(self, context: CommerceRuntimeContext, *, vendor_code, date_from, date_to):
        from commerce_agent.adapters.price_history import PriceHistoryRequest, PriceHistory
        from commerce_agent.debug_privacy import development_debug_enabled
        if development_debug_enabled():
            raise ToolAdapterError("unsupported_debug_source")
        return await self._operation(context, {
            "vendor_code": vendor_code, "date_from": date_from, "date_to": date_to,
        }, PriceHistoryRequest, PriceHistory, "/internal/v1/agent-tools/price/history")

    async def price_context(
        self, context: CommerceRuntimeContext, *, nm_id: str | None = None,
        vendor_code: str | None = None, product_keyword: str | None = None,
        limit: int = 20, include_evidence: bool = False,
    ) -> ToolResult[PriceContext] | ToolArtifactReference:
        return await self._operation(context, {
            "nm_id": nm_id, "vendor_code": vendor_code, "product_keyword": product_keyword,
            "limit": limit, "include_evidence": include_evidence,
        }, PriceContextRequest, PriceContext, "/internal/v1/agent-tools/price/context")

    async def price_decision_overview(
        self, context: CommerceRuntimeContext, *, product_keyword: str | None = None, limit: int = 20,
    ) -> ToolResult[PriceDecisionOverview] | ToolArtifactReference:
        return await self._operation(context, {"product_keyword": product_keyword, "limit": limit},
                                     PriceOverviewRequest, PriceDecisionOverview,
                                     "/internal/v1/agent-tools/price/decision-overview")

    async def price_competitor_overview(
        self, context: CommerceRuntimeContext, *, product_keyword: str | None = None, limit: int = 20,
    ) -> ToolResult[PriceCompetitorOverview] | ToolArtifactReference:
        return await self._operation(context, {"product_keyword": product_keyword, "limit": limit},
                                     PriceOverviewRequest, PriceCompetitorOverview,
                                     "/internal/v1/agent-tools/price/competitor-overview")

    async def price_competitor_details(
        self, context: CommerceRuntimeContext, *, nm_id: str | None = None,
        vendor_code: str | None = None, product_keyword: str | None = None, limit: int = 50,
    ) -> ToolResult[PriceCompetitorDetails] | ToolArtifactReference:
        return await self._operation(context, {
            "nm_id": nm_id, "vendor_code": vendor_code, "product_keyword": product_keyword, "limit": limit,
        }, PriceDetailsRequest, PriceCompetitorDetails, "/internal/v1/agent-tools/price/competitor-details")


    async def marketing_overview(
        self, context: CommerceRuntimeContext, *, product_keyword: str | None = None, limit: int = 5,
    ) -> ToolResult[MarketingOverview] | ToolArtifactReference:
        return await self._operation(context, {
            "product_keyword": product_keyword, "limit": limit,
        }, MarketingOverviewRequest, MarketingOverview, "/internal/v1/agent-tools/marketing/overview")


    async def marketing_campaign_period_report(
        self, context: CommerceRuntimeContext, *, date_from: str, date_to: str | None = None, limit: int = 200,
    ) -> ToolResult[MarketingCampaignPeriodReport] | ToolArtifactReference:
        return await self._operation(context, {
            "date_from": date_from, "date_to": date_to, "limit": limit,
        }, MarketingCampaignPeriodRequest, MarketingCampaignPeriodReport, "/internal/v1/agent-tools/marketing/campaign-period-report")


    async def marketing_drr_report(
        self, context: CommerceRuntimeContext, *, date_from: str, date_to: str | None = None, limit: int = 50,
    ) -> ToolResult[MarketingDrrReport] | ToolArtifactReference:
        return await self._operation(context, {
            "date_from": date_from, "date_to": date_to, "limit": limit,
        }, MarketingDrrRequest, MarketingDrrReport, "/internal/v1/agent-tools/marketing/drr-report")


    async def marketing_sku_period_comparison(
        self, context: CommerceRuntimeContext, *, date_from: str | None = None, date_to: str | None = None,
        sku_list: str | None = None, product_keyword: str | None = None, period_days: int = 7, limit: int = 100,
    ) -> ToolResult[MarketingSkuComparison] | ToolArtifactReference:
        return await self._operation(context, {
            "date_from": date_from, "date_to": date_to, "sku_list": sku_list, "product_keyword": product_keyword, "period_days": period_days, "limit": limit,
        }, MarketingSkuComparisonRequest, MarketingSkuComparison, "/internal/v1/agent-tools/marketing/sku-period-comparison")


    async def marketing_campaign_efficiency(
        self, context: CommerceRuntimeContext, *, product_keyword: str | None = None, days: int = 7, limit: int = 20,
    ) -> ToolResult[MarketingEfficiency] | ToolArtifactReference:
        return await self._operation(context, {
            "product_keyword": product_keyword, "days": days, "limit": limit,
        }, MarketingEfficiencyRequest, MarketingEfficiency, "/internal/v1/agent-tools/marketing/campaign-efficiency")


    async def marketing_query_performance(
        self, context: CommerceRuntimeContext, *, product_keyword: str | None = None, days: int = 7, limit: int = 30,
    ) -> ToolResult[MarketingQueryPerformance] | ToolArtifactReference:
        return await self._operation(context, {
            "product_keyword": product_keyword, "days": days, "limit": limit,
        }, MarketingQueryRequest, MarketingQueryPerformance, "/internal/v1/agent-tools/marketing/query-performance")


    async def marketing_bid_pressure(
        self, context: CommerceRuntimeContext, *, product_keyword: str | None = None, days: int = 7, limit: int = 30,
    ) -> ToolResult[MarketingBidPressure] | ToolArtifactReference:
        return await self._operation(context, {
            "product_keyword": product_keyword, "days": days, "limit": limit,
        }, MarketingQueryRequest, MarketingBidPressure, "/internal/v1/agent-tools/marketing/bid-pressure")


    async def marketing_visibility(
        self, context: CommerceRuntimeContext, *, product_keyword: str | None = None, limit: int = 30,
    ) -> ToolResult[MarketingVisibility] | ToolArtifactReference:
        return await self._operation(context, {
            "product_keyword": product_keyword, "limit": limit,
        }, MarketingVisibilityRequest, MarketingVisibility, "/internal/v1/agent-tools/marketing/visibility")


    async def marketing_search_demand(
        self, context: CommerceRuntimeContext, *, product_keyword: str | None = None, days: int = 7, limit: int = 30,
    ) -> ToolResult[MarketingSearchDemand] | ToolArtifactReference:
        return await self._operation(context, {
            "product_keyword": product_keyword, "days": days, "limit": limit,
        }, MarketingQueryRequest, MarketingSearchDemand, "/internal/v1/agent-tools/marketing/search-demand")

    async def marketing_exact_search(self, context: CommerceRuntimeContext, *, vendor_code, date_from, date_to):
        return await self._operation(context, {
            "vendor_code": vendor_code, "date_from": date_from, "date_to": date_to,
        }, MarketingExactSearchRequest, MarketingExactSearch, "/internal/v1/agent-tools/marketing/exact-search")

    async def historical_sources(self, context: CommerceRuntimeContext, *, vendor_code, date_from, date_to):
        from commerce_agent.adapters.historical_sources import HistoricalSources
        return await self._operation(context, {
            "vendor_code": vendor_code, "date_from": date_from, "date_to": date_to,
        }, MarketingExactSearchRequest, HistoricalSources, "/internal/v1/agent-tools/marketing/historical-sources")

    async def diagnostic_history(self, context: CommerceRuntimeContext, *, kind, vendor_code, date_from, date_to):
        paths = {"stock-history": "/internal/v1/agent-tools/supply/stock-history",
                 "ad-history": "/internal/v1/agent-tools/marketing/ad-history"}
        if kind not in paths:
            raise ValueError("unknown diagnostic history")
        return await self._operation(context, {"vendor_code": vendor_code, "date_from": date_from, "date_to": date_to},
            MarketingExactSearchRequest, ExactDiagnosticHistory, paths[kind])

    async def marketing_action_matrix(
        self, context: CommerceRuntimeContext, *, product_keyword=None, days=7, limit=20,
    ) -> ToolResult[MarketingActionMatrix] | ToolArtifactReference:
        return await self._operation(context, {
            "product_keyword": product_keyword, "days": days, "limit": limit,
        }, MarketingCompositeRequest, MarketingActionMatrix, "/internal/v1/agent-tools/marketing/action-matrix")

    async def marketing_deep_diagnostic(
        self, context: CommerceRuntimeContext, *, product_keyword=None, days=7, limit=20,
    ) -> ToolResult[MarketingDeepDiagnostic] | ToolArtifactReference:
        return await self._operation(context, {
            "product_keyword": product_keyword, "days": days, "limit": limit,
        }, MarketingCompositeRequest, MarketingDeepDiagnostic, "/internal/v1/agent-tools/marketing/deep-diagnostic")

    async def finance_overview(
        self, context: CommerceRuntimeContext, *, product_keyword: str | None = None, limit: int = 100,
    ) -> ToolResult[FinanceOverview] | ToolArtifactReference:
        return await self._operation(context, {"product_keyword": product_keyword, "limit": limit},
                                     FinanceOverviewRequest, FinanceOverview, "/internal/v1/agent-tools/finance/overview")

    async def finance_marketplace_report(
        self, context: CommerceRuntimeContext, *, date_from: str, date_to: str | None = None,
    ) -> ToolResult[FinanceMarketplaceReport] | ToolArtifactReference:
        return await self._operation(context, {"date_from": date_from, "date_to": date_to},
                                     FinancePeriodRequest, FinanceMarketplaceReport,
                                     "/internal/v1/agent-tools/finance/marketplace-report")

    async def finance_marketplace_report_text(
        self, context: CommerceRuntimeContext, *, date_from: str, date_to: str | None = None,
    ) -> ToolResult[FinanceMarketplaceText] | ToolArtifactReference:
        return await self._operation(context, {"date_from": date_from, "date_to": date_to},
                                     FinancePeriodRequest, FinanceMarketplaceText,
                                     "/internal/v1/agent-tools/finance/marketplace-report-text")

    async def finance_sku_report(
        self, context: CommerceRuntimeContext, *, date_from: str, date_to: str | None = None,
        product_keyword: str | None = None, limit: int = 100, sort_by: str = "net_revenue",
    ) -> ToolResult[FinanceSkuReport] | ToolArtifactReference:
        return await self._operation(context, {
            "date_from": date_from, "date_to": date_to, "product_keyword": product_keyword,
            "limit": limit, "sort_by": sort_by,
        }, FinanceSkuRequest, FinanceSkuReport, "/internal/v1/agent-tools/finance/sku-report")

    async def finance_sku_report_text(
        self, context: CommerceRuntimeContext, *, date_from: str, date_to: str | None = None,
        product_keyword: str | None = None, limit: int = 50, sort_by: str = "net_revenue",
    ) -> ToolResult[FinanceSkuText] | ToolArtifactReference:
        return await self._operation(context, {
            "date_from": date_from, "date_to": date_to, "product_keyword": product_keyword,
            "limit": limit, "sort_by": sort_by,
        }, FinanceSkuTextRequest, FinanceSkuText, "/internal/v1/agent-tools/finance/sku-report-text")

    async def finance_sku_profit_change(
        self, context: CommerceRuntimeContext, *, current_date_from: str,
        current_date_to: str | None = None, previous_date_from: str | None = None,
        previous_date_to: str | None = None, product_keyword: str | None = None, limit: int = 20,
    ) -> ToolResult[FinanceProfitChangeReport] | ToolArtifactReference:
        return await self._operation(context, {
            "current_date_from": current_date_from, "current_date_to": current_date_to,
            "previous_date_from": previous_date_from, "previous_date_to": previous_date_to,
            "product_keyword": product_keyword, "limit": limit,
        }, FinanceProfitChangeRequest, FinanceProfitChangeReport,
            "/internal/v1/agent-tools/finance/sku-profit-change")

    async def finance_sku_profit_change_text(
        self, context: CommerceRuntimeContext, *, current_date_from: str,
        current_date_to: str | None = None, previous_date_from: str | None = None,
        previous_date_to: str | None = None, product_keyword: str | None = None, limit: int = 20,
    ) -> ToolResult[FinanceProfitChangeText] | ToolArtifactReference:
        return await self._operation(context, {
            "current_date_from": current_date_from, "current_date_to": current_date_to,
            "previous_date_from": previous_date_from, "previous_date_to": previous_date_to,
            "product_keyword": product_keyword, "limit": limit,
        }, FinanceProfitChangeRequest, FinanceProfitChangeText,
            "/internal/v1/agent-tools/finance/sku-profit-change-text")

    async def _operation(self, context, arguments, request_type, result_type, path):
        from commerce_agent.debug_privacy import development_debug_enabled
        if development_debug_enabled():
            from langchain_core.runnables import RunnableLambda
            async def invoke(params):
                return await self._operation_untraced(context, params, request_type, result_type, path)
            operation = path.removeprefix("/internal/v1/agent-tools/").replace("/", ".").replace("-", "_")
            # Credentials/context stay in the closure, never in traced inputs.
            return await RunnableLambda(invoke, name="tool." + operation).ainvoke(arguments)
        return await self._operation_untraced(context, arguments, request_type, result_type, path)

    async def _operation_untraced(self, context, arguments, request_type, result_type, path):
        try:
            params = request_type(**arguments)
            assert_secret_free(params, path="tool.request")
            assert_no_direct_identifiers(params.model_dump(mode="json"))
            assert_secret_free(context, path="tool.context")
        except (ValidationError, SensitiveStateError, PrivacyError):
            raise ToolAdapterError("invalid_request") from None
        headers = {
            "X-Commerce-Service-Token": self._settings.service_token,
            "X-Commerce-Tenant-ID": context.tenant_id,
            "X-Commerce-User-ID": context.user_id,
            "X-Commerce-Role": context.roles[0],
            "X-Request-ID": context.request_id,
        }
        for attempt in range(2):
            try:
                async with asyncio.timeout(self._settings.timeout_seconds), self._client.stream(
                    "POST", path,
                    headers=headers, json=params.model_dump(mode="json"),
                ) as response:
                    status = response.status_code
                    if status in {503, 504}:
                        if attempt == 0:
                            await asyncio.sleep(0.05)
                            continue
                        raise ToolAdapterError("unavailable")
                    if status in {401, 403}:
                        raise ToolAdapterError("denied")
                    if status == 422:
                        raise ToolAdapterError("invalid_request")
                    if status != 200:
                        raise ToolAdapterError("invalid_response")
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        body.extend(chunk)
                        if len(body) > 262_144:
                            raise ToolAdapterError("result_too_large")
                    decoded = json.loads(body)
                    if isinstance(decoded, dict) and decoded.get("schema_version") == 2:
                        result = ToolArtifactReference.model_validate(decoded)
                        operation = path.removeprefix("/internal/v1/agent-tools/").replace("/", ".").replace("-", "_")
                        if result.operation != operation or result.result_schema != result_type.__name__:
                            raise ToolAdapterError("invalid_response")
                    else:
                        result = ToolResult[result_type].model_validate(decoded)
                    assert_secret_free(result, path="tool.result")
                    assert_no_direct_identifiers(result.model_dump(mode="json"))
                    return result
            except (TimeoutError, httpx.TimeoutException, httpx.NetworkError):
                if attempt == 0:
                    await asyncio.sleep(0.05)
                    continue
                raise ToolAdapterError("unavailable") from None
            except (ValidationError, ValueError, SensitiveStateError, httpx.HTTPError):
                raise ToolAdapterError("invalid_response") from None
        raise ToolAdapterError("unavailable")

    async def hydrate(self, context: CommerceRuntimeContext, reference: ToolArtifactReference,
                      *, result_type: type[T]) -> ToolResult[T]:
        from commerce_agent.debug_privacy import development_debug_enabled
        if development_debug_enabled() and isinstance(reference, ToolArtifactReference):
            from langchain_core.runnables import RunnableLambda
            async def invoke(ref):
                return await self._hydrate_untraced(context, ref, result_type=result_type)
            return await RunnableLambda(invoke, name="tool." + reference.operation + ".hydrate").ainvoke(reference)
        return await self._hydrate_untraced(context, reference, result_type=result_type)

    async def _hydrate_untraced(self, context: CommerceRuntimeContext, reference: ToolArtifactReference,
                               *, result_type: type[T]) -> ToolResult[T]:
        """Explicit full-envelope hydration; callers must not checkpoint raw artifacts."""
        try:
            reference = ToolArtifactReference.model_validate(
                reference.model_dump() if isinstance(reference, ToolArtifactReference) else reference,
            )
            assert_secret_free(reference, path="tool.artifact")
            assert_no_direct_identifiers(reference.model_dump(mode="json"))
            assert_secret_free(context, path="tool.context")
            if not isinstance(result_type, type) or not issubclass(result_type, BaseModel) or result_type.__name__ != reference.result_schema:
                raise ValueError("unexpected artifact schema")
        except (ValidationError, ValueError, SensitiveStateError):
            raise ToolAdapterError("invalid_request") from None
        if reference.expires_at <= datetime.now(timezone.utc):
            raise ToolAdapterError("artifact_unavailable")
        headers = {
            "X-Commerce-Service-Token": self._settings.service_token,
            "X-Commerce-Tenant-ID": context.tenant_id, "X-Commerce-User-ID": context.user_id,
            "X-Commerce-Role": context.roles[0], "X-Request-ID": context.request_id,
            "Accept-Encoding": "identity",
        }
        for attempt in range(2):
            try:
                async with asyncio.timeout(self._settings.timeout_seconds), self._client.stream(
                    "GET", "/internal/v1/agent-tool-artifacts/" + str(reference.id), headers=headers,
                ) as response:
                    if response.status_code in {503, 504}:
                        if attempt == 0:
                            await asyncio.sleep(0.05)
                            continue
                        raise ToolAdapterError("unavailable")
                    if response.status_code in {401, 403}:
                        raise ToolAdapterError("denied")
                    if response.status_code in {404, 410}:
                        raise ToolAdapterError("artifact_unavailable")
                    if (response.status_code != 200
                        or response.headers.get("content-encoding", "identity") != "identity"
                        or response.headers.get("x-artifact-sha256") != reference.sha256
                        or response.headers.get("x-artifact-bytes") != str(reference.byte_size)):
                        raise ToolAdapterError("invalid_response")
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        body.extend(chunk)
                        if len(body) > reference.byte_size:
                            raise ToolAdapterError("result_too_large")
                    if len(body) != reference.byte_size or sha256(body).hexdigest() != reference.sha256:
                        raise ToolAdapterError("invalid_response")
                    result = ToolResult[result_type].model_validate_json(body)
                    assert_secret_free(result, path="tool.artifact.result")
                    assert_no_direct_identifiers(result.model_dump(mode="json"))
                    return result
            except (TimeoutError, httpx.TimeoutException, httpx.NetworkError):
                if attempt == 0:
                    await asyncio.sleep(0.05)
                    continue
                raise ToolAdapterError("unavailable") from None
            except (ValidationError, ValueError, SensitiveStateError, httpx.HTTPError):
                raise ToolAdapterError("invalid_response") from None
        raise ToolAdapterError("unavailable")
