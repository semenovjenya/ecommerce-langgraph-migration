"""Domain tool wire contracts; mirrored across boundaries and parity-tested."""

from __future__ import annotations

from datetime import date, datetime
from typing import Generic, Literal, TypeVar, Union
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, RootModel, StrictBool, StrictFloat, StrictInt, StrictStr, field_validator, model_validator


class WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class EmptyToolRequest(WireModel):
    pass


class DatabaseTime(WireModel):
    db_current_date: date
    db_now: StrictStr

    @field_validator("db_now")
    @classmethod
    def preserve_database_timestamp(cls, value):
        # Preserve the exact native ISO string, including precision/offset.
        if "T" not in value and " " not in value:
            raise ValueError("ISO timestamp required")
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return value


class SalesVendorCodeMapRequest(WireModel):
    nm_ids: list[StrictInt | StrictStr] = Field(max_length=100)

    @field_validator("nm_ids")
    @classmethod
    def bound_identifier_size(cls, values):
        if any(isinstance(value, str) and len(value) > 64 for value in values):
            raise ValueError("identifier exceeds size limit")
        if any(isinstance(value, int) and not -(2 ** 63) <= value < 2 ** 63 for value in values):
            raise ValueError("identifier exceeds bigint bounds")
        for value in values:
            if isinstance(value, str):
                try:
                    number = int(value.strip())
                except ValueError:
                    continue  # Preserve the native invalid-text ignore behavior.
                if number >= 2 ** 63:
                    raise ValueError("identifier exceeds bigint bounds")
        return values


class SalesVendorCodeMap(RootModel[dict[StrictStr, StrictStr]]):
    @model_validator(mode="after")
    def bound_mapping(self):
        if len(self.root) > 100 or any(not key.isascii() or not key.isdigit()
                                      or int(key) <= 0 or int(key) >= 2 ** 63 for key in self.root):
            raise ValueError("invalid identity mapping")
        return self


class FormulaEntry(WireModel):
    name: StrictStr
    hypothesis: StrictStr
    shows: StrictStr
    formula: StrictStr
    grain: StrictStr
    interpretation: StrictStr
    requires_cross_agent: list[Literal["sales", "marketing", "supply", "price", "finance"]]


class FormulaCatalog(WireModel):
    catalog_name: StrictStr
    version: StrictStr
    language: StrictStr
    domain: Literal["sales", "marketing", "supply", "price", "finance"]
    usage_rule: StrictStr
    formulas: list[FormulaEntry]


class CatalogDatabase(WireModel):
    model_config = ConfigDict(extra="forbid", serialize_by_alias=True)
    schema_name: StrictStr | None = Field(default=None, alias="schema")
    schemas: list[StrictStr] | None = None
    note: StrictStr

    @model_validator(mode="after")
    def exactly_one_schema_shape(self):
        if (self.schema_name is None) == (self.schemas is None):
            raise ValueError("exactly one schema shape required")
        return self


class CatalogOutputContract(WireModel):
    agent_should_return: list[StrictStr]
    agent_should_not_return: list[StrictStr]


class CatalogActions(WireModel):
    workflow: list[StrictStr]
    output_contract: CatalogOutputContract


class CatalogRouting(WireModel):
    intent: StrictStr
    use_first: list[StrictStr]
    join: StrictStr
    output: list[StrictStr]
    warning: StrictStr | None = None


class CatalogTable(WireModel):
    name: StrictStr
    role: StrictStr
    use: StrictStr
    grain: StrictStr
    keys: list[StrictStr]
    empty_now: StrictBool
    metrics: list[StrictStr] | None = None
    rows: StrictInt | None = None
    date_range: StrictStr | None = None
    joins: list[StrictStr] | None = None
    technical_table: StrictBool | None = None
    dedupe_key: list[StrictStr] | None = None


class CatalogGap(WireModel):
    metric: StrictStr
    reason: StrictStr
    impact: StrictStr


class TableCatalog(WireModel):
    catalog_name: StrictStr
    version: StrictStr
    language: StrictStr
    database: CatalogDatabase
    purpose: StrictStr
    agent_actions: CatalogActions
    routing: list[CatalogRouting]
    tables: list[CatalogTable]
    data_gaps: list[StrictStr | CatalogGap]


class SalesSnapshotRequest(WireModel):
    product_keyword: StrictStr | None = Field(default=None, max_length=128)


class Period(WireModel):
    date_from: date
    date_to: date


class SalesComparison(WireModel):
    period: StrictStr
    current_period: Period
    previous_period: Period
    current_sales_units: StrictInt
    previous_sales_units: StrictInt
    sales_units_delta: StrictInt
    sales_units_delta_pct: StrictInt | StrictFloat | None
    current_revenue_rub: StrictInt | StrictFloat
    previous_revenue_rub: StrictInt | StrictFloat
    revenue_delta_rub: StrictInt | StrictFloat
    revenue_delta_pct: StrictInt | StrictFloat | None


class ReportTable(WireModel):
    title: StrictStr
    columns: list[StrictStr]
    rows: list[list[StrictStr | StrictInt | StrictFloat | None]]


class SalesSnapshot(WireModel):
    report_date: date | None
    focus_keyword: StrictStr | None
    source_table: Literal["sales.wb_supplier_sales"]
    required_fields: list[StrictStr]
    calculations: dict[StrictStr, StrictStr]
    rows: list[SalesComparison] = Field(max_length=3)
    report_table: ReportTable
    markdown_table: StrictStr



class MarketingSnapshotRequest(WireModel):
    product_keyword: StrictStr | None = Field(default=None, max_length=128)


class MarketingComparison(WireModel):
    period: StrictStr
    current_period: Period
    previous_period: Period
    current_campaign_count: StrictInt
    previous_campaign_count: StrictInt
    current_promoted_nm_count: StrictInt
    previous_promoted_nm_count: StrictInt
    current_spend_rub: StrictInt | StrictFloat
    previous_spend_rub: StrictInt | StrictFloat
    spend_delta_rub: StrictInt | StrictFloat
    spend_delta_pct: StrictInt | StrictFloat | None
    current_views: StrictInt
    previous_views: StrictInt
    views_delta: StrictInt
    views_delta_pct: StrictInt | StrictFloat | None
    current_clicks: StrictInt
    previous_clicks: StrictInt
    clicks_delta: StrictInt
    clicks_delta_pct: StrictInt | StrictFloat | None
    current_cpc_rub: StrictInt | StrictFloat | None
    previous_cpc_rub: StrictInt | StrictFloat | None
    current_ad_orders: StrictInt
    previous_ad_orders: StrictInt
    ad_orders_delta: StrictInt
    ad_orders_delta_pct: StrictInt | StrictFloat | None
    current_promoted_sku_revenue_rub: StrictInt | StrictFloat
    previous_promoted_sku_revenue_rub: StrictInt | StrictFloat
    promoted_sku_revenue_delta_rub: StrictInt | StrictFloat
    promoted_sku_revenue_delta_pct: StrictInt | StrictFloat | None
    current_drr_pct: StrictInt | StrictFloat | None
    previous_drr_pct: StrictInt | StrictFloat | None
    drr_delta_pp: StrictInt | StrictFloat | None


class MarketingSnapshot(WireModel):
    report_date: date | None
    focus_keyword: StrictStr | None
    source_tables: list[Literal[
        "marketing.wb_marketing_campaign_stats_daily",
        "marketing.wb_marketing_campaign_nms",
        "sales.wb_supplier_sales",
        "sales.wb_sales_funnel_products_daily",
    ]] = Field(min_length=4, max_length=4)
    required_fields: list[StrictStr]
    calculations: dict[StrictStr, StrictStr]
    interpretation_note: StrictStr | None = None
    rows: list[MarketingComparison] = Field(max_length=3)
    report_table: ReportTable
    markdown_table: StrictStr



Number = Union[StrictInt, StrictFloat]

class MarketingOverviewRequest(WireModel):
    product_keyword: StrictStr | None = Field(default=None, max_length=2048)
    limit: StrictInt = Field(default=5, ge=1, le=10)


class MarketingCampaignPeriodRequest(WireModel):
    date_from: StrictStr = Field(min_length=1, max_length=64)
    date_to: StrictStr | None = Field(default=None, max_length=32)
    limit: StrictInt = Field(default=200, ge=1, le=1000)


class MarketingDrrRequest(MarketingCampaignPeriodRequest):
    limit: StrictInt = Field(default=50, ge=1, le=100)


class MarketingSkuComparisonRequest(WireModel):
    date_from: StrictStr | None = Field(default=None, max_length=64)
    date_to: StrictStr | None = Field(default=None, max_length=32)
    sku_list: StrictStr | None = Field(default=None, max_length=2048)
    product_keyword: StrictStr | None = Field(default=None, max_length=2048)
    period_days: StrictInt = 7
    limit: StrictInt = Field(default=100, ge=1, le=200)

    @model_validator(mode="after")
    def validate_implicit_period(self):
        # Native explicit date ranges override this field and can exceed 31.
        if not self.date_from and not 1 <= self.period_days <= 31:
            raise ValueError("invalid implicit period")
        return self


class MarketingEfficiencyRequest(WireModel):
    product_keyword: StrictStr | None = Field(default=None, max_length=2048)
    days: StrictInt = Field(default=7, ge=1, le=30)
    limit: StrictInt = Field(default=20, ge=1, le=50)


class MarketingQueryRequest(MarketingEfficiencyRequest):
    limit: StrictInt = Field(default=30, ge=1, le=100)


class MarketingVisibilityRequest(WireModel):
    product_keyword: StrictStr | None = Field(default=None, max_length=2048)
    limit: StrictInt = Field(default=30, ge=1, le=100)


class MarketingOverviewTotals(WireModel):
    stat_date: date | None
    campaign_count_7d: StrictInt | None
    total_views_7d: Number | None
    total_clicks_7d: Number | None
    total_orders_7d: Number | None
    total_spend_7d: Number | None
    avg_ctr: Number | None
    avg_cr: Number | None


class MarketingBudget(WireModel):
    snapshot_date: date | None
    advert_count: StrictInt | None
    total_budget: Number | None
    total_cash: Number | None


class MarketingCampaignRow(WireModel):
    stat_date: date | None
    advert_id: StrictInt | None
    campaign_name: StrictStr | None
    subject_name: StrictStr | None
    views: Number | None
    clicks: Number | None
    orders: Number | None
    sum_spend: Number | None
    ctr: Number | None = None
    cpc: Number | None
    cr: Number | None
    active_nms_count: StrictInt | None = None


class MarketingOverview(WireModel):
    focus_keyword: StrictStr | None
    totals: MarketingOverviewTotals | None
    budget: MarketingBudget | None
    top_spend_campaigns: list[MarketingCampaignRow] = Field(max_length=10)
    inefficient_campaigns: list[MarketingCampaignRow] = Field(max_length=10)
    campaign_nm_rows: list[MarketingCampaignRow] = Field(max_length=10)


class MarketingAliasRow(WireModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, serialize_by_alias=True)


class MarketingCampaignPeriodRow(MarketingAliasRow):
    advert_id: StrictInt | None = Field(alias="ID кампании")
    campaign_type_label: StrictStr | None = Field(alias="Кампания, Тип")
    started_date: date | None = Field(alias="Дата начала")
    status: StrictStr | None = Field(alias="Статус")
    sku_list: StrictStr | None = Field(alias="Артикулы в кампании")
    sale_price_list: StrictStr | None = Field(alias="Цена продажи без СПП, ₽")
    impressions: Number | None = Field(alias="Показы")
    clicks: Number | None = Field(alias="Клики")
    ctr_pct: Number | None = Field(alias="Кликабельность, %")
    cpc_rub: Number | None = Field(alias="Цена клика, ₽")
    cr_pct: Number | None = Field(alias="Конверсия клика в заказ, %")
    add_to_cart: Number | None = Field(alias="Все добавления в корзину")
    ordered_items: Number | None = Field(alias="Все заказанные товары, шт")
    spend_rub: Number | None = Field(alias="Затраты, ₽")
    spend_share_pct: Number | None = Field(alias="Доля затрат (%)")


class MarketingCampaignPeriodReport(WireModel):
    date_from: date
    date_to: date
    source_tables: list[StrictStr]
    row_count: StrictInt
    rows: list[MarketingCampaignPeriodRow] = Field(max_length=1000)


class MarketingDrrTotals(WireModel):
    campaign_count: StrictInt | None
    promoted_nm_count: StrictInt | None
    marketing_spend: Number | None
    views: Number | None
    clicks: Number | None
    ad_orders: Number | None
    store_revenue: Number | None
    store_sales: StrictInt | None
    promoted_sku_revenue: Number | None
    promoted_sku_sales: StrictInt | None
    promoted_revenue_share_pct: Number | None
    promoted_sales_share_pct: Number | None
    estimated_drr_promoted_sku_pct: Number | None
    estimated_drr_store_pct: Number | None


class MarketingDrrCampaign(WireModel):
    advert_id: StrictInt | None
    campaign_name: StrictStr | None
    spend_rub: Number | None
    views: Number | None
    clicks: Number | None
    ad_orders: Number | None
    cpc_rub: Number | None
    nm_ids: StrictStr | None


class MarketingDrrSku(WireModel):
    nm_id: StrictInt | None
    vendor_code: StrictStr | None
    subject_name: StrictStr | None
    revenue: Number | None
    sales: StrictInt | None


class MarketingDrrReport(WireModel):
    date_from: date
    date_to: date
    spend_source: StrictStr
    sales_source: StrictStr
    interpretation_note: StrictStr
    totals: MarketingDrrTotals | None
    campaign_rows: list[MarketingDrrCampaign] = Field(max_length=100)
    promoted_sku_rows: list[MarketingDrrSku] = Field(max_length=100)


class MarketingSkuComparisonRow(WireModel):
    current_date_from: date | None
    current_date_to: date | None
    previous_date_from: date | None
    previous_date_to: date | None
    nm_id: StrictInt | None
    vendor_code: StrictStr | None
    subject_name: StrictStr | None
    current_campaigns: StrictInt | None
    previous_campaigns: StrictInt | None
    attribution_quality: Literal["estimated_allocated_from_campaign", "campaign_single_sku"]
    current_impressions: Number | None
    previous_impressions: Number | None
    impressions_delta: Number | None
    impressions_delta_pct: Number | None
    current_clicks: Number | None
    previous_clicks: Number | None
    clicks_delta: Number | None
    clicks_delta_pct: Number | None
    current_ctr_pct: Number | None
    previous_ctr_pct: Number | None
    current_cpc_rub: Number | None
    previous_cpc_rub: Number | None
    current_ad_orders: Number | None
    previous_ad_orders: Number | None
    ad_orders_delta: Number | None
    ad_orders_delta_pct: Number | None
    current_spend_rub: Number | None
    previous_spend_rub: Number | None
    spend_delta_rub: Number | None
    spend_delta_pct: Number | None
    current_sales_revenue_rub: Number | None
    previous_sales_revenue_rub: Number | None
    current_sales_units: StrictInt | None
    previous_sales_units: StrictInt | None
    current_estimated_drr_pct: Number | None
    previous_estimated_drr_pct: Number | None


class MarketingSkuComparison(WireModel):
    date_from: date | None
    date_to: date | None
    previous_date_from: date | None
    previous_date_to: date | None
    period_days: StrictInt
    focus_sku_list: list[StrictStr]
    focus_keyword: StrictStr | None
    metric_grain: StrictStr
    allocation_note: StrictStr
    source_tables: list[StrictStr]
    row_count: StrictInt
    rows: list[MarketingSkuComparisonRow] = Field(max_length=200)


class MarketingEfficiencyRow(WireModel):
    report_date: date | None
    advert_id: StrictInt | None
    campaign_name: StrictStr | None
    top_subject_name: StrictStr | None
    brand: StrictStr | None
    views: Number | None
    clicks: Number | None
    orders: Number | None
    sum_spend: Number | None
    ctr: Number | None
    cpc: Number | None
    cr: Number | None
    status: Literal["negative_outlier", "inefficient", "strong_leader", "working", "conditional", "investigate"]


class MarketingEfficiencyTotals(WireModel):
    report_date: date | None
    campaign_count: StrictInt | None
    total_views: Number | None
    total_clicks: Number | None
    total_orders: Number | None
    total_spend: Number | None


class MarketingConcentration(WireModel):
    top_campaign_spend: Number | None
    total_spend: Number | None
    top_campaign_spend_share_pct: Number | None


class MarketingEfficiency(WireModel):
    report_date: date | None
    period_days: StrictInt
    focus_keyword: StrictStr | None
    totals: MarketingEfficiencyTotals
    campaign_rows: list[MarketingEfficiencyRow] = Field(max_length=50)
    waste_rows: list[MarketingEfficiencyRow] = Field(max_length=50)
    scale_rows: list[MarketingEfficiencyRow] = Field(max_length=50)
    budget_concentration: MarketingConcentration


class MarketingSearchIdentity(WireModel):
    report_date: date | None
    nm_id: StrictInt | None
    vendor_code: StrictStr | None
    subject_name: StrictStr | None
    brand_name: StrictStr | None


class MarketingQueryRow(MarketingSearchIdentity):
    norm_query: StrictStr | None
    views: Number | None
    clicks: Number | None
    orders: Number | None
    spend: Number | None
    ctr: Number | None
    cpc: Number | None
    avg_pos: Number | None


class MarketingQueryPerformance(WireModel):
    report_date: date | None
    period_days: StrictInt
    focus_keyword: StrictStr | None
    top_query_rows: list[MarketingQueryRow] = Field(max_length=100)
    query_waste_rows: list[MarketingQueryRow] = Field(max_length=100)
    query_scale_rows: list[MarketingQueryRow] = Field(max_length=100)
    query_position_risk_rows: list[MarketingQueryRow] = Field(max_length=100)


class MarketingBidPressure(WireModel):
    report_date: date | None
    period_days: StrictInt
    focus_keyword: StrictStr | None
    # Native capability is absent. New populated shapes need explicit review.
    bid_rows: list[StrictStr] = Field(max_length=0)
    overbid_rows: list[StrictStr] = Field(max_length=0)
    underbid_rows: list[StrictStr] = Field(max_length=0)
    bid_efficiency_rows: list[StrictStr] = Field(max_length=0)


class MarketingVisibilityRow(WireModel):
    snapshot_date: date | None
    request_key: StrictStr | None
    nm_id: StrictInt | None
    name: StrictStr | None
    vendor_code: StrictStr | None
    subject_name: StrictStr | None
    brand_name: StrictStr | None
    avg_position_current: Number | None
    avg_position_dynamics: Number | None
    visibility_current: Number | None
    visibility_dynamics: Number | None
    open_card_current: Number | None
    open_card_dynamics: Number | None
    add_to_cart_current: Number | None
    add_to_cart_dynamics: Number | None
    orders_current: Number | None
    orders_dynamics: Number | None
    cart_to_order_current: Number | None
    cart_to_order_dynamics: Number | None


class MarketingVisibilityOverview(WireModel):
    snapshot_date: date | None
    total_products: StrictInt | None
    supplier_rating_current: Number | None
    position_average_current: Number | None
    position_average_dynamics: Number | None
    visibility_current: Number | None
    visibility_dynamics: Number | None
    open_card_current: Number | None
    open_card_dynamics: Number | None


class MarketingVisibility(WireModel):
    report_date: date | None
    focus_keyword: StrictStr | None
    overview: MarketingVisibilityOverview | None
    visibility_rows: list[MarketingVisibilityRow] = Field(max_length=100)
    position_loss_rows: list[MarketingVisibilityRow] = Field(max_length=100)
    open_card_gap_rows: list[MarketingVisibilityRow] = Field(max_length=100)
    post_click_conversion_rows: list[MarketingVisibilityRow] = Field(max_length=100)


class MarketingDemandRow(MarketingSearchIdentity):
    query_text: StrictStr | None
    frequency_sum: Number | None
    avg_position: Number | None
    orders_count: Number | None
    prior_orders_count: Number | None
    orders_delta: Number | None


class MarketingSearchDemand(WireModel):
    report_date: date | None
    period_days: StrictInt
    focus_keyword: StrictStr | None
    search_text_rows: list[MarketingDemandRow] = Field(max_length=100)
    growing_query_rows: list[MarketingDemandRow] = Field(max_length=100)
    high_frequency_low_order_rows: list[MarketingDemandRow] = Field(max_length=100)
    order_driving_query_rows: list[MarketingDemandRow] = Field(max_length=100)



class MarketingCompositeRequest(MarketingEfficiencyRequest):
    pass


class MarketingSourceFreshness(WireModel):
    report_date: date | None
    reference_date: date | None
    lag_days: StrictInt | None
    max_lag_days: StrictInt
    status: Literal["fresh", "stale", "future", "unknown", "unknown_reference"]


class MarketingMartFreshness(WireModel):
    campaign_mart: MarketingSourceFreshness
    query_bands_mart: MarketingSourceFreshness
    forecast_ready_mart: MarketingSourceFreshness
    query_demand_mart: MarketingSourceFreshness
    query_share_mart: MarketingSourceFreshness
    query_forecast_mart: MarketingSourceFreshness
    campaign_demand_mart: MarketingSourceFreshness


class MarketingMatrixFreshness(WireModel):
    reference_date: date | None
    stale_sources: list[StrictStr]
    invalid_sources: list[StrictStr]
    sources: MarketingMartFreshness
    legacy_campaign: MarketingSourceFreshness | None
    legacy_query: MarketingSourceFreshness | None
    interpretation: StrictStr


class MarketingActionProposal(WireModel):
    entity_type: Literal["кампания", "поисковый запрос", "артикул"]
    entity_id: StrictStr | StrictInt | None
    entity_name: StrictStr | None
    status: StrictStr
    action_mode: Literal["hold", "reduce", "needs_review", "test", "scale_review"]
    reason: StrictStr
    evidence: list[StrictStr]
    suggested_action: StrictStr
    marketing_source_status: StrictStr
    demand_source_status: StrictStr
    blockers: list[StrictStr]
    # Native mart, legacy and SKU variants omit different keys. Never fill
    # absent fields in the public payload; exclude_unset preserves that shape.
    decision_hint: StrictStr | None = None
    confidence: StrictStr | None = None
    safe_growth_pct: Number | None = None
    marginal_quality_coef: Number | None = None
    winner_query_share: Number | None = None
    waste_query_share: Number | None = None
    is_cut_candidate: StrictBool | None = None
    is_scale_candidate: StrictBool | None = None
    advert_id: StrictInt | None = None
    norm_query: StrictStr | None = None
    query_band: StrictStr | None = None
    nm_id: StrictInt | None = None
    vendor_code: StrictStr | None = None


class MarketingSupportingSources(WireModel):
    campaign_mart: StrictStr
    query_bands_mart: StrictStr
    forecast_ready_mart: StrictStr
    query_demand_mart: StrictStr
    query_share_mart: StrictStr
    query_forecast_mart: StrictStr
    campaign_demand_mart: StrictStr
    legacy_demand_fallback: Literal["used", "not_used"]
    legacy_visibility_fallback: Literal["used", "not_used"]


class MarketingActionSummary(WireModel):
    campaign_scale_count: StrictInt
    campaign_stop_count: StrictInt
    query_scale_count: StrictInt
    query_reduce_count: StrictInt
    visibility_investigate_count: StrictInt


class MarketingActionMatrix(WireModel):
    report_date: date | None
    period_days: StrictInt
    focus_keyword: StrictStr | None
    marketing_source_status: StrictStr
    demand_source_status: StrictStr
    source_freshness: MarketingMatrixFreshness
    campaign_actions: list[MarketingActionProposal] = Field(max_length=50)
    query_actions: list[MarketingActionProposal] = Field(max_length=50)
    sku_actions: list[MarketingActionProposal] = Field(max_length=50)
    supporting_sources: MarketingSupportingSources
    summary: MarketingActionSummary


class MarketingDiagnosticSummary(WireModel):
    top_campaign_scale_candidates: StrictInt
    top_campaign_waste_candidates: StrictInt
    query_scale_candidates: StrictInt
    query_waste_candidates: StrictInt
    overbid_candidates: StrictInt
    underbid_candidates: StrictInt
    visibility_risk_rows: StrictInt
    search_demand_growth_rows: StrictInt


class MarketingDiagnosticGap(WireModel):
    metric: StrictStr
    reason: StrictStr
    impact: StrictStr


class MarketingDeepDiagnostic(WireModel):
    report_date: date | None
    period_days: StrictInt
    focus_keyword: StrictStr | None
    summary: MarketingDiagnosticSummary
    campaign_efficiency: MarketingEfficiency
    query_performance: MarketingQueryPerformance
    bid_pressure: MarketingBidPressure
    visibility: MarketingVisibility
    search_demand: MarketingSearchDemand
    action_matrix: MarketingActionMatrix
    data_gaps: list[MarketingDiagnosticGap]


class FinancePeriodRequest(WireModel):
    date_from: StrictStr = Field(min_length=1, max_length=64)
    date_to: StrictStr | None = Field(default=None, max_length=32)


class FinanceOverviewRequest(WireModel):
    product_keyword: StrictStr | None = Field(default=None, max_length=2048)
    limit: StrictInt = Field(default=100, ge=1, le=100)


class FinanceSkuRequest(FinancePeriodRequest):
    product_keyword: StrictStr | None = Field(default=None, max_length=2048)
    limit: StrictInt = Field(default=100, ge=1, le=500)
    sort_by: Literal["net_revenue", "margin_rub", "profitability_pct"] = "net_revenue"


class FinanceSkuTextRequest(FinanceSkuRequest):
    limit: StrictInt = Field(default=50, ge=1, le=500)


class FinanceProfitChangeRequest(WireModel):
    current_date_from: StrictStr = Field(min_length=1, max_length=64)
    current_date_to: StrictStr | None = Field(default=None, max_length=32)
    previous_date_from: StrictStr | None = Field(default=None, max_length=64)
    previous_date_to: StrictStr | None = Field(default=None, max_length=32)
    product_keyword: StrictStr | None = Field(default=None, max_length=2048)
    limit: StrictInt = Field(default=20, ge=1, le=100)


class FinanceWatchRow(WireModel):
    nm_id: StrictInt
    vendor_code: StrictStr | None
    brand_name: StrictStr | None
    subject_name: StrictStr | None
    quantity_sum: Number | None = None
    retail_amount_sum: Number | None
    payout_sum: Number | None
    acquiring_fee_sum: Number | None = None
    delivery_rub_sum: Number | None = None
    storage_fee_sum: Number | None = None
    penalty_sum: Number | None = None
    payout_margin_before_cogs_pct: Number | None = None


class FinanceUnitRow(FinanceWatchRow):
    snapshot_date: date | None
    commission_sum: Number | None
    fee_burden_pct: Number | None


class FinanceMarginRow(FinanceWatchRow):
    report_date: date | None


class FinanceOverview(WireModel):
    focus_keyword: StrictStr | None
    unit_economics_watchlist: list[FinanceUnitRow] = Field(max_length=100)
    margin_watchlist: list[FinanceMarginRow] = Field(max_length=100)
    fee_pressure_rows: list[FinanceWatchRow] = Field(max_length=100)
    guardrails_scope: list[StrictStr]


class FinanceMetric(WireModel):
    label: StrictStr
    value: Number
    percent: Number | None = None


class FinanceMarketplaceTotals(WireModel):
    revenue: Number | None = None
    returns: Number | None = None
    sales_cogs: Number | None = None
    return_cogs: Number | None = None
    net_cogs: Number | None = None
    total_marketplace_expenses: Number | None = None
    commission_and_acquiring: Number | None = None
    marketing: Number | None = None
    logistics: Number | None = None
    acceptance: Number | None = None
    storage_fee: Number | None = None
    cross_docking: Number | None = None
    penalties_and_withholdings: Number | None = None
    misc_accruals: Number | None = None
    total_to_pay: Number | None = None
    margin_rub: Number | None = None
    margin_percent: Number | None = None
    missing_cost_row_count: StrictInt | None = None
    missing_cost_sales_quantity: Number | None = None
    missing_cost_return_quantity: Number | None = None


class FinanceMarketplaceReport(WireModel):
    date_from: date
    date_to: date
    source_table: StrictStr
    metrics: list[FinanceMetric] = Field(max_length=32)
    totals: FinanceMarketplaceTotals


class FinanceAliasRow(WireModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, serialize_by_alias=True)


class FinanceSkuRow(FinanceAliasRow):
    nm_id: StrictInt | None
    vendor_code: StrictStr | None = Field(alias="Артикул")
    unit_cost_rub: Number | None = Field(alias="Себестоимость")
    delivered_qty: Number | None = Field(alias="Доставлено, шт")
    net_revenue: Number | None = Field(alias="Выручка, руб (после возвратов)")
    return_qty: Number | None = Field(alias="Возвраты, шт")
    returns_rub: Number | None = Field(alias="Возвраты, руб")
    total_marketplace_expenses: Number | None = Field(alias="ИТОГО Расходы МП на артикул")
    commission_and_acquiring: Number | None = Field(alias="КОМИССИИ и ЭКВАЙРИНГ")
    marketing: Number | None = Field(alias="МАРКЕТИНГ")
    logistics: Number | None = Field(alias="ЛОГИСТИКА")
    return_cogs: Number | None = Field(alias="Себестоимость возвратов")
    cost_share_before_returns_pct: Number | None = Field(alias="Доля себестоимости в выручке до возвратов")
    abc_revenue_class: StrictStr | None = Field(alias="ABC по выручке")
    margin_rub: Number | None = Field(alias="Маржа, руб")
    profitability_pct: Number | None = Field(alias="Рентабельность, %")


class FinanceSkuTotals(WireModel):
    total_sku_count: StrictInt
    returned_sku_count: StrictInt
    missing_cost_sku_count: StrictInt


class FinanceSkuReport(WireModel):
    date_from: date
    date_to: date
    product_keyword: StrictStr | None
    sort_by: StrictStr
    source_tables: list[StrictStr]
    columns: list[StrictStr]
    rows: list[FinanceSkuRow] = Field(max_length=500)
    totals: FinanceSkuTotals


class FinanceProfitChangeRow(FinanceAliasRow):
    nm_id: StrictInt | None
    vendor_code: StrictStr | None = Field(alias="Артикул")
    current_margin_rub: Number = Field(alias="Маржа тек. период, руб")
    previous_margin_rub: Number = Field(alias="Маржа пред. период, руб")
    delta_margin_rub: Number = Field(alias="Δ маржи, руб")
    delta_margin_pct: Number | None = Field(alias="Δ маржи, %")
    current_revenue_rub: Number | None = Field(alias="Выручка тек. период, руб")
    previous_revenue_rub: Number | None = Field(alias="Выручка пред. период, руб")
    current_profitability_pct: Number | None = Field(alias="Рентабельность тек. период, %")
    driver_role: StrictStr = Field(alias="Роль")


class FinanceProfitChangeTotals(WireModel):
    current_total_sku_count: StrictInt
    previous_total_sku_count: StrictInt
    returned_sku_count: StrictInt
    missing_cost_sku_count: StrictInt


class FinanceProfitChangeReport(WireModel):
    current_period_start: date
    current_period_end: date
    previous_period_start: date
    previous_period_end: date
    product_keyword: StrictStr | None
    source_tables: list[StrictStr]
    columns: list[StrictStr]
    rows: list[FinanceProfitChangeRow] = Field(max_length=100)
    totals: FinanceProfitChangeTotals


class FinanceMarketplaceText(RootModel[StrictStr]):
    pass


class FinanceSkuText(RootModel[StrictStr]):
    pass


class FinanceProfitChangeText(RootModel[StrictStr]):
    pass

class PriceOverviewRequest(WireModel):
    product_keyword: StrictStr | None = Field(default=None, max_length=2048)
    limit: StrictInt = Field(default=20, ge=1, le=100)


class PriceIdentityRequest(PriceOverviewRequest):
    nm_id: StrictStr | None = Field(default=None, max_length=64)
    vendor_code: StrictStr | None = Field(default=None, max_length=128)


class PriceContextRequest(PriceIdentityRequest):
    include_evidence: StrictBool = False


class PriceDetailsRequest(PriceIdentityRequest):
    limit: StrictInt = Field(default=50, ge=1, le=100)


class PriceTimestampModel(WireModel):
    @field_validator("seller_last_synced_at", "analysis_last_synced_at",
                     "observed_at", "completed_at", check_fields=False)
    @classmethod
    def preserve_iso_timestamp(cls, value):
        if value is not None:
            if "T" not in value and " " not in value:
                raise ValueError("ISO timestamp required")
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        return value


class PriceContextRow(PriceTimestampModel):
    nm_id: StrictInt
    vendor_code: StrictStr | None = None
    brand: StrictStr | None = None
    subject_name: StrictStr | None = None
    title: StrictStr | None = None
    product_url: StrictStr | None = None
    seller_snapshot_date: date | None = None
    analysis_run_date: date | None = None
    source_snapshot_date: date | None = None
    context_snapshot_date: date | None = None
    context_as_of_date: date | None = None
    seller_price_regular: Number | None = None
    seller_price_report_promo: Number | None = None
    seller_controlled_price: Number | None = None
    seller_discount_percent: Number | None = None
    seller_club_discount_percent: Number | None = None
    seller_club_discounted_price: Number | None = None
    customer_price_display: Number | None = None
    customer_price_wallet: Number | None = None
    customer_price_effective: Number | None = None
    platform_discount_percent: Number | None = None
    wallet_discount_percent: Number | None = None
    report_to_customer_discount_percent: Number | None = None
    competitor_selection_mode: StrictStr | None = None
    broad_competitor_count: StrictInt | None = None
    broad_competitor_price_min: Number | None = None
    broad_market_price_median: Number | None = None
    broad_competitor_price_max: Number | None = None
    strict_competitor_count: StrictInt | None = None
    strict_competitor_price_min: Number | None = None
    strict_competitor_price_median: Number | None = None
    strict_competitor_price_max: Number | None = None
    market_candidate_count: StrictInt | None = None
    market_price_min: Number | None = None
    market_price_max: Number | None = None
    llm_checked_count: StrictInt | None = None
    llm_kept_count: StrictInt | None = None
    vision_checked_count: StrictInt | None = None
    vision_absolute_count: StrictInt | None = None
    vision_close_count: StrictInt | None = None
    vision_similar_count: StrictInt | None = None
    vision_not_analog_count: StrictInt | None = None
    used_market_fallback: StrictBool | None = None
    is_price_outlier: StrictBool | None = None
    analysis_status: StrictStr | None = None
    query_count: StrictInt | None = None
    source_found_query_count: StrictInt | None = None
    source_found_query_share: Number | None = None
    weighted_organic_position: Number | None = None
    organic_competitors_before_source: Number | None = None
    price_before_median: Number | None = None
    source_to_before_median_price_ratio: Number | None = None
    analysis_id: StrictInt | None = None
    seller_last_synced_at: StrictStr | None = None
    analysis_last_synced_at: StrictStr | None = None
    customer_price_basis: StrictStr | None = None
    competitor_quality: StrictStr | None = None
    freshness_status: StrictStr | None = None
    seller_age_days: StrictInt | None = None
    analysis_age_days: StrictInt | None = None
    context_age_days: StrictInt | None = None
    price_status: StrictStr | None = None
    customer_vs_strict_median_gap_percent: Number | None = None
    contract_version: StrictStr | None = None
    evidence_refs: list[StrictStr] = Field(max_length=3)


class PriceSemantics(WireModel):
    seller_controlled_price: StrictStr
    customer_price_display: StrictStr
    customer_price_wallet: StrictStr
    customer_price_effective: StrictStr
    broad_market_price_median: StrictStr
    strict_competitor_price_median: StrictStr


class PriceSerpEvidence(PriceTimestampModel):
    source_nm_id: StrictInt
    run_id: StrictStr
    batch_id: StrictStr | None
    query: StrictStr | None
    observed_at: StrictStr | None
    completed_at: StrictStr | None
    source_found_any: StrictBool | None
    source_position_organic: Number | None
    evidence_ref: StrictStr


class PriceContext(WireModel):
    contract_version: Literal["price-context-v1"]
    source_view: StrictStr
    grain: StrictStr
    price_semantics: PriceSemantics
    quality_rule: StrictStr
    row_count: StrictInt = Field(ge=0)
    rows: list[PriceContextRow] = Field(max_length=100)
    evidence: list[PriceSerpEvidence] = Field(max_length=500)


class PriceDecisionRow(WireModel):
    snapshot_date: date | None = None
    nm_id: StrictInt
    vendor_code: StrictStr | None = None
    brand_name: StrictStr | None = None
    subject_name: StrictStr | None = None
    source_price_promo: Number | None = None
    source_price_regular: Number | None = None
    competitor_count: StrictInt | None = None
    competitor_price_min: Number | None = None
    competitor_price_median: Number | None = None
    competitor_price_max: Number | None = None
    source_vs_median_gap_pct: Number | None = None
    source_vs_median_gap_rub: Number | None = None
    price_status: StrictStr | None = None
    elasticity_to_apply: Number | None = None
    elasticity_source: StrictStr | None = None
    marketing_decision_hint: StrictStr | None = None
    marketing_safe_growth_pct: Number | None = None
    marketing_marginal_quality_coef: Number | None = None
    marketing_winner_query_share_7d: Number | None = None
    marketing_waste_query_share_7d: Number | None = None
    marketing_confidence: StrictStr | Number | None = None
    scenario_id: StrictInt | StrictStr | None = None
    sales_model: StrictStr | None = None
    net_profit_rub_current: Number | None = None
    margin_pct_current: Number | None = None
    cost_price_rub: Number | None = None
    ad_percent: Number | None = None
    tax_percent: Number | None = None
    expected_outbound_per_buyout_rub: Number | None = None
    expected_return_logistics_rub: Number | None = None
    storage_total_rub: Number | None = None
    other_expenses_rub: Number | None = None
    stock_qty: Number | None = None
    baseline_orders_7d: Number | None = None
    baseline_revenue_gross_7d: Number | None = None
    baseline_revenue_net_7d: Number | None = None
    baseline_buyouts_7d: Number | None = None
    baseline_adv_spend_7d: Number | None = None
    source_price_segment_min: Number | None = None
    source_price_segment_max: Number | None = None
    candidate_price_min: Number | None = None
    candidate_price_max: Number | None = None
    candidate_step_rub: Number | None = None
    recommended_price: Number | None = None
    recommended_price_delta_abs: Number | None = None
    recommended_price_delta_pct: Number | None = None
    expected_orders_7d: Number | None = None
    expected_net_profit_7d: Number | None = None
    expected_margin_pct: Number | None = None
    decision_status: StrictStr | None = None
    decision_confidence: StrictStr | Number | None = None
    decision_reason_code: StrictStr | None = None
    market_ceiling_enabled: StrictBool | None = None
    current_net_profit_7d: Number | None = None
    price_ai_run_date: date | None = None
    price_context_as_of_date: date | None = None
    decision_lag_days: StrictInt | None = None
    decision_freshness: StrictStr | None = None


class PriceDecisionOverview(WireModel):
    focus_keyword: StrictStr | None
    source_table: StrictStr
    freshness_rule: StrictStr
    row_count: StrictInt = Field(ge=0)
    rows: list[PriceDecisionRow] = Field(max_length=100)


class PriceCompetitorRow(WireModel):
    run_date: date
    nm_id: StrictInt
    vendor_code: StrictStr | None
    brand_name: StrictStr | None
    subject_name: StrictStr | None
    title: StrictStr | None
    source_price_promo: Number | None
    source_price_regular: Number | None
    source_price_segment_min: Number | None
    source_price_segment_max: Number | None
    competitor_count: StrictInt
    cheaper_competitor_count: StrictInt
    pricier_competitor_count: StrictInt
    within_price_segment_count: StrictInt
    same_subject_count: StrictInt
    photo_ai_competitor_count: StrictInt
    photo_ai_unchecked_count: StrictInt
    competitor_price_min: Number | None
    competitor_price_median: Number | None
    competitor_price_max: Number | None
    competitor_price_avg: Number | None
    keyword_count: StrictInt
    raw_rows: StrictInt
    cheaper_competitor_share_pct: Number | None
    pricier_competitor_share_pct: Number | None
    source_vs_median_gap_pct: Number | None
    source_vs_median_gap_rub: Number | None
    price_status: StrictStr


class PriceCompetitorOverview(WireModel):
    run_date: date | None
    source_table: StrictStr
    sku_lookup_table: StrictStr
    current_price_table: StrictStr | None = None
    dedupe_key: StrictStr
    focus_keyword: StrictStr | None
    row_count: StrictInt = Field(ge=0)
    rows: list[PriceCompetitorRow] = Field(max_length=100)


class PriceCompetitorDetailRow(WireModel):
    run_date: date
    keyword: StrictStr | None
    source_nm_id: StrictInt
    vendor_code: StrictStr | None
    source_brand: StrictStr | None
    source_subject_name: StrictStr | None
    source_title: StrictStr | None
    source_price_regular: Number | None
    source_price_promo: Number | None
    source_price_segment_min: Number | None
    source_price_segment_max: Number | None
    competitor_nm_id: StrictInt
    competitor_brand: StrictStr | None
    competitor_supplier_name: StrictStr | None
    competitor_subject_name: StrictStr | None
    competitor_title: StrictStr | None
    competitor_product_url: StrictStr | None
    competitor_price_regular: Number | None
    competitor_price_promo: Number | None
    competitor_discount_percent: Number | None
    price_delta_abs: Number | None
    price_delta_percent: Number | None
    within_price_segment: StrictBool | None
    same_subject: StrictBool | None
    same_color: StrictBool | None
    photo_ai_is_competitor: StrictBool | None
    photo_ai_confidence: Number | None
    competitor_rating: Number | None
    competitor_feedbacks: StrictInt | None
    catalog_position_absolute: StrictInt | None


class PriceCompetitorDetails(WireModel):
    run_date: date | None
    source_table: StrictStr
    sku_lookup_table: StrictStr
    dedupe_key: StrictStr
    nm_id: StrictInt | None
    vendor_code: StrictStr | None
    focus_keyword: StrictStr | None
    row_count: StrictInt = Field(ge=0)
    rows: list[PriceCompetitorDetailRow] = Field(max_length=100)

class SalesOverviewRequest(WireModel):
    product_keyword: StrictStr | None = Field(default=None, max_length=2048)
    limit: StrictInt = Field(default=100, ge=1, le=100)


class DailyPriceSalesRequest(WireModel):
    product_keyword: StrictStr = Field(min_length=1, max_length=128)
    days: StrictInt = Field(default=14, ge=1, le=31)


class SalesRevenueChangeRequest(WireModel):
    date_from: StrictStr | None = Field(default=None, max_length=64)
    date_to: StrictStr | None = Field(default=None, max_length=32)
    product_keyword: StrictStr | None = Field(default=None, max_length=2048)
    limit: StrictInt = Field(default=20, ge=1, le=100)


class SalesTotals(WireModel):
    sales_date: date | None
    sales_1d_units: StrictInt
    sales_7d_units: StrictInt
    sold_sku_7d: StrictInt
    revenue_1d: Number | None
    revenue_7d: Number | None
    avg_price_with_disc_7d: Number | None


class SalesProductIdentity(WireModel):
    nm_id: StrictInt | None
    vendor_code: StrictStr | None


class TopRevenueRow(SalesProductIdentity):
    report_date: date | None
    brand_name: StrictStr | None
    subject_name: StrictStr | None
    sales_7d_units: StrictInt
    revenue_7d: Number
    avg_price_with_disc_7d: Number | None


class ConversionWatchRow(SalesProductIdentity):
    report_date: date
    brand_name: StrictStr | None
    subject_name: StrictStr | None
    open_count: StrictInt | None
    cart_count: StrictInt | None
    order_count: StrictInt | None
    add_to_cart_conversion: Number | None
    cart_to_order_conversion: Number | None
    buyout_percent: Number | None


class RawSalesRow(SalesProductIdentity):
    sales_date: date | None
    brand: StrictStr | None
    subject: StrictStr | None
    sales_count: StrictInt
    revenue: Number


class SalesOverview(WireModel):
    focus_keyword: StrictStr | None
    totals: SalesTotals | None
    top_revenue_rows: list[TopRevenueRow] = Field(max_length=100)
    conversion_watchlist: list[ConversionWatchRow] = Field(max_length=100)
    sales_raw_rows: list[RawSalesRow] = Field(max_length=100)


class DailyPriceSalesRow(SalesProductIdentity):
    report_date: date
    base_price_rub: Number | None
    seller_discount_pct: Number | None
    seller_effective_price_rub: Number | None
    observed_transaction_seller_price_avg_rub: Number | None
    observed_buyer_price_avg_rub: Number | None
    observed_spp_pct_avg: Number | None
    price_basis: Literal["seller_snapshot_exact_day", "observed_transaction_average", "missing"]
    non_cancelled_order_units: StrictInt | None
    realized_sales_units: StrictInt | None
    realized_for_pay_rub: Number | None
    open_count: StrictInt | None
    cart_count: StrictInt | None
    funnel_order_count: StrictInt | None
    add_to_cart_conversion: Number | None
    cart_to_order_conversion: Number | None
    buyout_percent: Number | None
    business_day_complete: StrictBool | None
    marketing_day_complete: StrictBool | None
    sales_day_complete: StrictBool | None


class DailyPriceSalesComparison(WireModel):
    focus_keyword: StrictStr
    days: StrictInt = Field(ge=1, le=31)
    source_tables: list[StrictStr] = Field(min_length=1, max_length=8)
    row_count: StrictInt = Field(ge=0)
    rows: list[DailyPriceSalesRow] = Field(max_length=3100)
    causal_valid: Literal[False]


class RevenueChangeRow(SalesProductIdentity):
    current_period_start: date
    current_period_end: date
    previous_period_start: date
    previous_period_end: date
    brand_name: StrictStr | None
    subject_name: StrictStr | None
    current_revenue: Number
    previous_revenue: Number
    delta_revenue: Number
    delta_revenue_pct: Number | None
    current_sales: StrictInt
    previous_sales: StrictInt
    delta_sales: StrictInt
    driver_role: Literal["new_in_current", "lost_in_current", "growth_driver", "decline_driver", "stable"]


class SalesRevenueChange(WireModel):
    date_from: date | None
    date_to: date | None
    current_period_start: date | None
    current_period_end: date | None
    previous_period_start: date | None
    previous_period_end: date | None
    period_days: StrictInt = Field(ge=1)
    focus_keyword: StrictStr | None
    source_table: StrictStr
    row_count: StrictInt = Field(ge=0)
    rows: list[RevenueChangeRow] = Field(max_length=100)


class SupplyOverviewRequest(WireModel):
    product_keyword: StrictStr | None = Field(default=None, max_length=2048)
    limit: StrictInt = Field(default=100, ge=1, le=100)


class InventoryValueRequest(SupplyOverviewRequest):
    pass


class InventoryTurnoverRequest(WireModel):
    date_from: StrictStr = Field(min_length=1, max_length=64)
    date_to: StrictStr | None = Field(default=None, max_length=32)
    vendor_codes: StrictStr | None = Field(default=None, max_length=2048)
    limit: StrictInt = Field(default=100, ge=1, le=1000)


class SupplySimulation(WireModel):
    enabled: StrictBool
    policy: Literal["latest_known_as_current"]
    source_snapshot_date: date | None
    effective_status: Literal["simulated_current", "observed_latest_known"]


class SupplyRow(WireModel):
    report_date: date | None = None
    nm_id: StrictInt | None = None
    vendor_code: StrictStr | None = None
    brand_name: StrictStr | None = None
    subject_name: StrictStr | None = None
    stock_qty: Number | None = None
    stock_qty_full: Number | None = None
    unavailable_stock_qty: Number | None = None
    sales_1d_units: Number | None = None
    sales_7d_units: Number | None = None
    avg_daily_sales_7d: Number | None = None
    days_of_stock: Number | None
    total_stock_days_for_audit: Number | None = None
    reorder_flag: Literal["critical", "warning", "ok", "overstock"] | None = None
    reorder_qty_recommended: Number | None = None
    in_way_to_client: Number | None = None
    in_way_from_client: Number | None = None
    supplier_audit_snapshot_date: date | None = None
    supplier_inbound_qty: Number | None = None
    supplier_goods_rows: Number | None = None
    supplier_eta_date: StrictStr | None = None
    coverage_band: StrictStr
    availability_status: StrictStr


class SupplyTotals(WireModel):
    report_date: date | None = None
    alert_sku_count: StrictInt | None = None
    critical_count: StrictInt | None = None
    warning_count: StrictInt | None = None
    reorder_qty_total: Number | None = None


class SupplyPipelineRow(WireModel):
    snapshot_date: date | None
    supply_id: StrictInt | None
    supply_date: StrictStr | None
    status_id: StrictInt | None
    warehouse_name: StrictStr | None
    quantity: Number | None
    ready_for_sale_quantity: Number | None
    accepted_quantity: Number | None
    unloading_quantity: Number | None
    sku_count: StrictInt
    goods_quantity: Number | None


class StockSnapshotRow(WireModel):
    snapshot_date: date | None
    nm_id: StrictInt | None
    supplier_article: StrictStr | None
    brand: StrictStr | None
    subject: StrictStr | None
    quantity: Number | None
    quantity_full: Number | None
    in_way_to_client: Number | None
    in_way_from_client: Number | None


class StockMatch(WireModel):
    snapshot_date: date | None
    nm_id: StrictInt | None
    vendor_code: StrictStr | None
    available_stock_qty: Number | None
    full_stock_qty: Number | None
    unavailable_stock_qty: Number | None
    in_way_to_client: Number | None
    in_way_from_client: Number | None
    supplier_audit_snapshot_date: date | None
    supplier_inbound_qty: Number | None
    supplier_goods_rows: Number | None
    supplier_eta_date: StrictStr | None


class StockLookup(WireModel):
    query: StrictStr | None
    match_count: StrictInt
    aggregation_method: StrictStr
    matches: list[StockMatch] = Field(max_length=100)
    status: Literal["found", "found_multiple", "partial", "no_data", "not_requested", "multiple_matches"]
    requested_skus: list[StrictStr] | None = None
    matched_queries: list[StrictStr] | None = None
    missing_queries: list[StrictStr] | None = None
    snapshot_date: date | None = None
    nm_id: StrictInt | None = None
    vendor_code: StrictStr | None = None
    available_stock_qty: Number | None
    full_stock_qty: Number | None = None
    unavailable_stock_qty: Number | None = None
    in_way_to_client: Number | None = None
    in_way_from_client: Number | None = None
    data_mode: Literal["simulation_latest_known", "observed_latest_known"]
    simulation: SupplySimulation


class SupplyOverview(WireModel):
    focus_keyword: StrictStr | None
    data_mode: Literal["simulation_latest_known", "observed_latest_known"]
    simulation: SupplySimulation
    sku_stock_lookup: StockLookup
    totals: SupplyTotals | None
    rows: list[SupplyRow] = Field(max_length=100)
    critical_reorder_rows: list[SupplyRow] = Field(max_length=100)
    overstock_rows: list[SupplyRow] = Field(max_length=100)
    pipeline_rows: list[SupplyPipelineRow] = Field(max_length=100)
    stock_snapshot_rows: list[StockSnapshotRow] = Field(max_length=100)
    calculation_methods: dict[StrictStr, StrictStr]


class InventoryValueRow(WireModel):
    report_date: date | None
    nm_id: StrictInt | None
    vendor_code: StrictStr | None
    brand_name: StrictStr | None = None
    subject_name: StrictStr | None = None
    available_stock_qty: Number | None
    unit_cost_rub: Number | None
    current_stock_value_rub: Number | None
    valuation_status: Literal["missing_cogs", "confirmed"]


class InventoryValueTotals(WireModel):
    confirmed_stock_value_rub: Number
    valued_sku_count: StrictInt
    missing_cogs_sku_count: StrictInt


class InventoryValue(WireModel):
    report_date: date | None
    focus_keyword: StrictStr | None
    rows: list[InventoryValueRow] = Field(max_length=100)
    totals: InventoryValueTotals
    calculation_method: StrictStr


class TurnoverMetrics(WireModel):
    stock_snapshot_days: StrictInt | None = None
    avg_stock_qty: Number | None = None
    avg_stock_qty_full: Number | None = None
    sales_units: Number | None = None
    sales_revenue: Number | None = None
    turnover_ratio: Number | None = None
    days_on_hand: Number | None
    latest_snapshot_date: date | None = None
    current_stock_qty: Number | None = None
    current_stock_qty_full: Number | None = None
    current_in_way_to_client: Number | None = None
    current_in_way_from_client: Number | None = None
    turnover_band: StrictStr
    availability_status: StrictStr


class TurnoverTotals(TurnoverMetrics):
    date_from: date
    date_to: date
    period_days: StrictInt


class TurnoverRow(TurnoverMetrics):
    nm_id: StrictInt | None
    vendor_code: StrictStr | None
    brand_name: StrictStr | None = None
    subject_name: StrictStr | None = None
    sales_7d_units: Number | None = None
    min_stock_qty_full: Number | None = None
    max_stock_qty_full: Number | None = None
    sku_conclusion: StrictStr


class TurnoverMatrixRow(WireModel):
    sku: StrictStr | None
    sales_units: Number | None
    sales_7d_units: Number | None
    avg_stock_qty: Number | None
    avg_stock_qty_full: Number | None
    current_stock_qty: Number | None
    current_stock_qty_full: Number | None
    current_in_way_to_client: Number | None
    current_in_way_from_client: Number | None
    turnover_ratio: Number | None
    days_on_hand: Number | None
    turnover_band: StrictStr
    availability_status: StrictStr
    conclusion: StrictStr


class AlternativeMethod(WireModel):
    name: Literal["boundary_snapshot_average", "movement_reconstruction", "latest_stock_velocity_proxy"]
    meaning: StrictStr
    when_to_use: StrictStr


class SourceCoverage(WireModel):
    stock_rows: StrictInt = Field(ge=0)
    sales_rows: StrictInt = Field(ge=0)


class InventoryTurnover(WireModel):
    date_from: date
    date_to: date
    vendor_codes: list[StrictStr]
    method: StrictStr
    coverage_formula: StrictStr
    coverage_band_policy: dict[StrictStr, StrictStr]
    calculation_status: Literal["exact", "needs_alternative_search"]
    exact_available: StrictBool
    limitation: StrictStr | None
    alternative_methods: list[AlternativeMethod]
    totals: TurnoverTotals | None
    rows: list[TurnoverRow] = Field(max_length=1000)
    sku_metric_matrix: list[TurnoverMatrixRow] = Field(max_length=1000)
    sku_metric_matrix_contract: StrictStr
    source_coverage: SourceCoverage



class InventoryTurnoverAlternativesRequest(WireModel):
    date_from: StrictStr = Field(min_length=10, max_length=10)
    date_to: StrictStr = Field(min_length=10, max_length=10)
    vendor_codes: StrictStr | None = Field(default=None, max_length=2048)
    limit: StrictInt = Field(default=100, ge=1, le=100)
    boundary_tolerance_days: StrictInt = Field(default=0, ge=0, le=31)

    @model_validator(mode="after")
    def exact_bounded_scope(self):
        start, end = date.fromisoformat(self.date_from), date.fromisoformat(self.date_to)
        if start.isoformat() != self.date_from or end.isoformat() != self.date_to or not 1 <= (end-start).days+1 <= 366:
            raise ValueError("exact bounded ISO period required")
        if self.vendor_codes is not None:
            tokens = self.vendor_codes.split(",")
            if (not 1 <= len(tokens) <= 20 or any(not token.strip() or len(token.strip()) > 64
                    or ";" in token or any(ord(c) < 32 for c in token) for token in tokens)
                    or len({token.strip() for token in tokens}) != len(tokens)):
                raise ValueError("distinct bounded canonical vendor codes required")
        return self


TurnoverAlternativeMethod = Literal["boundary_snapshot_average", "movement_reconstruction", "latest_stock_velocity_proxy"]


class TurnoverStockObservation(WireModel):
    snapshot_date: date
    distance_days: StrictInt = Field(ge=0)
    observed_sellable_stock_qty: Number | None = Field(ge=0)
    source_rows: StrictInt = Field(ge=1)
    quantity_rows: StrictInt = Field(ge=0)
    covered_warehouses: list[StrictStr] = Field(max_length=1000)
    snapshot_completeness: Literal["unverified"]

    @model_validator(mode="after")
    def native_observation_counts(self):
        if self.quantity_rows > self.source_rows or (self.quantity_rows != self.source_rows and self.observed_sellable_stock_qty is not None):
            raise ValueError("stock quantities require observed source rows")
        if any(len(value) > 256 for value in self.covered_warehouses):
            raise ValueError("warehouse label too large")
        return self


class TurnoverAlternativeAttempt(WireModel):
    method: TurnoverAlternativeMethod
    status: Literal["estimate_available", "unavailable"]
    reason: StrictStr = Field(min_length=1, max_length=128)
    boundary_average_stock_qty_estimate: Number | None
    boundary_stock_coverage_days_estimate: Number | None
    observed_stock_coverage_days_estimate: Number | None
    used_stock_dates: list[date] = Field(max_length=2)
    caveats: list[StrictStr] = Field(max_length=20)

    @model_validator(mode="after")
    def no_unavailable_or_unverified_exact_number(self):
        values = (self.boundary_average_stock_qty_estimate, self.boundary_stock_coverage_days_estimate,
                  self.observed_stock_coverage_days_estimate)
        if self.status == "unavailable" and any(value is not None for value in values):
            raise ValueError("unavailable method cannot carry calculated values")
        # This version has no registered complete snapshot or movement journal.
        if self.method != "latest_stock_velocity_proxy":
            if self.status != "unavailable" or any(value is not None for value in values):
                raise ValueError("unregistered complete source cannot yield an estimate")
        elif self.boundary_average_stock_qty_estimate is not None or self.boundary_stock_coverage_days_estimate is not None:
            raise ValueError("observed-stock proxy is not a period average")
        if self.status == "estimate_available":
            if self.observed_stock_coverage_days_estimate is None or self.observed_stock_coverage_days_estimate < 0 or not self.used_stock_dates or not self.caveats:
                raise ValueError("estimate requires native date and explicit limitations")
        return self


class TurnoverAlternativeRow(WireModel):
    vendor_code: StrictStr = Field(min_length=1, max_length=64)
    nm_id: StrictInt = Field(gt=0)
    sales_units: StrictInt | None = Field(ge=0)
    sales_source_rows: StrictInt = Field(ge=0)
    sales_observed_from: date | None
    sales_observed_to: date | None
    boundary_start: TurnoverStockObservation | None
    boundary_end: TurnoverStockObservation | None
    latest_stock: TurnoverStockObservation | None
    method_attempts: list[TurnoverAlternativeAttempt] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def methods_and_proxy_prerequisites(self):
        if [item.method for item in self.method_attempts] != [
                "boundary_snapshot_average", "movement_reconstruction", "latest_stock_velocity_proxy"]:
            raise ValueError("exact alternative method order required")
        if self.sales_source_rows == 0 and any(value is not None for value in (
                self.sales_units, self.sales_observed_from, self.sales_observed_to)):
            raise ValueError("missing sales source is unknown, not zero")
        if self.sales_source_rows > 0 and (self.sales_units is None or self.sales_units > self.sales_source_rows
                or self.sales_observed_from is None or self.sales_observed_to is None):
            raise ValueError("sales counts and dates must describe actual source rows")
        if (self.sales_observed_from is None) != (self.sales_observed_to is None):
            raise ValueError("incomplete sales source window")
        if self.sales_observed_from and self.sales_observed_from > self.sales_observed_to:
            raise ValueError("reversed sales source window")
        proxy = self.method_attempts[-1]
        if proxy.status == "estimate_available":
            stock = self.latest_stock
            if (not self.sales_units or self.sales_source_rows == 0 or not self.sales_observed_from
                    or stock is None or stock.quantity_rows == 0
                    or stock.observed_sellable_stock_qty is None or stock.observed_sellable_stock_qty < 0
                    or proxy.used_stock_dates != [stock.snapshot_date]):
                raise ValueError("proxy prerequisites missing")
        return self


class InventoryTurnoverAlternatives(WireModel):
    policy_version: Literal[1]
    date_from: date
    date_to: date
    period_days: StrictInt = Field(ge=1, le=366)
    database_date: date
    vendor_codes: list[StrictStr] = Field(max_length=20)
    boundary_tolerance_days: StrictInt = Field(ge=0, le=31)
    boundary_tie_policy: Literal["nearest_then_earlier"]
    exact_available: Literal[False]
    calculation_status: Literal["estimates_available", "unavailable"]
    source_completeness: Literal["unverified"]
    matched_sku_count: StrictInt = Field(ge=0)
    returned_sku_count: StrictInt = Field(ge=0, le=100)
    row_limit_reached: StrictBool
    unreturned_vendor_codes: list[StrictStr] = Field(max_length=20)
    rows: list[TurnoverAlternativeRow] = Field(max_length=100)
    warnings: list[StrictStr] = Field(max_length=100)

    @field_validator("policy_version", mode="before")
    @classmethod
    def strict_policy_version(cls, value):
        if type(value) is not int or value != 1:
            raise ValueError("unsupported alternative policy version")
        return value

    @field_validator("exact_available", mode="before")
    @classmethod
    def never_exact(cls, value):
        if value is not False:
            raise ValueError("alternatives are never exact")
        return value

    @model_validator(mode="after")
    def scope_and_method_availability(self):
        if self.date_to >= self.database_date or self.period_days != (self.date_to-self.date_from).days+1:
            raise ValueError("requested period is not a bounded past period")
        if self.returned_sku_count != len(self.rows) or self.matched_sku_count < self.returned_sku_count:
            raise ValueError("alternative row counts mismatch")
        if self.row_limit_reached != (self.matched_sku_count > self.returned_sku_count):
            raise ValueError("alternative cap coverage mismatch")
        identities = [(row.vendor_code, row.nm_id) for row in self.rows]
        if len(identities) != len(set(identities)):
            raise ValueError("duplicate alternative source identity")
        if len({row.vendor_code for row in self.rows}) != len(self.rows):
            raise ValueError("ambiguous canonical alternative identity")
        if len(self.vendor_codes) != len(set(self.vendor_codes)) or not set(self.unreturned_vendor_codes) <= set(self.vendor_codes):
            raise ValueError("invalid requested alternative identities")
        if self.vendor_codes and any(row.vendor_code not in self.vendor_codes for row in self.rows):
            raise ValueError("alternative row outside canonical request")
        for row in self.rows:
            if row.sales_observed_from and not self.date_from <= row.sales_observed_from <= row.sales_observed_to <= self.date_to:
                raise ValueError("sales observations outside requested period")
            for observation, boundary in ((row.boundary_start,self.date_from),(row.boundary_end,self.date_to)):
                if observation and (observation.snapshot_date > self.database_date
                        or observation.distance_days != abs((observation.snapshot_date-boundary).days)
                        or observation.distance_days > self.boundary_tolerance_days):
                    raise ValueError("stock boundary outside explicit tolerance")
            if row.latest_stock and row.latest_stock.snapshot_date > self.database_date:
                raise ValueError("future stock observation")
        available = any(attempt.status == "estimate_available" for row in self.rows for attempt in row.method_attempts)
        if (self.calculation_status == "estimates_available") != available:
            raise ValueError("alternative availability mismatch")
        return self


class Evidence(WireModel):
    id: StrictStr
    source: StrictStr
    grain: Literal["portfolio", "product_filter"]
    period: Period | None


class Freshness(WireModel):
    as_of: date | None
    observed_at: datetime
    status: Literal["available", "missing"]
    # Observation time is not proof that the underlying source is fresh.


T = TypeVar("T")


class ToolArtifactReference(WireModel):
    schema_version: Literal[2] = 2
    kind: Literal["tool_artifact"] = "tool_artifact"
    id: UUID
    operation: StrictStr = Field(pattern=r"^(sales|marketing|supply|price|finance|shared)\.[a-z_]+$", max_length=100)
    result_schema: StrictStr = Field(pattern=r"^[A-Z][A-Za-z0-9]+$", max_length=100)
    payload_schema_version: Literal[1] = 1
    sha256: StrictStr = Field(pattern=r"^[a-f0-9]{64}$")
    byte_size: StrictInt = Field(ge=1, le=10_485_760)
    expires_at: datetime

    @field_validator("expires_at")
    @classmethod
    def aware_expiry(cls, value):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timezone-aware expiry required")
        return value


class ToolResult(WireModel, Generic[T]):
    schema_version: Literal[1] = 1
    data: T
    evidence: list[Evidence]
    freshness: Freshness
    source: StrictStr
    period: Period | None
    warnings: list[StrictStr]

class MarketingExactSearchRequest(WireModel):
    vendor_code: StrictStr = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    date_from: date
    date_to: date

    @field_validator("date_from", "date_to", mode="before")
    @classmethod
    def exact_iso_date(cls, value):
        if type(value) is date:
            return value
        if not isinstance(value, str) or len(value) != 10 or date.fromisoformat(value).isoformat() != value:
            raise ValueError("ISO date required")
        return value

    @model_validator(mode="after")
    def bounded_window(self):
        if not 0 <= (self.date_to - self.date_from).days <= 30:
            raise ValueError("exact search window must contain one to thirty-one days")
        return self


class DiagnosticStockDay(WireModel):
    day: date
    nm_id: StrictInt = Field(gt=0)
    stock_qty: StrictInt | StrictFloat
    warehouse_count: StrictInt = Field(ge=0)


class DiagnosticAdDay(WireModel):
    day: date
    nm_id: StrictInt = Field(gt=0)
    advert_id: StrictInt = Field(gt=0)
    stats_evidence_origin: Literal["raw_fullstats_row", "complete_day_absence", "missing"]
    ad_metrics_complete: StrictBool
    views: StrictInt | None
    clicks: StrictInt | None
    ad_orders: StrictInt | None
    ad_spend_rub: StrictInt | StrictFloat | None


class ExactDiagnosticHistory(WireModel):
    version: Literal[1]
    kind: Literal["stock-history", "ad-history"]
    vendor_code: StrictStr
    nm_id: StrictInt | None
    identity_status: Literal["resolved", "missing", "ambiguous"]
    requested_period: Period
    previous_period: Period
    source: Literal["agent_marts.v_agent_supply_sku_coverage_daily", "agent_marts.v_agent_ad_campaign_sku_daily"]
    rows: list[DiagnosticStockDay | DiagnosticAdDay] = Field(max_length=4000)
    limitations: list[StrictStr]

    @model_validator(mode="after")
    def coherent_history(self):
        from datetime import timedelta
        start, end = self.requested_period.date_from, self.requested_period.date_to
        n = (end-start).days + 1
        if not 1 <= n <= 31 or self.previous_period.date_from != start-timedelta(days=n) or self.previous_period.date_to != start-timedelta(days=1):
            raise ValueError("invalid diagnostic windows")
        if (self.identity_status == "resolved") != (self.nm_id is not None and self.nm_id > 0):
            raise ValueError("invalid diagnostic identity")
        if self.identity_status != "resolved" and (self.nm_id is not None or self.rows):
            raise ValueError("unresolved diagnostic identity has id or rows")
        stock = self.kind == "stock-history"
        source = "agent_marts.v_agent_supply_sku_coverage_daily" if stock else "agent_marts.v_agent_ad_campaign_sku_daily"
        if self.source != source:
            raise ValueError("diagnostic source mismatch")
        keys = []
        for r in self.rows:
            if not isinstance(r, DiagnosticStockDay if stock else DiagnosticAdDay) or r.nm_id != self.nm_id or not self.previous_period.date_from <= r.day <= end:
                raise ValueError("diagnostic row outside source scope")
            keys.append((r.day, None if stock else r.advert_id))
            if not stock:
                metrics = [r.views, r.clicks, r.ad_orders, r.ad_spend_rub]
                if r.stats_evidence_origin == "missing" and (r.ad_metrics_complete or any(v is not None for v in metrics)):
                    raise ValueError("missing ad rows cannot become observed metrics")
                if r.stats_evidence_origin == "complete_day_absence" and (not r.ad_metrics_complete or any(v != 0 for v in metrics)):
                    raise ValueError("watermarked absence must be complete zero")
                if r.ad_metrics_complete and any(v is None for v in metrics):
                    raise ValueError("complete ad row lacks metrics")
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate diagnostic grain")
        return self


class MarketingSearchDay(WireModel):
    day: date
    rows: StrictInt
    distinct_queries: StrictInt
    repeated_query_rows: StrictInt
    frequency: StrictInt | StrictFloat | None
    orders: StrictInt | StrictFloat | None


class MarketingExactSearch(WireModel):
    version: Literal[1]
    vendor_code: StrictStr
    nm_id: StrictInt | None
    identity_status: Literal["resolved", "missing", "ambiguous"]
    requested_period: Period
    previous_period: Period
    source_rows: StrictInt
    target_rows: StrictInt
    current_rows: StrictInt
    previous_rows: StrictInt
    source_date_from: date | None
    source_date_to: date | None
    target_date_from: date | None
    target_date_to: date | None
    current_days: list[MarketingSearchDay] = Field(max_length=31)
    previous_days: list[MarketingSearchDay] = Field(max_length=31)
    coverage: Literal["identity_missing", "identity_ambiguous", "source_empty", "target_absent",
                      "window_absent", "partial_coverage", "available"]
    limitations: list[StrictStr]

    @model_validator(mode="after")
    def coherent_coverage(self):
        from datetime import timedelta
        start, end = self.requested_period.date_from, self.requested_period.date_to
        n = (end-start).days + 1
        if not 1 <= n <= 31 or self.previous_period.date_to != start-timedelta(days=1) or self.previous_period.date_from != start-timedelta(days=n):
            raise ValueError("inconsistent exact search windows")
        if min(self.source_rows, self.target_rows, self.current_rows, self.previous_rows) < 0 or self.target_rows > self.source_rows or self.current_rows+self.previous_rows > self.target_rows:
            raise ValueError("inconsistent search coverage counts")
        if (self.identity_status == "resolved") != (self.nm_id is not None and self.nm_id > 0):
            raise ValueError("inconsistent search identity")
        for days, period, count in ((self.current_days, self.requested_period, self.current_rows), (self.previous_days, self.previous_period, self.previous_rows)):
            if len({d.day for d in days}) != len(days) or sum(d.rows for d in days) != count:
                raise ValueError("inconsistent search daily coverage")
            if any(not period.date_from <= d.day <= period.date_to or d.rows < 1 or d.distinct_queries < 0 or d.distinct_queries > d.rows or d.repeated_query_rows != d.rows-d.distinct_queries for d in days):
                raise ValueError("invalid search day")
        for count, lo, hi in ((self.source_rows, self.source_date_from, self.source_date_to), (self.target_rows, self.target_date_from, self.target_date_to)):
            if (count == 0 and (lo is not None or hi is not None)) or (count > 0 and (lo is None or hi is None or lo > hi)):
                raise ValueError("inconsistent observed window")
        if self.target_rows and (self.source_date_from > self.target_date_from or self.source_date_to < self.target_date_to):
            raise ValueError("target outside source coverage")
        if self.identity_status != "resolved" and self.target_rows:
            raise ValueError("unresolved identity cannot have target data")
        if any(self.target_date_from is None or not self.target_date_from <= d.day <= self.target_date_to
               for d in self.current_days + self.previous_days):
            raise ValueError("daily rows outside observed coverage")
        expected = ("identity_" + self.identity_status if self.identity_status != "resolved" else
            "source_empty" if not self.source_rows else "target_absent" if not self.target_rows else
            "window_absent" if not self.current_rows else
            "partial_coverage" if len(self.current_days) != n or len(self.previous_days) != n else "available")
        if self.coverage != expected:
            raise ValueError("coverage label inconsistent with observations")
        return self


