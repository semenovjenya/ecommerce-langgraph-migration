"""Bounded historical observations, with measurement windows and replay lineage."""
from datetime import date, datetime, timedelta
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictFloat, StrictBool, model_validator, field_validator
from zoneinfo import ZoneInfo
from decimal import Decimal

Number = StrictInt | StrictFloat
FUNNEL = "sales.wb_sales_funnel_products_daily"
DETAILS = "marketing.wb_search_query_details"
TEXTS = "marketing.wb_search_query_texts"
SERP = "analytics_ai.wb_serp_runs"
DELIVERY_FIELDS = (
    "time_to_ready_days", "time_to_ready_hours", "time_to_ready_mins", "localization_percent",
    "past_time_to_ready_days", "past_time_to_ready_hours", "past_time_to_ready_mins", "past_localization_percent",
    "comparison_time_to_ready_days", "comparison_time_to_ready_hours", "comparison_time_to_ready_mins",
    "comparison_localization_percent_dynamic",
)
DELIVERY_LIMITATIONS = [
    "funnel_delivery_is_source_period_average_not_supplier_eta_or_region_promise",
    "daily_delivery_and_localization_have_no_valid_weekly_weights_do_not_average_or_sum",
    "all_zero_delivery_components_are_unknown_not_instant_delivery",
    "past_and_comparison_delivery_use_row_past_period_not_requested_previous_window",
    "missing_lineage_row_does_not_prove_observed_origin",
]


class Closed(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Window(Closed):
    date_from: date
    date_to: date


class Origin(Closed):
    replay_batch: str | None
    original_date: date | None

    @model_validator(mode="after")
    def paired(self):
        if (self.replay_batch is None) != (self.original_date is None):
            raise ValueError("incomplete lineage")
        return self


class FunnelDay(Origin):
    id: StrictInt
    nm_id: StrictInt
    day: date
    selected_period_start: date | None
    selected_period_end: date | None
    open_count: StrictInt | None
    cart_count: StrictInt | None
    order_count: StrictInt | None
    order_sum: Number | None
    buyout_count: StrictInt | None
    buyout_sum: Number | None
    cancel_count: StrictInt | None
    add_to_cart_conversion: Number | None
    cart_to_order_conversion: Number | None


class FunnelDeliveryDay(FunnelDay):
    # Required nullable fields distinguish a v2 read from old, uncollected v1.
    observation_origin: Literal["observed", "replay", "unknown"]
    past_period_start: date | None
    past_period_end: date | None
    time_to_ready_days: StrictInt | None
    time_to_ready_hours: StrictInt | None
    time_to_ready_mins: StrictInt | None
    localization_percent: StrictInt | None
    past_time_to_ready_days: StrictInt | None
    past_time_to_ready_hours: StrictInt | None
    past_time_to_ready_mins: StrictInt | None
    past_localization_percent: StrictInt | None
    comparison_time_to_ready_days: StrictInt | None
    comparison_time_to_ready_hours: StrictInt | None
    comparison_time_to_ready_mins: StrictInt | None
    comparison_localization_percent_dynamic: Number | None

    @model_validator(mode="after")
    def delivery_windows(self):
        if (self.observation_origin == "replay") != (self.replay_batch is not None):
            raise ValueError("funnel observation origin mismatch")
        if (self.past_period_start is None) != (self.past_period_end is None):
            raise ValueError("incomplete funnel past period")
        if self.past_period_start is not None and self.past_period_start > self.past_period_end:
            raise ValueError("reversed funnel past period")
        return self


class FunnelWindow(Closed):
    period: Window
    expected_days: StrictInt
    observed_days: StrictInt
    missing_days: list[date]
    complete: StrictBool
    # SQL totals are returned only for fully observed, non-null, exact-day rows.
    open_count: StrictInt | None
    cart_count: StrictInt | None
    order_count: StrictInt | None
    order_sum: Number | None


class SearchDetail(Origin):
    id: StrictInt
    nm_id: StrictInt
    snapshot_hour: datetime
    current_period_start: date
    current_period_end: date
    past_period_start: date | None
    past_period_end: date | None
    avg_position_current: Number | None
    visibility_current: Number | None
    open_card_current: Number | None
    add_to_cart_current: Number | None
    orders_current: Number | None

    @field_validator("snapshot_hour")
    @classmethod
    def aware_snapshot(cls, value):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timezone-aware observation required")
        return value


class SearchText(SearchDetail):
    text: str = Field(max_length=1000)
    top_order_by: str
    frequency_current: Number | None
    week_frequency: StrictInt | None


class SerpObservation(Origin):
    run_id: str
    source_nm_id: StrictInt
    observed_at: datetime
    query_normalized: str = Field(max_length=1000)
    region_dest: str
    source_found_any: StrictBool | None
    source_found_organic: StrictBool | None
    source_position_absolute: StrictInt | None
    source_position_organic: StrictInt | None
    max_position_scanned: StrictInt | None
    parser_status: str
    @field_validator("observed_at")
    @classmethod
    def aware_observation(cls, value):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timezone-aware observation required")
        return value


class RelatedOrigin(Closed):
    source: str
    rows: StrictInt = Field(ge=0)
    replayed_rows: StrictInt | None = Field(ge=0)
    date_from: date | None
    date_to: date | None
    original_date_from: date | None
    original_date_to: date | None


class HistoricalSources(Closed):
    version: Literal[1, 2]
    vendor_code: str
    nm_id: StrictInt | None
    identity_status: Literal["resolved", "missing", "ambiguous"]
    requested_period: Window
    previous_period: Window
    sources: list[str]
    funnel: list[FunnelDay | FunnelDeliveryDay] = Field(max_length=62)
    funnel_periods: list[FunnelWindow] = Field(min_length=2, max_length=2)
    search_details: list[SearchDetail] = Field(max_length=4000)
    search_texts: list[SearchText] = Field(max_length=4000)
    serp: list[SerpObservation] = Field(max_length=4000)
    lineage_available: StrictBool
    related_provenance: list[RelatedOrigin] = Field(max_length=7)
    limitations: list[str]

    @model_validator(mode="after")
    def coherent(self):
        if any(isinstance(r, FunnelDeliveryDay) != (self.version == 2) for r in self.funnel):
            raise ValueError("funnel fields do not match historical source version")
        a, b = self.requested_period.date_from, self.requested_period.date_to
        n = (b-a).days+1
        if not 1 <= n <= 31 or self.previous_period != Window(date_from=a-timedelta(days=n), date_to=a-timedelta(days=1)):
            raise ValueError("historical windows mismatch")
        if self.sources != [FUNNEL, DETAILS, TEXTS, SERP]:
            raise ValueError("historical source mismatch")
        rows = [*self.funnel, *self.search_details, *self.search_texts, *self.serp]
        if (self.identity_status == "resolved") != (self.nm_id is not None and self.nm_id > 0):
            raise ValueError("historical identity mismatch")
        if self.identity_status != "resolved" and (rows or self.nm_id is not None):
            raise ValueError("unresolved source has observations")
        if not self.lineage_available and any(r.replay_batch for r in rows):
            raise ValueError("unavailable lineage cannot prove replay")
        for origin in self.related_provenance:
            if origin.replayed_rows is not None and origin.replayed_rows > origin.rows:
                raise ValueError("replay counts exceed observed rows")
            if (not self.lineage_available) != (origin.replayed_rows is None):
                raise ValueError("unknown lineage counts cannot be zero")
            if origin.rows and (origin.date_from is None or origin.date_to is None or not self.previous_period.date_from <= origin.date_from <= origin.date_to <= b):
                raise ValueError("related provenance outside requested window")
        if any(getattr(r, "nm_id", getattr(r, "source_nm_id", None)) != self.nm_id for r in rows):
            raise ValueError("foreign historical identity")
        if len({r.day for r in self.funnel}) != len(self.funnel) or any(not self.previous_period.date_from <= r.day <= b for r in self.funnel):
            raise ValueError("duplicate or out of scope funnel day")
        for group in (self.search_details, self.search_texts):
            if len({r.id for r in group}) != len(group):
                raise ValueError("duplicate search row")
            if any(r.current_period_start > r.current_period_end or r.current_period_end < self.previous_period.date_from or r.current_period_start > b for r in group):
                raise ValueError("search measurement window outside scope")
        if len({r.run_id for r in self.serp}) != len(self.serp):
            raise ValueError("duplicate SERP observation")
        if any(not self.previous_period.date_from <= r.observed_at.astimezone(ZoneInfo("Europe/Moscow")).date() <= b for r in self.serp):
            raise ValueError("SERP observation outside requested windows")
        for summary, period in zip(self.funnel_periods, (self.previous_period, self.requested_period)):
            selected = [r for r in self.funnel if period.date_from <= r.day <= period.date_to]
            missing = sorted({period.date_from+timedelta(days=i) for i in range(n)} - {r.day for r in selected})
            complete = not missing and all(r.selected_period_start == r.day == r.selected_period_end
                and all(getattr(r, f) is not None for f in ("open_count", "cart_count", "order_count", "order_sum")) for r in selected)
            if summary.period != period or summary.expected_days != n or summary.observed_days != len(selected) or summary.missing_days != missing or summary.complete != complete:
                raise ValueError("funnel coverage mismatch")
            for metric in ("open_count", "cart_count", "order_count", "order_sum"):
                expected = sum(Decimal(str(getattr(r, metric))) for r in selected) if complete else None
                actual = Decimal(str(getattr(summary, metric))) if getattr(summary, metric) is not None else None
                if actual != expected:
                    raise ValueError("funnel totals mismatch")
        return self
