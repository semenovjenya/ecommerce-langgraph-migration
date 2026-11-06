"""Exact-period Price evidence; no latest-snapshot fallback."""
from __future__ import annotations
from datetime import date, timedelta
from typing import Literal
from pydantic import Field, StrictInt, model_validator
from commerce_agent.adapters.contracts import WireModel, Number, Period, MarketingExactSearchRequest


class PriceHistoryRequest(MarketingExactSearchRequest):
    pass


class SellerPriceDay(WireModel):
    snapshot_date: date
    nm_id: StrictInt
    price: Number | None
    discounted_price: Number | None
    discount: Number | None
    club_discount: Number | None
    club_discounted_price: Number | None


class CustomerPriceDay(WireModel):
    snapshot_date: date
    source_nm_id: StrictInt
    source_price_site_display: Number | None
    source_price_site_wallet: Number | None
    source_price_site_effective: Number | None
    source_platform_discount_percent: Number | None
    source_wallet_discount_percent: Number | None
    market_price_min: Number | None
    market_price_median: Number | None
    market_price_max: Number | None


class HistoricalCompetitor(WireModel):
    run_date: date
    source_snapshot_date: date | None
    source_nm_id: StrictInt
    competitor_nm_id: StrictInt
    source_price_regular: Number | None
    source_price_promo: Number | None
    competitor_price_regular: Number | None
    competitor_price_promo: Number | None


class PriceComparison(WireModel):
    layer: Literal["seller", "customer", "competitors"]
    metric: str
    competitor_nm_id: StrictInt | None = None
    direction: Literal["higher", "lower", "unchanged", "mixed", "insufficient_data"]
    previous_values: list[Number]
    current_values: list[Number]


def compare_observed_prices(seller, customer, competitors, start, end):
    previous = start - (end - start + timedelta(days=1))
    expected = (end - start).days + 1
    comparisons = []
    groups = [
        ("seller", metric, None, seller, "snapshot_date")
        for metric in ("price", "discounted_price", "club_discounted_price")
    ] + [
        ("customer", metric, None, customer, "snapshot_date")
        for metric in ("source_price_site_display", "source_price_site_wallet", "source_price_site_effective", "market_price_median")
    ] + [
        ("competitors", metric, identity, [row for row in competitors if row["competitor_nm_id"] == identity], "run_date")
        for identity in sorted({row["competitor_nm_id"] for row in competitors})
        for metric in ("competitor_price_regular", "competitor_price_promo")
    ]
    for layer, metric, identity, rows, day_key in groups:
        sides = []
        complete = True
        for first, last in ((previous, start - timedelta(days=1)), (start, end)):
            selected = [row for row in rows if str(first) <= str(row[day_key]) <= str(last)]
            days = {str(row[day_key]) for row in selected}
            values = sorted({row[metric] for row in selected if row[metric] is not None})
            complete = complete and len(days) == expected and len(selected) == expected and all(
                row[metric] is not None for row in selected)
            sides.append(values)
        before, after = sides
        direction = ("insufficient_data" if not complete or not before or not after else
            "higher" if min(after) > max(before) else "lower" if max(after) < min(before) else
            "unchanged" if len(before) == len(after) == 1 and before == after else "mixed")
        comparisons.append(dict(layer=layer, metric=metric, competitor_nm_id=identity,
            direction=direction, previous_values=before, current_values=after))
    return comparisons


class PriceHistory(WireModel):
    version: Literal[1] = 1
    vendor_code: str
    nm_id: StrictInt | None
    identity_status: Literal["resolved", "missing", "ambiguous"]
    requested_period: Period
    previous_period: Period
    seller: list[SellerPriceDay] = Field(max_length=62)
    customer: list[CustomerPriceDay] = Field(max_length=62)
    competitors: list[HistoricalCompetitor] = Field(max_length=2000)
    limitations: list[str]
    comparisons: list[PriceComparison] = Field(max_length=4007)

    @model_validator(mode="after")
    def exact_windows(self):
        start, end = self.requested_period.date_from, self.requested_period.date_to
        if not start or not end:
            raise ValueError("Price history requires exact dates")
        PriceHistoryRequest(vendor_code=self.vendor_code, date_from=start, date_to=end)
        previous = start - (end - start + timedelta(days=1))
        if self.previous_period.date_from != previous or self.previous_period.date_to != start - timedelta(days=1):
            raise ValueError("Price history previous period mismatch")
        if (self.identity_status == "resolved") != (self.nm_id is not None and self.nm_id > 0):
            raise ValueError("Price history identity mismatch")
        if self.identity_status != "resolved" and (self.nm_id is not None or self.seller or self.customer or self.competitors):
            raise ValueError("Unresolved Price identity has rows")
        for rows, day_key, identity_key in (
            (self.seller, "snapshot_date", "nm_id"),
            (self.customer, "snapshot_date", "source_nm_id"),
            (self.competitors, "run_date", "source_nm_id"),
        ):
            for row in rows:
                if getattr(row, identity_key) != self.nm_id or not previous <= getattr(row, day_key) <= end:
                    raise ValueError("Price history row outside scope")
            if day_key == "snapshot_date" and len({row.snapshot_date for row in rows}) != len(rows):
                raise ValueError("Duplicate Price day")
        expected_comparisons = compare_observed_prices(
            [row.model_dump(mode="json") for row in self.seller],
            [row.model_dump(mode="json") for row in self.customer],
            [row.model_dump(mode="json") for row in self.competitors], start, end)
        if [item.model_dump(mode="json") for item in self.comparisons] != expected_comparisons:
            raise ValueError("Price comparison differs from observations")
        return self

