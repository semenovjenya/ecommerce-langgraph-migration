"""Independent coverage axes. A successful read does not prove model consumption."""
from datetime import date, datetime
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator


class Closed(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Window(Closed):
    date_from: date
    date_to: date

    @model_validator(mode="after")
    def ordered(self):
        if self.date_from > self.date_to:
            raise ValueError("coverage window reversed")
        return self


class Availability(Closed):
    status: Literal["available", "empty", "not_collected", "unavailable", "error", "unknown"]
    reason: str = Field(min_length=1)


class Coverage(Closed):
    status: Literal["complete", "partial", "unknown"]
    unit: Literal["days", "rows", "values"]
    expected: StrictInt | None = Field(ge=0)
    observed: StrictInt | None = Field(ge=0)
    missing_refs: list[str]
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def coherent(self):
        if self.expected is not None and self.observed is not None and self.observed > self.expected:
            raise ValueError("coverage observed exceeds expected")
        if self.status == "complete" and (self.expected is None or self.observed != self.expected or self.missing_refs):
            raise ValueError("complete coverage requires known full count")
        if self.status == "partial" and (self.expected is None or self.observed is None or self.observed >= self.expected):
            raise ValueError("partial coverage requires a known deficit")
        return self


class Comparability(Closed):
    status: Literal["comparable", "incomparable", "unknown"]
    reason: str = Field(min_length=1)


class OriginGroup(Closed):
    origin: Literal["observed", "replay", "carry_forward", "proxy", "derived", "unknown"]
    row_refs: list[str]
    reason: str = Field(min_length=1)
    input_origins: list[Literal["observed", "replay", "carry_forward", "proxy", "derived", "unknown"]]

    @model_validator(mode="after")
    def derived_inputs(self):
        if self.origin == "derived" and not self.input_origins:
            raise ValueError("derived origin requires input origins")
        if len(self.row_refs) != len(set(self.row_refs)):
            raise ValueError("duplicate origin row")
        return self


class Consumption(Closed):
    status: Literal["not_requested", "fetched", "delivered_to_model", "used_in_check", "unused"]
    reason: str = Field(min_length=1)
    proof_scope: Literal["source_read", "offline_transport", "provider_transport", "accepted_binding", "unknown"]
    boundary_ref: str | None
    check_receipt_ref: str | None
    observation_refs: list[str]
    accepted_event_ref: str | None

    @model_validator(mode="after")
    def requires_proof(self):
        if self.status == "delivered_to_model" and (not self.boundary_ref or self.proof_scope not in ("offline_transport", "provider_transport")):
            raise ValueError("delivery requires a transport-bound proof")
        if self.status == "used_in_check" and (not self.check_receipt_ref or not self.observation_refs or not self.accepted_event_ref or self.proof_scope != "accepted_binding"):
            raise ValueError("usage requires accepted observation and check binding")
        return self


class SourceCoverage(Closed):
    source_id: str = Field(pattern=r"^[a-z_][a-z0-9_]*\.[a-z_][a-z0-9_]*$")
    field_ids: list[str]
    sku: str
    nm_id: StrictInt | None
    grain: str = Field(min_length=1)
    requested_window: Window
    previous_window: Window
    measurement_windows: list[Window]
    observed_at: datetime | None
    snapshot_at: datetime | None
    time_reason: str = Field(min_length=1)
    fetched_at: datetime
    rows_ref: str
    row_refs: list[str]
    availability: Availability
    row_coverage: Coverage
    field_nonnull_coverage: dict[str, Coverage]
    comparability: Comparability
    origins: list[OriginGroup]
    consumption: Consumption
    evidence_refs: list[str]

    @model_validator(mode="after")
    def source_scope(self):
        if (len(self.field_ids) != len(set(self.field_ids)) or any(not f.startswith(self.source_id + ".") for f in self.field_ids)
                or set(self.field_nonnull_coverage) != set(self.field_ids)):
            raise ValueError("coverage fields outside source")
        if self.observed_at is not None and (self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None):
            raise ValueError("coverage observation time must be timezone-aware")
        if self.fetched_at.tzinfo is None or self.fetched_at.utcoffset() is None:
            raise ValueError("coverage fetch time must be timezone-aware")
        if self.snapshot_at is not None and (self.snapshot_at.tzinfo is None or self.snapshot_at.utcoffset() is None):
            raise ValueError("coverage snapshot time must be timezone-aware")
        origin_refs = [ref for group in self.origins for ref in group.row_refs]
        if len(set(self.row_refs)) != len(self.row_refs) or len(origin_refs) != len(set(origin_refs)) or set(origin_refs) != set(self.row_refs):
            raise ValueError("origin groups must partition returned rows")
        if self.availability.status == "empty" and self.row_refs:
            raise ValueError("empty source cannot have rows")
        if self.availability.status == "available" and not self.row_refs:
            raise ValueError("available source requires returned rows")
        return self


class CoverageManifest(Closed):
    version: Literal[1]
    sources: list[SourceCoverage]
    scope: Literal["historical_sources_only"]
    limitations: list[str]

    @model_validator(mode="after")
    def unique_source_scope(self):
        keys = [(s.source_id,s.sku,s.requested_window.date_from,s.requested_window.date_to,s.rows_ref) for s in self.sources]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate source coverage scope")
        return self


class AcceptedUsage(Closed):
    observation_ref: str
    observation_kind: Literal["data", "coverage", "status"]
    source_id: str | None
    field_ids: list[str]
    hypothesis_id: str
    check_receipt_ref: str = Field(pattern=r"^r[0-4]$")
    read_key: str = Field(pattern=r"^[a-f0-9]{64}$")
    sku: str
    requested_window: Window
    accepted_event_ref: str
    consumption: Consumption


class UsageManifest(Closed):
    version: Literal[1]
    owner: str = Field(pattern=r"^[a-f0-9]{64}$")
    bindings: list[AcceptedUsage]
    model_delivery: Literal["unverified"]
    limitation: Literal["accepted_binding_does_not_prove_provider_delivery_or_semantic_acceptance"]
