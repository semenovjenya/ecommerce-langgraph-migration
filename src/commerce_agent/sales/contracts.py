"""Legacy Sales output schemas, copied verbatim and parity-tested."""
from __future__ import annotations
from datetime import date, datetime
from enum import Enum
import re
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

class QueryIntent(str, Enum):
    FINANCE = "finance"
    MARKETING = "marketing"
    PRICE = "price"
    SALES = "sales"
    SUPPLY = "supply"


class RoundType(str, Enum):
    ROUTER = "router"
    PRIMARY = "primary"
    CROSS_CHECK = "cross_check"
    FOLLOWUP = "followup"
    GAP_RESOLUTION = "gap_resolution"
    CLAIM_VERIFICATION = "claim_verification"
    SYNTHESIS = "synthesis"
    TERMINAL_ACTION = "terminal_action"


class QueryMode(str, Enum):
    DECISION = "decision"
    DIAGNOSTIC = "diagnostic"
    REPORT = "report"
    LOOKUP = "lookup"


class DataGap(BaseModel):
    metric: str = Field(..., description="Missing metric or business dimension.")
    reason: str = Field(..., description="Why the metric is unavailable.")
    impact: str = Field(..., description="How the missing metric limits the report.")


class CrossAgentRequest(BaseModel):
    target_agent: QueryIntent = Field(..., description="Domain agent that can close the evidence gap.")
    reason: str = Field(..., description="Why the primary agent needs this cross-domain check.")
    question: str = Field(..., description="Specific narrow question for the target agent.")
    period: str | None = Field(default=None, description="Requested period, if known.")
    vendor_codes: list[str] = Field(default_factory=list, description="Vendor codes to check.")
    nm_ids: list[int | str] = Field(default_factory=list, description="WB nm_id values to check.")
    required_checks: list[str] = Field(
        default_factory=list,
        description="Specific facts or metrics the target agent should verify.",
    )


class DriverClass(str, Enum):
    PRIMARY_DRIVER = "primary_driver"
    SECONDARY_DRIVER = "secondary_driver"
    AMPLIFIER = "amplifier"
    NOT_A_DRIVER = "not_a_driver"


class ReportTable(BaseModel):
    title: str
    columns: list[str]
    rows: list[list[str | int | float | None]] = Field(default_factory=list)


class DiagnosticFinding(BaseModel):
    factor: str
    observation: str
    impact: str
    cause: str
    action: str
    evidence: list[str] = Field(default_factory=list)
    confidence: str
    driver_class: DriverClass

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, value):
        normalized = str(value or "").strip().lower().replace("ё", "е")
        aliases = {
            "высокая": "high",
            "высокий": "high",
            "высокое": "high",
            "high": "high",
            "средняя": "medium",
            "средний": "medium",
            "среднее": "medium",
            "medium": "medium",
            "низкая": "low",
            "низкий": "low",
            "низкое": "low",
            "low": "low",
        }
        return aliases.get(normalized, value)


class EvidenceFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: str = Field(..., min_length=1)
    value: str | int | float | bool | None = None
    unit: str | None = None
    period: str | None = None
    period_start: date | datetime | str | None = None
    period_end: date | datetime | str | None = None
    sku: str | None = None
    target_type: str | None = None
    target_id: str | None = None
    source_table: str | None = None
    source_snapshot_at: date | datetime | str | None = None
    retrieved_at: datetime | str | None = None
    metric_definition: str | None = None
    freshness_status: Literal["fresh", "stale", "future", "unknown"] = "unknown"
    freshness_lag_days: int | None = None
    statement: str | None = None
    evidence_ref: str = Field(..., min_length=1)
    source_agent: str | None = None
    status: Literal["confirmed", "refuted", "unconfirmed"] = "confirmed"
    confidence: Literal["low", "medium", "high"] = "medium"

    @staticmethod
    def _metric_is_identifier(metric: str) -> bool:
        normalized = re.sub(
            r"[^a-zа-я0-9]+",
            "_",
            metric.lower().replace("ё", "е"),
        ).strip("_")
        if normalized in {
            "nm_id",
            "nmid",
            "sku",
            "sku_id",
            "vendor_code",
            "vendorcode",
            "артикул",
            "артикул_продавца",
            "код_продавца",
        }:
            return True
        return bool(re.fullmatch(r"[a-z][a-z0-9_]*_id", normalized))

    @model_validator(mode="after")
    def validate_numeric_context(self) -> "EvidenceFact":
        metric_is_identifier = self._metric_is_identifier(self.metric)
        value_is_numeric = isinstance(self.value, (int, float)) and not isinstance(
            self.value,
            bool,
        )
        if (
            value_is_numeric
            and not metric_is_identifier
            and not str(self.unit or "").strip()
        ):
            raise ValueError("unit is required when numeric value is present")
        if (
            self.value is not None
            and not metric_is_identifier
            and not str(self.period or "").strip()
        ):
            raise ValueError("period is required when value is present")
        return self


class SalesDiagnosticResult(BaseModel):
    agent_name: str
    summary: str
    facts: list[EvidenceFact] = Field(default_factory=list)
    sales_report_snapshot: ReportTable | None = Field(
        default=None,
        description="Mandatory top table for sales reports: yesterday/day-before, week/week, month/month.",
    )
    sales_overview: ReportTable
    sales_dynamics: ReportTable
    sales_decomposition: ReportTable
    findings: list[DiagnosticFinding] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    data_gaps: list[DataGap] = Field(default_factory=list)
    cross_agent_requests: list[CrossAgentRequest] = Field(default_factory=list)

    def as_tables(self) -> list[ReportTable]:
        return [self.sales_overview, self.sales_dynamics, self.sales_decomposition]
