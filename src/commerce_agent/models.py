from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import core_schema

from commerce_agent.sales.contracts import SalesDiagnosticResult
from commerce_agent.marketing.contracts import MarketingDiagnosticResult
from commerce_agent.supply.contracts import AgentDiagnosticResult as SupplyDiagnosticResult
from commerce_agent.finance.contracts import AgentDiagnosticResult as FinanceDiagnosticResult
from commerce_agent.price.contracts import AgentDiagnosticResult as PriceDiagnosticResult


class SchemaVersion(int):
    """Reducer identity that makes even LangGraph's step -1 checkpoint versioned."""

    def __new__(cls, value: int = 1):
        if int(value) != 1:
            raise ValueError("only CommerceState schema_version=1 is supported")
        return super().__new__(cls, 1)

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Any):
        del source_type, handler
        return core_schema.no_info_after_validator_function(
            cls,
            core_schema.literal_schema([1]),
        )


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DomainName(str, Enum):
    SALES = "sales"
    MARKETING = "marketing"
    SUPPLY = "supply"
    PRICE = "price"
    FINANCE = "finance"


DOMAIN_NAMES: tuple[DomainName, ...] = tuple(DomainName)


class QueryPlan(FrozenModel):
    objective: str = "answer_user_request"
    intent: str = "commerce_analysis"
    required_domains: tuple[DomainName, ...] = DOMAIN_NAMES
    requested_analyses: tuple[str, ...] = ()
    allowed_action_families: tuple[Literal["price", "marketing", "supply"], ...] = ()
    constraints: dict[str, Any] = Field(default_factory=dict)


class AgentStatus(FrozenModel):
    domain: DomainName
    status: Literal[
        "pending",
        "running",
        "planned",
        "completed",
        "failed",
        "skipped",
        "interrupted",
        "cancelled",
    ] = "pending"
    attempt: int = Field(default=0, ge=0)
    node: str | None = None
    detail: str | None = None


class DomainAnalysis(FrozenModel):
    domain: DomainName
    revision: int = Field(default=1, ge=1)
    status: Literal["not_implemented", "completed", "partial", "failed"]
    summary: str
    evidence_refs: tuple[str, ...] = ()
    # Additive v1 extension: old skeleton checkpoints omit this field. Only
    # Implemented domains publish only their own validated structured result.
    result: SalesDiagnosticResult | MarketingDiagnosticResult | SupplyDiagnosticResult | FinanceDiagnosticResult | PriceDiagnosticResult | None = None

    @model_validator(mode="before")
    @classmethod
    def parse_structured_result_by_owner(cls, value):
        if isinstance(value, dict) and value.get("result") is not None:
            value = dict(value)
            owners = {DomainName.SALES: SalesDiagnosticResult, DomainName.MARKETING: MarketingDiagnosticResult,
                      DomainName.SUPPLY: SupplyDiagnosticResult, DomainName.FINANCE: FinanceDiagnosticResult,
                      DomainName.PRICE: PriceDiagnosticResult}
            model = owners.get(value.get("domain"))
            if model is None:
                raise ValueError("structured result has no implemented domain owner")
            result = value["result"]
            if isinstance(result, BaseModel) and not isinstance(result, model):
                raise ValueError("structured result belongs to another domain")
            if isinstance(result, dict) and set(result) - set(model.model_fields):
                raise ValueError("structured result contains foreign domain fields")
            value["result"] = model.model_validate(result)
        return value

    @model_validator(mode="after")
    def enforce_structured_result_owner(self):
        owners = {DomainName.SALES: SalesDiagnosticResult, DomainName.MARKETING: MarketingDiagnosticResult,
                  DomainName.SUPPLY: SupplyDiagnosticResult, DomainName.FINANCE: FinanceDiagnosticResult,
                  DomainName.PRICE: PriceDiagnosticResult}
        if self.result is not None and not isinstance(self.result, owners.get(self.domain, type(None))):
            raise ValueError("structured result belongs to another domain")
        if self.domain == DomainName.SUPPLY and self.result is not None and self.result.agent_name != "Supply Agent":
            raise ValueError("Supply structured result has a foreign agent identity")
        if self.domain == DomainName.FINANCE and self.result is not None and self.result.agent_name != "Finance Agent":
            raise ValueError("Finance structured result has a foreign agent identity")
        if self.domain == DomainName.PRICE and self.result is not None and self.result.agent_name != "Price Agent":
            raise ValueError("Price structured result has a foreign agent identity")
        return self


class EvidenceItem(FrozenModel):
    evidence_ref: str = Field(min_length=1)
    domain: DomainName
    claim: str
    source: str
    period: str | None = None
    freshness: Literal["fresh", "stale", "future", "unknown"] = "unknown"


class DataGap(FrozenModel):
    gap_id: str = Field(min_length=1)
    domain: DomainName | None = None
    reason: str
    blocking: bool = True
    status: Literal["open", "resolved", "accepted_limitation"] = "open"
    revision: int = Field(default=1, ge=1)


class Incident(FrozenModel):
    incident_id: str = Field(min_length=1)
    severity: Literal["info", "warning", "critical"]
    summary: str


class ActionTask(FrozenModel):
    action_id: str = Field(min_length=1)
    action_family: Literal["price", "marketing", "supply"]
    summary: str
    status: Literal["proposed", "blocked", "approved", "rejected", "executed"] = "proposed"


class ConfidenceReport(FrozenModel):
    score: float = Field(ge=0.0, le=1.0)
    level: Literal["low", "medium", "high"]
    limitations: tuple[str, ...] = ()


class ManagerDecision(FrozenModel):
    decision_id: str
    status: Literal["needs_data", "hold", "ready_for_approval"]
    summary: str


class ApprovalRequest(FrozenModel):
    interrupt_id: str
    action_id: str
    summary: str
    idempotency_key: str


class ActionReceipt(FrozenModel):
    receipt_id: str
    action_id: str
    idempotency_key: str
    status: Literal["executed", "verified", "failed", "unknown"]
    audit_ref: str | None = None
    verification_ref: str | None = None


class WorkflowError(FrozenModel):
    error_id: str
    code: str
    message: str
    node: str
    retryable: bool = False
    severity: Literal["info", "warning", "error", "critical"] = "error"


class ResearchResult(FrozenModel):
    brief_id: str
    status: Literal["completed", "partial", "failed", "cancelled"]
    findings: tuple[str, ...] = ()
    evidence: tuple[EvidenceItem, ...] = ()
    limitations: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    proposed_checks: tuple[str, ...] = ()
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


__all__ = [
    "DOMAIN_NAMES",
    "ActionReceipt",
    "ActionTask",
    "AgentStatus",
    "ApprovalRequest",
    "ConfidenceReport",
    "DataGap",
    "DomainAnalysis",
    "DomainName",
    "EvidenceItem",
    "Incident",
    "ManagerDecision",
    "QueryPlan",
    "ResearchResult",
    "SchemaVersion",
    "WorkflowError",
]
