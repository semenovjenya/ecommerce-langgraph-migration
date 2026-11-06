"""Legacy Marketing schema baseline, not permission to publish unchecked data.

Shared legacy evidence shapes are already isolated/pure in sales.contracts.
Importing them does not invoke Sales or the backend; do not fork their meaning.
"""
from pydantic import BaseModel, Field

from commerce_agent.sales.contracts import (
    CrossAgentRequest, DataGap, DiagnosticFinding, EvidenceFact, ReportTable,
)


class AgentDiagnosticResult(BaseModel):
    agent_name: str
    summary: str
    facts: list["EvidenceFact"] = Field(default_factory=list)
    tables: list[ReportTable] = Field(default_factory=list)
    findings: list[DiagnosticFinding] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    data_gaps: list[DataGap] = Field(default_factory=list)
    cross_agent_requests: list[CrossAgentRequest] = Field(default_factory=list)


class MarketingDiagnosticResult(AgentDiagnosticResult):
    marketing_report_snapshot: ReportTable | None = Field(
        default=None,
        description="Mandatory top table for marketing reports: yesterday/day-before, week/week, month/month with ДРР.",
    )
