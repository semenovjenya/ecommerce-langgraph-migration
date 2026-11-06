"""Exact legacy Finance result schema; publication enforces native evidence."""
from pydantic import BaseModel, Field
from commerce_agent.sales.contracts import CrossAgentRequest, DataGap, DiagnosticFinding, EvidenceFact, ReportTable

class AgentDiagnosticResult(BaseModel):
    agent_name: str
    summary: str
    facts: list["EvidenceFact"] = Field(default_factory=list)
    tables: list[ReportTable] = Field(default_factory=list)
    findings: list[DiagnosticFinding] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    data_gaps: list[DataGap] = Field(default_factory=list)
    cross_agent_requests: list[CrossAgentRequest] = Field(default_factory=list)
