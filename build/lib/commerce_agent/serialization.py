from __future__ import annotations

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer


# The self-managed runtime owns serialization. This explicit list is used by
# every production/development checkpointer and must be reviewed before upgrades.
CHECKPOINT_MODEL_ALLOWLIST = (
    ("commerce_agent.models", "ActionReceipt"),
    ("commerce_agent.models", "ActionTask"),
    ("commerce_agent.models", "AgentStatus"),
    ("commerce_agent.models", "ApprovalRequest"),
    ("commerce_agent.models", "ConfidenceReport"),
    ("commerce_agent.models", "DataGap"),
    ("commerce_agent.models", "DomainAnalysis"),
    ("commerce_agent.models", "DomainName"),
    ("commerce_agent.models", "EvidenceItem"),
    ("commerce_agent.models", "Incident"),
    ("commerce_agent.models", "ManagerDecision"),
    ("commerce_agent.models", "QueryPlan"),
    ("commerce_agent.models", "ResearchResult"),
    ("commerce_agent.models", "WorkflowError"),
    ("commerce_agent.sales.contracts", "QueryIntent"),
    ("commerce_agent.sales.contracts", "DriverClass"),
    ("commerce_agent.sales.contracts", "DataGap"),
    ("commerce_agent.sales.contracts", "CrossAgentRequest"),
    ("commerce_agent.sales.contracts", "ReportTable"),
    ("commerce_agent.sales.contracts", "DiagnosticFinding"),
    ("commerce_agent.sales.contracts", "EvidenceFact"),
    ("commerce_agent.sales.contracts", "SalesDiagnosticResult"),
    ("commerce_agent.marketing.contracts", "MarketingDiagnosticResult"),
)


def checkpoint_serializer() -> JsonPlusSerializer:
    return JsonPlusSerializer(
        allowed_msgpack_modules=CHECKPOINT_MODEL_ALLOWLIST,
    )


__all__ = ["CHECKPOINT_MODEL_ALLOWLIST", "checkpoint_serializer"]
