import pytest

from commerce_agent.models import DomainAnalysis, DomainName
from commerce_agent.reducers import merge_sales_analysis, StateConflictError
from commerce_agent.sales.contracts import SalesDiagnosticResult, ReportTable
from commerce_agent.serialization import checkpoint_serializer


def sales_result():
    table = ReportTable(title="Обзор", columns=["Метрика", "Значение"], rows=[["Выручка", 0]])
    return SalesDiagnosticResult(agent_name="Sales Agent", summary="Срез подтверждён.",
                                 sales_overview=table, sales_dynamics=table, sales_decomposition=table)


def test_old_v1_checkpoint_remains_readable_with_additive_result_default():
    old = {"domain": "sales", "revision": 1, "status": "not_implemented",
           "summary": "M1 skeleton", "evidence_refs": []}
    result = merge_sales_analysis(None, old)
    assert result.result is None and result.revision == 1


def test_sales_result_restores_typed_without_losing_numeric_zero():
    value = DomainAnalysis(domain=DomainName.SALES, revision=2, status="completed",
                           summary="Срез подтверждён.", result=sales_result())
    serde = checkpoint_serializer()
    restored = serde.loads_typed(serde.dumps_typed(value))
    assert isinstance(restored, DomainAnalysis)
    assert isinstance(restored.result, SalesDiagnosticResult)
    assert isinstance(restored.result.sales_overview, ReportTable)
    assert restored.result.sales_overview.rows[0][1] == 0
    assert restored == value
    assert merge_sales_analysis(None, value.model_dump(mode="json")) == value


def test_other_domain_cannot_publish_sales_result():
    with pytest.raises(ValueError):
        DomainAnalysis(domain=DomainName.FINANCE, status="completed", summary="invalid", result=sales_result())
    with pytest.raises(StateConflictError):
        merge_sales_analysis(None, DomainAnalysis(domain=DomainName.FINANCE, status="completed", summary="invalid"))
