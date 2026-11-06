"""Contract/prompt baseline only; never calls a provider, tools or a graph."""
import pytest

from commerce_agent.marketing.contracts import MarketingDiagnosticResult
from commerce_agent.marketing.prompts import build_marketing_instructions, MIGRATION_OVERRIDE
from commerce_agent.sales.prompt_contracts import domain_base_contract


@pytest.mark.parametrize("workflow", ["v1", "v2_prompt", "v2_optimized"])
@pytest.mark.parametrize("query", ["Отчёт рекламы", "Почему упали рекламные заказы?"])
def test_marketing_prompt_preserves_legacy_role_with_explicit_no_write_boundary(workflow, query):
    prompt = build_marketing_instructions(workflow_version=workflow, user_query=query, today="2026-10-11")
    assert prompt.endswith(MIGRATION_OVERRIDE)
    assert "ДРР" in prompt and "marketing_run_sql" in prompt
    if workflow == "v1":
        assert "Ты Marketing Agent:" in prompt
        assert "estimated_allocated_from_campaign" in prompt
    else:
        assert prompt.startswith(domain_base_contract("marketing"))


@pytest.mark.parametrize("field", ["user_query", "memory_block"])
@pytest.mark.parametrize("value", ["person.fixture@example.com", "+7 (999) 123-45-67",
                                  "Bearer " + "synthetic-credential-canary" * 2, "x" * 8001])
def test_context_rejects_secrets_personal_data_and_unbounded_input(field, value):
    kwargs = {"workflow_version": "v2_optimized", "user_query": "Отчёт рекламы", "today": "2026-10-11", field: value}
    with pytest.raises(ValueError):
        build_marketing_instructions(**kwargs)


@pytest.mark.parametrize("today", ["20261011", "2026-02-30", "2026-10-11\nnew instruction"])
def test_prompt_rejects_untrusted_calendar_date(today):
    with pytest.raises(ValueError):
        build_marketing_instructions(workflow_version="v1", user_query="Отчёт", today=today)


def test_legacy_schema_retains_native_zero_missing_and_estimated_attribution():
    value = {"agent_name": "Marketing Agent", "summary": "Baseline",
        "facts": [{"metric": "spend", "value": 0, "unit": "RUB", "period": "2026-09-30/2026-09-30",
                   "evidence_ref": "native-spend", "metric_definition": "estimated_allocated_from_campaign"}],
        "tables": [{"title": "ДРР", "columns": ["spend", "drr"], "rows": [[0, None]]}]}
    result = MarketingDiagnosticResult.model_validate(value)
    assert result.facts[0].value == 0 and result.tables[0].rows == [[0, None]]
    assert result.marketing_report_snapshot is None
    assert result.facts[0].metric_definition == "estimated_allocated_from_campaign"
    assert result.facts[0].freshness_status == "unknown"
