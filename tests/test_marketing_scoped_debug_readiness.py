"""Current projection is not source approval: no scoped trace may expose labels."""
import json
import pytest
from commerce_agent.debug_privacy import sanitize_debug_payload
from commerce_agent.marketing.runtime import MarketingDependencies,build_marketing_subgraph


def test_malformed_query_proposal_is_now_suppressed_not_source_approved():
    # The former leak characterization now proves its export-side fix.
    # Full scoped source/model/wire approval remains a separate gate.
    person="Synthetic Person Canary"
    source={"source":"marketing.dm_marketing_campaign_daily","data":{"query_actions":[{
        "entity_type":"поисковый запрос","entity_id":person,"entity_name":person,"norm_query":person,
        "reason":person,"action_mode":"needs_review","spend":123.45}]}}
    projected=sanitize_debug_payload(source)
    assert person not in json.dumps(projected)
    assert projected["data"]["query_actions"]==["[omitted-private]"]


def test_current_scoped_debug_is_rejected_before_model_or_tool_callbacks(monkeypatch):
    monkeypatch.setenv("COMMERCE_AGENT_ENVIRONMENT","development")
    monkeypatch.setenv("COMMERCE_AGENT_LANGSMITH_DEBUG","true")
    calls=[]
    def client():calls.append("tools");raise AssertionError("must not run")
    class Model:
        async def analyze(self,**kwargs):calls.append("model");raise AssertionError("must not run")
    dependencies=MarketingDependencies(client_factory=client,model=Model(),enable_scoped_reports=True)
    with pytest.raises(ValueError,match="scoped report debug not accepted"):build_marketing_subgraph(dependencies)
    assert calls==[]
