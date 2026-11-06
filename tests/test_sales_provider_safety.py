"""Actual Responses SDK failures through Sales and the final tracing projection."""
import asyncio
import json
from unittest.mock import PropertyMock, patch
from uuid import uuid4

import httpx
import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langsmith import Client
from langsmith.schemas import LangSmithInfo

from commerce_agent.graph import build_graph
from commerce_agent.ingress import GuardedGraph
from commerce_agent.sales.model import LangChainSalesModel, SalesModelSettings
from commerce_agent.serialization import checkpoint_serializer
from test_observability import _dual_environment
from test_sales_evidence import CONTEXT
from test_sales_model import response
from test_sales_runtime import dependencies


CANARY = "lsv2_synthetic_provider_error_canary_123456"
NARRATIVE = {"summary": "", "findings": [],
             "recommendations": [], "data_gaps": [], "cross_agent_requests": []}


@pytest.mark.parametrize("mode", ["valid", "rate_limit", "refusal", "malformed", "incomplete",
                                  "content_filter", "empty", "tool_call", "mixed_refusal",
                                  "builtin_tool", "sensitive_output"])
def test_provider_response_cannot_leak_secrets_or_publish_incomplete_analysis(mode, monkeypatch):
    environment = _dual_environment() | {
        "LANGFUSE_ENABLED": "false", "COMMERCE_AGENT_ENVIRONMENT": "development",
        "COMMERCE_AGENT_LANGSMITH_STATE_SUMMARY": "true",
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    exported, network, model_requests = [], [], []

    def provider(request):
        model_requests.append(request)
        if mode == "rate_limit":
            return httpx.Response(429, json={"error": {"message": CANARY,
                "type": "rate_limit_error", "code": "rate_limit_exceeded"}})
        payload = response(NARRATIVE)
        content = payload["output"][0]["content"]
        if mode == "refusal":
            content[:] = [{"type": "refusal", "refusal": CANARY}]
        elif mode == "malformed":
            content[0]["text"] = "{" + CANARY
        elif mode in {"incomplete", "content_filter"}:
            # Valid JSON does NOT prove the provider finished its response.
            payload["status"] = "incomplete"
            payload["incomplete_details"] = {"reason": "max_output_tokens" if mode == "incomplete" else "content_filter"}
        elif mode == "empty":
            payload["output"] = []
        elif mode == "tool_call":
            payload["output"].append({"id": "fc_fixture", "type": "function_call",
                "call_id": "call_fixture", "name": "forbidden_write", "arguments": "{}", "status": "completed"})
        elif mode == "mixed_refusal":
            content.append({"type": "refusal", "refusal": CANARY})
        elif mode == "builtin_tool":
            payload["output"].append({"id": "ws_fixture", "type": "web_search_call",
                "status": "completed", "action": {"type": "search", "query": "fixture"}})
        elif mode == "sensitive_output":
            content[0]["text"] = json.dumps({**NARRATIVE, "summary": CANARY})
        return httpx.Response(200, json=payload)

    def capture(_client, run, **_kwargs):
        exported.append(run)

    def no_network(_client, _method, url, **_kwargs):
        network.append(url)
        raise AssertionError("unexpected network")

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as http:
            model = LangChainSalesModel(SalesModelSettings(api_key="synthetic-model-key"), http_async_client=http)
            graph = GuardedGraph(build_graph(
                checkpointer=InMemorySaver(serde=checkpoint_serializer()),
                sales_dependencies=dependencies(model, model_profile="openai_default"),
            ))
            config = {"configurable": {"thread_id": "m3-sales-provider-" + str(uuid4())}}
            ctx = CONTEXT.model_copy(update={"model_profile": "openai_default"})
            result = await graph.ainvoke({"schema_version": 1,
                "messages": [HumanMessage(content="Отчет продаж")]}, config=config, context=ctx)
            graph.flush_traces(timeout=5)
            assert len(model_requests) == 1  # no automatic retry of provider failure
            if mode == "valid":
                assert result["sales_analysis"].result is not None
            else:
                assert result["sales_analysis"].status == "failed"
                assert result["sales_analysis"].result is None
                assert "SALES_ANALYSIS_REJECTED" in {item.code for item in result["errors"]}
            async for snapshot in graph.aget_state_history(config):
                assert CANARY not in json.dumps(snapshot.values, default=str)

    with patch.object(Client, "_create_run", autospec=True, side_effect=capture), \
         patch.object(Client, "_update_run", autospec=True, side_effect=capture), \
         patch.object(Client, "_send_multipart_req", autospec=True), \
         patch.object(Client, "info", new_callable=PropertyMock, return_value=LangSmithInfo()), \
         patch.object(Client, "request_with_retries", autospec=True, side_effect=no_network):
        asyncio.run(exercise())
    assert not network
    assert exported
    assert any(run.get("run_type") == "llm" for run in exported)
    assert CANARY not in json.dumps(exported, default=str)
    assert "synthetic-model-key" not in json.dumps(exported, default=str)
    roots = [run for run in exported if run.get("name") == "commerce_agent" and run.get("outputs")]
    assert roots and "state_summary" in roots[-1]["outputs"]
