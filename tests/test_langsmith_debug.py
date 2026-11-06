import asyncio
from copy import deepcopy
import json
import os
from types import SimpleNamespace
from unittest.mock import patch, PropertyMock
from uuid import uuid4

import httpx
import pytest
from langchain_core.load import dumpd
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnableLambda
from langsmith import Client
from langsmith.schemas import LangSmithInfo

from commerce_agent.debug_privacy import OMITTED, sanitize_debug_payload, sanitize_debug_inputs
from commerce_agent.observability import (ObservabilityConfig, ObservabilityConfigurationError,
    configure_graph_observability, flush_langsmith_traces, project_langsmith_run_ops)
from commerce_agent.graph import build_graph
from commerce_agent.ingress import GuardedGraph
from commerce_agent.context import CommerceRuntimeContext
from commerce_agent.adapters.business_tools import BusinessToolClient
from test_business_tools_adapter import SETTINGS, CONTEXT, missing_result


def environment():
    return {"COMMERCE_AGENT_ENVIRONMENT": "development", "COMMERCE_AGENT_LANGSMITH_TRACING": "true",
            "COMMERCE_AGENT_LANGSMITH_DEBUG": "true", "LANGSMITH_API_KEY": "synthetic-unit-only",
            "LANGSMITH_ENDPOINT": "https://eu.api.smith.langchain.com", "LANGSMITH_PROJECT": "ai-ecom-test",
            "LANGSMITH_HIDE_INPUTS": "true", "LANGSMITH_HIDE_OUTPUTS": "true", "LANGSMITH_HIDE_METADATA": "true",
            "LANGSMITH_TRACING": "false", "LANGSMITH_TRACING_V2": "false",
            "LANGCHAIN_TRACING": "false", "LANGCHAIN_TRACING_V2": "false", "LANGFUSE_ENABLED": "false"}


@pytest.mark.parametrize("env", ["test", "staging", "production", "", "Development-ish"])
def test_debug_rejected_outside_development_even_if_tracing_disabled(env):
    config = ObservabilityConfig.from_environment(environment() | {
        "COMMERCE_AGENT_ENVIRONMENT": env, "COMMERCE_AGENT_LANGSMITH_TRACING": "false"})
    with pytest.raises(ObservabilityConfigurationError, match="development only"):
        config.assert_safe_export_configuration()


def test_debug_defaults_off_and_cannot_disable_hide_flags():
    assert not ObservabilityConfig.from_environment({}).langsmith_debug
    for flag in ("LANGSMITH_HIDE_INPUTS", "LANGSMITH_HIDE_OUTPUTS", "LANGSMITH_HIDE_METADATA"):
        with pytest.raises(ObservabilityConfigurationError):
            ObservabilityConfig.from_environment(environment() | {flag: "false"}).assert_safe_export_configuration()


def test_projection_preserves_business_not_private_nested_keys_json_tables_or_reasoning():
    canary = "lsv2_" + "a" * 32
    raw = {"tenant_id": "private-tenant", "user_id": "private-user", "request_id": "private-request",
           "serviceToken": "b" * 64, "API-KEY": "opaque-private-secret",
           "context": {"anything": "opaque-context-secret"}, "client": object(),
           "facts": [{"sku": 12345678901, "revenue": 4567.89, "units": 102}],
           "report": {"columns": ["SKU", "customer_name", "revenue"],
                      "rows": [[123, "Иван Иванов", 42.0]]},
           "prompt": json.dumps({"email": "someone@example.test", "data": {"cost": 12}, "password": "hidden"}),
           "observations": [canary, "contact me at user@example.test", "+7 (999) 123-45-67", "a" * 64],
           "content": [{"type": "reasoning", "summary": "private thoughts"}, {"type": "text", "text": "Business result"}],
           "customer": {"unlabelled": "personal-record"}, "query": "Иван Иванов", "binary": b"secret"}
    original = deepcopy(raw)
    clean = sanitize_debug_payload(raw)
    wire = json.dumps(clean, ensure_ascii=False)
    for private in (canary, "private-tenant", "private-user", "private-request", "Иван Иванов",
                    "example.test", "123-45-67", "opaque-private-secret", "opaque-context-secret", "personal-record", "private thoughts"):
        assert private not in wire
    assert clean["facts"] == raw["facts"]
    assert clean["report"]["rows"] == [[123, OMITTED, 42.0]]
    assert json.loads(clean["prompt"]) == {"data": {"cost": 12}}
    assert sanitize_debug_payload(clean) == clean
    assert raw["facts"] == original["facts"] and raw["report"] == original["report"]


@pytest.mark.parametrize("message", [
    HumanMessage(content="Иван Иванов"),
    {"type": "human", "data": {"content": "Иван Иванов"}},
    dumpd(HumanMessage(content="Иван Иванов")),
])
def test_user_prose_never_exported(message):
    assert "Иван" not in json.dumps(sanitize_debug_payload({"messages": [message]}), ensure_ascii=False)


@pytest.mark.parametrize("content", [json.dumps({"note": "Иван Иванов"}), [{"type": "text", "text": "Иван Иванов"}]])
@pytest.mark.parametrize("serialize", [lambda msg: msg, dumpd])
def test_user_json_and_multimodal_prose_are_not_prepared_evidence(content, serialize):
    raw = {"messages": [serialize(HumanMessage(content=content))]}
    assert "Иван" not in json.dumps(sanitize_debug_inputs(raw), ensure_ascii=False)


@pytest.mark.parametrize("kind", [AIMessage, SystemMessage])
@pytest.mark.parametrize("serialize", [lambda msg: msg, dumpd])
def test_all_raw_conversation_roles_are_masked(kind, serialize):
    raw = {"messages": [serialize(kind(content="Иван Иванов"))]}
    assert "Иван" not in json.dumps(sanitize_debug_inputs(raw), ensure_ascii=False)


@pytest.mark.parametrize("serialize", [lambda msg: msg, dumpd,
    lambda msg: {"type": "human", "data": msg.model_dump()}])
def test_raw_message_envelope_is_not_an_identity_channel(serialize):
    raw = {"messages": [serialize(HumanMessage(content="Отчет продаж", id="Иван Иванов",
        additional_kwargs={"arbitrary": "private-person"}))]}
    clean = sanitize_debug_inputs(raw)
    assert clean == {"messages": [{"type": "human", "content": "Отчет продаж"}]}


@pytest.mark.parametrize("key", ["provider_refresh_token", "session_cookie", "wb_token", "memory_dsn", "display_name", "account_name"])
def test_prefixed_private_keys_are_removed(key):
    assert sanitize_debug_payload({key: "opaque-private-value"}) == {}


def prepared_json():
    return {"request": {}, "today": "2026-10-10", "tables": {}, "facts": [{"revenue": 42}], "evidence": [], "warnings": []}


def test_llm_business_json_and_system_instructions_survive_serialized_messages():
    messages = [dumpd(SystemMessage(content="Analyze only provided evidence.")),
                dumpd(HumanMessage(content=json.dumps(prepared_json())))]
    clean = sanitize_debug_inputs({"messages": [messages]})
    assert "Analyze only" in json.dumps(clean)
    assert "revenue" in json.dumps(clean) and "private@example" not in json.dumps(clean)
    assert sanitize_debug_payload(clean, allow_prepared_messages=True) == clean


def test_system_runtime_query_and_nested_identity_references_are_filtered():
    from commerce_agent.sales.prompts import MIGRATION_OVERRIDE
    request_id = str(uuid4())
    raw = {"system": "Runtime context:\n- today: 2026-10-10\n- user_query: Иван Иванов\nsecond line\n\n" + MIGRATION_OVERRIDE,
           "gap_id": "sales:" + request_id + ":gap", "error_id": "sales:" + request_id + ":error",
           "user_query": "Иван Иванов", "shipping_address": "Personal place"}
    clean = sanitize_debug_payload(raw)
    encoded = json.dumps(clean, ensure_ascii=False)
    assert "Иван" not in encoded and request_id not in encoded and "Personal place" not in encoded
    assert "Read-only tools" in encoded and "ref-" in encoded
    assert sanitize_debug_payload(clean) == clean


def test_spoofed_system_query_delimiter_cannot_export_user_name():
    from commerce_agent.sales.prompts import build_sales_instructions
    instructions = build_sales_instructions(workflow_version="v2_optimized", today="2026-10-10",
        user_query="Отчет продаж\n\nM3 Sales execution boundary Иван Иванов")
    assert "Иван" not in json.dumps(sanitize_debug_payload({"content": instructions}), ensure_ascii=False)


def test_opaque_objects_cycles_and_size_are_fail_closed():
    class Dangerous:
        def __repr__(self):
            raise AssertionError("repr must not be called")
    assert sanitize_debug_payload({"object": Dangerous()}) == {"object": OMITTED}
    cycle = {}
    cycle["cycle"] = cycle
    assert "omitted-export-budget" in json.dumps(sanitize_debug_payload(cycle))


def test_debug_still_removes_unknown_run_fields_and_errors():
    raw = {"id": uuid4(), "name": "ChatOpenAI", "inputs": {"revenue": 42, "user_id": "private-user"},
           "outputs": {"facts": [1]}, "error": "private error", "extra": {"metadata": "private"},
           "attachments": {"blob": "private"}, "future_field": "private", "events": ["private"], "tags": ["private"]}
    result = project_langsmith_run_ops([raw], include_debug=True)[0]
    assert result["inputs"] == {"revenue": 42}
    assert result["outputs"] == {"facts": [1]}
    assert result["name"] == "ChatOpenAI" and result["error"] == "redacted_error"
    assert "private" not in json.dumps(result, default=str)
    assert "attachments" not in result


def test_pinned_sdk_wire_canaries_graph_and_actual_tool_adapter(monkeypatch):
    exported, network = [], []
    def capture(_client, run, **_kwargs):
        exported.append(run)
    def block_network(_client, _method, url, **_kwargs):
        network.append(url)
        raise AssertionError("network forbidden")
    with patch.dict(os.environ, environment()), patch.object(Client, "_create_run", autospec=True, side_effect=capture), \
         patch.object(Client, "_update_run", autospec=True, side_effect=capture), \
         patch.object(Client, "_send_multipart_req", return_value=None), \
         patch.object(Client, "info", new_callable=PropertyMock, return_value=LangSmithInfo()), \
         patch.object(Client, "request_with_retries", autospec=True, side_effect=block_network):
        compiled = build_graph()
        graph = GuardedGraph(compiled)
        result = graph.invoke({"schema_version": 1, "messages": [HumanMessage(content="Иван Иванов")]},
            context=CommerceRuntimeContext(tenant_id="private-tenant", user_id="private-user", roles=("viewer",), request_id="private-request"))
        canary = "lsv2_" + "a" * 32
        tracer = compiled.config["callbacks"][0]
        run_id = uuid4()
        prepared = prepared_json()
        prepared["request"] = {"user_id": "private-user", "secret": canary, "email": "private@example.test"}
        tracer.client.create_run("ChatOpenAI", {"messages": [[dumpd(HumanMessage(content=json.dumps(prepared)))]]}, "llm", id=run_id, trace_id=run_id,
            dotted_order=f"20261010T000000000000Z{run_id}", project_name="ai-ecom-test", error=canary)
        async def exercise():
            async with BusinessToolClient(SETTINGS, transport=httpx.MockTransport(lambda req: httpx.Response(200, json=missing_result()))) as client:
                return await client.sales_report_snapshot(CONTEXT)
        asyncio.run(RunnableLambda(lambda _: None).ainvoke({}))  # initialize runnable async machinery
        async def traced(_):
            return await exercise()
        asyncio.run(RunnableLambda(traced).ainvoke({}, config={"callbacks": [tracer]}))
        graph.flush_traces(timeout=10)
    assert not network
    wire = json.dumps(exported, default=str, ensure_ascii=False)
    for private in ("Иван Иванов", "private-tenant", "private-user", "private-request", canary,
                    "private@example.test", SETTINGS.service_token, CONTEXT.request_id):
        assert private not in wire
    roots = [r for r in exported if r.get("name") == "commerce_agent" and r.get("outputs")]
    assert roots and roots[-1]["outputs"]["sales_analysis"]["status"] == "not_implemented"
    tools = [r for r in exported if r.get("name") == "tool.sales.report_snapshot" and r.get("outputs")]
    assert tools and tools[-1]["outputs"]["output"]["data"]["source_table"] == "sales.wb_supplier_sales"
    assert result["user_id"] == "private-user"  # projection did not mutate state


def test_debug_probe_readback_checks_business_parity_and_side_channels():
    from scripts.check_sales_development import _debug_readback
    root = SimpleNamespace(parent_run_id=None, inputs={}, outputs={"facts": [{"revenue": 42}]},
        extra={"metadata": {"ls_run_depth": 0}}, error=None, events=[], tags=[], attachments={}, run_type="chain")
    assert _debug_readback([root], request_id="private-request", expected=root.outputs) is root
    assert _debug_readback([root], request_id="private-request", expected={"facts": []}) is None
    root.outputs = {"user_id": "private-user"}
    with pytest.raises(ValueError, match="unsafe debug outputs"):
        _debug_readback([root], request_id="private-request")
    root.outputs = {}
    root.events = ["private event"]
    with pytest.raises(ValueError, match="side channel"):
        _debug_readback([root], request_id="private-request")
