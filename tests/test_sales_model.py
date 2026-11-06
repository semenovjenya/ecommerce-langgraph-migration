"""Real LangChain/OpenAI serialization over MockTransport; no provider requests."""
import asyncio
import json

import httpx
import pytest

from commerce_agent.sales.model import LangChainSalesModel, SalesModelSettings
from commerce_agent.sales.policy import SalesRequest
from commerce_agent.sales.publication import validate_sales_publication
from test_sales_evidence import collect


def response(narrative):
    return {"id": "resp_fixture", "object": "response", "created_at": 1791626400,
        "status": "completed", "model": "gpt-4.1-mini", "error": None, "incomplete_details": None,
        "output": [{"id": "msg_fixture", "type": "message", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": json.dumps(narrative, ensure_ascii=False), "annotations": []}]}],
        "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150,
                  "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}}}


def test_responses_wire_strict_schema_no_tools_no_store_and_authoritative_projection():
    prepared, _ = collect(SalesRequest(user_query="Отчет продаж"))
    seen = []
    def transport(req):
        body = json.loads(req.content)
        seen.append(body)
        assert req.url.path == "/v1/responses"
        assert body["model"] == "gpt-4.1-mini" and body["store"] is False
        assert not body.get("tools")
        format = body["text"]["format"]
        assert format["type"] == "json_schema" and format["strict"] is True
        assert "sales_overview" not in format["schema"]["properties"]
        return httpx.Response(200, json=response({"summary": "",
            "findings": [], "recommendations": [{"kind": "source_completeness", "claim_index": None}], "data_gaps": [], "cross_agent_requests": []}))
    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as http:
            model = LangChainSalesModel(SalesModelSettings(api_key="synthetic-model-key"), http_async_client=http)
            result = await model.analyze(instructions="Fixture instructions", prepared=prepared)
        published = validate_sales_publication(result, prepared)
        assert published.summary == "Проверены доступные данные."
        assert published.sales_decomposition.rows[0][2] == 0
        assert published.facts[1].value == 0
    asyncio.run(exercise())
    assert len(seen) == 1


@pytest.mark.parametrize("profile", ["openrouter_gemma_free", "unknown"])
def test_profile_cannot_silently_fallback_to_openai(profile):
    with pytest.raises(ValueError):
        SalesModelSettings(api_key="fixture", profile=profile)


def test_secret_not_in_settings_repr_and_legacy_model_override_preserved(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-private-model-key")
    monkeypatch.setenv("OPENAI_MODEL", "configured-model-fixture")
    monkeypatch.setenv("LLM_MODEL_PROFILE", "openai_default")
    settings = SalesModelSettings.from_environment()
    assert settings.model == "configured-model-fixture"
    assert "synthetic-private" not in repr(settings)
