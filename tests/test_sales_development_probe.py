import asyncio
import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from openai import BadRequestError

from scripts import check_sales_development as probe


def test_private_token_requires_owned_regular_private_file(tmp_path):
    token = tmp_path / "token"
    token.write_text("a" * 64)
    token.chmod(0o600)
    assert probe._private_token(token) == "a" * 64
    token.chmod(0o644)
    with pytest.raises(ValueError):
        probe._private_token(token)
    token.chmod(0o600)
    alias = tmp_path / "alias"
    alias.symlink_to(token)
    with pytest.raises(OSError):
        probe._private_token(alias)
    token.write_text("synthetic-invalid-value")
    with pytest.raises(ValueError):
        probe._private_token(token)


@pytest.mark.parametrize("actor,environment,uri", [
    (0, "development", "postgresql://local:synthetic@localhost:55432/commerce_agent"),
    (1, "production", "postgresql://local:synthetic@localhost:55432/commerce_agent"),
    (1, "development", "postgresql://local:synthetic@production.example/commerce_agent"),
    (1, "development", "postgresql://local:synthetic@localhost:55432/commerce_agent?host=production.example"),
])
def test_probe_rejects_target_before_credentials_network_or_graph(monkeypatch, actor, environment, uri, capsys):
    monkeypatch.setenv("COMMERCE_AGENT_ENVIRONMENT", environment)
    monkeypatch.setenv("COMMERCE_AGENT_DATABASE_URI", uri)
    monkeypatch.setattr("sys.argv", ["probe", "--actor-id", str(actor), "--tools-token-file", "absent"])
    with patch.object(probe, "_private_token") as token, patch.object(probe.httpx, "get") as network, \
         patch.object(probe, "_run_sales") as graph:
        assert probe.main() == 1
        token.assert_not_called()
        network.assert_not_called()
        graph.assert_not_called()
    assert "synthetic" not in capsys.readouterr().out


def test_probe_eu_preflight_denial_prevents_model_and_graph(monkeypatch, capsys):
    monkeypatch.setenv("COMMERCE_AGENT_ENVIRONMENT", "development")
    monkeypatch.setenv("COMMERCE_AGENT_DATABASE_URI", "postgresql://local:synthetic@localhost:55432/commerce_agent")
    monkeypatch.setattr("sys.argv", ["probe", "--actor-id", "1", "--tools-token-file", "absent", "--trace"])
    response = SimpleNamespace(raise_for_status=lambda: None,
        json=lambda: {"provider": "openai", "profile": "openai_default", "model": "test-model"})
    with patch.dict(os.environ), patch.object(probe, "_private_token", return_value="a" * 64), \
         patch.object(probe.httpx, "get", return_value=response), patch.object(probe, "Client") as client, \
         patch.object(probe, "_run_sales") as graph:
        client.return_value.list_projects.side_effect = PermissionError("synthetic-private-provider-body")
        assert probe.main() == 1
        graph.assert_not_called()
        assert client.call_args.kwargs["api_url"] == probe.EU_LANGSMITH_API_ENDPOINT
        assert client.call_args.kwargs["hide_inputs"] is True
        assert client.call_args.kwargs["hide_outputs"] is True
    assert "synthetic-private" not in capsys.readouterr().out


@pytest.mark.parametrize("error,expected", [
    (ValueError("Sales grounded summary cannot contain provider prose"), "mixed_grounded_summary"),
    (ValueError("synthetic-private-model-output"), "suppressed"),
])
def test_local_diagnostic_suppresses_unknown_model_bodies(error, expected):
    model = probe.DiagnosticSalesModel(SimpleNamespace(analyze=AsyncMock(side_effect=error)))
    with pytest.raises(ValueError):
        asyncio.run(model.analyze(instructions="", prepared={}))
    assert model.diagnostic == {"stage": "model", "reason": expected}
    assert "synthetic-private" not in json.dumps(model.diagnostic)


def test_local_diagnostic_provider_error_exports_only_numeric_status():
    response = httpx.Response(400, request=httpx.Request("POST", "https://example.invalid"))
    error = BadRequestError("synthetic-private-body", response=response,
                            body={"private": "synthetic-private-body"})
    model = probe.DiagnosticSalesModel(SimpleNamespace(analyze=AsyncMock(side_effect=error)))
    with pytest.raises(BadRequestError):
        asyncio.run(model.analyze(instructions="", prepared={}))
    assert model.diagnostic == {"stage": "model", "reason": "suppressed", "http_status": 400}


def test_local_diagnostic_publication_failure_is_not_a_model_failure():
    model = probe.DiagnosticSalesModel(SimpleNamespace(analyze=AsyncMock(return_value=object())))
    with patch("commerce_agent.sales.publication.validate_sales_publication",
               side_effect=ValueError("Sales directional prose needs a bound native comparison")):
        with pytest.raises(ValueError):
            asyncio.run(model.analyze(instructions="", prepared={}))
    assert model.diagnostic == {"stage": "publication", "reason": "unbound_direction"}


def test_local_diagnostic_success_returns_validated_value():
    candidate, validated = object(), object()
    model = probe.DiagnosticSalesModel(SimpleNamespace(analyze=AsyncMock(return_value=candidate)))
    with patch("commerce_agent.sales.publication.validate_sales_publication", return_value=validated) as check:
        assert asyncio.run(model.analyze(instructions="", prepared={})) is validated
        check.assert_called_once_with(candidate, {})
    assert model.diagnostic == {}
