from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts import check_langsmith_state_summary as probe


def _run(*, outputs=None, inputs=None, metadata=None, name="commerce_agent"):
    extra = {"metadata": metadata or {}}
    return SimpleNamespace(
        name=name,
        inputs=inputs,
        outputs=outputs,
        extra=extra,
        error=None,
        model_dump=lambda **_kwargs: {
            "name": name, "inputs": inputs, "outputs": outputs, "extra": extra
        },
    )


def test_state_summary_readback_accepts_only_allowlisted_progress() -> None:
    safe = {
        "state_summary": {
            "agent_statuses": {"sales": "planned"},
            "analysis_statuses": {"finance": "not_implemented"},
            "data_gap_count": 5,
        }
    }
    root = _run(outputs=safe, inputs={}, metadata={"ls_run_depth": 0})
    assert probe._safe_readback([root], request_id="synthetic-request") is root
    assert probe._safe_readback(
        [_run(outputs={"messages": [{"content": "private"}]})],
        request_id="synthetic-request",
    ) is None
    assert probe._safe_readback(
        [_run(outputs={"state_summary": {**safe["state_summary"], "private": "value"}})],
        request_id="synthetic-request",
    ) is None
    assert probe._safe_readback(
        [_run(outputs=safe, inputs={"private": "value"})],
        request_id="synthetic-request",
    ) is None
    assert probe._safe_readback(
        [_run(outputs=safe, metadata={"tenant_id": "private"})],
        request_id="synthetic-request",
    ) is None


def test_state_summary_probe_stops_before_graph_without_eu_access(monkeypatch) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "synthetic-test-key")
    monkeypatch.setenv(
        "COMMERCE_AGENT_DATABASE_URI",
        "postgresql://local:synthetic@localhost:55432/commerce_agent",
    )
    monkeypatch.setenv("COMMERCE_AGENT_ENVIRONMENT", "development")
    with patch.object(probe, "Client") as client, patch.object(
        probe, "_invoke_current_graph"
    ) as graph_run:
        client.return_value.list_projects.side_effect = PermissionError("forbidden")
        assert probe.main() == 1
        graph_run.assert_not_called()


def test_state_summary_probe_rejects_nonlocal_database_before_network(monkeypatch) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "synthetic-test-key")
    monkeypatch.setenv(
        "COMMERCE_AGENT_DATABASE_URI",
        "postgresql://user:synthetic@production-db.example/commerce_agent",
    )
    with patch.object(probe, "Client") as client:
        assert probe.main() == 2
        client.assert_not_called()


@pytest.mark.parametrize(
    "database_uri",
    [
        "postgresql://user:synthetic@localhost:55432/commerce_agent?host=production-db.example",
        "postgresql://user:synthetic@localhost:55432/commerce_agent?hostaddr=203.0.113.4",
        "postgresql://user:synthetic@localhost:55432/commerce_agent?service=production",
        "postgresql://user:synthetic@localhost:55432/commerce_agent?port=5432",
    ],
)
def test_state_summary_probe_rejects_libpq_target_overrides(database_uri: str) -> None:
    assert not probe._is_local_development_database_uri(database_uri, {})


@pytest.mark.parametrize("name", ["PGHOSTADDR", "PGSERVICE", "PGHOST"])
def test_state_summary_probe_rejects_environment_target_overrides(name: str) -> None:
    database_uri = "postgresql://user:synthetic@localhost:55432/commerce_agent"
    assert not probe._is_local_development_database_uri(
        database_uri, {name: "production-db.example"}
    )
