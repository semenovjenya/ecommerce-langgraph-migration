"""Sales graph must hydrate complete M2 reports without persisting raw artifacts."""
import asyncio
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from uuid import uuid4

import httpx
import pytest
from langgraph.checkpoint.memory import InMemorySaver

from commerce_agent.adapters.contracts import ToolArtifactReference
from commerce_agent.sales.runtime import build_sales_subgraph
from commerce_agent.serialization import checkpoint_serializer
from test_sales_evidence import CONTEXT, respond
from test_sales_runtime import FixtureModel, dependencies, invocation


RAW_MARKER = "native-uncheckpointed-markdown-"


class ArtifactTransport:
    """Real adapter wire; reports use fixture data, never real store/provider."""
    def __init__(self, failure=None):
        self.failure = failure
        self.gets = []
        self.references = {}

    def __call__(self, request):
        if request.method == "POST":
            response = respond(request)
            if not request.url.path.endswith("/report-snapshot"):
                return response
            payload = response.json()
            payload["data"]["markdown_table"] = RAW_MARKER * 12_000
            if self.failure == "secret":
                payload["data"]["markdown_table"] += "lsv2_synthetic_artifact_canary_123456"
            encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
            assert len(encoded) > 262_144
            reference = ToolArtifactReference(
                id=uuid4(), operation="sales.report_snapshot", result_schema="SalesSnapshot",
                sha256=sha256(encoded).hexdigest(), byte_size=len(encoded),
                expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            )
            self.references[str(reference.id)] = (reference, encoded)
            return httpx.Response(200, json=reference.model_dump(mode="json"))
        assert request.method == "GET"
        reference, encoded = self.references[request.url.path.rsplit("/", 1)[-1]]
        self.gets.append(request)
        assert request.headers["x-commerce-tenant-id"] == CONTEXT.tenant_id
        assert request.headers["x-commerce-user-id"] == CONTEXT.user_id
        assert request.headers["accept-encoding"] == "identity"
        if self.failure in {"denied", "expired"}:
            return httpx.Response(403 if self.failure == "denied" else 410)
        if self.failure == "corrupt":
            encoded = encoded[:-1] + b" "
        return httpx.Response(200, content=encoded, headers={
            "X-Artifact-SHA256": reference.sha256, "X-Artifact-Bytes": str(reference.byte_size),
        })


def assert_no_raw_checkpoint(graph_state, transport):
    encoded = json.dumps(graph_state, ensure_ascii=False, default=str)
    assert RAW_MARKER not in encoded
    assert "lsv2_synthetic_artifact_canary" not in encoded
    assert "schema_version\": 2" not in encoded
    for artifact_id, (reference, _) in transport.references.items():
        assert artifact_id not in encoded and reference.sha256 not in encoded


def test_large_report_hydrates_before_model_and_checkpoint_contains_only_projection():
    async def exercise():
        transport, model = ArtifactTransport(), FixtureModel()
        saver = InMemorySaver(serde=checkpoint_serializer())
        graph = build_sales_subgraph(dependencies(model, transport), checkpointer=saver)
        config = {"configurable": {"thread_id": "m3-sales-artifact-" + str(uuid4())}}
        result = await graph.ainvoke(invocation(), config, context=CONTEXT)
        assert len(transport.gets) == 1 and len(model.calls) == 1
        assert result["sales_analysis"].result.facts[1].value == 0
        assert result["agent_statuses"]["sales"].status == "completed"
        inline = await build_sales_subgraph(dependencies()).ainvoke(invocation(), context=CONTEXT)
        for name in ("sales_report_snapshot", "sales_overview", "sales_dynamics", "sales_decomposition"):
            assert getattr(result["sales_analysis"].result, name) == getattr(inline["sales_analysis"].result, name)
        for actual, expected in zip(result["sales_analysis"].result.facts, inline["sales_analysis"].result.facts, strict=True):
            assert actual.model_dump(exclude={"evidence_ref"}) == expected.model_dump(exclude={"evidence_ref"})
        assert_no_raw_checkpoint(model.calls[0][1], transport)
        async for snapshot in graph.aget_state_history(config):
            assert_no_raw_checkpoint(snapshot.values, transport)
    asyncio.run(exercise())


@pytest.mark.parametrize("failure", ["denied", "expired", "corrupt", "secret"])
def test_unavailable_or_unsafe_large_report_never_calls_model_or_publishes_result(failure):
    async def exercise():
        transport, model = ArtifactTransport(failure), FixtureModel()
        graph = build_sales_subgraph(dependencies(model, transport),
                                     checkpointer=InMemorySaver(serde=checkpoint_serializer()))
        config = {"configurable": {"thread_id": "m3-sales-artifact-" + str(uuid4())}}
        result = await graph.ainvoke(invocation(), config, context=CONTEXT)
        assert len(transport.gets) == 1 and not model.calls
        assert result["sales_analysis"].status == "failed"
        assert result["sales_analysis"].result is None
        assert result["errors"][0].code == "SALES_NEEDS_DATA"
        async for snapshot in graph.aget_state_history(config):
            assert_no_raw_checkpoint(snapshot.values, transport)
    asyncio.run(exercise())
