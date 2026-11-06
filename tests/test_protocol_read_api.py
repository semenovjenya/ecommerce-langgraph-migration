from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
from uuid import uuid4

import httpx
from langchain_core.messages import HumanMessage
import pytest

from commerce_agent.context import CommerceRuntimeContext
from commerce_agent.persistence import (
    PersistenceSettings,
    initialize_checkpoint_schema,
    open_postgres_graph,
)
from commerce_agent.protocol_read_api import (
    ReadResources,
    StateSchemaMismatch,
    create_read_app,
    project_live_state,
    project_snapshot,
)
from commerce_agent.protocol_store import ProtocolRecordError, ThreadOwner, open_protocol_store
from commerce_agent.safety import SensitiveStateError


def test_state_projection_rejects_checkpoint_secrets() -> None:
    snapshot = SimpleNamespace(
        values={"messages": [], "api_key": "synthetic-canary"},
        next=(),
        config={"configurable": {"checkpoint_id": "synthetic-checkpoint"}},
    )
    with pytest.raises(SensitiveStateError):
        project_snapshot(snapshot)


def test_live_state_projection_uses_run_lifecycle_for_sdk_hydration() -> None:
    run_id = str(uuid4())
    task = SimpleNamespace(interrupts=(
        SimpleNamespace(id="approval-1", value={"kind": "approval"}),
    ))
    snapshot = SimpleNamespace(values={}, next=(), tasks=(task,), config={})

    queued = project_live_state(snapshot, "queued", run_id)
    assert queued["run_status"] == "queued"
    assert queued["run_id"] == run_id
    assert "next" not in queued
    assert queued["tasks"] == []

    interrupted = project_live_state(snapshot, "interrupted", run_id)
    assert interrupted["tasks"] == [{
        "interrupts": [{"id": "approval-1", "value": {"kind": "approval"}}],
    }]

    cancelled = project_live_state(snapshot, "cancelled", run_id)
    assert cancelled["next"] == []
    assert cancelled["tasks"] == []
    assert cancelled["run_status"] == "cancelled"


def test_live_state_projection_rejects_secret_in_interrupt() -> None:
    task = SimpleNamespace(interrupts=(
        SimpleNamespace(id="approval-1", value={"api_key": "synthetic-canary"}),
    ))
    snapshot = SimpleNamespace(values={}, next=(), tasks=(task,), config={})
    with pytest.raises(SensitiveStateError):
        project_live_state(snapshot, "interrupted")


def test_state_projection_rejects_extra_or_unversioned_checkpoint_fields() -> None:
    values = {
        "schema_version": 1,
        "messages": [],
        "tenant_id": "tenant",
        "user_id": "user",
        "request_id": "request",
    }
    snapshot = SimpleNamespace(
        values={**values, "internal_debug_payload": "private-but-not-a-secret-token"},
        next=(),
        config={"configurable": {"checkpoint_id": "synthetic-checkpoint"}},
    )
    with pytest.raises(StateSchemaMismatch, match="schema mismatch"):
        project_snapshot(snapshot)
    snapshot.values = {**values, "schema_version": 2}
    with pytest.raises(StateSchemaMismatch, match="schema mismatch"):
        project_snapshot(snapshot)


def test_internal_read_api_reports_schema_mismatch_without_state_details() -> None:
    class OwnedStore:
        async def require_owned(self, *_args):
            return None

        async def latest_run_lifecycle(self, *_args):
            return None

    class BadGraph:
        async def aget_state(self, _config):
            return SimpleNamespace(
                values={"schema_version": 2, "internal_debug_payload": "private"},
                next=(), config={},
            )

    token = "synthetic-internal-service-token-" + "x" * 32
    app = create_read_app(
        resources=ReadResources(store=OwnedStore(), graph=BadGraph()),
        service_token=token,
    )

    async def exercise() -> None:
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://internal.test"
            ) as client:
                response = await client.get(
                    f"/threads/{uuid4()}/state",
                    headers={
                        "X-Commerce-Service-Token": token,
                        "X-Commerce-Tenant-ID": "tenant",
                        "X-Commerce-User-ID": "user",
                    },
                )
                assert response.status_code == 409
                assert response.json() == {"detail": "state schema mismatch"}
                assert "internal_debug_payload" not in response.text

    asyncio.run(exercise())


def test_state_read_retries_when_another_tab_starts_a_run_between_reads() -> None:
    old_run, new_run, thread_id = str(uuid4()), str(uuid4()), str(uuid4())

    class RacingStore:
        reads = 0

        async def require_owned(self, *_args):
            return None

        async def latest_run_lifecycle(self, *_args):
            self.reads += 1
            return (old_run, "completed") if self.reads == 1 else (new_run, "queued")

    class RacingGraph:
        reads = 0

        async def aget_state(self, _config):
            self.reads += 1
            return SimpleNamespace(values={}, next=(), tasks=(), config={})

    store, graph = RacingStore(), RacingGraph()
    token = "synthetic-internal-service-token-" + "x" * 32
    app = create_read_app(
        resources=ReadResources(store=store, graph=graph), service_token=token,
    )

    async def exercise() -> None:
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://internal.test",
            ) as client:
                response = await client.get(
                    f"/threads/{thread_id}/state",
                    headers={
                        "X-Commerce-Service-Token": token,
                        "X-Commerce-Tenant-ID": "tenant",
                        "X-Commerce-User-ID": "user",
                    },
                )
        assert response.status_code == 200, response.text
        assert response.json()["run_status"] == "queued"
        assert response.json()["run_id"] == new_run
        assert "next" not in response.json()
        assert store.reads == 4
        assert graph.reads == 2

    asyncio.run(exercise())


def test_state_read_fails_retryably_when_lifecycle_never_stabilizes() -> None:
    thread_id = str(uuid4())

    class FlappingStore:
        reads = 0

        async def require_owned(self, *_args):
            return None

        async def latest_run_lifecycle(self, *_args):
            self.reads += 1
            return (str(self.reads), "running")

    class Graph:
        async def aget_state(self, _config):
            return SimpleNamespace(values={}, next=(), tasks=(), config={})

    token = "synthetic-internal-service-token-" + "x" * 32
    store = FlappingStore()
    app = create_read_app(
        resources=ReadResources(store=store, graph=Graph()), service_token=token,
    )

    async def exercise() -> None:
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://internal.test",
            ) as client:
                response = await client.get(
                    f"/threads/{thread_id}/state",
                    headers={
                        "X-Commerce-Service-Token": token,
                        "X-Commerce-Tenant-ID": "tenant",
                        "X-Commerce-User-ID": "user",
                    },
                )
        assert response.status_code == 503
        assert response.headers["retry-after"] == "1"
        assert response.json() == {"detail": "state changed during read"}
        assert store.reads == 6

    asyncio.run(exercise())


def test_state_projection_limits_response_size() -> None:
    snapshot = SimpleNamespace(
        values={
            "schema_version": 1,
            "messages": [{"content": "x" * 1_048_576}],
            "tenant_id": "tenant",
            "user_id": "user",
            "request_id": "request",
        },
        next=(),
        config={"configurable": {"checkpoint_id": "synthetic-checkpoint"}},
    )
    with pytest.raises(ProtocolRecordError, match="size limit"):
        project_snapshot(snapshot)


def test_real_postgres_checkpoint_is_readable_only_by_thread_owner() -> None:
    if not os.getenv("COMMERCE_AGENT_DATABASE_URI"):
        pytest.skip("dedicated development PostgreSQL is not configured")
    settings = PersistenceSettings.from_environment()
    thread_id = str(uuid4())
    owner = ThreadOwner(tenant_id="read-api-tenant", user_id="user-47")
    context = CommerceRuntimeContext(
        tenant_id=owner.tenant_id,
        user_id=owner.user_id,
        roles=("viewer",),
        request_id="read-api-synthetic-request",
    )
    token = "synthetic-internal-service-token-" + "x" * 32
    path = f"/threads/{thread_id}"

    async def cleanup(store) -> None:
        # Only this test's minted UUID is removed; no production or shared rows.
        async with store._pool.connection() as conn:
            await conn.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s", (thread_id,))
            await conn.execute("DELETE FROM checkpoint_blobs WHERE thread_id = %s", (thread_id,))
            await conn.execute("DELETE FROM checkpoints WHERE thread_id = %s", (thread_id,))
            await conn.execute("DELETE FROM commerce_agent_threads WHERE thread_id = %s", (thread_id,))

    async def exercise() -> None:
        await initialize_checkpoint_schema(settings)
        async with open_protocol_store(settings) as store:
            await store.setup()
            await store.create_thread(owner, thread_id)
            try:
                async with open_postgres_graph(settings) as graph:
                    await graph.ainvoke(
                        {
                            "schema_version": 1,
                            "messages": [HumanMessage(content="safe synthetic state query")],
                        },
                        config={"configurable": {"thread_id": thread_id}},
                        context=context,
                    )
                    app = create_read_app(
                        resources=ReadResources(store=store, graph=graph),
                        service_token=token,
                    )
                    transport = httpx.ASGITransport(app=app)
                    async with app.router.lifespan_context(app):
                        async with httpx.AsyncClient(
                            transport=transport, base_url="http://internal.test"
                        ) as client:
                            headers = {
                                "X-Commerce-Service-Token": token,
                                "X-Commerce-Tenant-ID": owner.tenant_id,
                                "X-Commerce-User-ID": owner.user_id,
                            }
                            state = await client.get(f"{path}/state", headers=headers)
                            assert state.status_code == 200, state.text
                            payload = state.json()
                            assert payload["values"]["schema_version"] == 1
                            assert set(payload["values"]["agent_statuses"]) == {
                                "sales", "marketing", "supply", "price", "finance"
                            }
                            assert payload["checkpoint"]["checkpoint_id"]
                            assert payload["next"] == []
                            assert state.headers["cache-control"] == "no-store"

                            history = await client.get(f"{path}/history?limit=3", headers=headers)
                            assert history.status_code == 200, history.text
                            assert len(history.json()["items"]) == 3
                            assert history.json()["items"][0]["checkpoint"]
                            all_history = await client.get(
                                f"{path}/history?limit=100", headers=headers
                            )
                            assert all_history.status_code == 200, all_history.text
                            assert len(all_history.json()["items"]) >= 3

                            assert (await client.get(f"{path}/state")).status_code == 401
                            assert (await client.get(
                                f"{path}/state",
                                headers={**headers, "X-Commerce-Service-Token": "wrong"},
                            )).status_code == 401
                            assert (await client.get(
                                f"{path}/state",
                                headers={**headers, "X-Commerce-User-ID": "other-user"},
                            )).status_code == 404
                            assert (await client.get(
                                f"{path}/history",
                                headers={**headers, "X-Commerce-Tenant-ID": "other-tenant"},
                            )).status_code == 404
                            assert (await client.get(
                                "/threads/not-a-uuid/state", headers=headers
                            )).status_code == 422
                            assert (await client.get(
                                f"{path}/history?limit=101", headers=headers
                            )).status_code == 422
            finally:
                await cleanup(store)

    asyncio.run(exercise())
