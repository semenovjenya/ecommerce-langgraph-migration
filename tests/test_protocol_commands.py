from __future__ import annotations

import asyncio
import json
import os
from operator import add
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Annotated
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
import pytest
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from psycopg.errors import LockNotAvailable
from typing_extensions import TypedDict

from commerce_agent.context import CommerceRuntimeContext
from commerce_agent.persistence import (
    PersistenceSettings, initialize_checkpoint_schema, open_postgres_checkpointer,
    open_postgres_graph,
)
from commerce_agent.protocol_commands import parse_respond_command, parse_start_command
from commerce_agent.protocol_events import normalize_graph_event, protocol_json_value
from commerce_agent.protocol_read_api import ReadResources, create_read_app
from commerce_agent.protocol_store import (
    ActiveRunConflict, ProtocolRecordError, ThreadNotOwned, ThreadOwner,
    cancellation_events, lifecycle_event, open_protocol_store,
)
from commerce_agent.safety import SensitiveStateError
from commerce_agent.protocol_worker import execute_claimed_run, run_once
from commerce_agent.ingress import GuardedGraph


def _start_command(
    command_id: str | int,
    text: str = "safe synthetic question",
    *,
    thread_id: str | None = None,
) -> dict:
    return {
        "id": command_id,
        "method": "run.start",
        "params": {
            "assistant_id": "commerce_agent",
            "input": {"messages": [{"type": "human", "content": text}]},
            **({"config": {"configurable": {"thread_id": thread_id}}}
               if thread_id is not None else {}),
        },
    }


def test_start_command_is_text_only_and_identity_is_server_owned() -> None:
    owner = ThreadOwner(tenant_id="tenant", user_id="user")
    thread_id = str(uuid4())
    command = parse_start_command(
        _start_command("command-1", thread_id=thread_id), thread_id=thread_id, owner=owner,
        role="viewer", request_id=str(uuid4()),
    )
    assert command.graph_input["schema_version"] == 1
    assert command.graph_input["messages"][0]["type"] == "human"
    assert command.runtime_context["tenant_id"] == "tenant"
    assert command.runtime_context["user_id"] == "user"
    assert command.runtime_context["roles"] == ["viewer"]

    forged = _start_command("command-2")
    forged["params"]["context"] = {"tenant_id": "other"}
    with pytest.raises(ProtocolRecordError, match="unsupported"):
        parse_start_command(forged, thread_id=thread_id, owner=owner, role="viewer", request_id=str(uuid4()))
    forged = _start_command("command-3")
    forged["params"]["input"]["messages"][0]["type"] = "ai"
    with pytest.raises(ProtocolRecordError, match="only user"):
        parse_start_command(forged, thread_id=thread_id, owner=owner, role="viewer", request_id=str(uuid4()))
    with pytest.raises(SensitiveStateError):
        parse_start_command(
            _start_command("command-4", "lsv2_credential_like_canary_12345"),
            thread_id=thread_id, owner=owner, role="viewer", request_id=str(uuid4()),
        )
    wrong_thread = _start_command("command-5", thread_id=str(uuid4()))
    with pytest.raises(ProtocolRecordError, match="owned thread"):
        parse_start_command(
            wrong_thread, thread_id=thread_id, owner=owner,
            role="viewer", request_id=str(uuid4()),
        )


def test_pinned_react_sdk_run_start_fixture_matches_parser() -> None:
    fixture_path = (
        Path(__file__).resolve().parents[1]
        / "frontend-probe/fixtures/run-start-v1.json"
    )
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    thread_id = payload["params"]["config"]["configurable"]["thread_id"]
    parsed = parse_start_command(
        payload,
        thread_id=thread_id,
        owner=ThreadOwner(tenant_id="probe-tenant", user_id="probe-user"),
        role="viewer",
        request_id=str(uuid4()),
    )
    assert parsed.response_id == 1
    assert parsed.command_id.startswith("sdk-v1:")
    assert parsed.graph_input["messages"][0]["content"] == "safe question"


def test_numeric_sdk_correlation_is_not_durable_idempotency_identity() -> None:
    owner = ThreadOwner(tenant_id="probe-tenant", user_id="probe-user")
    thread_id = str(uuid4())

    def parse(message_id: str, content: str):
        payload = _start_command(1, content, thread_id=thread_id)
        payload["params"]["input"]["messages"][0]["id"] = message_id
        return parse_start_command(
            payload, thread_id=thread_id, owner=owner, role="viewer",
            request_id=str(uuid4()),
        )

    first = parse("stable-message-1", "first")
    retry = parse("stable-message-1", "first")
    second = parse("stable-message-2", "second")
    assert first.response_id == retry.response_id == second.response_id == 1
    assert first.command_id == retry.command_id
    assert first.command_id != second.command_id
    without_message_id = _start_command(1, thread_id=thread_id)
    with pytest.raises(ProtocolRecordError, match="stable message IDs"):
        parse_start_command(
            without_message_id, thread_id=thread_id, owner=owner,
            role="viewer", request_id=str(uuid4()),
        )
    boolean_id = _start_command(True, thread_id=thread_id)
    with pytest.raises(ProtocolRecordError, match="invalid command ID"):
        parse_start_command(
            boolean_id, thread_id=thread_id, owner=owner,
            role="viewer", request_id=str(uuid4()),
        )


def test_respond_parser_is_manager_only_and_rejects_graph_mutations() -> None:
    payload = {
        "id": 2, "method": "input.respond",
        "params": {
            "namespace": [], "interrupt_id": "synthetic-interrupt",
            "response": {"approved": False},
        },
    }
    parsed = parse_respond_command(payload, role="manager")
    assert parsed.response_id == 2
    assert parsed.response == {"approved": False}
    with pytest.raises(ProtocolRecordError, match="manager approval"):
        parse_respond_command(payload, role="viewer")
    forged = json.loads(json.dumps(payload))
    forged["params"]["goto"] = "action_execution"
    with pytest.raises(ProtocolRecordError, match="unsupported"):
        parse_respond_command(forged, role="manager")
    forged = json.loads(json.dumps(payload))
    forged["params"]["response"]["approved"] = "yes"
    with pytest.raises(ProtocolRecordError, match="synthetic approval"):
        parse_respond_command(forged, role="manager")

def test_command_http_persists_one_owned_queued_run_and_retries_idempotently() -> None:
    if not os.getenv("COMMERCE_AGENT_DATABASE_URI"):
        pytest.skip("dedicated development PostgreSQL is not configured")
    settings = PersistenceSettings.from_environment()
    owner = ThreadOwner(tenant_id="commands-tenant", user_id="user-47")
    thread_id = str(uuid4())
    service_token = "synthetic-internal-service-token-" + "x" * 32
    path = f"/threads/{thread_id}/commands"

    async def exercise() -> None:
        await initialize_checkpoint_schema(settings)
        async with open_protocol_store(settings) as store:
            await store.setup()
            app = create_read_app(
                resources=ReadResources(store=store, graph=object()),
                service_token=service_token,
            )
            headers = {
                "X-Commerce-Service-Token": service_token,
                "X-Commerce-Tenant-ID": owner.tenant_id,
                "X-Commerce-User-ID": owner.user_id,
                "X-Commerce-Role": "viewer",
                "X-Request-ID": str(uuid4()),
            }
            try:
                async with app.router.lifespan_context(app):
                    async with httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=app),
                        base_url="http://internal.test",
                    ) as client:
                        first = await client.post(
                            path,
                            json=_start_command("command-1", thread_id=thread_id),
                            headers=headers,
                        )
                        assert first.status_code == 200, first.text
                        run_id = first.json()["result"]["run_id"]
                        assert first.json()["id"] == "command-1"
                        retry = await client.post(
                            path, json=_start_command("command-1", thread_id=thread_id),
                            headers={**headers, "X-Request-ID": str(uuid4())},
                        )
                        assert retry.status_code == 200
                        assert retry.json()["result"]["run_id"] == run_id
                        conflict = await client.post(
                            path, json=_start_command("command-2"), headers=headers
                        )
                        assert conflict.status_code == 409
                        changed = await client.post(
                            path, json=_start_command(
                                "command-1", "changed text", thread_id=thread_id
                            ),
                            headers=headers,
                        )
                        assert changed.status_code == 409
                        assert (await client.post(path, json=_start_command("command-1"))).status_code == 401
                        assert (await client.post(
                            path, json=_start_command("command-3"),
                            headers={**headers, "X-Commerce-User-ID": "other-user"},
                        )).status_code == 404
                        assert (await client.post(
                            path, json=_start_command("command-4"),
                            headers={**headers, "X-Commerce-Tenant-ID": "other-tenant"},
                        )).status_code == 404
                        assert (await client.post(
                            path,
                            json=_start_command("command-5", "lsv2_credential_like_canary_12345"),
                            headers=headers,
                        )).status_code == 422
                        async with store._pool.connection() as conn:
                            cursor = await conn.execute(
                                "SELECT status, input_json, context_json FROM commerce_agent_runs WHERE run_id = %s",
                                (run_id,),
                            )
                            row = await cursor.fetchone()
                            assert row["status"] == "queued"
                            assert row["input_json"]["messages"][0]["content"] == "safe synthetic question"
                            assert row["context_json"]["tenant_id"] == owner.tenant_id
                            assert row["context_json"]["user_id"] == owner.user_id
                            assert "lsv2_credential_like_canary_12345" not in str(row)
            finally:
                async with store._pool.connection() as conn:
                    await conn.execute("DELETE FROM commerce_agent_events WHERE thread_id = %s", (thread_id,))
                    await conn.execute("DELETE FROM commerce_agent_runs WHERE thread_id = %s", (thread_id,))
                    await conn.execute("DELETE FROM commerce_agent_threads WHERE thread_id = %s", (thread_id,))

    asyncio.run(exercise())


def test_cancel_queued_run_is_owned_idempotent_and_terminal() -> None:
    if not os.getenv("COMMERCE_AGENT_DATABASE_URI"):
        pytest.skip("dedicated development PostgreSQL is not configured")
    settings = PersistenceSettings.from_environment()
    owner = ThreadOwner(tenant_id=f"cancel-queued-{uuid4().hex}", user_id="user-47")
    thread_id = str(uuid4())
    token = "synthetic-internal-service-token-" + "x" * 32

    async def exercise() -> None:
        await initialize_checkpoint_schema(settings)
        async with open_protocol_store(settings) as store:
            await store.setup()
            run_id = await store.create_run(
                owner, thread_id, command_id="cancel-queued-command",
                input_payload={"schema_version": 1, "messages": [{
                    "type": "human", "content": "safe queued question", "id": "queued-message",
                }]},
                runtime_context={
                    "tenant_id": owner.tenant_id, "user_id": owner.user_id,
                    "roles": ["viewer"], "request_id": str(uuid4()),
                },
            )
            app = create_read_app(
                resources=ReadResources(store=store, graph=object()),
                service_token=token,
            )
            path = f"/threads/{thread_id}/runs/{run_id}/cancel"
            headers = {
                "X-Commerce-Service-Token": token,
                "X-Commerce-Tenant-ID": owner.tenant_id,
                "X-Commerce-User-ID": owner.user_id,
            }
            try:
                async with app.router.lifespan_context(app):
                    async with httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=app),
                        base_url="http://internal.test",
                    ) as client:
                        params = {"wait": "0", "action": "interrupt"}
                        first = await client.post(path, params=params, headers=headers)
                        again = await client.post(path, params=params, headers=headers)
                        assert first.status_code == again.status_code == 204
                        assert (await client.post(path, params=params)).status_code == 401
                        wrong_owner = await client.post(
                            path, params=params,
                            headers={**headers, "X-Commerce-User-ID": "other-user"},
                        )
                        assert wrong_owner.status_code == 404
                        assert (await client.post(
                            path, params={"wait": "1", "action": "interrupt"},
                            headers=headers,
                        )).status_code == 422
                        assert (await client.post(
                            path, params={"wait": "0", "action": "rollback"},
                            headers=headers,
                        )).status_code == 422
                async with store._pool.connection() as conn:
                    cursor = await conn.execute(
                        "SELECT status, attempts, cancel_requested FROM commerce_agent_runs WHERE run_id = %s",
                        (run_id,),
                    )
                    assert await cursor.fetchone() == {
                        "status": "cancelled", "attempts": 0, "cancel_requested": True,
                    }
                events = await store.read_events(owner, thread_id)
                assert [event["params"]["data"]["event"] for event in events] == [
                    "cancelled", "interrupted",
                ]
                assert events[-1]["params"]["data"]["reason"] == "cancelled"
            finally:
                async with store._pool.connection() as conn:
                    await conn.execute("DELETE FROM commerce_agent_events WHERE thread_id = %s", (thread_id,))
                    await conn.execute("DELETE FROM commerce_agent_runs WHERE thread_id = %s", (thread_id,))
                    await conn.execute("DELETE FROM commerce_agent_threads WHERE thread_id = %s", (thread_id,))

    asyncio.run(exercise())


def test_cancel_running_worker_stops_before_graph_completes() -> None:
    if not os.getenv("COMMERCE_AGENT_DATABASE_URI"):
        pytest.skip("dedicated development PostgreSQL is not configured")
    settings = PersistenceSettings.from_environment()
    owner = ThreadOwner(tenant_id=f"cancel-running-{uuid4().hex}", user_id="user-47")
    thread_id = str(uuid4())
    entered = asyncio.Event()
    exited = asyncio.Event()

    class SlowGraph:
        async def aget_state(self, _config):
            return SimpleNamespace(values={}, config={}, tasks=(), next=())

        async def astream_protocol_events(self, _input, *, config, context):
            async def events():
                entered.set()
                try:
                    await asyncio.Event().wait()
                    yield {"type": "event"}
                finally:
                    exited.set()

            return events()

    async def exercise() -> None:
        await initialize_checkpoint_schema(settings)
        async with open_protocol_store(settings) as api_store:
            await api_store.setup()
            run_id = await api_store.create_run(
                owner, thread_id, command_id="cancel-running-command",
                input_payload={"schema_version": 1, "messages": [{
                    "type": "human", "content": "safe running question", "id": "running-message",
                }]},
                runtime_context={
                    "tenant_id": owner.tenant_id, "user_id": owner.user_id,
                    "roles": ["viewer"], "request_id": str(uuid4()),
                },
            )
            try:
                async with open_protocol_store(settings) as worker_store:
                    worker_id = str(uuid4())
                    claim = await worker_store.claim_next_run(worker_id, lease_seconds=6)
                    assert claim is not None and claim.run_id == run_id
                    task = asyncio.create_task(execute_claimed_run(
                        worker_store, SlowGraph(), claim, worker_id=worker_id,
                        lease_seconds=6,
                    ))
                    try:
                        await asyncio.wait_for(entered.wait(), timeout=5)
                        assert await api_store.request_cancel(owner, thread_id, run_id) == "requested"
                        assert await api_store.request_cancel(owner, thread_id, run_id) == "requested"
                        assert await asyncio.wait_for(task, timeout=5) == "cancelled"
                        assert exited.is_set()
                    finally:
                        if not task.done():
                            task.cancel()
                            await asyncio.gather(task, return_exceptions=True)
                async with api_store._pool.connection() as conn:
                    cursor = await conn.execute(
                        "SELECT status, attempts, lease_owner FROM commerce_agent_runs WHERE run_id = %s",
                        (run_id,),
                    )
                    assert await cursor.fetchone() == {
                        "status": "cancelled", "attempts": 1, "lease_owner": None,
                    }
                events = await api_store.read_events(owner, thread_id)
                assert [event["params"]["data"]["event"] for event in events] == [
                    "running", "cancelled", "interrupted",
                ]
                assert events[-1]["params"]["data"]["reason"] == "cancelled"
            finally:
                async with api_store._pool.connection() as conn:
                    await conn.execute("DELETE FROM commerce_agent_events WHERE thread_id = %s", (thread_id,))
                    await conn.execute("DELETE FROM commerce_agent_runs WHERE thread_id = %s", (thread_id,))
                    await conn.execute("DELETE FROM commerce_agent_threads WHERE thread_id = %s", (thread_id,))

    asyncio.run(exercise())


def test_cancel_after_graph_stream_before_terminalization_settles_sdk() -> None:
    if not os.getenv("COMMERCE_AGENT_DATABASE_URI"):
        pytest.skip("dedicated development PostgreSQL is not configured")
    settings = PersistenceSettings.from_environment()
    owner = ThreadOwner(tenant_id=f"cancel-tail-{uuid4().hex}", user_id="user-47")
    thread_id = str(uuid4())
    worker_id = str(uuid4())

    class FinishedGraph:
        async def aget_state(self, _config):
            return SimpleNamespace(values={}, config={}, tasks=(), next=())

        async def astream_protocol_events(self, _input, *, config, context):
            async def events():
                yield lifecycle_event("completed")

            return events()

    async def exercise() -> None:
        await initialize_checkpoint_schema(settings)
        async with open_protocol_store(settings) as store:
            await store.setup()
            run_id = await store.create_run(
                owner, thread_id, command_id="cancel-tail-command",
                input_payload={"schema_version": 1, "messages": []},
                runtime_context={
                    "tenant_id": owner.tenant_id, "user_id": owner.user_id,
                    "roles": ["viewer"], "request_id": str(uuid4()),
                },
            )
            try:
                claim = await store.claim_next_run(worker_id, lease_seconds=10)
                assert claim is not None and claim.run_id == run_id
                original_check = store.is_cancel_requested
                executing_task = asyncio.current_task()
                checks = 0

                async def cancel_after_stream(run: str, worker: str) -> bool:
                    nonlocal checks
                    if asyncio.current_task() is executing_task:
                        checks += 1
                        if checks == 2:
                            assert await store.request_cancel(
                                owner, thread_id, run_id,
                            ) == "requested"
                    return await original_check(run, worker)

                store.is_cancel_requested = cancel_after_stream
                assert await execute_claimed_run(
                    store, FinishedGraph(), claim, worker_id=worker_id,
                    lease_seconds=10,
                ) == "cancelled"
                assert checks == 2
                events = await store.read_events(owner, thread_id)
                assert [event["params"]["data"]["event"] for event in events] == [
                    "running", "cancelled", "interrupted",
                ]
                assert events[-1]["params"]["data"]["reason"] == "cancelled"
            finally:
                async with store._pool.connection() as conn:
                    await conn.execute("DELETE FROM commerce_agent_events WHERE thread_id = %s", (thread_id,))
                    await conn.execute("DELETE FROM commerce_agent_runs WHERE thread_id = %s", (thread_id,))
                    await conn.execute("DELETE FROM commerce_agent_threads WHERE thread_id = %s", (thread_id,))

    asyncio.run(exercise())


def test_cancel_interrupted_run_is_terminal_and_idempotent() -> None:
    if not os.getenv("COMMERCE_AGENT_DATABASE_URI"):
        pytest.skip("dedicated development PostgreSQL is not configured")
    settings = PersistenceSettings.from_environment()
    owner = ThreadOwner(tenant_id=f"cancel-paused-{uuid4().hex}", user_id="user-47")
    thread_id = str(uuid4())
    worker_id = str(uuid4())

    async def exercise() -> None:
        await initialize_checkpoint_schema(settings)
        async with open_protocol_store(settings) as store:
            await store.setup()
            run_id = await store.create_run(
                owner, thread_id, command_id="cancel-paused-command",
                input_payload={"schema_version": 1, "messages": [{
                    "type": "human", "content": "safe paused question", "id": "paused-message",
                }]},
                runtime_context={
                    "tenant_id": owner.tenant_id, "user_id": owner.user_id,
                    "roles": ["viewer"], "request_id": str(uuid4()),
                },
            )
            try:
                claim = await store.claim_next_run(worker_id, lease_seconds=6)
                assert claim is not None and claim.run_id == run_id
                assert await store.finish_run(
                    run_id, worker_id, status="interrupted",
                    terminal_events=(lifecycle_event("interrupted"),),
                )
                assert await store.request_cancel(owner, thread_id, run_id) == "cancelled"
                assert await store.request_cancel(owner, thread_id, run_id) == "cancelled"
                events = await store.read_events(owner, thread_id)
                assert [event["params"]["data"]["event"] for event in events] == [
                    "interrupted", "cancelled", "interrupted",
                ]
                assert events[-1]["params"]["data"]["reason"] == "cancelled"
                async with store._pool.connection() as conn:
                    cursor = await conn.execute(
                        "SELECT status, attempts FROM commerce_agent_runs WHERE run_id = %s",
                        (run_id,),
                    )
                    assert await cursor.fetchone() == {"status": "cancelled", "attempts": 1}
            finally:
                async with store._pool.connection() as conn:
                    await conn.execute("DELETE FROM commerce_agent_events WHERE thread_id = %s", (thread_id,))
                    await conn.execute("DELETE FROM commerce_agent_runs WHERE thread_id = %s", (thread_id,))
                    await conn.execute("DELETE FROM commerce_agent_threads WHERE thread_id = %s", (thread_id,))

    asyncio.run(exercise())


def test_cancel_serializes_with_worker_event_and_rejects_late_events() -> None:
    if not os.getenv("COMMERCE_AGENT_DATABASE_URI"):
        pytest.skip("dedicated development PostgreSQL is not configured")
    settings = PersistenceSettings.from_environment()
    owner = ThreadOwner(tenant_id=f"cancel-event-{uuid4().hex}", user_id="user-47")
    thread_id = str(uuid4())
    worker_id = str(uuid4())

    async def exercise() -> None:
        await initialize_checkpoint_schema(settings)
        async with open_protocol_store(settings) as store:
            await store.setup()
            run_id = await store.create_run(
                owner, thread_id, command_id="cancel-event-command",
                input_payload={"schema_version": 1, "messages": []},
                runtime_context={
                    "tenant_id": owner.tenant_id, "user_id": owner.user_id,
                    "roles": ["viewer"], "request_id": str(uuid4()),
                },
            )
            claim = await store.claim_next_run(worker_id, lease_seconds=10)
            assert claim is not None and claim.run_id == run_id
            progress_event = {
                "type": "event", "method": "lifecycle",
                "params": {"namespace": [], "data": {"event": "running"}},
            }
            append_task: asyncio.Task | None = None
            cancel_task: asyncio.Task | None = None
            try:
                # Hold the run row so append_event must wait after locking
                # the thread. Cancellation must serialize behind that thread
                # lock, never commit before the in-flight event.
                async with store._pool.connection() as blocker:
                    async with blocker.transaction():
                        await blocker.execute(
                            "SELECT 1 FROM commerce_agent_runs WHERE run_id = %s FOR UPDATE",
                            (run_id,),
                        )
                        append_task = asyncio.create_task(store.append_event(
                            owner, thread_id, run_id, progress_event,
                            worker_id=worker_id,
                        ))
                        for _ in range(200):
                            async with store._pool.connection() as observer:
                                try:
                                    await observer.execute(
                                        "SELECT 1 FROM commerce_agent_threads "
                                        "WHERE thread_id = %s FOR UPDATE NOWAIT",
                                        (thread_id,),
                                    )
                                except LockNotAvailable:
                                    break
                            await asyncio.sleep(0.01)
                        else:
                            raise AssertionError("worker event did not lock thread")
                        cancel_task = asyncio.create_task(
                            store.request_cancel(owner, thread_id, run_id)
                        )
                assert append_task is not None and cancel_task is not None
                appended = await asyncio.wait_for(append_task, timeout=5)
                assert await asyncio.wait_for(cancel_task, timeout=5) == "requested"
                assert appended["seq"] == 0
                with pytest.raises(ThreadNotOwned):
                    await store.append_event(
                        owner, thread_id, run_id, progress_event,
                        worker_id=worker_id,
                    )
                assert await store.finish_run(
                    run_id, worker_id, status="cancelled",
                    terminal_events=cancellation_events(),
                )
                events = await store.read_events(owner, thread_id)
                assert [event["seq"] for event in events] == [0, 1, 2]
                assert [event["params"]["data"]["event"] for event in events] == [
                    "running", "cancelled", "interrupted",
                ]
            finally:
                for task in (append_task, cancel_task):
                    if task is not None and not task.done():
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                async with store._pool.connection() as conn:
                    await conn.execute("DELETE FROM commerce_agent_events WHERE thread_id = %s", (thread_id,))
                    await conn.execute("DELETE FROM commerce_agent_runs WHERE thread_id = %s", (thread_id,))
                    await conn.execute("DELETE FROM commerce_agent_threads WHERE thread_id = %s", (thread_id,))

    asyncio.run(exercise())


def test_approved_response_cancelled_before_resume_is_not_applied() -> None:
    if not os.getenv("COMMERCE_AGENT_DATABASE_URI"):
        pytest.skip("dedicated development PostgreSQL is not configured")
    settings = PersistenceSettings.from_environment()
    owner = ThreadOwner(tenant_id=f"cancel-resume-{uuid4().hex}", user_id="user-47")
    thread_id = str(uuid4())
    worker_id = str(uuid4())
    response = {"approved": False}

    async def exercise() -> None:
        await initialize_checkpoint_schema(settings)
        async with open_protocol_store(settings) as store:
            await store.setup()
            run_id = await store.create_run(
                owner, thread_id, command_id="cancel-resume-command",
                input_payload={"schema_version": 1, "messages": []},
                runtime_context={
                    "tenant_id": owner.tenant_id, "user_id": owner.user_id,
                    "roles": ["manager"], "request_id": str(uuid4()),
                },
            )
            try:
                claim = await store.claim_next_run(worker_id, lease_seconds=10)
                assert claim is not None and claim.run_id == run_id
                assert await store.finish_run(run_id, worker_id, status="interrupted")
                assert await store.accept_resume(
                    owner, thread_id, interrupt_id="cancel-resume-interrupt",
                    response=response, checkpoint_id="synthetic-checkpoint",
                    role="manager",
                ) == run_id
                claim = await store.claim_next_run(worker_id, lease_seconds=10)
                assert claim is not None and claim.run_id == run_id
                assert await store.request_cancel(owner, thread_id, run_id) == "requested"
                assert await store.finish_run(
                    run_id, worker_id, status="cancelled",
                    terminal_events=cancellation_events(),
                )
                async with store._pool.connection() as conn:
                    cursor = await conn.execute(
                        "SELECT applied FROM commerce_agent_interrupt_responses "
                        "WHERE run_id = %s AND interrupt_id = %s",
                        (run_id, "cancel-resume-interrupt"),
                    )
                    assert await cursor.fetchone() == {"applied": False}
                with pytest.raises(ActiveRunConflict, match="cancelled"):
                    await store.accept_resume(
                        owner, thread_id, interrupt_id="cancel-resume-interrupt",
                        response=response, checkpoint_id="synthetic-checkpoint",
                        role="manager",
                    )
            finally:
                async with store._pool.connection() as conn:
                    await conn.execute(
                        "DELETE FROM commerce_agent_interrupt_responses WHERE thread_id = %s",
                        (thread_id,),
                    )
                    await conn.execute("DELETE FROM commerce_agent_events WHERE thread_id = %s", (thread_id,))
                    await conn.execute("DELETE FROM commerce_agent_runs WHERE thread_id = %s", (thread_id,))
                    await conn.execute("DELETE FROM commerce_agent_threads WHERE thread_id = %s", (thread_id,))

    asyncio.run(exercise())


def test_cancel_and_resume_lock_thread_before_run_without_deadlock() -> None:
    if not os.getenv("COMMERCE_AGENT_DATABASE_URI"):
        pytest.skip("dedicated development PostgreSQL is not configured")
    settings = PersistenceSettings.from_environment()
    owner = ThreadOwner(tenant_id=f"cancel-race-{uuid4().hex}", user_id="user-47")
    thread_id = str(uuid4())
    worker_id = str(uuid4())

    async def exercise() -> None:
        await initialize_checkpoint_schema(settings)
        async with open_protocol_store(settings) as store:
            await store.setup()
            run_id = await store.create_run(
                owner, thread_id, command_id="cancel-race-command",
                input_payload={"schema_version": 1, "messages": []},
                runtime_context={
                    "tenant_id": owner.tenant_id, "user_id": owner.user_id,
                    "roles": ["manager"], "request_id": str(uuid4()),
                },
            )
            cancel_task: asyncio.Task | None = None
            resume_task: asyncio.Task | None = None
            try:
                claim = await store.claim_next_run(worker_id, lease_seconds=10)
                assert claim is not None and claim.run_id == run_id
                assert await store.finish_run(run_id, worker_id, status="interrupted")
                async with store._pool.connection() as thread_blocker:
                    async with thread_blocker.transaction():
                        await thread_blocker.execute(
                            "SELECT 1 FROM commerce_agent_threads "
                            "WHERE thread_id = %s FOR UPDATE",
                            (thread_id,),
                        )
                        async with store._pool.connection() as run_blocker:
                            async with run_blocker.transaction():
                                await run_blocker.execute(
                                    "SELECT 1 FROM commerce_agent_runs "
                                    "WHERE run_id = %s FOR UPDATE",
                                    (run_id,),
                                )
                                cancel_task = asyncio.create_task(
                                    store.request_cancel(owner, thread_id, run_id)
                                )
                                await asyncio.sleep(0.05)
                                resume_task = asyncio.create_task(store.accept_resume(
                                    owner, thread_id, interrupt_id="cancel-race-interrupt",
                                    response={"approved": False},
                                    checkpoint_id="synthetic-checkpoint", role="manager",
                                ))
                                await asyncio.sleep(0.05)
                            # The run row is free while the thread row remains
                            # blocked. Opposite lock orders would form a cycle.
                            await asyncio.sleep(0.05)
                assert cancel_task is not None and resume_task is not None
                outcomes = await asyncio.wait_for(asyncio.gather(
                    cancel_task, resume_task, return_exceptions=True,
                ), timeout=5)
                assert outcomes[0] == "cancelled"
                assert outcomes[1] == run_id or isinstance(outcomes[1], ActiveRunConflict)
                async with store._pool.connection() as conn:
                    cursor = await conn.execute(
                        "SELECT status FROM commerce_agent_runs WHERE run_id = %s", (run_id,),
                    )
                    assert await cursor.fetchone() == {"status": "cancelled"}
            finally:
                for task in (cancel_task, resume_task):
                    if task is not None and not task.done():
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                async with store._pool.connection() as conn:
                    await conn.execute(
                        "DELETE FROM commerce_agent_interrupt_responses WHERE thread_id = %s",
                        (thread_id,),
                    )
                    await conn.execute("DELETE FROM commerce_agent_events WHERE thread_id = %s", (thread_id,))
                    await conn.execute("DELETE FROM commerce_agent_runs WHERE thread_id = %s", (thread_id,))
                    await conn.execute("DELETE FROM commerce_agent_threads WHERE thread_id = %s", (thread_id,))

    asyncio.run(exercise())


def test_command_worker_checkpoint_and_replay_survive_pool_change() -> None:
    if not os.getenv("COMMERCE_AGENT_DATABASE_URI"):
        pytest.skip("dedicated development PostgreSQL is not configured")
    settings = PersistenceSettings.from_environment()
    owner = ThreadOwner(tenant_id="worker-tenant", user_id="user-47")
    thread_id = str(uuid4())
    token = "synthetic-internal-service-token-" + "x" * 32
    worker_one, worker_two = str(uuid4()), str(uuid4())

    async def cleanup(store) -> None:
        async with store._pool.connection() as conn:
            await conn.execute("DELETE FROM commerce_agent_events WHERE thread_id = %s", (thread_id,))
            await conn.execute("DELETE FROM commerce_agent_runs WHERE thread_id = %s", (thread_id,))
            await conn.execute("DELETE FROM commerce_agent_threads WHERE thread_id = %s", (thread_id,))
            await conn.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s", (thread_id,))
            await conn.execute("DELETE FROM checkpoint_blobs WHERE thread_id = %s", (thread_id,))
            await conn.execute("DELETE FROM checkpoints WHERE thread_id = %s", (thread_id,))

    async def exercise() -> None:
        await initialize_checkpoint_schema(settings)
        async with open_protocol_store(settings) as api_store:
            await api_store.setup()
            headers = {
                "X-Commerce-Service-Token": token,
                "X-Commerce-Tenant-ID": owner.tenant_id,
                "X-Commerce-User-ID": owner.user_id,
                "X-Commerce-Role": "viewer",
                "X-Request-ID": str(uuid4()),
            }
            app = create_read_app(
                resources=ReadResources(store=api_store, graph=object()),
                service_token=token,
            )
            try:
                async with app.router.lifespan_context(app):
                    async with httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=app),
                        base_url="http://internal.test",
                    ) as client:
                        command = await client.post(
                            f"/threads/{thread_id}/commands",
                            json=_start_command("worker-command"), headers=headers,
                        )
                        assert command.status_code == 200, command.text
                        run_id = command.json()["result"]["run_id"]
                # Another pool represents a separate worker process. First
                # worker claims but dies before graph invocation; its lease
                # expires, then the second worker must recover the same run.
                async with open_protocol_store(settings) as first_store:
                    claim = await first_store.claim_next_run(worker_one, lease_seconds=2)
                    assert claim is not None and claim.run_id == run_id
                    async with first_store._pool.connection() as conn:
                        await conn.execute(
                            "UPDATE commerce_agent_runs SET lease_until = now() - interval '1 second' WHERE run_id = %s",
                            (run_id,),
                        )
                async with open_protocol_store(settings) as worker_store:
                    async with open_postgres_graph(settings) as graph:
                        assert await run_once(
                            worker_store, graph, worker_id=worker_two,
                            lease_seconds=20,
                        )
                        snapshot = await graph.aget_state(
                            {"configurable": {"thread_id": thread_id}}
                        )
                        assert snapshot.values["schema_version"] == 1
                        assert snapshot.next == ()
                    events = await worker_store.read_events(owner, thread_id)
                    assert events
                    assert [event["seq"] for event in events] == list(range(len(events)))
                    assert any(event["method"] == "values" for event in events)
                    assert any(
                        event["method"] == "lifecycle"
                        and event["params"]["data"].get("event") == "completed"
                        for event in events
                    )
                    assert await worker_store.read_events(owner, thread_id, since=events[-2]["seq"]) == [events[-1]]
                    assert not await worker_store.finish_run(
                        run_id, worker_one, status="completed"
                    )
                    async with worker_store._pool.connection() as conn:
                        cursor = await conn.execute(
                            "SELECT status, attempts, lease_owner FROM commerce_agent_runs WHERE run_id = %s",
                            (run_id,),
                        )
                        row = await cursor.fetchone()
                        assert row["status"] == "completed"
                        assert row["attempts"] == 2
                        assert row["lease_owner"] is None
                        second_request_id = str(uuid4())
                        second_run_id = await api_store.create_run(
                            owner,
                            thread_id,
                            command_id="worker-command-2",
                            input_payload={
                                "schema_version": 1,
                                "messages": [{
                                    "type": "human",
                                    "content": "second safe synthetic question",
                                    "id": "second-message",
                                }],
                            },
                            runtime_context={
                                "tenant_id": owner.tenant_id,
                                "user_id": owner.user_id,
                                "roles": ["viewer"],
                                "request_id": second_request_id,
                            },
                        )
                        async with open_postgres_graph(settings) as second_graph:
                            assert await run_once(
                                worker_store, second_graph, worker_id=worker_two,
                                lease_seconds=20,
                            )
                            second_snapshot = await second_graph.aget_state(
                                {"configurable": {"thread_id": thread_id}}
                            )
                        assert second_snapshot.values["request_id"] == second_request_id
                        assert len(second_snapshot.values["messages"]) == 2
                        cursor = await conn.execute(
                            "SELECT status FROM commerce_agent_runs WHERE run_id = %s",
                            (second_run_id,),
                        )
                        assert (await cursor.fetchone())["status"] == "completed"
            finally:
                await cleanup(api_store)

    asyncio.run(exercise())


def test_worker_reconciles_completed_checkpoint_without_replaying_input_or_events() -> None:
    if not os.getenv("COMMERCE_AGENT_DATABASE_URI"):
        pytest.skip("dedicated development PostgreSQL is not configured")
    settings = PersistenceSettings.from_environment()
    owner = ThreadOwner(tenant_id="reconcile-tenant", user_id="user-47")
    thread_id = str(uuid4())
    first_worker, recovery_worker = str(uuid4()), str(uuid4())

    async def exercise() -> None:
        await initialize_checkpoint_schema(settings)
        async with open_protocol_store(settings) as store:
            await store.setup()
            run_id = await store.create_run(
                owner, thread_id, command_id="reconcile-command",
                input_payload={
                    "schema_version": 1,
                    "messages": [{
                        "type": "human", "content": "safe reconcile question",
                        "id": "reconcile-message",
                    }],
                },
                runtime_context={
                    "tenant_id": owner.tenant_id, "user_id": owner.user_id,
                    "roles": ["viewer"], "request_id": str(uuid4()),
                },
            )
            try:
                claim = await store.claim_next_run(first_worker, lease_seconds=20)
                assert claim is not None and claim.run_id == run_id
                async with open_postgres_graph(settings) as graph:
                    initial = await graph.aget_state(
                        {"configurable": {"thread_id": thread_id}}
                    )
                    assert initial.values == {}
                    assert await store.record_start_checkpoint(
                        run_id, first_worker, None
                    ) is None
                    stream = await graph.astream_protocol_events(
                        claim.input_payload,
                        config={"configurable": {"thread_id": thread_id}},
                        context=CommerceRuntimeContext.model_validate(claim.runtime_context),
                    )
                    async for raw in stream:
                        await store.append_event(
                            owner, thread_id, run_id,
                            normalize_graph_event(raw), worker_id=first_worker,
                        )
                before = await store.read_events(owner, thread_id)
                assert before
                async with store._pool.connection() as conn:
                    await conn.execute(
                        "UPDATE commerce_agent_runs SET lease_until = now() - interval '1 second' WHERE run_id = %s",
                        (run_id,),
                    )
                async with open_protocol_store(settings) as recovered_store:
                    async with open_postgres_graph(settings) as recovered_graph:
                        assert await run_once(
                            recovered_store, recovered_graph,
                            worker_id=recovery_worker, lease_seconds=20,
                        )
                        snapshot = await recovered_graph.aget_state(
                            {"configurable": {"thread_id": thread_id}}
                        )
                        assert len(snapshot.values["messages"]) == 1
                    after = await recovered_store.read_events(owner, thread_id)
                    assert after[:len(before)] == before
                    assert len(after) == len(before) + 1
                    assert after[-1]["method"] == "lifecycle"
                    assert after[-1]["params"]["data"]["event"] == "completed"
                    async with recovered_store._pool.connection() as conn:
                        cursor = await conn.execute(
                            "SELECT status, attempts FROM commerce_agent_runs WHERE run_id = %s",
                            (run_id,),
                        )
                        row = await cursor.fetchone()
                        assert row == {"status": "completed", "attempts": 2}
            finally:
                async with store._pool.connection() as conn:
                    await conn.execute("DELETE FROM commerce_agent_events WHERE thread_id = %s", (thread_id,))
                    await conn.execute("DELETE FROM commerce_agent_runs WHERE thread_id = %s", (thread_id,))
                    await conn.execute("DELETE FROM commerce_agent_threads WHERE thread_id = %s", (thread_id,))
                    await conn.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s", (thread_id,))
                    await conn.execute("DELETE FROM checkpoint_blobs WHERE thread_id = %s", (thread_id,))
                    await conn.execute("DELETE FROM checkpoints WHERE thread_id = %s", (thread_id,))

    asyncio.run(exercise())


def test_worker_reconciles_final_checkpoint_when_values_event_was_not_journaled() -> None:
    if not os.getenv("COMMERCE_AGENT_DATABASE_URI"):
        pytest.skip("dedicated development PostgreSQL is not configured")
    settings = PersistenceSettings.from_environment()
    owner = ThreadOwner(tenant_id=f"checkpoint-gap-{uuid4().hex}", user_id="user-47")
    thread_id = str(uuid4())
    first_worker, recovery_worker = str(uuid4()), str(uuid4())

    async def exercise() -> None:
        await initialize_checkpoint_schema(settings)
        async with open_protocol_store(settings) as store:
            await store.setup()
            run_id = await store.create_run(
                owner, thread_id, command_id="checkpoint-gap-command",
                input_payload={"schema_version": 1, "messages": [{
                    "type": "human", "content": "safe checkpoint gap question",
                    "id": "checkpoint-gap-message",
                }]},
                runtime_context={
                    "tenant_id": owner.tenant_id, "user_id": owner.user_id,
                    "roles": ["viewer"], "request_id": str(uuid4()),
                },
            )
            try:
                claim = await store.claim_next_run(
                    first_worker, lease_seconds=20, target_run_id=run_id,
                )
                assert claim is not None
                assert await store.record_start_checkpoint(
                    run_id, first_worker, None,
                ) is None
                async with open_postgres_graph(settings) as graph:
                    stream = await graph.astream_protocol_events(
                        claim.input_payload,
                        config={"configurable": {"thread_id": thread_id}},
                        context=CommerceRuntimeContext.model_validate(claim.runtime_context),
                    )
                    async for _ in stream:
                        # Simulate process loss after synchronous checkpoint commit
                        # but before a protocol event is written to its journal.
                        pass
                    committed = await graph.aget_state(
                        {"configurable": {"thread_id": thread_id}}
                    )
                    assert committed.next == ()
                    assert len(committed.values["messages"]) == 1
                assert await store.read_events(owner, thread_id) == []
                async with store._pool.connection() as conn:
                    await conn.execute(
                        "UPDATE commerce_agent_runs SET lease_until = now() - interval '1 second' "
                        "WHERE run_id = %s",
                        (run_id,),
                    )
                async with open_protocol_store(settings) as recovered_store:
                    async with open_postgres_graph(settings) as recovered_graph:
                        recovered = await recovered_store.claim_next_run(
                            recovery_worker, lease_seconds=20, target_run_id=run_id,
                        )
                        assert recovered is not None
                        assert await execute_claimed_run(
                            recovered_store, recovered_graph, recovered,
                            worker_id=recovery_worker, lease_seconds=20,
                        ) == "completed"
                    after = await recovered_store.read_events(owner, thread_id)
                    assert [event["method"] for event in after] == ["values", "lifecycle"]
                    assert after[0]["params"]["namespace"] == []
                    assert len(after[0]["params"]["data"]["messages"]) == 1
                    assert after[0]["params"]["data"]["schema_version"] == 1
                    assert after[1]["params"]["data"]["event"] == "completed"
                    assert [event["seq"] for event in after] == [0, 1]
            finally:
                async with store._pool.connection() as conn:
                    await conn.execute("DELETE FROM commerce_agent_events WHERE thread_id = %s", (thread_id,))
                    await conn.execute("DELETE FROM commerce_agent_runs WHERE thread_id = %s", (thread_id,))
                    await conn.execute("DELETE FROM commerce_agent_threads WHERE thread_id = %s", (thread_id,))
                    await conn.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s", (thread_id,))
                    await conn.execute("DELETE FROM checkpoint_blobs WHERE thread_id = %s", (thread_id,))
                    await conn.execute("DELETE FROM checkpoints WHERE thread_id = %s", (thread_id,))

    asyncio.run(exercise())


def test_worker_repairs_partial_values_journal_once_across_recovery() -> None:
    if not os.getenv("COMMERCE_AGENT_DATABASE_URI"):
        pytest.skip("dedicated development PostgreSQL is not configured")
    settings = PersistenceSettings.from_environment()
    owner = ThreadOwner(tenant_id=f"partial-journal-{uuid4().hex}", user_id="user-47")
    thread_id = str(uuid4())
    first_worker, second_worker, third_worker = str(uuid4()), str(uuid4()), str(uuid4())

    async def expire_lease(store, run_id: str) -> None:
        async with store._pool.connection() as conn:
            await conn.execute(
                "UPDATE commerce_agent_runs SET lease_until = now() - interval '1 second' "
                "WHERE run_id = %s",
                (run_id,),
            )

    async def exercise() -> None:
        await initialize_checkpoint_schema(settings)
        async with open_protocol_store(settings) as store:
            await store.setup()
            run_id = await store.create_run(
                owner, thread_id, command_id="partial-journal-command",
                input_payload={"schema_version": 1, "messages": [{
                    "type": "human", "content": "safe partial journal question",
                    "id": "partial-journal-message",
                }]},
                runtime_context={
                    "tenant_id": owner.tenant_id, "user_id": owner.user_id,
                    "roles": ["viewer"], "request_id": str(uuid4()),
                },
            )
            try:
                claim = await store.claim_next_run(
                    first_worker, lease_seconds=20, target_run_id=run_id,
                )
                assert claim is not None
                assert await store.record_start_checkpoint(
                    run_id, first_worker, None,
                ) is None
                root_event_count = 0
                async with open_postgres_graph(settings) as graph:
                    stream = await graph.astream_protocol_events(
                        claim.input_payload,
                        config={"configurable": {"thread_id": thread_id}},
                        context=CommerceRuntimeContext.model_validate(claim.runtime_context),
                    )
                    async for raw in stream:
                        event = normalize_graph_event(raw)
                        if event["method"] == "values" and event["params"]["namespace"] == []:
                            root_event_count += 1
                            if root_event_count == 1:
                                await store.append_event(
                                    owner, thread_id, run_id, event, worker_id=first_worker,
                                )
                    final = await graph.aget_state(
                        {"configurable": {"thread_id": thread_id}}
                    )
                    assert final.next == ()
                    assert root_event_count >= 2
                before = await store.read_events(owner, thread_id)
                assert len(before) == 1
                assert before[0]["params"]["data"] != protocol_json_value(final.values)
                await expire_lease(store, run_id)

                async with open_protocol_store(settings) as second_store:
                    async with open_postgres_graph(settings) as second_graph:
                        second_claim = await second_store.claim_next_run(
                            second_worker, lease_seconds=20, target_run_id=run_id,
                        )
                        assert second_claim is not None

                        async def crash_before_terminal(*_args, **_kwargs):
                            raise asyncio.CancelledError

                        second_store.finish_run = crash_before_terminal
                        with pytest.raises(asyncio.CancelledError):
                            await execute_claimed_run(
                                second_store, second_graph, second_claim,
                                worker_id=second_worker, lease_seconds=20,
                            )
                    repaired = await second_store.read_events(owner, thread_id)
                    assert [event["method"] for event in repaired] == ["values", "values"]
                    assert repaired[0] == before[0]
                    assert repaired[1]["params"]["data"] == protocol_json_value(final.values)
                    await expire_lease(second_store, run_id)

                async with open_protocol_store(settings) as third_store:
                    async with open_postgres_graph(settings) as third_graph:
                        third_claim = await third_store.claim_next_run(
                            third_worker, lease_seconds=20, target_run_id=run_id,
                        )
                        assert third_claim is not None
                        assert await execute_claimed_run(
                            third_store, third_graph, third_claim,
                            worker_id=third_worker, lease_seconds=20,
                        ) == "completed"
                    after = await third_store.read_events(owner, thread_id)
                    assert after[:len(repaired)] == repaired
                    assert [event["method"] for event in after] == [
                        "values", "values", "lifecycle",
                    ]
                    assert [event["seq"] for event in after] == [0, 1, 2]
                    assert after[-1]["params"]["data"]["event"] == "completed"
                    async with third_store._pool.connection() as conn:
                        cursor = await conn.execute(
                            "SELECT status, attempts FROM commerce_agent_runs WHERE run_id = %s",
                            (run_id,),
                        )
                        assert await cursor.fetchone() == {"status": "completed", "attempts": 3}
            finally:
                async with store._pool.connection() as conn:
                    await conn.execute("DELETE FROM commerce_agent_events WHERE thread_id = %s", (thread_id,))
                    await conn.execute("DELETE FROM commerce_agent_runs WHERE thread_id = %s", (thread_id,))
                    await conn.execute("DELETE FROM commerce_agent_threads WHERE thread_id = %s", (thread_id,))
                    await conn.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s", (thread_id,))
                    await conn.execute("DELETE FROM checkpoint_blobs WHERE thread_id = %s", (thread_id,))
                    await conn.execute("DELETE FROM checkpoints WHERE thread_id = %s", (thread_id,))

    asyncio.run(exercise())


def test_worker_recovers_mid_run_after_process_loss_without_replaying_completed_domains() -> None:
    if not os.getenv("COMMERCE_AGENT_DATABASE_URI"):
        pytest.skip("dedicated development PostgreSQL is not configured")
    settings = PersistenceSettings.from_environment()
    owner = ThreadOwner(tenant_id=f"mid-run-{uuid4().hex}", user_id="user-47")
    thread_id = str(uuid4())
    first_worker, recovery_worker = str(uuid4()), str(uuid4())

    async def exercise() -> None:
        entered = asyncio.Event()
        release = asyncio.Event()
        approval_calls = 0

        async def slow_approval_node(_state):
            nonlocal approval_calls
            approval_calls += 1
            entered.set()
            await release.wait()
            return {}

        await initialize_checkpoint_schema(settings)
        async with open_protocol_store(settings) as store:
            await store.setup()
            run_id = await store.create_run(
                owner, thread_id, command_id="mid-run-command",
                input_payload={
                    "schema_version": 1,
                    "messages": [{
                        "type": "human", "content": "safe mid-run recovery question",
                        "id": "mid-run-message",
                    }],
                },
                runtime_context={
                    "tenant_id": owner.tenant_id, "user_id": owner.user_id,
                    "roles": ["viewer"], "request_id": str(uuid4()),
                },
            )
            try:
                async with open_postgres_graph(
                    settings, approval_node=slow_approval_node,
                ) as first_graph:
                    first_task = asyncio.create_task(run_once(
                        store, first_graph, worker_id=first_worker,
                        lease_seconds=20,
                    ))
                    try:
                        await asyncio.wait_for(entered.wait(), timeout=10)
                        async with asyncio.timeout(10):
                            while True:
                                before = await store.read_events(owner, thread_id)
                                if any(
                                    event["method"] == "values"
                                    and event["params"]["namespace"] == []
                                    and "agent_statuses" in event["params"]["data"]
                                    for event in before
                                ):
                                    break
                                await asyncio.sleep(0.01)
                        snapshot = await first_graph.aget_state(
                            {"configurable": {"thread_id": thread_id}}
                        )
                        assert snapshot.values["messages"][0].content == (
                            "safe mid-run recovery question"
                        )
                        assert not first_task.done(), "approval node must still be running"
                        assert approval_calls == 1
                    finally:
                        first_task.cancel()
                        await asyncio.gather(first_task, return_exceptions=True)
                    before = await store.read_events(owner, thread_id)

                async with store._pool.connection() as conn:
                    await conn.execute(
                        "UPDATE commerce_agent_runs SET lease_until = now() - interval '1 second' WHERE run_id = %s",
                        (run_id,),
                    )
                release.set()
                async with open_protocol_store(settings) as recovered_store:
                    async with open_postgres_graph(
                        settings, approval_node=slow_approval_node,
                    ) as recovered_graph:
                        assert await run_once(
                            recovered_store, recovered_graph,
                            worker_id=recovery_worker, lease_seconds=20,
                        )
                        final = await recovered_graph.aget_state(
                            {"configurable": {"thread_id": thread_id}}
                        )
                    after = await recovered_store.read_events(owner, thread_id)
                    assert after[:len(before)] == before
                    assert [event["seq"] for event in after] == list(range(len(after)))
                    assert final.next == ()
                    assert len(final.values["messages"]) == 1
                    assert approval_calls == 2, "recovery must finish the in-flight node"
                    assert after[-1]["params"]["data"]["event"] == "completed"
                    assert sum(
                        event["method"] == "lifecycle"
                        and event["params"]["data"].get("event") == "completed"
                        for event in after
                    ) == 1
                    for domain in ("sales", "marketing", "supply", "price", "finance"):
                        before_domain_events = sum(
                            event["method"] == "values"
                            and any(part.startswith(domain + ":") for part in event["params"]["namespace"])
                            for event in before
                        )
                        assert before_domain_events > 0, f"{domain} was not observed before failure"
                        assert sum(
                            event["method"] == "values"
                            and any(part.startswith(domain + ":") for part in event["params"]["namespace"])
                            for event in after
                        ) == before_domain_events
                    async with recovered_store._pool.connection() as conn:
                        cursor = await conn.execute(
                            "SELECT status, attempts FROM commerce_agent_runs WHERE run_id = %s",
                            (run_id,),
                        )
                        assert await cursor.fetchone() == {"status": "completed", "attempts": 2}
            finally:
                async with store._pool.connection() as conn:
                    await conn.execute("DELETE FROM commerce_agent_events WHERE thread_id = %s", (thread_id,))
                    await conn.execute("DELETE FROM commerce_agent_runs WHERE thread_id = %s", (thread_id,))
                    await conn.execute("DELETE FROM commerce_agent_threads WHERE thread_id = %s", (thread_id,))
                    await conn.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s", (thread_id,))
                    await conn.execute("DELETE FROM checkpoint_blobs WHERE thread_id = %s", (thread_id,))
                    await conn.execute("DELETE FROM checkpoints WHERE thread_id = %s", (thread_id,))

    asyncio.run(exercise())


def test_worker_reuses_pending_write_from_completed_parallel_node() -> None:
    if not os.getenv("COMMERCE_AGENT_DATABASE_URI"):
        pytest.skip("dedicated development PostgreSQL is not configured")
    settings = PersistenceSettings.from_environment()
    owner = ThreadOwner(tenant_id=f"pending-write-{uuid4().hex}", user_id="user-47")
    thread_id = str(uuid4())
    first_worker, recovery_worker = str(uuid4()), str(uuid4())

    class ParallelState(TypedDict, total=False):
        schema_version: int
        messages: list[dict[str, str]]
        completed: Annotated[list[str], add]

    async def exercise() -> None:
        slow_entered = asyncio.Event()
        release_slow = asyncio.Event()
        fast_calls = 0
        slow_calls = 0

        async def fast_node(_state: ParallelState) -> dict[str, list[str]]:
            nonlocal fast_calls
            fast_calls += 1
            return {"completed": ["fast"]}

        async def slow_node(_state: ParallelState) -> dict[str, list[str]]:
            nonlocal slow_calls
            slow_calls += 1
            slow_entered.set()
            await release_slow.wait()
            return {"completed": ["slow"]}

        def guarded_parallel_graph(checkpointer):
            builder = StateGraph(ParallelState)
            builder.add_node("fast", fast_node)
            builder.add_node("slow", slow_node)
            builder.add_edge(START, "fast")
            builder.add_edge(START, "slow")
            builder.add_edge("fast", END)
            builder.add_edge("slow", END)
            return GuardedGraph(builder.compile(checkpointer=checkpointer))

        await initialize_checkpoint_schema(settings)
        async with open_protocol_store(settings) as store:
            await store.setup()
            run_id = await store.create_run(
                owner, thread_id, command_id="pending-write-command",
                input_payload={
                    "schema_version": 1,
                    "messages": [{"type": "human", "content": "safe pending write question"}],
                },
                runtime_context={
                    "tenant_id": owner.tenant_id, "user_id": owner.user_id,
                    "roles": ["viewer"], "request_id": str(uuid4()),
                },
            )
            try:
                async with open_postgres_checkpointer(settings) as checkpointer:
                    graph = guarded_parallel_graph(checkpointer)
                    first_task = asyncio.create_task(run_once(
                        store, graph, worker_id=first_worker, lease_seconds=20,
                    ))
                    try:
                        await asyncio.wait_for(slow_entered.wait(), timeout=10)
                        async with asyncio.timeout(10):
                            while True:
                                async with store._pool.connection() as conn:
                                    cursor = await conn.execute(
                                        "SELECT count(*) AS n FROM checkpoint_writes "
                                        "WHERE thread_id = %s AND channel = 'completed'",
                                        (thread_id,),
                                    )
                                    pending_write_count = (await cursor.fetchone())["n"]
                                if pending_write_count > 0:
                                    break
                                await asyncio.sleep(0.01)
                        snapshot = await graph.aget_state(
                            {"configurable": {"thread_id": thread_id}}
                        )
                        assert snapshot.values["completed"] == ["fast"]
                        assert any(task.name == "slow" for task in snapshot.tasks)
                        assert not first_task.done()
                        assert fast_calls == slow_calls == 1
                    finally:
                        first_task.cancel()
                        await asyncio.gather(first_task, return_exceptions=True)

                async with store._pool.connection() as conn:
                    await conn.execute(
                        "UPDATE commerce_agent_runs SET lease_until = now() - interval '1 second' "
                        "WHERE run_id = %s",
                        (run_id,),
                    )
                release_slow.set()
                async with open_protocol_store(settings) as recovered_store:
                    async with open_postgres_checkpointer(settings) as checkpointer:
                        recovered_graph = guarded_parallel_graph(checkpointer)
                        assert await run_once(
                            recovered_store, recovered_graph,
                            worker_id=recovery_worker, lease_seconds=20,
                        )
                        final = await recovered_graph.aget_state(
                            {"configurable": {"thread_id": thread_id}}
                        )
                    assert final.next == ()
                    assert sorted(final.values["completed"]) == ["fast", "slow"]
                    assert fast_calls == 1, "completed parallel node must not replay"
                    assert slow_calls == 2, "in-flight parallel node must resume"
                    async with recovered_store._pool.connection() as conn:
                        cursor = await conn.execute(
                            "SELECT status, attempts FROM commerce_agent_runs WHERE run_id = %s",
                            (run_id,),
                        )
                        assert await cursor.fetchone() == {"status": "completed", "attempts": 2}
            finally:
                async with store._pool.connection() as conn:
                    await conn.execute("DELETE FROM commerce_agent_events WHERE thread_id = %s", (thread_id,))
                    await conn.execute("DELETE FROM commerce_agent_runs WHERE thread_id = %s", (thread_id,))
                    await conn.execute("DELETE FROM commerce_agent_threads WHERE thread_id = %s", (thread_id,))
                    await conn.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s", (thread_id,))
                    await conn.execute("DELETE FROM checkpoint_blobs WHERE thread_id = %s", (thread_id,))
                    await conn.execute("DELETE FROM checkpoints WHERE thread_id = %s", (thread_id,))

    asyncio.run(exercise())


def test_worker_recovers_after_real_process_kill_and_replays_durable_events() -> None:
    if not os.getenv("COMMERCE_AGENT_DATABASE_URI"):
        pytest.skip("dedicated development PostgreSQL is not configured")
    settings = PersistenceSettings.from_environment()
    if settings.environment != "development" or urlsplit(settings.database_uri).hostname not in {
        "localhost", "127.0.0.1", "::1",
    }:
        pytest.skip("process-kill injection only runs against loopback development DB")
    owner = ThreadOwner(tenant_id=f"process-kill-{uuid4().hex}", user_id="user-47")
    thread_id = str(uuid4())
    worker_id = str(uuid4())
    recovery_worker = str(uuid4())

    worker_code = """
import asyncio
import sys
from commerce_agent.persistence import PersistenceSettings, open_postgres_graph
from commerce_agent.protocol_store import open_protocol_store
from commerce_agent.protocol_worker import execute_claimed_run

async def approval_node(_state):
    print("APPROVAL_ENTERED", flush=True)
    await asyncio.Event().wait()

async def main():
    settings = PersistenceSettings.from_environment()
    async with open_protocol_store(settings) as store:
        async with open_postgres_graph(settings, approval_node=approval_node) as graph:
            claim = await store.claim_next_run(
                sys.argv[1], lease_seconds=20, target_run_id=sys.argv[2]
            )
            assert claim is not None
            await execute_claimed_run(
                store, graph, claim, worker_id=sys.argv[1], lease_seconds=20
            )

asyncio.run(main())
"""

    async def exercise() -> None:
        await initialize_checkpoint_schema(settings)
        async with open_protocol_store(settings) as store:
            await store.setup()
            run_id = await store.create_run(
                owner, thread_id, command_id="process-kill-command",
                input_payload={
                    "schema_version": 1,
                    "messages": [{
                        "type": "human", "content": "safe process kill question",
                        "id": "process-kill-message",
                    }],
                },
                runtime_context={
                    "tenant_id": owner.tenant_id, "user_id": owner.user_id,
                    "roles": ["viewer"], "request_id": str(uuid4()),
                },
            )
            process = None
            try:
                process = await asyncio.create_subprocess_exec(
                    sys.executable, "-u", "-c", worker_code, worker_id, run_id,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                    cwd=str(Path(__file__).resolve().parents[1]),
                )
                assert process.stdout is not None
                marker = await asyncio.wait_for(process.stdout.readline(), timeout=10)
                assert marker == b"APPROVAL_ENTERED\n", "worker did not reach approval node"
                async with asyncio.timeout(10):
                    while True:
                        before = await store.read_events(owner, thread_id)
                        if all(
                            any(
                                event["method"] == "values"
                                and any(part.startswith(domain + ":") for part in event["params"]["namespace"])
                                for event in before
                            )
                            for domain in ("sales", "marketing", "supply", "price", "finance")
                        ):
                            break
                        await asyncio.sleep(0.01)
                assert process.returncode is None
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                await asyncio.wait_for(process.communicate(), timeout=10)
                assert process.returncode is not None and process.returncode != 0

                async with store._pool.connection() as conn:
                    await conn.execute(
                        "UPDATE commerce_agent_runs SET lease_until = now() - interval '1 second' "
                        "WHERE run_id = %s",
                        (run_id,),
                    )
                async with open_protocol_store(settings) as recovered_store:
                    async with open_postgres_graph(settings) as recovered_graph:
                        claim = await recovered_store.claim_next_run(
                            recovery_worker, lease_seconds=20, target_run_id=run_id,
                        )
                        assert claim is not None
                        assert await execute_claimed_run(
                            recovered_store, recovered_graph, claim,
                            worker_id=recovery_worker, lease_seconds=20,
                        )
                        final = await recovered_graph.aget_state(
                            {"configurable": {"thread_id": thread_id}}
                        )
                    after = await recovered_store.read_events(owner, thread_id)
                    assert after[:len(before)] == before
                    assert [event["seq"] for event in after] == list(range(len(after)))
                    assert len(final.values["messages"]) == 1
                    assert final.next == ()
                    assert sum(
                        event["method"] == "lifecycle"
                        and event["params"]["data"].get("event") == "completed"
                        for event in after
                    ) == 1
                    for domain in ("sales", "marketing", "supply", "price", "finance"):
                        count = lambda events: sum(
                            event["method"] == "values"
                            and any(part.startswith(domain + ":") for part in event["params"]["namespace"])
                            for event in events
                        )
                        assert count(before) > 0
                        assert count(after) == count(before)
                    async with recovered_store._pool.connection() as conn:
                        cursor = await conn.execute(
                            "SELECT status, attempts FROM commerce_agent_runs WHERE run_id = %s",
                            (run_id,),
                        )
                        assert await cursor.fetchone() == {"status": "completed", "attempts": 2}
            finally:
                if process is not None:
                    if process.returncode is None:
                        try:
                            process.kill()
                        except ProcessLookupError:
                            pass
                    await asyncio.wait_for(process.communicate(), timeout=10)
                async with store._pool.connection() as conn:
                    await conn.execute("DELETE FROM commerce_agent_events WHERE thread_id = %s", (thread_id,))
                    await conn.execute("DELETE FROM commerce_agent_runs WHERE thread_id = %s", (thread_id,))
                    await conn.execute("DELETE FROM commerce_agent_threads WHERE thread_id = %s", (thread_id,))
                    await conn.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s", (thread_id,))
                    await conn.execute("DELETE FROM checkpoint_blobs WHERE thread_id = %s", (thread_id,))
                    await conn.execute("DELETE FROM checkpoints WHERE thread_id = %s", (thread_id,))

    asyncio.run(exercise())


def test_targeted_claim_leaves_an_older_unrelated_run_queued() -> None:
    if not os.getenv("COMMERCE_AGENT_DATABASE_URI"):
        pytest.skip("dedicated development PostgreSQL is not configured")
    settings = PersistenceSettings.from_environment()
    owner = ThreadOwner(tenant_id=f"targeted-claim-{uuid4().hex}", user_id="user-47")
    other_thread, target_thread = str(uuid4()), str(uuid4())
    context = {
        "tenant_id": owner.tenant_id, "user_id": owner.user_id,
        "roles": ["viewer"], "request_id": str(uuid4()),
    }

    async def exercise() -> None:
        await initialize_checkpoint_schema(settings)
        async with open_protocol_store(settings) as store:
            await store.setup()
            try:
                other_run = await store.create_run(
                    owner, other_thread, command_id="other-command",
                    input_payload={"schema_version": 1, "messages": []},
                    runtime_context=context,
                )
                target_run = await store.create_run(
                    owner, target_thread, command_id="target-command",
                    input_payload={"schema_version": 1, "messages": []},
                    runtime_context=context,
                )
                with pytest.raises(ProtocolRecordError, match="target run ID"):
                    await store.claim_next_run(str(uuid4()), target_run_id="not-a-uuid")
                claim = await store.claim_next_run(
                    str(uuid4()), lease_seconds=20, target_run_id=target_run,
                )
                assert claim is not None and claim.run_id == target_run
                async with store._pool.connection() as conn:
                    cursor = await conn.execute(
                        "SELECT run_id, status, attempts FROM commerce_agent_runs "
                        "WHERE run_id IN (%s, %s) ORDER BY run_id",
                        (other_run, target_run),
                    )
                    rows = {str(row["run_id"]): row for row in await cursor.fetchall()}
                assert rows[other_run]["status"] == "queued"
                assert rows[other_run]["attempts"] == 0
                assert rows[target_run]["status"] == "running"
                assert rows[target_run]["attempts"] == 1
            finally:
                async with store._pool.connection() as conn:
                    await conn.execute(
                        "DELETE FROM commerce_agent_events WHERE thread_id IN (%s, %s)",
                        (other_thread, target_thread),
                    )
                    await conn.execute(
                        "DELETE FROM commerce_agent_runs WHERE thread_id IN (%s, %s)",
                        (other_thread, target_thread),
                    )
                    await conn.execute(
                        "DELETE FROM commerce_agent_threads WHERE thread_id IN (%s, %s)",
                        (other_thread, target_thread),
                    )

    asyncio.run(exercise())


def test_synthetic_interrupt_resumes_same_run_with_manager_and_no_write() -> None:
    if not os.getenv("COMMERCE_AGENT_DATABASE_URI"):
        pytest.skip("dedicated development PostgreSQL is not configured")
    settings = PersistenceSettings.from_environment()
    owner = ThreadOwner(tenant_id="interrupt-tenant", user_id="manager-47")
    thread_id = str(uuid4())
    service_token = "synthetic-internal-service-token-" + "x" * 32
    headers = {
        "X-Commerce-Service-Token": service_token,
        "X-Commerce-Tenant-ID": owner.tenant_id,
        "X-Commerce-User-ID": owner.user_id,
        "X-Commerce-Role": "manager",
        "X-Request-ID": str(uuid4()),
    }

    def approval_node(_state):
        response = interrupt({"kind": "approval", "action_id": "synthetic-no-write"})
        assert response == {"approved": False}
        return {}

    async def exercise() -> None:
        await initialize_checkpoint_schema(settings)
        async with open_protocol_store(settings) as api_store:
            await api_store.setup()
            try:
                async with open_postgres_graph(settings, approval_node=approval_node) as api_graph:
                    app = create_read_app(
                        resources=ReadResources(store=api_store, graph=api_graph),
                        service_token=service_token,
                    )
                    async with app.router.lifespan_context(app):
                        async with httpx.AsyncClient(
                            transport=httpx.ASGITransport(app=app),
                            base_url="http://internal.test",
                        ) as client:
                            path = f"/threads/{thread_id}/commands"
                            started = await client.post(
                                path, headers=headers,
                                json=_start_command("synthetic-start", thread_id=thread_id),
                            )
                            assert started.status_code == 200, started.text
                            run_id = started.json()["result"]["run_id"]
                            async with open_protocol_store(settings) as worker_store:
                                async with open_postgres_graph(
                                    settings, approval_node=approval_node,
                                ) as worker_graph:
                                    assert await run_once(
                                        worker_store, worker_graph,
                                        worker_id=str(uuid4()), lease_seconds=20,
                                    )
                            events = await api_store.read_events(owner, thread_id)
                            requested = [e for e in events if e["method"] == "input.requested"]
                            assert len(requested) == 1
                            interrupt_id = requested[0]["params"]["data"]["interrupt_id"]
                            assert requested[0]["params"]["data"]["payload"]["action_id"] == "synthetic-no-write"
                            assert events[-1]["params"]["data"]["event"] == "interrupted"
                            response = {
                                "id": 2, "method": "input.respond",
                                "params": {
                                    "namespace": [], "interrupt_id": interrupt_id,
                                    "response": {"approved": False},
                                },
                            }
                            viewer = await client.post(
                                path, headers={**headers, "X-Commerce-Role": "viewer"},
                                json=response,
                            )
                            assert viewer.status_code == 403
                            wrong = await client.post(
                                path, headers=headers,
                                json={**response, "params": {
                                    **response["params"], "interrupt_id": "wrong-id",
                                }},
                            )
                            assert wrong.status_code == 409
                            approved = await client.post(path, headers=headers, json=response)
                            again = await client.post(path, headers=headers, json=response)
                            assert approved.status_code == again.status_code == 200
                            assert approved.json()["result"]["run_id"] == run_id
                            assert again.json()["result"]["run_id"] == run_id
                            conflict = await client.post(
                                path, headers=headers,
                                json={**response, "params": {
                                    **response["params"], "response": {"approved": True},
                                }},
                            )
                            assert conflict.status_code == 409
                            async with open_protocol_store(settings) as worker_store:
                                async with open_postgres_graph(
                                    settings, approval_node=approval_node,
                                ) as worker_graph:
                                    assert await run_once(
                                        worker_store, worker_graph,
                                        worker_id=str(uuid4()), lease_seconds=20,
                                    )
                                    snapshot = await worker_graph.aget_state({
                                        "configurable": {"thread_id": thread_id},
                                    })
                                    assert snapshot.next == ()
                                    assert snapshot.values.get("execution_receipts") == []
                            replayed = await client.post(path, headers=headers, json=response)
                            assert replayed.status_code == 200
                            assert replayed.json()["result"]["run_id"] == run_id
                            async with api_store._pool.connection() as conn:
                                cursor = await conn.execute(
                                    "SELECT status, attempts FROM commerce_agent_runs WHERE thread_id = %s",
                                    (thread_id,),
                                )
                                assert await cursor.fetchall() == [{
                                    "status": "completed", "attempts": 2,
                                }]
            finally:
                async with api_store._pool.connection() as conn:
                    await conn.execute("DELETE FROM commerce_agent_events WHERE thread_id = %s", (thread_id,))
                    await conn.execute("DELETE FROM commerce_agent_runs WHERE thread_id = %s", (thread_id,))
                    await conn.execute("DELETE FROM commerce_agent_threads WHERE thread_id = %s", (thread_id,))
                    await conn.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s", (thread_id,))
                    await conn.execute("DELETE FROM checkpoint_blobs WHERE thread_id = %s", (thread_id,))
                    await conn.execute("DELETE FROM checkpoints WHERE thread_id = %s", (thread_id,))

    asyncio.run(exercise())
