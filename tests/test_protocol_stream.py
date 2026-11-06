from __future__ import annotations

import asyncio
import json
import os
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import httpx
import pytest

import commerce_agent.protocol_stream as protocol_stream
from commerce_agent.persistence import PersistenceSettings, initialize_checkpoint_schema
from commerce_agent.protocol_read_api import (
    ReadResources, _TimedStreamingResponse, create_read_app,
)
from commerce_agent.protocol_store import (
    ProtocolRecordError, ThreadNotOwned, ThreadOwner, open_protocol_store,
)
from commerce_agent.protocol_stream import (
    encode_sse, event_matches, parse_subscribe_params, stream_events,
)


def _event(seq: int, method: str, namespace: list[str]) -> dict:
    return {
        "type": "event", "method": method, "seq": seq,
        "event_id": f"synthetic-thread:{seq}",
        "params": {"namespace": namespace, "data": {"status": "planned"}},
    }


class _StreamClock:
    """Advance stream policy time without changing the event loop's real clock."""

    def __init__(self, *, wake_at: float | None = None) -> None:
        self.now = 0.0
        self.wake_at = wake_at
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        assert seconds >= 0
        self.sleeps.append(seconds)
        self.now += seconds
        if self.wake_at is not None:
            self.now = max(self.now, self.wake_at)
        await asyncio.sleep(0)

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(protocol_stream, "monotonic", self.monotonic)
        # Replace only this module's facade, not asyncio.sleep globally. DB
        # wait_for and HTTP/concurrency deadlines still use the real event loop.
        monkeypatch.setattr(protocol_stream, "asyncio", SimpleNamespace(
            wait_for=asyncio.wait_for, sleep=self.sleep,
        ))


def test_subscription_validation_and_selector_filters() -> None:
    params = parse_subscribe_params({
        "channels": ["values", "custom:research"],
        "namespaces": [["sales"]], "depth": 1, "since": 4,
    })
    assert params.since == 4
    assert event_matches(_event(5, "values", ["sales:run-uuid"]), params)
    assert event_matches(_event(6, "values", ["sales:run-uuid", "analyze"]), params)
    assert not event_matches(_event(7, "values", ["sales:run-uuid", "analyze", "tool"]), params)
    assert not event_matches(_event(8, "values", ["finance"]), params)
    assert not event_matches(_event(9, "tools", ["sales:run-uuid"]), params)
    custom = _event(10, "custom", ["sales:run-uuid"])
    custom["params"]["data"] = {"name": "research", "payload": {"status": "planned"}}
    assert event_matches(custom, params)
    custom["params"]["data"]["name"] = "other"
    assert not event_matches(custom, params)
    exact = parse_subscribe_params({"channels": ["values"], "namespaces": [["sales:exact"]]})
    assert not event_matches(_event(5, "values", ["sales:other"]), exact)
    assert event_matches(_event(5, "values", ["sales:exact"]), exact)
    root = parse_subscribe_params({"channels": ["values"], "namespaces": [[]], "depth": 0})
    assert event_matches(_event(5, "values", []), root)
    assert not event_matches(_event(5, "values", ["sales"]), root)
    sdk_root = parse_subscribe_params({
        "channels": ["values", "checkpoints", "lifecycle", "input", "messages", "tools"],
        "namespaces": [[]], "depth": 0,
    })
    assert event_matches(_event(11, "input.requested", []), sdk_root)
    assert not event_matches(_event(11, "input.requested", ["sales"]), sdk_root)


@pytest.mark.parametrize("body", [
    {}, {"channels": []}, {"channels": ["unknown"]},
    {"channels": ["values"], "since": True},
    {"channels": ["values"], "since": -2},
    {"channels": ["values"], "since": 9_223_372_036_854_775_808},
    {"channels": ["values"], "depth": 33},
    {"channels": ["values"], "namespaces": [["a\nprivate"]]},
    {"channels": ["values"], "private": "value"},
])
def test_subscription_rejects_unknown_and_unbounded_input(body: dict) -> None:
    with pytest.raises(ProtocolRecordError):
        parse_subscribe_params(body)


def test_sse_preserves_protocol_event_and_rejects_header_injection() -> None:
    event = _event(2, "values", ["sales"])
    frame = encode_sse(event)
    assert frame.startswith(b"id: synthetic-thread:2\nevent: message\ndata: ")
    assert json.loads(frame.split(b"data: ", 1)[1]) == event
    with pytest.raises(ProtocolRecordError, match="event ID"):
        encode_sse({**event, "event_id": "safe\r\nevent: private"})


def test_internal_stream_rejects_missing_auth_bad_selector_and_other_owner() -> None:
    class Store:
        async def require_owned(self, owner, _thread):
            if owner.user_id != "owner":
                raise ThreadNotOwned("thread not found")

    app = create_read_app(
        resources=ReadResources(store=Store(), graph=SimpleNamespace()),
        service_token="synthetic-internal-service-token-" + "x" * 32,
    )
    path = f"/threads/{uuid4()}/stream"
    headers = {
        "X-Commerce-Service-Token": "synthetic-internal-service-token-" + "x" * 32,
        "X-Commerce-Tenant-ID": "tenant", "X-Commerce-User-ID": "owner",
    }

    async def exercise() -> None:
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://internal.test"
            ) as client:
                assert (await client.post(path, json={"channels": ["values"]})).status_code == 401
                assert (await client.post(path, headers=headers, json={"channels": []})).status_code == 422
                assert (await client.post(
                    path, headers={**headers, "X-Commerce-User-ID": "other"},
                    json={"channels": ["values"]},
                )).status_code == 404

    asyncio.run(exercise())


def test_internal_stream_emits_protocol_sse_with_no_buffering() -> None:
    class Store:
        async def require_owned(self, owner, _thread):
            assert owner == ThreadOwner(tenant_id="tenant", user_id="owner")

    async def finite_stream(_store, _owner, _thread_id, params):
        assert params.since == 1
        yield encode_sse(_event(2, "values", []))

    token = "synthetic-internal-service-token-" + "x" * 32
    app = create_read_app(
        resources=ReadResources(store=Store(), graph=SimpleNamespace()),
        service_token=token,
    )

    async def exercise() -> None:
        with patch("commerce_agent.protocol_read_api.stream_events", finite_stream):
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://internal.test"
                ) as client:
                    response = await client.post(
                        f"/threads/{uuid4()}/stream",
                        headers={
                            "X-Commerce-Service-Token": token,
                            "X-Commerce-Tenant-ID": "tenant",
                            "X-Commerce-User-ID": "owner",
                        },
                        json={"channels": ["values"], "since": 1},
                    )
                    assert response.status_code == 200
                    assert response.content.startswith(b"id: synthetic-thread:2")
                    assert response.headers["content-type"].startswith("text/event-stream")
                    assert response.headers["x-accel-buffering"] == "no"
                    assert app.state.stream_slots._value == 16

    asyncio.run(exercise())


def test_stream_has_heartbeat_deadline_and_bounded_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _StreamClock()
    poll_times: list[float] = []

    class Store:
        async def require_owned(self, *_args):
            return None

        async def read_events(self, *_args, **_kwargs):
            poll_times.append(clock.now)
            return []

    store = Store()
    token = "synthetic-internal-service-token-" + "x" * 32
    app = create_read_app(
        resources=ReadResources(store=store, graph=SimpleNamespace()),
        service_token=token,
    )

    async def exercise() -> None:
        params = parse_subscribe_params({"channels": ["values"]})
        frames = []
        heartbeat_times = []
        with monkeypatch.context() as stream_patch:
            clock.install(stream_patch)
            async for frame in stream_events(
                store, ThreadOwner("tenant", "owner"), str(uuid4()), params,
                poll_seconds=0.005, heartbeat_seconds=0.02, max_seconds=0.06,
                db_timeout_seconds=0.02,
            ):
                frames.append(frame)
                heartbeat_times.append(clock.now)
        assert frames == [b": keepalive\n\n", b": keepalive\n\n"]
        assert heartbeat_times == pytest.approx([0.02, 0.04])
        assert clock.now == 0.06
        assert all(0 <= value < 0.06 for value in poll_times)
        assert all(0 < seconds <= 0.005 for seconds in clock.sleeps)
        async with app.router.lifespan_context(app):
            for _ in range(16):
                await app.state.stream_slots.acquire()
            try:
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://internal.test"
                ) as client:
                    response = await client.post(
                        f"/threads/{uuid4()}/stream",
                        headers={
                            "X-Commerce-Service-Token": token,
                            "X-Commerce-Tenant-ID": "tenant",
                            "X-Commerce-User-ID": "owner",
                        },
                        json={"channels": ["values"]},
                    )
                    assert response.status_code == 429
                    assert app.state.stream_slots._value == 0
            finally:
                for _ in range(16):
                    app.state.stream_slots.release()
            assert app.state.stream_slots._value == 16

    asyncio.run(exercise())


@pytest.mark.parametrize("wake_at", [0.06, 0.08])
def test_stream_delayed_wakeup_closes_at_deadline_without_late_heartbeat(
    monkeypatch: pytest.MonkeyPatch, wake_at: float,
) -> None:
    clock = _StreamClock(wake_at=wake_at)
    poll_times: list[float] = []

    class Store:
        async def read_events(self, *_args, **_kwargs):
            poll_times.append(clock.now)
            return []

    async def exercise() -> None:
        with monkeypatch.context() as stream_patch:
            clock.install(stream_patch)
            frames = [frame async for frame in stream_events(
                Store(), ThreadOwner("tenant", "owner"), str(uuid4()),
                parse_subscribe_params({"channels": ["values"]}),
                poll_seconds=0.005, heartbeat_seconds=0.02, max_seconds=0.06,
                db_timeout_seconds=0.02,
            )]
        assert frames == []
        assert poll_times == [0.0]
        assert clock.sleeps == [0.005]
        assert clock.now == wake_at

    asyncio.run(exercise())


@pytest.mark.parametrize("read_finishes_at", [0.06, 0.08])
def test_stream_empty_poll_finishing_at_deadline_does_not_emit_late_heartbeat(
    monkeypatch: pytest.MonkeyPatch, read_finishes_at: float,
) -> None:
    clock = _StreamClock()
    poll_times: list[float] = []

    class Store:
        async def read_events(self, *_args, **_kwargs):
            poll_times.append(clock.now)
            clock.now = read_finishes_at
            return []

    async def exercise() -> None:
        with monkeypatch.context() as stream_patch:
            clock.install(stream_patch)
            frames = [frame async for frame in stream_events(
                Store(), ThreadOwner("tenant", "owner"), str(uuid4()),
                parse_subscribe_params({"channels": ["values"]}),
                poll_seconds=0.005, heartbeat_seconds=0.02, max_seconds=0.06,
                db_timeout_seconds=0.02,
            )]
        assert frames == []
        assert poll_times == [0.0]
        assert clock.sleeps == [0]
        assert clock.now == read_finishes_at

    asyncio.run(exercise())


def test_asgi_deadline_releases_slot_when_downstream_send_stalls() -> None:
    released = False
    body_send_started = False

    async def frames():
        yield b": keepalive\n\n"

    async def receive():
        await asyncio.Event().wait()

    async def send(message):
        nonlocal body_send_started
        if message["type"] == "http.response.body":
            body_send_started = True
            await asyncio.Event().wait()

    def release() -> None:
        nonlocal released
        released = True

    response = _TimedStreamingResponse(
        frames(), release_slot=release, max_seconds=0.05,
        media_type="text/event-stream",
    )
    asyncio.run(response(
        {"type": "http", "asgi": {"version": "3.0"}, "method": "GET", "path": "/", "headers": []},
        receive, send,
    ))
    assert body_send_started
    assert released


def test_database_poll_does_not_hold_stream_forever() -> None:
    class StalledStore:
        async def read_events(self, *_args, **_kwargs):
            await asyncio.Event().wait()

    async def exercise() -> None:
        stream = stream_events(
            StalledStore(), ThreadOwner("tenant", "owner"), str(uuid4()),
            parse_subscribe_params({"channels": ["values"]}),
            poll_seconds=0.005, heartbeat_seconds=0.02, max_seconds=0.06,
            db_timeout_seconds=0.01,
        )
        with pytest.raises(TimeoutError):
            await anext(stream)
        await stream.aclose()

    asyncio.run(exercise())


def test_postgres_replay_follow_and_owner_isolation_across_pools() -> None:
    if not os.getenv("COMMERCE_AGENT_DATABASE_URI"):
        pytest.skip("dedicated development PostgreSQL is not configured")
    settings = PersistenceSettings.from_environment()
    thread_id = str(uuid4())
    owner = ThreadOwner(tenant_id="stream-test-tenant", user_id="owner")
    other = ThreadOwner(tenant_id="stream-test-tenant", user_id="other")
    context = {
        "tenant_id": owner.tenant_id, "user_id": owner.user_id,
        "roles": ["viewer"], "request_id": "stream-test-request",
    }

    async def cleanup(store) -> None:
        async with store._pool.connection() as conn:
            await conn.execute("DELETE FROM commerce_agent_events WHERE thread_id = %s", (thread_id,))
            await conn.execute("DELETE FROM commerce_agent_runs WHERE thread_id = %s", (thread_id,))
            await conn.execute("DELETE FROM commerce_agent_threads WHERE thread_id = %s", (thread_id,))

    async def exercise() -> None:
        await initialize_checkpoint_schema(settings)
        async with open_protocol_store(settings) as writer:
            await writer.setup()
            try:
                run_id = await writer.create_run(
                    owner, thread_id, command_id="stream-test-command",
                    input_payload={"schema_version": 1, "messages": []},
                    runtime_context=context,
                )
                for method, namespace in (("tools", []), ("values", ["sales:run-1"])):
                    await writer.append_event(owner, thread_id, run_id, {
                        "type": "event", "method": method,
                        "params": {"namespace": namespace, "data": {"status": "planned"}},
                    })
                async with open_protocol_store(settings) as reader:
                    params = parse_subscribe_params({
                        "channels": ["values"], "namespaces": [["sales"]], "since": -1,
                    })
                    stream = stream_events(reader, owner, thread_id, params, poll_seconds=0.01)
                    first = await asyncio.wait_for(anext(stream), timeout=2)
                    assert b"id: " + thread_id.encode() + b":1" in first
                    await writer.append_event(owner, thread_id, run_id, {
                        "type": "event", "method": "values",
                        "params": {"namespace": ["sales:run-2"], "data": {"status": "completed"}},
                    })
                    second = await asyncio.wait_for(anext(stream), timeout=2)
                    assert b"id: " + thread_id.encode() + b":2" in second
                    await stream.aclose()
                    denied = stream_events(reader, other, thread_id, params)
                    with pytest.raises(ThreadNotOwned):
                        await anext(denied)
                    await denied.aclose()
            finally:
                await cleanup(writer)

    asyncio.run(exercise())
