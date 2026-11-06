"""Owner-scoped HTTP reads and durable event subscriptions.

This is an internal service boundary. The browser reaches it only through the
authenticated FastAPI BFF; no user bearer token or provider key is accepted.
Schema setup remains an explicit deployment step, not a startup side effect.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
import hmac
import json
import os
import asyncio
from typing import Any
from uuid import UUID

from pydantic import ValidationError
from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from commerce_agent.persistence import PersistenceSettings, open_postgres_graph
from commerce_agent.protocol_commands import parse_respond_command, parse_start_command
from commerce_agent.protocol_events import protocol_json_value
from commerce_agent.protocol_stream import parse_subscribe_params, stream_events
from commerce_agent.protocol_store import (
    ActiveRunConflict,
    IdempotencyConflict,
    ProtocolRecordError,
    ProtocolStore,
    ThreadNotOwned,
    ThreadOwner,
    open_protocol_store,
)
from commerce_agent.safety import SensitiveStateError, assert_secret_free
from commerce_agent.state import CommerceStateSnapshot, STATE_FIELD_NAMES


SERVICE_TOKEN_ENV = "COMMERCE_AGENT_SERVICE_TOKEN"
MAX_STATE_BYTES = 1_048_576
MAX_HISTORY_BYTES = 4_194_304


class StateSchemaMismatch(ProtocolRecordError):
    """The persisted state is not compatible with this API release."""


class _TimedStreamingResponse(StreamingResponse):
    """Cancel an idle or backpressured ASGI stream, not just its source iterator."""

    def __init__(
        self, *args, release_slot: Callable[[], None], max_seconds: float = 35.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._release_slot = release_slot
        self._max_seconds = max_seconds

    async def __call__(self, scope, receive, send) -> None:
        try:
            await asyncio.wait_for(
                super().__call__(scope, receive, send), timeout=self._max_seconds
            )
        except TimeoutError:
            # The HTTP headers may already be sent; closing the connection
            # causes the browser to reauthorize and replay from its last seq.
            return
        finally:
            self._release_slot()


@dataclass(frozen=True)
class ReadResources:
    store: ProtocolStore
    graph: Any


def _checkpoint_id(config: Any) -> str | None:
    if not isinstance(config, Mapping):
        return None
    configurable = config.get("configurable")
    if not isinstance(configurable, Mapping):
        return None
    value = configurable.get("checkpoint_id")
    return value if isinstance(value, str) else None


def project_snapshot(snapshot: Any) -> dict[str, Any]:
    """Expose state and checkpoint identity, never trace/checkpoint metadata."""

    values = snapshot.values or {}
    assert_secret_free(values, path="checkpoint.values")
    if values:
        # The first LangGraph checkpoints precede intake. A plain per-run
        # request_id channel is absent there; only intake may populate it.
        # Keep history readable without accepting a missing ID after intake.
        if (
            "request_id" not in values
            and set(snapshot.next or ()) in ({"__start__"}, {"intake"})
        ):
            values = {**values, "request_id": ""}
        if (
            not isinstance(values, Mapping)
            or set(values) - STATE_FIELD_NAMES
            or values.get("schema_version") != 1
        ):
            raise StateSchemaMismatch("checkpoint state schema mismatch")
        try:
            CommerceStateSnapshot.model_validate(values)
        except ValidationError as exc:
            raise StateSchemaMismatch("checkpoint state schema mismatch") from exc
    payload = {
        "values": protocol_json_value(values),
        "checkpoint": (
            {"checkpoint_id": checkpoint_id}
            if (checkpoint_id := _checkpoint_id(snapshot.config)) else None
        ),
        "next": protocol_json_value(list(snapshot.next or ())),
    }
    assert_secret_free(payload, path="http.state")
    if len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) > MAX_STATE_BYTES:
        raise ProtocolRecordError("checkpoint state response exceeds size limit")
    return payload


def project_live_state(
    snapshot: Any, run_status: str | None, run_id: str | None = None,
) -> dict[str, Any]:
    """Join checkpoint truth with transport lifecycle for SDK hydration.

    A queued run has no new checkpoint yet, and a cancelled paused run still
    has its old interrupted checkpoint. Neither checkpoint alone describes
    whether the transport should reconnect or show an approval.
    """

    payload = project_snapshot(snapshot)
    if run_status is None:
        return payload
    if run_status not in {
        "queued", "running", "interrupted", "completed", "failed", "cancelled",
    }:
        raise ProtocolRecordError("invalid run status")
    payload["run_status"] = run_status
    if run_id is not None:
        try:
            payload["run_id"] = str(UUID(run_id))
        except (TypeError, ValueError, AttributeError) as exc:
            raise ProtocolRecordError("invalid run ID") from exc
    if run_status in {"queued", "running"}:
        # An empty checkpoint `next` is not proof of an idle thread while a
        # run is queued or its final checkpoint awaits terminalization. The
        # pinned SDK treats an absent `next` as active and rejoins SSE.
        if not payload["next"]:
            del payload["next"]
        payload["tasks"] = []
    elif run_status == "interrupted":
        payload["tasks"] = [
            {
                "interrupts": [
                    {"id": item.id, "value": protocol_json_value(item.value)}
                    for item in task.interrupts
                ],
            }
            for task in getattr(snapshot, "tasks", ())
            if task.interrupts
        ]
    else:
        # A cancelled run can leave a physical LangGraph interrupt behind.
        # It is no longer actionable; suppress stale approval on remount and
        # let the UI use run_status for the terminal outcome.
        payload["next"] = []
        payload["tasks"] = []
    assert_secret_free(payload, path="http.state")
    if len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) > MAX_STATE_BYTES:
        raise ProtocolRecordError("checkpoint state response exceeds size limit")
    return payload


def _owner(request: Request) -> ThreadOwner:
    expected = request.app.state.service_token
    supplied = request.headers.get("x-commerce-service-token", "")
    if not supplied or not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="service authentication required")
    try:
        return ThreadOwner(
            tenant_id=request.headers.get("x-commerce-tenant-id", ""),
            user_id=request.headers.get("x-commerce-user-id", ""),
        )
    except (ProtocolRecordError, SensitiveStateError):
        raise HTTPException(status_code=400, detail="verified actor context required") from None


def _thread_id(request: Request) -> str:
    raw = request.path_params["thread_id"]
    try:
        return str(UUID(raw))
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail="invalid thread ID") from None


def _run_id(request: Request) -> str:
    raw = request.path_params["run_id"]
    try:
        return str(UUID(raw))
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail="invalid run ID") from None


async def _read_state(request: Request) -> JSONResponse:
    owner = _owner(request)
    resources: ReadResources = request.app.state.resources
    thread_id = _thread_id(request)
    try:
        await resources.store.require_owned(owner, thread_id)
        # A second tab can start/finish another run while the checkpointer
        # read is in flight. Never pair an old terminal run with a new active
        # checkpoint: that would make the SDK skip SSE on remount.
        for _ in range(3):
            before = await resources.store.latest_run_lifecycle(owner, thread_id)
            snapshot = await resources.graph.aget_state(
                {"configurable": {"thread_id": thread_id}}
            )
            after = await resources.store.latest_run_lifecycle(owner, thread_id)
            if before == after:
                break
        else:
            raise HTTPException(
                status_code=503, detail="state changed during read",
                headers={"Retry-After": "1"},
            )
        return JSONResponse(
            project_live_state(
                snapshot,
                before[1] if before is not None else None,
                before[0] if before is not None else None,
            ),
            headers={"Cache-Control": "no-store"},
        )
    except ThreadNotOwned:
        raise HTTPException(status_code=404, detail="thread not found") from None
    except StateSchemaMismatch:
        raise HTTPException(status_code=409, detail="state schema mismatch") from None
    except (ProtocolRecordError, SensitiveStateError):
        raise HTTPException(status_code=500, detail="state unavailable") from None


async def _read_history(request: Request) -> JSONResponse:
    owner = _owner(request)
    resources: ReadResources = request.app.state.resources
    thread_id = _thread_id(request)
    try:
        limit = int(request.query_params.get("limit", "20"))
    except ValueError:
        raise HTTPException(status_code=422, detail="invalid history limit") from None
    if not 1 <= limit <= 100:
        raise HTTPException(status_code=422, detail="invalid history limit")
    try:
        await resources.store.require_owned(owner, thread_id)
        snapshots = resources.graph.aget_state_history(
            {"configurable": {"thread_id": thread_id}}, limit=limit
        )
        items: list[dict[str, Any]] = []
        total_bytes = 0
        try:
            async for snapshot in snapshots:
                item = project_snapshot(snapshot)
                total_bytes += len(json.dumps(item, ensure_ascii=False).encode("utf-8"))
                if total_bytes > MAX_HISTORY_BYTES:
                    raise ProtocolRecordError("checkpoint history response exceeds size limit")
                items.append(item)
        finally:
            close = getattr(snapshots, "aclose", None)
            if close is not None:
                await close()
        return JSONResponse({"items": items}, headers={"Cache-Control": "no-store"})
    except ThreadNotOwned:
        raise HTTPException(status_code=404, detail="thread not found") from None
    except StateSchemaMismatch:
        raise HTTPException(status_code=409, detail="state schema mismatch") from None
    except (ProtocolRecordError, SensitiveStateError):
        raise HTTPException(status_code=500, detail="state unavailable") from None


async def _read_subscription(request: Request) -> dict[str, Any]:
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > 16_384:
            raise HTTPException(status_code=413, detail="stream subscription too large")
        body.extend(chunk)
    try:
        value = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        raise HTTPException(status_code=422, detail="invalid stream subscription") from None
    if not isinstance(value, dict):
        raise HTTPException(status_code=422, detail="invalid stream subscription")
    return value


async def _command(request: Request) -> JSONResponse:
    owner = _owner(request)
    resources: ReadResources = request.app.state.resources
    thread_id = _thread_id(request)
    role = request.headers.get("x-commerce-role", "")
    request_id = request.headers.get("x-request-id", "")
    if role not in {"viewer", "manager", "admin"}:
        raise HTTPException(status_code=400, detail="verified role required")
    try:
        request_id = str(UUID(request_id))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="verified request ID required") from None
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > 32_768:
            raise HTTPException(status_code=413, detail="command too large")
        body.extend(chunk)
    try:
        raw = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        raise HTTPException(status_code=422, detail="invalid command") from None
    try:
        if isinstance(raw, Mapping) and raw.get("method") == "input.respond":
            if role not in {"manager", "admin"}:
                raise HTTPException(status_code=403, detail="manager approval required")
            command = parse_respond_command(raw, role=role)
            await resources.store.require_owned(owner, thread_id)
            snapshot = await resources.graph.aget_state(
                {"configurable": {"thread_id": thread_id}}
            )
            pending = any(
                item.id == command.interrupt_id
                for task in snapshot.tasks for item in task.interrupts
            )
            run_id = await resources.store.accept_resume(
                owner, thread_id,
                interrupt_id=command.interrupt_id,
                response=command.response,
                checkpoint_id=_checkpoint_id(snapshot.config) if pending else None,
                role=role,
            )
        else:
            command = parse_start_command(
                raw, thread_id=thread_id, owner=owner, role=role, request_id=request_id
            )
            run_id = await resources.store.create_run(
                owner,
                thread_id,
                command_id=command.command_id,
                input_payload=command.graph_input,
                runtime_context=command.runtime_context,
            )
    except ThreadNotOwned:
        raise HTTPException(status_code=404, detail="thread not found") from None
    except (ActiveRunConflict, IdempotencyConflict):
        raise HTTPException(status_code=409, detail="run conflict") from None
    except (ProtocolRecordError, SensitiveStateError):
        raise HTTPException(status_code=422, detail="invalid command") from None
    return JSONResponse(
        {"type": "success", "id": command.response_id, "result": {"run_id": run_id}},
        headers={"Cache-Control": "no-store"},
    )


async def _cancel_run(request: Request) -> Response:
    owner = _owner(request)
    resources: ReadResources = request.app.state.resources
    thread_id = _thread_id(request)
    run_id = _run_id(request)
    if len(request.query_params.multi_items()) != 2 or dict(request.query_params) != {
        "wait": "0", "action": "interrupt",
    }:
        raise HTTPException(status_code=422, detail="unsupported cancellation option")
    async for chunk in request.stream():
        if chunk:
            raise HTTPException(status_code=422, detail="cancellation body is not accepted")
    try:
        await resources.store.request_cancel(owner, thread_id, run_id)
    except ThreadNotOwned:
        raise HTTPException(status_code=404, detail="run not found") from None
    except ProtocolRecordError:
        raise HTTPException(status_code=422, detail="invalid cancellation") from None
    return Response(status_code=204, headers={"Cache-Control": "no-store"})


async def _stream(request: Request) -> StreamingResponse:
    owner = _owner(request)
    resources: ReadResources = request.app.state.resources
    thread_id = _thread_id(request)
    try:
        params = parse_subscribe_params(await _read_subscription(request))
        await resources.store.require_owned(owner, thread_id)
    except ThreadNotOwned:
        raise HTTPException(status_code=404, detail="thread not found") from None
    except ProtocolRecordError:
        raise HTTPException(status_code=422, detail="invalid stream subscription") from None
    semaphore: asyncio.Semaphore = request.app.state.stream_slots
    try:
        await asyncio.wait_for(semaphore.acquire(), timeout=0.01)
    except TimeoutError:
        raise HTTPException(status_code=429, detail="too many active streams") from None

    released = False

    def release_slot() -> None:
        nonlocal released
        if not released:
            released = True
            semaphore.release()

    async def bounded_stream() -> AsyncIterator[bytes]:
        try:
            async for frame in stream_events(resources.store, owner, thread_id, params):
                yield frame
        finally:
            release_slot()

    return _TimedStreamingResponse(
        bounded_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store",
            "X-Accel-Buffering": "no",
        },
        release_slot=release_slot,
    )


def create_read_app(
    *,
    resources: ReadResources | None = None,
    service_token: str | None = None,
) -> Starlette:
    """Create the internal read API; tests may inject isolated resources."""

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        token = service_token if service_token is not None else os.getenv(SERVICE_TOKEN_ENV, "")
        if len(token.encode("utf-8")) < 32:
            raise RuntimeError(f"{SERVICE_TOKEN_ENV} must have at least 32 bytes")
        app.state.service_token = token
        app.state.stream_slots = asyncio.Semaphore(16)
        if resources is not None:
            app.state.resources = resources
            yield
            return
        settings = PersistenceSettings.from_environment()
        async with open_protocol_store(settings) as store:
            async with open_postgres_graph(settings) as graph:
                app.state.resources = ReadResources(store=store, graph=graph)
                yield

    async def http_error(_request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            {"detail": str(exc.detail)},
            status_code=exc.status_code,
            headers={**(exc.headers or {}), "Cache-Control": "no-store"},
        )

    return Starlette(
        routes=[
            Route("/threads/{thread_id}/state", _read_state, methods=["GET"]),
            Route("/threads/{thread_id}/history", _read_history, methods=["GET"]),
            Route("/threads/{thread_id}/stream", _stream, methods=["POST"]),
            Route("/threads/{thread_id}/commands", _command, methods=["POST"]),
            Route(
                "/threads/{thread_id}/runs/{run_id}/cancel", _cancel_run,
                methods=["POST"],
            ),
        ],
        lifespan=lifespan,
        exception_handlers={HTTPException: http_error},
    )


app = create_read_app()


__all__ = [
    "ReadResources", "SERVICE_TOKEN_ENV", "StateSchemaMismatch", "app",
    "create_read_app", "project_snapshot",
]
