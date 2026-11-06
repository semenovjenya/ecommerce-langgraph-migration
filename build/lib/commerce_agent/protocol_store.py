"""Durable ownership, run and replay records for the self-managed protocol.

The LangGraph checkpointer owns graph state. These tables own transport/run
lifecycle and are deliberately separate from business data and legacy chats.
Schema setup is an explicit deployment step, never a service-startup side effect.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from hashlib import sha256
import json
import re
from time import time
from typing import Any
from uuid import UUID, uuid4

from psycopg.types.json import Jsonb
from psycopg.rows import dict_row
from psycopg.errors import UniqueViolation
from psycopg_pool import AsyncConnectionPool

from commerce_agent.persistence import PersistenceSettings
from commerce_agent.safety import assert_secret_free


_COMMAND_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_EVENT_METHOD = re.compile(r"[a-z][a-z0-9_.:-]{0,127}\Z")


class ThreadNotOwned(LookupError):
    """The thread does not exist or does not belong to the verified actor."""


class IdempotencyConflict(ValueError):
    """A command ID was reused with different input or runtime context."""


class ActiveRunConflict(ValueError):
    """A second run cannot start while this thread has unfinished work."""


class ProtocolRecordError(ValueError):
    """An untrusted protocol record failed validation."""


@dataclass(frozen=True)
class ThreadOwner:
    tenant_id: str
    user_id: str

    def __post_init__(self) -> None:
        if not self.tenant_id or not self.user_id:
            raise ProtocolRecordError("verified tenant_id and user_id are required")
        if len(self.tenant_id) > 128 or len(self.user_id) > 128:
            raise ProtocolRecordError("verified identity exceeds its size limit")
        assert_secret_free(
            {"tenant_id": self.tenant_id, "user_id": self.user_id},
            path="verified_identity",
        )


@dataclass(frozen=True)
class ClaimedRun:
    run_id: str
    thread_id: str
    owner: ThreadOwner
    input_payload: dict[str, Any]
    runtime_context: dict[str, Any]
    attempt: int
    resume_interrupt_id: str | None = None
    resume_response: dict[str, Any] | None = None
    resume_checkpoint_id: str | None = None
    cancel_requested: bool = False


def _thread_uuid(thread_id: str) -> UUID:
    try:
        return UUID(thread_id)
    except (TypeError, ValueError, AttributeError) as exc:
        raise ProtocolRecordError("thread_id must be a UUID") from exc


def _json_payload(value: Any, *, path: str) -> tuple[str, Any]:
    assert_secret_free(value, path=path)
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ProtocolRecordError("protocol payload must be JSON serializable") from exc
    if len(encoded.encode("utf-8")) > 262_144:
        raise ProtocolRecordError("protocol payload exceeds 256 KiB")
    return encoded, json.loads(encoded)


def _safe_protocol_event(event: Mapping[str, Any]) -> tuple[dict[str, Any], str, list[str]]:
    if not isinstance(event, Mapping):
        raise ProtocolRecordError("event must be an object")
    _, safe_event = _json_payload(dict(event), path="protocol.event")
    method = safe_event.get("method")
    params = safe_event.get("params")
    namespace = params.get("namespace") if isinstance(params, dict) else None
    if (
        safe_event.get("type") != "event"
        or not isinstance(method, str)
        or not _EVENT_METHOD.fullmatch(method)
        or not isinstance(namespace, list)
        or not all(isinstance(part, str) and len(part) <= 128 for part in namespace)
    ):
        raise ProtocolRecordError("invalid Agent Protocol event envelope")
    if "seq" in safe_event or "event_id" in safe_event:
        raise ProtocolRecordError("event sequence is assigned by PostgreSQL")
    return safe_event, method, namespace


def lifecycle_event(status: str) -> dict[str, Any]:
    if status not in {"running", "completed", "failed", "interrupted", "cancelled"}:
        raise ProtocolRecordError("invalid lifecycle status")
    return {
        "type": "event", "method": "lifecycle",
        "params": {
            "namespace": [], "timestamp": int(time() * 1000),
            "data": {"event": status},
        },
    }


def cancellation_events() -> tuple[dict[str, Any], dict[str, Any]]:
    """Preserve the cancelled outcome and settle pinned SDK v1 clients.

    The v1 lifecycle controller only treats completed/failed/interrupted as
    terminal. An interrupted transport marker is not an approval interrupt:
    the durable run status stays cancelled and no input.requested is emitted.
    """

    terminal = lifecycle_event("interrupted")
    terminal["params"]["data"]["reason"] = "cancelled"
    return lifecycle_event("cancelled"), terminal


class ProtocolStore:
    """Every data read/write is scoped by verified tenant and user identity."""

    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool

    async def setup(self) -> None:
        """Run the idempotent M1.5 schema migration explicitly."""
        async with self._pool.connection() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS commerce_agent_threads (
                        thread_id uuid PRIMARY KEY,
                        tenant_id text NOT NULL,
                        user_id text NOT NULL,
                        last_seq bigint NOT NULL DEFAULT -1,
                        created_at timestamptz NOT NULL DEFAULT now()
                    )
                    """
                )
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS commerce_agent_runs (
                        run_id uuid PRIMARY KEY,
                        thread_id uuid NOT NULL REFERENCES commerce_agent_threads(thread_id),
                        command_id text NOT NULL,
                        request_sha256 char(64) NOT NULL,
                        input_json jsonb NOT NULL,
                        context_json jsonb NOT NULL,
                        start_checkpoint_id text,
                        start_checkpoint_recorded boolean NOT NULL DEFAULT false,
                        cancel_requested boolean NOT NULL DEFAULT false,
                        status text NOT NULL DEFAULT 'queued'
                            CHECK (status IN (
                                'queued', 'running', 'interrupted',
                                'completed', 'failed', 'cancelled'
                            )),
                        created_at timestamptz NOT NULL DEFAULT now(),
                        updated_at timestamptz NOT NULL DEFAULT now(),
                        UNIQUE (thread_id, command_id),
                        UNIQUE (thread_id, run_id)
                    )
                    """
                )
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS commerce_agent_events (
                        thread_id uuid NOT NULL,
                        seq bigint NOT NULL,
                        run_id uuid NOT NULL,
                        method text NOT NULL,
                        namespace text[] NOT NULL,
                        payload jsonb NOT NULL,
                        created_at timestamptz NOT NULL DEFAULT now(),
                        PRIMARY KEY (thread_id, seq),
                        FOREIGN KEY (thread_id, run_id)
                            REFERENCES commerce_agent_runs(thread_id, run_id)
                    )
                    """
                )
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS commerce_agent_interrupt_responses (
                        thread_id uuid NOT NULL,
                        run_id uuid NOT NULL,
                        interrupt_id text NOT NULL,
                        response_json jsonb NOT NULL,
                        response_sha256 char(64) NOT NULL,
                        checkpoint_id text NOT NULL,
                        accepted_by_user_id text NOT NULL,
                        accepted_role text NOT NULL,
                        applied boolean NOT NULL DEFAULT false,
                        created_at timestamptz NOT NULL DEFAULT now(),
                        PRIMARY KEY (thread_id, interrupt_id),
                        FOREIGN KEY (thread_id, run_id)
                            REFERENCES commerce_agent_runs(thread_id, run_id)
                            ON DELETE CASCADE
                    )
                    """
                )
                await conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS commerce_agent_runs_thread_status_idx
                    ON commerce_agent_runs(thread_id, status, created_at)
                    """
                )
                await conn.execute(
                    """
                    ALTER TABLE commerce_agent_runs
                    ADD COLUMN IF NOT EXISTS lease_owner uuid,
                    ADD COLUMN IF NOT EXISTS lease_until timestamptz,
                    ADD COLUMN IF NOT EXISTS attempts integer NOT NULL DEFAULT 0,
                    ADD COLUMN IF NOT EXISTS start_checkpoint_id text,
                    ADD COLUMN IF NOT EXISTS start_checkpoint_recorded boolean NOT NULL DEFAULT false,
                    ADD COLUMN IF NOT EXISTS cancel_requested boolean NOT NULL DEFAULT false
                    """
                )
                await conn.execute(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS commerce_agent_one_active_run_idx
                    ON commerce_agent_runs(thread_id)
                    WHERE status IN ('queued', 'running', 'interrupted')
                    """
                )
                await conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS commerce_agent_runs_claim_idx
                    ON commerce_agent_runs(status, lease_until, created_at)
                    """
                )

    async def create_thread(self, owner: ThreadOwner, thread_id: str) -> None:
        ident = _thread_uuid(thread_id)
        async with self._pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO commerce_agent_threads(thread_id, tenant_id, user_id)
                SELECT %s, %s, %s
                WHERE NOT EXISTS (
                    SELECT 1 FROM checkpoints WHERE thread_id = %s
                )
                ON CONFLICT (thread_id) DO NOTHING
                """,
                (ident, owner.tenant_id, owner.user_id, str(ident)),
            )
        await self.require_owned(owner, thread_id)

    async def require_owned(self, owner: ThreadOwner, thread_id: str) -> None:
        ident = _thread_uuid(thread_id)
        async with self._pool.connection() as conn:
            cursor = await conn.execute(
                """
                SELECT 1 FROM commerce_agent_threads
                WHERE thread_id = %s AND tenant_id = %s AND user_id = %s
                """,
                (ident, owner.tenant_id, owner.user_id),
            )
            if await cursor.fetchone() is None:
                raise ThreadNotOwned("thread not found")

    async def latest_run_lifecycle(
        self, owner: ThreadOwner, thread_id: str
    ) -> tuple[str, str] | None:
        """Return the latest owned run identity and status, independent of checkpoints."""

        ident = _thread_uuid(thread_id)
        async with self._pool.connection() as conn:
            cursor = await conn.execute(
                """
                SELECT r.run_id, r.status FROM commerce_agent_runs r
                JOIN commerce_agent_threads t ON t.thread_id = r.thread_id
                WHERE r.thread_id = %s AND t.tenant_id = %s AND t.user_id = %s
                ORDER BY r.created_at DESC, r.run_id DESC
                LIMIT 1
                """,
                (ident, owner.tenant_id, owner.user_id),
            )
            row = await cursor.fetchone()
            return (str(row["run_id"]), row["status"]) if row is not None else None

    async def accept_resume(
        self,
        owner: ThreadOwner,
        thread_id: str,
        *,
        interrupt_id: str,
        response: Mapping[str, Any],
        checkpoint_id: str | None,
        role: str,
    ) -> str:
        """Idempotently queue the same interrupted run, never a new run."""

        if role not in {"manager", "admin"}:
            raise ProtocolRecordError("manager approval role required")
        if not isinstance(interrupt_id, str) or not 1 <= len(interrupt_id) <= 128:
            raise ProtocolRecordError("invalid interrupt ID")
        if checkpoint_id is not None and (
            not isinstance(checkpoint_id, str) or not 1 <= len(checkpoint_id) <= 255
        ):
            raise ProtocolRecordError("invalid resume checkpoint ID")
        _, response_data = _json_payload(dict(response), path="interrupt.response")
        digest = sha256(json.dumps(
            response_data, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        ident = _thread_uuid(thread_id)
        async with self._pool.connection() as conn:
            async with conn.transaction():
                cursor = await conn.execute(
                    """
                    SELECT 1 FROM commerce_agent_threads
                    WHERE thread_id = %s AND tenant_id = %s AND user_id = %s
                    FOR UPDATE
                    """,
                    (ident, owner.tenant_id, owner.user_id),
                )
                if await cursor.fetchone() is None:
                    raise ThreadNotOwned("thread not found")
                cursor = await conn.execute(
                    """
                    SELECT i.run_id, i.response_sha256, r.status
                    FROM commerce_agent_interrupt_responses i
                    JOIN commerce_agent_runs r ON r.run_id = i.run_id
                    WHERE i.thread_id = %s AND i.interrupt_id = %s
                    """,
                    (ident, interrupt_id),
                )
                existing = await cursor.fetchone()
                if existing is not None:
                    if existing["response_sha256"] != digest:
                        raise IdempotencyConflict("interrupt response conflicts with prior decision")
                    if existing["status"] == "cancelled":
                        raise ActiveRunConflict("run was cancelled after approval")
                    return str(existing["run_id"])
                if checkpoint_id is None:
                    raise ActiveRunConflict("interrupt is not pending")
                cursor = await conn.execute(
                    """
                    SELECT run_id FROM commerce_agent_runs
                    WHERE thread_id = %s AND status = 'interrupted'
                    FOR UPDATE
                    """,
                    (ident,),
                )
                interrupted = await cursor.fetchone()
                if interrupted is None:
                    raise ActiveRunConflict("interrupt is not pending")
                run_id = interrupted["run_id"]
                await conn.execute(
                    """
                    INSERT INTO commerce_agent_interrupt_responses (
                        thread_id, run_id, interrupt_id, response_json,
                        response_sha256, checkpoint_id, accepted_by_user_id,
                        accepted_role
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        ident, run_id, interrupt_id, Jsonb(response_data), digest,
                        checkpoint_id, owner.user_id, role,
                    ),
                )
                await conn.execute(
                    """
                    UPDATE commerce_agent_runs
                    SET status = 'queued', updated_at = now()
                    WHERE run_id = %s AND status = 'interrupted'
                    """,
                    (run_id,),
                )
                return str(run_id)

    async def request_cancel(
        self, owner: ThreadOwner, thread_id: str, run_id: str
    ) -> str:
        """Cancel queued/paused work now; signal an active worker durably."""

        thread_ident = _thread_uuid(thread_id)
        try:
            run_ident = UUID(run_id)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ProtocolRecordError("run_id must be a UUID") from exc
        safe_events = [_safe_protocol_event(item) for item in cancellation_events()]
        async with self._pool.connection() as conn:
            async with conn.transaction():
                # Every operation that may also advance a thread's event
                # sequence locks thread -> run, including resume and finish.
                cursor = await conn.execute(
                    """
                    SELECT 1 FROM commerce_agent_threads
                    WHERE thread_id = %s AND tenant_id = %s AND user_id = %s
                    FOR UPDATE
                    """,
                    (thread_ident, owner.tenant_id, owner.user_id),
                )
                if await cursor.fetchone() is None:
                    raise ThreadNotOwned("run not found")
                cursor = await conn.execute(
                    """
                    SELECT r.status FROM commerce_agent_runs r
                    WHERE r.thread_id = %s AND r.run_id = %s
                    FOR UPDATE OF r
                    """,
                    (thread_ident, run_ident),
                )
                row = await cursor.fetchone()
                if row is None:
                    raise ThreadNotOwned("run not found")
                status = row["status"]
                if status in {"completed", "failed", "cancelled"}:
                    return status
                if status == "running":
                    await conn.execute(
                        """
                        UPDATE commerce_agent_runs
                        SET cancel_requested = true, updated_at = now()
                        WHERE run_id = %s
                        """,
                        (run_ident,),
                    )
                    return "requested"
                if status not in {"queued", "interrupted"}:
                    raise ProtocolRecordError("invalid run status")
                await conn.execute(
                    """
                    UPDATE commerce_agent_runs
                    SET status = 'cancelled', cancel_requested = true,
                        lease_owner = NULL, lease_until = NULL, updated_at = now()
                    WHERE run_id = %s
                    """,
                    (run_ident,),
                )
                cursor = await conn.execute(
                    """
                    UPDATE commerce_agent_threads SET last_seq = last_seq + %s
                    WHERE thread_id = %s RETURNING last_seq
                    """,
                    (len(safe_events), thread_ident),
                )
                last_seq = (await cursor.fetchone())["last_seq"]
                for index, (event, method, namespace) in enumerate(safe_events):
                    seq = last_seq - len(safe_events) + index + 1
                    event["seq"] = seq
                    event["event_id"] = f"{thread_ident}:{seq}"
                    await conn.execute(
                        """
                        INSERT INTO commerce_agent_events (
                            thread_id, seq, run_id, method, namespace, payload
                        ) VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (thread_ident, seq, run_ident, method, namespace, Jsonb(event)),
                    )
                return "cancelled"

    async def is_cancel_requested(self, run_id: str, worker_id: str) -> bool:
        try:
            run_ident, worker_ident = UUID(run_id), UUID(worker_id)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ProtocolRecordError("invalid run or worker ID") from exc
        async with self._pool.connection() as conn:
            cursor = await conn.execute(
                """
                SELECT cancel_requested FROM commerce_agent_runs
                WHERE run_id = %s AND status = 'running'
                  AND lease_owner = %s AND lease_until > now()
                """,
                (run_ident, worker_ident),
            )
            row = await cursor.fetchone()
            return bool(row and row["cancel_requested"])

    async def create_run(
        self,
        owner: ThreadOwner,
        thread_id: str,
        *,
        command_id: str,
        input_payload: Mapping[str, Any],
        runtime_context: Mapping[str, Any],
    ) -> str:
        if not isinstance(command_id, str) or not _COMMAND_ID.fullmatch(command_id):
            raise ProtocolRecordError("invalid command ID")
        if not isinstance(input_payload, Mapping) or not isinstance(runtime_context, Mapping):
            raise ProtocolRecordError("run input and context must be objects")
        _, input_data = _json_payload(dict(input_payload), path="run.input")
        _, context_data = _json_payload(dict(runtime_context), path="run.context")
        if (
            context_data.get("tenant_id") != owner.tenant_id
            or context_data.get("user_id") != owner.user_id
        ):
            raise ProtocolRecordError("runtime identity disagrees with verified owner")
        # A transport retry may have a new correlation request_id. Compare
        # every other policy/context field so a command ID cannot silently
        # alias a run started under different roles or model policy.
        stable_context = {
            key: value for key, value in context_data.items() if key != "request_id"
        }
        digest = sha256(
            json.dumps(
                {"input": input_data, "context": stable_context},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        await self.create_thread(owner, thread_id)
        ident = _thread_uuid(thread_id)
        new_run_id = uuid4()
        async with self._pool.connection() as conn:
            try:
                cursor = await conn.execute(
                    """
                    INSERT INTO commerce_agent_runs (
                        run_id, thread_id, command_id, request_sha256,
                        input_json, context_json
                    )
                    SELECT %s, t.thread_id, %s, %s, %s, %s
                    FROM commerce_agent_threads t
                    WHERE t.thread_id = %s AND t.tenant_id = %s AND t.user_id = %s
                    ON CONFLICT (thread_id, command_id) DO NOTHING
                    RETURNING run_id, request_sha256
                    """,
                    (
                        new_run_id, command_id, digest,
                        Jsonb(input_data), Jsonb(context_data),
                        ident, owner.tenant_id, owner.user_id,
                    ),
                )
            except UniqueViolation as exc:
                raise ActiveRunConflict("thread already has an active run") from exc
            row = await cursor.fetchone()
            if row is None:
                cursor = await conn.execute(
                    """
                    SELECT r.run_id, r.request_sha256 FROM commerce_agent_runs r
                    JOIN commerce_agent_threads t ON t.thread_id = r.thread_id
                    WHERE r.thread_id = %s AND r.command_id = %s
                        AND t.tenant_id = %s AND t.user_id = %s
                    """,
                    (ident, command_id, owner.tenant_id, owner.user_id),
                )
                row = await cursor.fetchone()
            if row is None:
                raise ThreadNotOwned("thread not found")
            if row["request_sha256"] != digest:
                raise IdempotencyConflict("command ID conflicts with an existing run")
            return str(row["run_id"])

    async def claim_next_run(
        self, worker_id: str, *, lease_seconds: int = 60,
        target_run_id: str | None = None,
    ) -> ClaimedRun | None:
        try:
            worker_uuid = UUID(worker_id)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ProtocolRecordError("worker ID must be a UUID") from exc
        try:
            target_uuid = UUID(target_run_id) if target_run_id is not None else None
        except (TypeError, ValueError, AttributeError) as exc:
            raise ProtocolRecordError("target run ID must be a UUID") from exc
        if type(lease_seconds) is not int or not 2 <= lease_seconds <= 600:
            raise ProtocolRecordError("invalid run lease")
        async with self._pool.connection() as conn:
            async with conn.transaction():
                cursor = await conn.execute(
                    """
                    WITH picked AS (
                        SELECT run_id FROM commerce_agent_runs
                        WHERE (status = 'queued'
                           OR (status = 'running' AND lease_until < now()))
                          AND (%s::uuid IS NULL OR run_id = %s::uuid)
                        ORDER BY created_at, run_id
                        FOR UPDATE SKIP LOCKED
                        LIMIT 1
                    ), claimed AS (
                        UPDATE commerce_agent_runs r
                        SET status = 'running', lease_owner = %s,
                            lease_until = now() + (%s * interval '1 second'),
                            attempts = attempts + 1, updated_at = now()
                        FROM picked
                        WHERE r.run_id = picked.run_id
                        RETURNING r.run_id, r.thread_id, r.input_json,
                                  r.context_json, r.attempts, r.cancel_requested
                    )
                    SELECT claimed.run_id, claimed.thread_id, claimed.input_json,
                           claimed.context_json, claimed.attempts,
                           claimed.cancel_requested,
                           t.tenant_id, t.user_id,
                           pending_resume.interrupt_id AS resume_interrupt_id,
                           pending_resume.response_json AS resume_response,
                           pending_resume.checkpoint_id AS resume_checkpoint_id
                    FROM claimed
                    JOIN commerce_agent_threads t ON t.thread_id = claimed.thread_id
                    LEFT JOIN LATERAL (
                        SELECT interrupt_id, response_json, checkpoint_id
                        FROM commerce_agent_interrupt_responses
                        WHERE run_id = claimed.run_id AND NOT applied
                        ORDER BY created_at DESC LIMIT 1
                    ) pending_resume ON true
                    """,
                    (target_uuid, target_uuid, worker_uuid, lease_seconds),
                )
                row = await cursor.fetchone()
        if row is None:
            return None
        return ClaimedRun(
            run_id=str(row["run_id"]),
            thread_id=str(row["thread_id"]),
            owner=ThreadOwner(tenant_id=row["tenant_id"], user_id=row["user_id"]),
            input_payload=row["input_json"],
            runtime_context=row["context_json"],
            attempt=row["attempts"],
            resume_interrupt_id=row["resume_interrupt_id"],
            resume_response=row["resume_response"],
            resume_checkpoint_id=row["resume_checkpoint_id"],
            cancel_requested=row["cancel_requested"],
        )

    async def renew_run(
        self, run_id: str, worker_id: str, *, lease_seconds: int = 60
    ) -> bool:
        if type(lease_seconds) is not int or not 2 <= lease_seconds <= 600:
            raise ProtocolRecordError("invalid run lease")
        try:
            run_uuid, worker_uuid = UUID(run_id), UUID(worker_id)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ProtocolRecordError("invalid run or worker ID") from exc
        async with self._pool.connection() as conn:
            cursor = await conn.execute(
                """
                UPDATE commerce_agent_runs
                SET lease_until = now() + (%s * interval '1 second'), updated_at = now()
                WHERE run_id = %s AND status = 'running'
                  AND NOT cancel_requested
                  AND lease_owner = %s AND lease_until > now()
                RETURNING 1
                """,
                (lease_seconds, run_uuid, worker_uuid),
            )
            return await cursor.fetchone() is not None

    async def record_start_checkpoint(
        self,
        run_id: str,
        worker_id: str,
        checkpoint_id: str | None,
    ) -> str | None:
        """Fence and persist the checkpoint preceding this run's first input.

        A recovered worker compares the current checkpoint with this marker:
        equality means input was not durably applied yet; a later checkpoint
        means it must resume, not submit the browser message again.
        """

        try:
            run_uuid, worker_uuid = UUID(run_id), UUID(worker_id)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ProtocolRecordError("run_id and worker_id must be UUIDs") from exc
        if checkpoint_id is not None and (
            not isinstance(checkpoint_id, str) or not 1 <= len(checkpoint_id) <= 255
        ):
            raise ProtocolRecordError("invalid start checkpoint ID")
        async with self._pool.connection() as conn:
            async with conn.transaction():
                cursor = await conn.execute(
                    """
                    UPDATE commerce_agent_runs
                    SET start_checkpoint_id = %s, start_checkpoint_recorded = true,
                        updated_at = now()
                    WHERE run_id = %s AND status = 'running'
                      AND NOT cancel_requested
                      AND lease_owner = %s AND lease_until > now()
                      AND NOT start_checkpoint_recorded
                    RETURNING start_checkpoint_id
                    """,
                    (checkpoint_id, run_uuid, worker_uuid),
                )
                row = await cursor.fetchone()
                if row is not None:
                    return row["start_checkpoint_id"]
                cursor = await conn.execute(
                    """
                    SELECT start_checkpoint_id FROM commerce_agent_runs
                    WHERE run_id = %s AND status = 'running'
                      AND NOT cancel_requested
                      AND lease_owner = %s AND lease_until > now()
                      AND start_checkpoint_recorded
                    """,
                    (run_uuid, worker_uuid),
                )
                row = await cursor.fetchone()
                if row is None:
                    raise ProtocolRecordError("run lease was lost before checkpoint recording")
                return row["start_checkpoint_id"]

    async def finish_run(
        self, run_id: str, worker_id: str, *, status: str,
        terminal_events: tuple[Mapping[str, Any], ...] = (),
    ) -> bool:
        if status not in {"completed", "failed", "interrupted", "cancelled"}:
            raise ProtocolRecordError("invalid terminal run status")
        try:
            run_uuid, worker_uuid = UUID(run_id), UUID(worker_id)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ProtocolRecordError("invalid run or worker ID") from exc
        safe_events = [_safe_protocol_event(event) for event in terminal_events]
        if len(safe_events) > 32:
            raise ProtocolRecordError("too many terminal events")
        async with self._pool.connection() as conn:
            async with conn.transaction():
                cursor = await conn.execute(
                    """
                    SELECT t.thread_id FROM commerce_agent_threads t
                    JOIN commerce_agent_runs r ON r.thread_id = t.thread_id
                    WHERE r.run_id = %s
                    FOR UPDATE OF t
                    """,
                    (run_uuid,),
                )
                if await cursor.fetchone() is None:
                    return False
                cursor = await conn.execute(
                    """
                    UPDATE commerce_agent_runs
                    SET status = %s, lease_owner = NULL, lease_until = NULL,
                        updated_at = now()
                    WHERE run_id = %s AND status = 'running'
                      AND (%s = 'cancelled' OR NOT cancel_requested)
                      AND lease_owner = %s AND lease_until > now()
                    RETURNING thread_id
                    """,
                    (status, run_uuid, status, worker_uuid),
                )
                row = await cursor.fetchone()
                if row is None:
                    return False
                if status in {"completed", "interrupted"}:
                    await conn.execute(
                        """
                        UPDATE commerce_agent_interrupt_responses
                        SET applied = true
                        WHERE run_id = %s AND NOT applied
                        """,
                        (run_uuid,),
                    )
                if safe_events:
                    thread_id = row["thread_id"]
                    cursor = await conn.execute(
                        """
                        UPDATE commerce_agent_threads
                        SET last_seq = last_seq + %s
                        WHERE thread_id = %s
                        RETURNING last_seq
                        """,
                        (len(safe_events), thread_id),
                    )
                    last_seq = (await cursor.fetchone())["last_seq"]
                    for index, (event, method, namespace) in enumerate(safe_events):
                        seq = last_seq - len(safe_events) + index + 1
                        event["seq"] = seq
                        event["event_id"] = f"{thread_id}:{seq}"
                        await conn.execute(
                            """
                            INSERT INTO commerce_agent_events (
                                thread_id, seq, run_id, method, namespace, payload
                            ) VALUES (%s, %s, %s, %s, %s, %s)
                            """,
                            (thread_id, seq, run_uuid, method, namespace, Jsonb(event)),
                        )
                return True

    async def append_event(
        self,
        owner: ThreadOwner,
        thread_id: str,
        run_id: str,
        event: Mapping[str, Any],
        *,
        worker_id: str | None = None,
    ) -> dict[str, Any]:
        safe_event, method, namespace = _safe_protocol_event(event)
        ident = _thread_uuid(thread_id)
        try:
            run_ident = UUID(run_id)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ProtocolRecordError("run_id must be a UUID") from exc
        try:
            worker_ident = UUID(worker_id) if worker_id is not None else None
        except (TypeError, ValueError, AttributeError) as exc:
            raise ProtocolRecordError("worker_id must be a UUID") from exc
        async with self._pool.connection() as conn:
            async with conn.transaction():
                cursor = await conn.execute(
                    """
                    SELECT 1 FROM commerce_agent_threads
                    WHERE thread_id = %s AND tenant_id = %s AND user_id = %s
                    FOR UPDATE
                    """,
                    (ident, owner.tenant_id, owner.user_id),
                )
                if await cursor.fetchone() is None:
                    raise ThreadNotOwned("thread or run not found")
                # Serialize with request_cancel and finish_run before deciding
                # whether a worker event may be committed. The old unlocked
                # EXISTS check could commit after an accepted cancellation.
                cursor = await conn.execute(
                    """
                    SELECT 1 FROM commerce_agent_runs
                    WHERE thread_id = %s AND run_id = %s
                      AND NOT cancel_requested
                      AND (%s::uuid IS NULL OR (
                          status = 'running' AND lease_owner = %s
                          AND lease_until > now()
                      ))
                    FOR UPDATE
                    """,
                    (ident, run_ident, worker_ident, worker_ident),
                )
                if await cursor.fetchone() is None:
                    raise ThreadNotOwned("thread or run not found")
                cursor = await conn.execute(
                    """
                    UPDATE commerce_agent_threads
                    SET last_seq = last_seq + 1
                    WHERE thread_id = %s
                    RETURNING last_seq
                    """,
                    (ident,),
                )
                row = await cursor.fetchone()
                seq = row["last_seq"]
                safe_event["seq"] = seq
                safe_event["event_id"] = f"{ident}:{seq}"
                await conn.execute(
                    """
                    INSERT INTO commerce_agent_events (
                        thread_id, seq, run_id, method, namespace, payload
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        ident, seq, run_ident, method, namespace,
                        Jsonb(safe_event),
                    ),
                )
                return safe_event

    async def read_events(
        self,
        owner: ThreadOwner,
        thread_id: str,
        *,
        since: int = -1,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        if type(since) is not int or not -1 <= since <= 9_223_372_036_854_775_807:
            raise ProtocolRecordError("invalid replay cursor")
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ProtocolRecordError("invalid replay page size")
        ident = _thread_uuid(thread_id)
        async with self._pool.connection() as conn:
            cursor = await conn.execute(
                """
                SELECT e.payload FROM commerce_agent_threads t
                LEFT JOIN LATERAL (
                    SELECT payload, seq FROM commerce_agent_events
                    WHERE thread_id = t.thread_id AND seq > %s
                    ORDER BY seq ASC LIMIT %s
                ) e ON true
                WHERE t.thread_id = %s AND t.tenant_id = %s AND t.user_id = %s
                ORDER BY e.seq ASC NULLS LAST
                """,
                (since, limit, ident, owner.tenant_id, owner.user_id),
            )
            rows = await cursor.fetchall()
            if not rows:
                raise ThreadNotOwned("thread not found")
            return [row["payload"] for row in rows if row["payload"] is not None]

    async def latest_run_root_values(
        self, owner: ThreadOwner, thread_id: str, run_id: str
    ) -> Any | None:
        """Return the last journaled root state for checkpoint reconciliation."""

        ident = _thread_uuid(thread_id)
        try:
            run_ident = UUID(run_id)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ProtocolRecordError("run_id must be a UUID") from exc
        async with self._pool.connection() as conn:
            cursor = await conn.execute(
                """
                SELECT e.payload FROM commerce_agent_events e
                JOIN commerce_agent_threads t ON t.thread_id = e.thread_id
                WHERE e.thread_id = %s AND e.run_id = %s
                  AND e.method = 'values' AND e.namespace = ARRAY[]::text[]
                  AND t.tenant_id = %s AND t.user_id = %s
                ORDER BY e.seq DESC LIMIT 1
                """,
                (ident, run_ident, owner.tenant_id, owner.user_id),
            )
            row = await cursor.fetchone()
            return row["payload"]["params"]["data"] if row is not None else None


@asynccontextmanager
async def open_protocol_store(settings: PersistenceSettings) -> AsyncIterator[ProtocolStore]:
    pool = AsyncConnectionPool(
        conninfo=settings.database_uri,
        min_size=settings.pool_min_size,
        max_size=settings.pool_max_size,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        },
        open=False,
        name="commerce-agent-protocol",
    )
    async with pool:
        yield ProtocolStore(pool)


async def initialize_protocol_schema(settings: PersistenceSettings) -> None:
    async with open_protocol_store(settings) as store:
        await store.setup()


__all__ = [
    "ActiveRunConflict",
    "ClaimedRun",
    "IdempotencyConflict",
    "ProtocolRecordError",
    "ProtocolStore",
    "ThreadNotOwned",
    "ThreadOwner",
    "cancellation_events",
    "initialize_protocol_schema",
    "open_protocol_store",
]
