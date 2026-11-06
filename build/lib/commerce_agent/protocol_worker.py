"""Development M1.5 worker for durable queued LangGraph runs.

The API only commits commands. This separate process claims a PostgreSQL run,
executes the guarded graph, and appends its v3 events to the durable journal.
No business tools or external writes are enabled in the current stub graph.
"""

from __future__ import annotations

import asyncio
import logging
from time import time
from typing import Any
from uuid import uuid4

from langgraph.types import Command

from commerce_agent.context import CommerceRuntimeContext
from commerce_agent.persistence import PersistenceSettings, open_postgres_graph
from commerce_agent.protocol_events import (
    normalize_graph_event, protocol_json_value, requested_input_events,
)
from commerce_agent.protocol_store import (
    ClaimedRun, ProtocolStore, cancellation_events, lifecycle_event,
    open_protocol_store,
)
from commerce_agent.safety import assert_secret_free


logger = logging.getLogger(__name__)


class RunLeaseLost(RuntimeError):
    """The worker may not continue publishing after losing its claim."""


async def execute_claimed_run(
    store: ProtocolStore,
    graph: Any,
    claim: ClaimedRun,
    *,
    worker_id: str,
    lease_seconds: int = 60,
    run_timeout_seconds: float = 120.0,
) -> str:
    """Resume a checkpoint after a process loss; never replay browser input then."""

    assert_secret_free(claim.input_payload, path="worker.input")
    assert_secret_free(claim.runtime_context, path="worker.context")
    assert_secret_free(claim.resume_response, path="worker.resume")
    context = CommerceRuntimeContext.model_validate(claim.runtime_context)
    if context.tenant_id != claim.owner.tenant_id or context.user_id != claim.owner.user_id:
        raise ValueError("run identity disagrees with its durable owner")
    config = {"configurable": {"thread_id": claim.thread_id}}
    current_task = asyncio.current_task()
    assert current_task is not None
    stop_heartbeat = asyncio.Event()
    cancellation_seen = False

    async def finish_cancelled() -> str:
        if not await store.finish_run(
            claim.run_id, worker_id, status="cancelled",
            terminal_events=cancellation_events(),
        ):
            raise RunLeaseLost("run lease was lost before cancellation")
        return "cancelled"

    async def cancellation_requested() -> bool:
        try:
            return await store.is_cancel_requested(claim.run_id, worker_id)
        except Exception:
            return False

    async def maintain_lease() -> None:
        nonlocal cancellation_seen
        while not stop_heartbeat.is_set():
            try:
                await asyncio.wait_for(
                    stop_heartbeat.wait(), timeout=min(1.0, lease_seconds / 3),
                )
                return
            except TimeoutError:
                pass
            if await cancellation_requested():
                cancellation_seen = True
                current_task.cancel()
                return
            try:
                renewed = await store.renew_run(
                    claim.run_id, worker_id, lease_seconds=lease_seconds
                )
            except Exception:
                renewed = False
            if not renewed:
                cancellation_seen = await cancellation_requested()
                current_task.cancel()
                return

    heartbeat = asyncio.create_task(maintain_lease())
    try:
        if claim.cancel_requested or await cancellation_requested():
            return await finish_cancelled()
        async with asyncio.timeout(run_timeout_seconds):
            snapshot = await graph.aget_state(config)
            values = snapshot.values or {}
            checkpoint_config = (snapshot.config or {}).get("configurable", {})
            checkpoint_id = checkpoint_config.get("checkpoint_id")
            if values and not isinstance(checkpoint_id, str):
                raise ValueError("durable state has no checkpoint identity")
            start_checkpoint_id = await store.record_start_checkpoint(
                claim.run_id, worker_id, checkpoint_id
            )
            if checkpoint_id != start_checkpoint_id and not values:
                raise ValueError("run checkpoint disappeared before recovery")
            status: str | None = None
            if claim.resume_interrupt_id is not None and checkpoint_id == claim.resume_checkpoint_id:
                pending_ids = {
                    item.id for task in snapshot.tasks for item in task.interrupts
                }
                if claim.resume_interrupt_id not in pending_ids:
                    raise ValueError("resume interrupt is absent from checkpoint")
                graph_input = Command(resume={
                    claim.resume_interrupt_id: claim.resume_response,
                })
            elif checkpoint_id != start_checkpoint_id and values and not snapshot.next:
                # A synchronous final checkpoint can commit before the
                # corresponding v3 values event reaches the separate journal.
                # Repair the authoritative root state once before marking the
                # run terminal; never replay the browser input or graph nodes.
                final_values = protocol_json_value(values)
                journaled_values = await store.latest_run_root_values(
                    claim.owner, claim.thread_id, claim.run_id
                )
                if journaled_values != final_values:
                    await store.append_event(
                        claim.owner, claim.thread_id, claim.run_id,
                        {
                            "type": "event", "method": "values",
                            "params": {
                                "namespace": [], "timestamp": int(time() * 1000),
                                "data": final_values,
                            },
                        },
                        worker_id=worker_id,
                    )
                status = "completed"
            else:
                graph_input = (
                    claim.input_payload
                    if claim.resume_interrupt_id is None
                    and checkpoint_id == start_checkpoint_id
                    else None
                )
            terminal_interrupt_events: dict[str, dict[str, Any]] = {}
            if status is None:
                await store.append_event(
                    claim.owner, claim.thread_id, claim.run_id,
                    lifecycle_event("running"), worker_id=worker_id,
                )
                stream = await graph.astream_protocol_events(
                    graph_input, config=config, context=context
                )
                async for raw in stream:
                    event = normalize_graph_event(raw)
                    for requested in requested_input_events(event):
                        interrupt_id = requested["params"]["data"]["interrupt_id"]
                        terminal_interrupt_events[interrupt_id] = requested
                    if (
                        event["method"] == "lifecycle"
                        and isinstance(event["params"]["data"], dict)
                        and event["params"]["data"].get("event")
                        in {"completed", "failed", "interrupted"}
                    ):
                        continue
                    await store.append_event(
                        claim.owner, claim.thread_id, claim.run_id, event,
                        worker_id=worker_id,
                    )
                status = "interrupted" if terminal_interrupt_events else "completed"
            if await cancellation_requested():
                status = "cancelled"
                terminal_interrupt_events.clear()
        assert status is not None
        terminal_events = (
            cancellation_events() if status == "cancelled"
            else tuple(terminal_interrupt_events.values()) + (lifecycle_event(status),)
        )
        if not await store.finish_run(
            claim.run_id, worker_id, status=status, terminal_events=terminal_events,
        ):
            if await cancellation_requested():
                return await finish_cancelled()
            raise RunLeaseLost("run lease was lost before completion")
        return status
    except asyncio.CancelledError:
        if cancellation_seen or await cancellation_requested():
            return await finish_cancelled()
        # Process shutdown or lease loss: leave the running row for lease-based
        # recovery. Never mark it failed or replay input without checkpointing.
        raise
    except Exception:
        if await cancellation_requested():
            return await finish_cancelled()
        try:
            await store.finish_run(
                claim.run_id, worker_id, status="failed",
                terminal_events=(lifecycle_event("failed"),),
            )
        except Exception:
            pass
        raise
    finally:
        stop_heartbeat.set()
        heartbeat.cancel()
        await asyncio.gather(heartbeat, return_exceptions=True)


async def run_once(
    store: ProtocolStore,
    graph: Any,
    *,
    worker_id: str,
    lease_seconds: int = 60,
) -> bool:
    claim = await store.claim_next_run(worker_id, lease_seconds=lease_seconds)
    if claim is None:
        return False
    await execute_claimed_run(
        store, graph, claim, worker_id=worker_id, lease_seconds=lease_seconds
    )
    return True


async def run_forever() -> None:
    settings = PersistenceSettings.from_environment()
    worker_id = str(uuid4())
    async with open_protocol_store(settings) as store:
        async with open_postgres_graph(settings) as graph:
            try:
                while True:
                    try:
                        worked = await run_once(store, graph, worker_id=worker_id)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        # Exceptions may contain user text or connection details.
                        logger.warning("commerce_agent_worker_run_failed")
                        worked = False
                    if not worked:
                        await asyncio.sleep(0.5)
            finally:
                graph.flush_traces()


if __name__ == "__main__":
    asyncio.run(run_forever())


__all__ = ["RunLeaseLost", "execute_claimed_run", "run_forever", "run_once"]
