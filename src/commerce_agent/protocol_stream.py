"""Bounded, owner-scoped replay and live SSE from the durable event journal."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
import json
import re
from time import monotonic
from typing import Any

from commerce_agent.protocol_store import ProtocolRecordError, ProtocolStore, ThreadOwner
from commerce_agent.safety import assert_secret_free


_CHANNELS = frozenset({
    "values", "messages", "updates", "checkpoints", "tasks", "tools",
    "custom", "lifecycle", "input", "input.requested",
})
_CUSTOM_CHANNEL = re.compile(r"custom:[A-Za-z][A-Za-z0-9_.:-]{0,127}\Z")
_NAMESPACE_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")


@dataclass(frozen=True)
class SubscribeParams:
    channels: frozenset[str]
    namespaces: tuple[tuple[str, ...], ...] | None
    depth: int | None
    since: int


def parse_subscribe_params(body: Any) -> SubscribeParams:
    """Reject ambiguous or unbounded browser subscriptions before streaming."""

    if not isinstance(body, Mapping) or set(body) - {
        "channels", "namespaces", "depth", "since"
    }:
        raise ProtocolRecordError("invalid stream subscription")
    channels = body.get("channels")
    if (
        not isinstance(channels, list)
        or not 1 <= len(channels) <= 32
        or not all(
            isinstance(channel, str)
            and (channel in _CHANNELS or _CUSTOM_CHANNEL.fullmatch(channel))
            for channel in channels
        )
    ):
        raise ProtocolRecordError("invalid stream channels")
    raw_namespaces = body.get("namespaces")
    namespaces: tuple[tuple[str, ...], ...] | None = None
    if raw_namespaces is not None:
        if not isinstance(raw_namespaces, list) or len(raw_namespaces) > 32:
            raise ProtocolRecordError("invalid stream namespaces")
        parsed_namespaces: list[tuple[str, ...]] = []
        for namespace in raw_namespaces:
            if (
                not isinstance(namespace, list)
                or len(namespace) > 32
                or not all(
                    isinstance(segment, str)
                    and _NAMESPACE_SEGMENT.fullmatch(segment)
                    for segment in namespace
                )
            ):
                raise ProtocolRecordError("invalid stream namespace")
            parsed_namespaces.append(tuple(namespace))
        namespaces = tuple(parsed_namespaces) or None
    depth = body.get("depth")
    if depth is not None and (type(depth) is not int or not 0 <= depth <= 32):
        raise ProtocolRecordError("invalid stream depth")
    since = body.get("since", -1)
    if type(since) is not int or not -1 <= since <= 9_223_372_036_854_775_807:
        raise ProtocolRecordError("invalid replay cursor")
    return SubscribeParams(
        channels=frozenset(channels), namespaces=namespaces, depth=depth, since=since
    )


def _segment_matches(pattern: str, segment: str) -> bool:
    if ":" in pattern:
        return pattern == segment
    return segment == pattern or segment.startswith(f"{pattern}:")


def _namespace_matches(
    namespace: list[str], prefixes: tuple[tuple[str, ...], ...] | None,
    depth: int | None,
) -> bool:
    candidates = prefixes if prefixes is not None else ((),)
    for prefix in candidates:
        if len(prefix) > len(namespace):
            continue
        if all(_segment_matches(pattern, segment) for pattern, segment in zip(prefix, namespace)):
            if depth is None or len(namespace) - len(prefix) <= depth:
                return True
    return False


def event_matches(event: Mapping[str, Any], params: SubscribeParams) -> bool:
    method = event.get("method")
    envelope = event.get("params")
    if not isinstance(method, str) or not isinstance(envelope, Mapping):
        raise ProtocolRecordError("invalid stored event")
    namespace = envelope.get("namespace")
    if not isinstance(namespace, list) or not all(isinstance(item, str) for item in namespace):
        raise ProtocolRecordError("invalid stored event namespace")
    if method == "custom":
        data = envelope.get("data")
        name = data.get("name") if isinstance(data, Mapping) else None
        channel_match = "custom" in params.channels or (
            isinstance(name, str) and f"custom:{name}" in params.channels
        )
    else:
        # The pinned React SDK subscribes to the `input` channel while the
        # protocol event method is `input.requested`.
        channel_match = method in params.channels or (
            method == "input.requested" and "input" in params.channels
        )
    return channel_match and _namespace_matches(namespace, params.namespaces, params.depth)


def encode_sse(event: Mapping[str, Any]) -> bytes:
    """Keep the complete protocol event in data; SSE id mirrors its journal ID."""

    assert_secret_free(event, path="protocol.sse")
    seq = event.get("seq")
    event_id = event.get("event_id")
    if type(seq) is not int or seq < 0 or not isinstance(event_id, str):
        raise ProtocolRecordError("invalid stored event identity")
    if not isinstance(event.get("params"), Mapping) or event.get("type") != "event":
        raise ProtocolRecordError("invalid stored event envelope")
    if any(char in event_id for char in "\r\n\x00"):
        raise ProtocolRecordError("invalid stored event ID")
    payload = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
    return f"id: {event_id}\nevent: message\ndata: {payload}\n\n".encode("utf-8")


async def stream_events(
    store: ProtocolStore,
    owner: ThreadOwner,
    thread_id: str,
    params: SubscribeParams,
    *,
    poll_seconds: float = 2.0,
    max_seconds: float = 30.0,
    heartbeat_seconds: float = 5.0,
    db_timeout_seconds: float = 5.0,
) -> AsyncIterator[bytes]:
    """Replay/follow briefly; reconnect must reauthorize through the BFF."""

    if not 0 < poll_seconds <= heartbeat_seconds < max_seconds <= 60:
        raise ValueError("invalid stream timing limits")
    if not 0 < db_timeout_seconds <= max_seconds:
        raise ValueError("invalid stream database timeout")
    cursor = params.since
    deadline = monotonic() + max_seconds
    heartbeat_at = monotonic() + heartbeat_seconds
    while monotonic() < deadline:
        events = await asyncio.wait_for(
            store.read_events(owner, thread_id, since=cursor, limit=250),
            timeout=db_timeout_seconds,
        )
        for event in events:
            seq = event.get("seq")
            if type(seq) is not int or seq <= cursor:
                raise ProtocolRecordError("stored event sequence is not monotonic")
            cursor = seq  # Advance even when a channel/namespace filter hides the event.
            if event_matches(event, params):
                yield encode_sse(event)
        now = monotonic()
        if now >= heartbeat_at and now < deadline:
            yield b": keepalive\n\n"
            heartbeat_at = now + heartbeat_seconds
        if not events:
            await asyncio.sleep(min(poll_seconds, max(0, deadline - now)))


__all__ = [
    "SubscribeParams", "encode_sse", "event_matches", "parse_subscribe_params",
    "stream_events",
]
