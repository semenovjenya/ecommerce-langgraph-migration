"""Fail-closed command parsing for the M1.5 self-managed Agent Protocol."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import re
from typing import Any

from collections.abc import Mapping
from langchain_core.messages import HumanMessage

from commerce_agent.context import CommerceRuntimeContext
from commerce_agent.graph import GRAPH_ID
from commerce_agent.protocol_store import ProtocolRecordError, ThreadOwner
from commerce_agent.safety import assert_secret_free


@dataclass(frozen=True)
class StartCommand:
    response_id: str | int
    command_id: str
    graph_input: dict[str, Any]
    runtime_context: dict[str, Any]


@dataclass(frozen=True)
class RespondCommand:
    response_id: str | int
    interrupt_id: str
    response: dict[str, bool]


_INTERRUPT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")


def _wire_id(value: Any) -> str | int:
    if (
        type(value) is not str and type(value) is not int
    ) or (
        type(value) is str and (not value or len(value) > 128)
    ) or (
        type(value) is int and not 0 <= value <= 9_007_199_254_740_991
    ):
        raise ProtocolRecordError("invalid command ID")
    return value


def parse_start_command(
    value: Any,
    *,
    thread_id: str,
    owner: ThreadOwner,
    role: str,
    request_id: str,
) -> StartCommand:
    """Accept only a text-only stub invocation; identity is service-derived."""

    assert_secret_free(value, path="protocol.command")
    if not isinstance(value, Mapping) or set(value) not in (
        {"id", "method", "params"},
        {"type", "id", "method", "params"},
    ):
        raise ProtocolRecordError("invalid command envelope")
    if value.get("type", "command") != "command" or value["method"] != "run.start":
        raise ProtocolRecordError("unsupported command")
    response_id = _wire_id(value["id"])
    params = value["params"]
    if not isinstance(params, Mapping) or set(params) - {"assistant_id", "input", "config"}:
        raise ProtocolRecordError("unsupported run.start parameter")
    if params.get("assistant_id", GRAPH_ID) != GRAPH_ID:
        raise ProtocolRecordError("wrong assistant ID")
    config = params.get("config")
    if config is not None and (
        not isinstance(config, Mapping)
        or set(config) != {"configurable"}
        or not isinstance(config["configurable"], Mapping)
        or dict(config["configurable"]) != {"thread_id": thread_id}
    ):
        raise ProtocolRecordError("run config must bind the owned thread")
    source = params.get("input")
    if not isinstance(source, Mapping) or set(source) - {"schema_version", "messages"}:
        raise ProtocolRecordError("invalid run input")
    if source.get("schema_version", 1) != 1 or type(source.get("schema_version", 1)) is not int:
        raise ProtocolRecordError("unsupported state schema")
    messages = source.get("messages")
    if not isinstance(messages, list) or not 1 <= len(messages) <= 20:
        raise ProtocolRecordError("text messages are required")

    normalized: list[dict[str, Any]] = []
    message_ids: list[str] = []
    for message in messages:
        if not isinstance(message, Mapping) or set(message) - {
            "type", "role", "content", "id", "additional_kwargs", "response_metadata",
        }:
            raise ProtocolRecordError("unsupported browser message")
        if message.get("type", "human") not in {"human", "user"}:
            raise ProtocolRecordError("only user messages are accepted")
        if message.get("role", "user") != "user":
            raise ProtocolRecordError("only user messages are accepted")
        if message.get("additional_kwargs", {}) != {} or message.get("response_metadata", {}) != {}:
            raise ProtocolRecordError("message metadata is not accepted")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip() or len(content) > 8_192:
            raise ProtocolRecordError("invalid user message text")
        message_id = message.get("id")
        if message_id is not None and (
            not isinstance(message_id, str) or not 1 <= len(message_id) <= 128
        ):
            raise ProtocolRecordError("invalid message ID")
        if type(response_id) is int:
            # SDK transport IDs are JS-runtime-local counters. A page reload
            # can restart at 1, so they are never durable run identities.
            if message_id is None:
                raise ProtocolRecordError("numeric SDK commands require stable message IDs")
            message_ids.append(message_id)
        normalized.append(
            HumanMessage(content=content, id=message_id).model_dump(mode="json")
        )

    command_id = response_id
    if type(response_id) is int:
        fingerprint = sha256(json.dumps(
            message_ids, ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        command_id = f"sdk-v1:{fingerprint}"

    try:
        context = CommerceRuntimeContext(
            tenant_id=owner.tenant_id,
            user_id=owner.user_id,
            roles=(role,),
            request_id=request_id,
        ).model_dump(mode="json")
    except ValueError as exc:
        raise ProtocolRecordError("invalid verified actor context") from exc
    assert_secret_free(context, path="protocol.runtime_context")
    return StartCommand(
        response_id=response_id,
        command_id=command_id,
        graph_input={"schema_version": 1, "messages": normalized},
        runtime_context=context,
    )


def parse_respond_command(value: Any, *, role: str) -> RespondCommand:
    """Accept only a root, boolean synthetic approval; no updates or goto."""

    assert_secret_free(value, path="protocol.command")
    if role not in {"manager", "admin"}:
        raise ProtocolRecordError("manager approval role required")
    if not isinstance(value, Mapping) or set(value) not in (
        {"id", "method", "params"}, {"type", "id", "method", "params"},
    ):
        raise ProtocolRecordError("invalid command envelope")
    if value.get("type", "command") != "command" or value["method"] != "input.respond":
        raise ProtocolRecordError("unsupported command")
    response_id = _wire_id(value["id"])
    params = value["params"]
    if not isinstance(params, Mapping) or set(params) != {
        "namespace", "interrupt_id", "response",
    }:
        raise ProtocolRecordError("unsupported input.respond parameter")
    if params["namespace"] != []:
        raise ProtocolRecordError("only root interrupts may be resumed")
    interrupt_id = params["interrupt_id"]
    if not isinstance(interrupt_id, str) or not _INTERRUPT_ID.fullmatch(interrupt_id):
        raise ProtocolRecordError("invalid interrupt ID")
    response = params["response"]
    if not isinstance(response, Mapping) or set(response) != {"approved"} or type(response["approved"]) is not bool:
        raise ProtocolRecordError("invalid synthetic approval response")
    return RespondCommand(
        response_id=response_id,
        interrupt_id=interrupt_id,
        response={"approved": response["approved"]},
    )


__all__ = [
    "RespondCommand", "StartCommand", "parse_respond_command", "parse_start_command",
]
