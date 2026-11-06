"""Demonstrate a real M1 skeleton invocation with safe state in LangSmith EU.

This is a development-only diagnostic. The fixed domain subgraphs are still
stubs, so their `planned`/`not_implemented` state is not a business analysis.
Only categorical/count projections leave the process; raw state stays in the
dedicated PostgreSQL checkpointer.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Mapping
from urllib.parse import urlsplit
from uuid import UUID, uuid4

from langsmith import Client
from psycopg.conninfo import conninfo_to_dict

from commerce_agent.observability import (
    EU_LANGSMITH_API_ENDPOINT,
    project_langsmith_state_outputs,
)


_QUESTION = "Проверь продажи и остатки за последние 30 дней"
_LOCAL_DATABASE_PARAMETERS = {"user", "password", "host", "port", "dbname", "sslmode"}
_DATABASE_TARGET_ENV_VARS = (
    "PGHOST",
    "PGHOSTADDR",
    "PGPORT",
    "PGDATABASE",
    "PGSERVICE",
    "PGSERVICEFILE",
)


def _is_local_development_database_uri(
    database_uri: str, environment: Mapping[str, str]
) -> bool:
    """Validate the effective libpq target, not merely the visible URI host."""

    if any(environment.get(name) for name in _DATABASE_TARGET_ENV_VARS):
        return False
    try:
        parsed = urlsplit(database_uri)
        parameters = conninfo_to_dict(database_uri)
        return (
            parsed.scheme in {"postgres", "postgresql"}
            and parsed.hostname in {"localhost", "127.0.0.1"}
            and parsed.port == 55432
            and parsed.path == "/commerce_agent"
            and not parsed.fragment
            and set(parameters) <= _LOCAL_DATABASE_PARAMETERS
            and parameters.get("host") in {"localhost", "127.0.0.1"}
            and parameters.get("port") == "55432"
            and parameters.get("dbname") == "commerce_agent"
        )
    except Exception:
        return False


def _pin_eu_environment(project: str, *, tracing: bool) -> None:
    os.environ["LANGSMITH_ENDPOINT"] = EU_LANGSMITH_API_ENDPOINT
    os.environ["LANGCHAIN_ENDPOINT"] = EU_LANGSMITH_API_ENDPOINT
    os.environ["LANGSMITH_PROJECT"] = project
    os.environ["LANGCHAIN_PROJECT"] = project
    os.environ["COMMERCE_AGENT_ENVIRONMENT"] = "development"
    os.environ["COMMERCE_AGENT_LANGSMITH_TRACING"] = "true" if tracing else "false"
    os.environ["COMMERCE_AGENT_LANGSMITH_STATE_SUMMARY"] = "true" if tracing else "false"
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGSMITH_TRACING_V2"] = "false"
    os.environ["LANGCHAIN_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGSMITH_HIDE_INPUTS"] = "true"
    os.environ["LANGSMITH_HIDE_OUTPUTS"] = "true"
    os.environ["LANGSMITH_HIDE_METADATA"] = "true"
    os.environ["LANGFUSE_ENABLED"] = "false"

    from langsmith.utils import get_env_var

    get_env_var.cache_clear()


async def _invoke_current_graph() -> tuple[UUID, str]:
    from langchain_core.messages import HumanMessage

    from commerce_agent.context import CommerceRuntimeContext
    from commerce_agent.persistence import PersistenceSettings, open_postgres_graph

    run_id = uuid4()
    thread_id = str(uuid4())
    context = CommerceRuntimeContext(
        tenant_id="langsmith-state-demo-tenant",
        user_id="langsmith-state-demo-user",
        roles=("manager",),
        request_id=f"langsmith-state-demo-{run_id}",
    )
    async with open_postgres_graph(PersistenceSettings.from_environment()) as graph:
        result = await graph.ainvoke(
            {
                "schema_version": 1,
                "messages": [HumanMessage(content=_QUESTION, id=f"demo-{run_id}")],
            },
            config={
                "run_id": run_id,
                "configurable": {"thread_id": thread_id},
            },
            context=context,
        )
        snapshot = await graph.aget_state(
            {"configurable": {"thread_id": thread_id}}
        )
        if snapshot.next or result.get("schema_version") != 1:
            raise RuntimeError("development graph did not reach a completed checkpoint")
        for domain in ("sales", "marketing", "supply", "price", "finance"):
            status = result["agent_statuses"][domain]
            status_value = status.get("status") if isinstance(status, Mapping) else status.status
            if status_value != "planned":
                raise RuntimeError("unexpected M1 domain status")
            analysis = result[f"{domain}_analysis"]
            analysis_status = (
                analysis.get("status") if isinstance(analysis, Mapping) else analysis.status
            )
            if analysis_status != "not_implemented":
                raise RuntimeError("unexpected M1 analysis status")
        graph.flush_traces(timeout=30)
    return run_id, context.request_id


def _safe_readback(runs: list[object], *, request_id: str) -> object | None:
    root = None
    for run in runs:
        if getattr(run, "inputs", None) not in (None, {}):
            return None
        outputs = getattr(run, "outputs", None)
        if outputs not in (None, {}) and (
            not isinstance(outputs, Mapping)
            or set(outputs) != {"state_summary"}
            or project_langsmith_state_outputs(outputs) != outputs
        ):
            return None
        extra = getattr(run, "extra", None) or {}
        if not isinstance(extra, Mapping) or set(extra) - {"metadata"}:
            return None
        metadata = extra.get("metadata") or {}
        if not isinstance(metadata, Mapping) or set(metadata) - {"ls_run_depth"}:
            return None
        if getattr(run, "error", None) not in (None, "redacted_error"):
            return None
        if getattr(run, "events", None) not in (None, []):
            return None
        if getattr(run, "tags", None) not in (None, []):
            return None
        if getattr(run, "attachments", None) not in (None, {}):
            return None
        serialized = str(run.model_dump(mode="json"))
        if any(value in serialized for value in (
            _QUESTION, "langsmith-state-demo-tenant",
            "langsmith-state-demo-user", request_id,
        )):
            return None
        if getattr(run, "name", None) == "commerce_agent":
            root = run
    return root


def main() -> int:
    api_key = os.environ.get("LANGSMITH_API_KEY")
    if not api_key:
        print("LANGSMITH_API_KEY is not configured")
        return 2
    database_uri = os.environ.get("COMMERCE_AGENT_DATABASE_URI")
    if not database_uri:
        print("COMMERCE_AGENT_DATABASE_URI is not configured")
        return 2
    local_database = _is_local_development_database_uri(database_uri, os.environ)
    if (
        os.environ.get("COMMERCE_AGENT_ENVIRONMENT", "development").strip().lower()
        not in {"development", "test"}
        or not local_database
    ):
        print("State-summary demo requires a loopback development DB URI; verify no port-forward")
        return 2
    project = os.environ.get(
        "LANGSMITH_PROJECT", "ai-ecom-commerce-agent-development"
    )
    _pin_eu_environment(project, tracing=False)
    client = Client(
        api_url=EU_LANGSMITH_API_ENDPOINT,
        api_key=api_key,
        workspace_id=os.environ.get("LANGSMITH_WORKSPACE_ID") or None,
        hide_inputs=True,
        hide_outputs=True,
        hide_metadata=True,
    )
    try:
        next(client.list_projects(limit=1), None)
    except Exception as exc:
        print(f"LangSmith EU preflight failed: {type(exc).__name__}")
        return 1

    _pin_eu_environment(project, tracing=True)
    try:
        trace_id, request_id = asyncio.run(_invoke_current_graph())
    except Exception as exc:
        print(f"Graph invocation failed: {type(exc).__name__}")
        return 1

    for _ in range(15):
        try:
            runs = list(client.list_runs(project_name=project, trace_id=trace_id))
        except Exception as exc:
            print(f"LangSmith EU readback failed: {type(exc).__name__}")
            return 1
        if runs:
            root = _safe_readback(runs, request_id=request_id)
            if root is None:
                print(f"LangSmith trace failed privacy/readback checks: {trace_id}")
                return 1
            summary = (root.outputs or {}).get("state_summary", {})
            if (
                summary.get("agent_statuses", {}).get("sales") == "planned"
                and summary.get("analysis_statuses", {}).get("finance")
                == "not_implemented"
            ):
                try:
                    url = client.get_run_url(run=root, project_name=project)
                except Exception:
                    url = "https://eu.smith.langchain.com"
                print(f"LangSmith EU state-summary trace verified: {trace_id}")
                print(f"Project: {project}")
                print("Domain status: planned; analysis status: not_implemented (M1 skeleton)")
                print(f"State summary: {json.dumps(summary, ensure_ascii=False, sort_keys=True)}")
                print(f"Trace URL: {url}")
                return 0
        time.sleep(2)

    print(f"LangSmith EU state summary not visible after graph run: {trace_id}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
