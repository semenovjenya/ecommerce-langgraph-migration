from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import Future, wait
from dataclasses import dataclass
from functools import partial
import json
import logging
import math
import os
import re
from threading import Lock
from time import monotonic, time
from typing import Any
from uuid import UUID

from langsmith import Client
from pydantic import BaseModel

from commerce_agent.safety import SensitiveStateError, assert_secret_free
from commerce_agent.debug_privacy import sanitize_debug_payload, sanitize_debug_inputs


logger = logging.getLogger(__name__)

EU_LANGSMITH_API_ENDPOINT = "https://eu.api.smith.langchain.com"
DEFAULT_LANGFUSE_BASE_URL = "https://cloud.langfuse.com"
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_MODEL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/+-]{0,127}\Z")
_SAFE_SPAN_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_.:/-]{0,127}\Z")
_NUMERIC_ATTRIBUTE_PREFIXES = ("gen_ai.usage.", "gen_ai.client.token.usage.")
_METRIC_NAMES = frozenset(
    {
        "input", "output", "total", "input_tokens", "output_tokens",
        "prompt_tokens", "completion_tokens", "total_tokens",
        "cache_read_input_tokens", "cache_creation_input_tokens",
        "reasoning_tokens", "input_cost", "output_cost", "total_cost",
    }
)
_JSON_METRIC_ATTRIBUTES = frozenset(
    {"langfuse.observation.usage_details", "langfuse.observation.cost_details"}
)
_OBSERVATION_TYPES = frozenset(
    {"agent", "chain", "embedding", "evaluator", "event", "generation", "guardrail", "retriever", "span", "tool"}
)
_OBSERVATION_LEVELS = frozenset({"DEFAULT", "DEBUG", "WARNING", "ERROR"})
_EXPORT_ENVIRONMENTS = frozenset({"development", "staging", "production", "test"})
_RESOURCE_KEYS = frozenset(
    {
        "langfuse.environment", "langfuse.release", "service.name",
        "telemetry.sdk.language", "telemetry.sdk.name", "telemetry.sdk.version",
    }
)
_LANGSMITH_RUN_TYPES = frozenset(
    {"chain", "llm", "tool", "retriever", "embedding", "prompt", "parser"}
)
_LANGSMITH_DOTTED_ORDER = re.compile(r"[0-9A-Fa-fTZ.:-]{1,4096}\Z")
_LANGSMITH_TIMESTAMP = re.compile(r"[0-9T:+.Z-]{1,64}\Z")
_LANGSMITH_NODE_NAMES = frozenset(
    {
        "commerce_agent", "intake", "router", "query_plan", "sales",
        "marketing", "supply", "price", "finance", "prepare",
        "collect_evidence", "analyze", "validate_and_publish",
        "evidence_gate", "gap_resolution", "deep_research", "synthesis",
        "approval_interrupt", "action_execution", "effect_measurement_schedule",
    }
)
_STATE_SUMMARY_DOMAINS = (
    "sales", "marketing", "supply", "price", "finance", "deep_research"
)
_STATE_SUMMARY_AGENT_STATUSES = frozenset(
    {"pending", "running", "planned", "completed", "failed", "skipped", "interrupted", "cancelled"}
)
_STATE_SUMMARY_ANALYSIS_STATUSES = frozenset(
    {"not_implemented", "completed", "partial", "failed"}
)
_STATE_SUMMARY_COUNTS = {
    "detected_incidents": "incident_count",
    "evidence_matrix": "evidence_count",
    "data_gaps": "data_gap_count",
    "action_tasks": "action_task_count",
    "execution_receipts": "execution_receipt_count",
    "errors": "error_count",
}


class ObservabilityConfigurationError(ValueError):
    """An enabled exporter would violate the graph's privacy or region contract."""


def _enabled(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in _TRUE_VALUES


@dataclass(frozen=True)
class ObservabilityConfig:
    """Non-secret observability settings safe to attach to graph metadata."""

    langsmith_tracing: bool
    langsmith_implicit_tracing: bool
    langsmith_endpoint: str
    langsmith_project: str
    langsmith_has_api_key: bool
    langsmith_hide_inputs: bool
    langsmith_hide_outputs: bool
    langsmith_hide_metadata: bool
    langsmith_state_summary: bool
    langsmith_debug: bool
    runtime_environment: str
    langfuse_enabled: bool
    langfuse_base_url: str
    langfuse_environment: str
    langfuse_has_public_key: bool
    langfuse_has_secret_key: bool

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> "ObservabilityConfig":
        env = os.environ if environment is None else environment
        return cls(
            langsmith_tracing=_enabled(env.get("COMMERCE_AGENT_LANGSMITH_TRACING")),
            langsmith_implicit_tracing=any(
                _enabled(env.get(name))
                for name in (
                    "LANGSMITH_TRACING",
                    "LANGSMITH_TRACING_V2",
                    "LANGCHAIN_TRACING",
                    "LANGCHAIN_TRACING_V2",
                )
            ),
            langsmith_endpoint=(
                env.get("LANGSMITH_ENDPOINT", EU_LANGSMITH_API_ENDPOINT).rstrip("/")
            ),
            langsmith_project=env.get(
                "LANGSMITH_PROJECT",
                "ai-ecom-commerce-agent-development",
            ),
            langsmith_has_api_key=bool(env.get("LANGSMITH_API_KEY")),
            langsmith_hide_inputs=_enabled(env.get("LANGSMITH_HIDE_INPUTS")),
            langsmith_hide_outputs=_enabled(env.get("LANGSMITH_HIDE_OUTPUTS")),
            langsmith_hide_metadata=_enabled(env.get("LANGSMITH_HIDE_METADATA")),
            langsmith_state_summary=_enabled(
                env.get("COMMERCE_AGENT_LANGSMITH_STATE_SUMMARY")
            ),
            langsmith_debug=_enabled(env.get("COMMERCE_AGENT_LANGSMITH_DEBUG")),
            runtime_environment=env.get("COMMERCE_AGENT_ENVIRONMENT", "").strip().lower(),
            langfuse_enabled=_enabled(env.get("LANGFUSE_ENABLED")),
            langfuse_base_url=env.get(
                "LANGFUSE_BASE_URL",
                DEFAULT_LANGFUSE_BASE_URL,
            ).rstrip("/"),
            langfuse_environment=env.get(
                "LANGFUSE_TRACING_ENVIRONMENT",
                "development",
            ),
            langfuse_has_public_key=bool(env.get("LANGFUSE_PUBLIC_KEY")),
            langfuse_has_secret_key=bool(env.get("LANGFUSE_SECRET_KEY")),
        )

    def warnings(self) -> tuple[str, ...]:
        warnings: list[str] = []
        if self.langsmith_tracing and not self.langsmith_has_api_key:
            warnings.append("LangSmith tracing disabled: LANGSMITH_API_KEY is missing")
        if self.langfuse_enabled and not (
            self.langfuse_has_public_key and self.langfuse_has_secret_key
        ):
            warnings.append(
                "Langfuse tracing disabled: LANGFUSE_PUBLIC_KEY or "
                "LANGFUSE_SECRET_KEY is missing"
            )
        return tuple(warnings)

    def assert_safe_export_configuration(self) -> None:
        _safe_langsmith_project(self.langsmith_project)
        if self.langsmith_debug and self.runtime_environment != "development":
            raise ObservabilityConfigurationError("LangSmith debug is development only")
        if self.langsmith_state_summary and self.runtime_environment not in {
            "development", "test"
        }:
            raise ObservabilityConfigurationError(
                "LangSmith state summary is development/test only"
            )
        if self.langsmith_implicit_tracing:
            raise ObservabilityConfigurationError(
                "commerce-agent requires implicit LangSmith tracing disabled"
            )
        if not (self.langsmith_tracing and self.langsmith_has_api_key):
            return
        if self.langsmith_endpoint != EU_LANGSMITH_API_ENDPOINT:
            raise ObservabilityConfigurationError(
                "LangSmith tracing requires the approved EU API endpoint"
            )
        if not (
            self.langsmith_hide_inputs
            and self.langsmith_hide_outputs
            and self.langsmith_hide_metadata
        ):
            raise ObservabilityConfigurationError(
                "LangSmith tracing requires hidden inputs, outputs and metadata"
            )

    @property
    def langsmith_ready(self) -> bool:
        return (
            self.langsmith_tracing
            and self.langsmith_has_api_key
            and self.langsmith_endpoint == EU_LANGSMITH_API_ENDPOINT
            and self.langsmith_hide_inputs
            and self.langsmith_hide_outputs
            and self.langsmith_hide_metadata
        )

    @property
    def langfuse_ready(self) -> bool:
        return (
            self.langfuse_enabled
            and self.langfuse_has_public_key
            and self.langfuse_has_secret_key
        )

    def mode(self, *, langfuse_attached: bool) -> str:
        if self.langsmith_ready and langfuse_attached:
            return "dual"
        if self.langsmith_ready:
            return "langsmith"
        if langfuse_attached:
            return "langfuse"
        return "disabled"


def _safe_langsmith_project(value: str) -> str:
    if not _SAFE_SPAN_NAME.fullmatch(value):
        raise ObservabilityConfigurationError("invalid LangSmith project name")
    try:
        assert_secret_free(value, path="trace.name")
    except SensitiveStateError as exc:
        raise ObservabilityConfigurationError("unsafe LangSmith project name") from exc
    return value


def _summary_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="python")
    return value if isinstance(value, Mapping) else None


def project_langsmith_state_outputs(outputs: Any) -> dict[str, Any]:
    """Export only bounded, categorical CommerceState progress, never raw data.

    The pinned SDK calls this before its buffered-run exporter. The final
    exporter calls it again as a fail-closed allowlist, so an SDK change or an
    unexpected callback shape cannot smuggle a raw output into LangSmith.
    """

    candidate = _summary_mapping(outputs)
    if candidate is None:
        return {}
    for wrapper in ("output", "state_summary"):
        nested = _summary_mapping(candidate.get(wrapper))
        if nested is not None:
            candidate = nested

    summary: dict[str, Any] = {}
    if type(candidate.get("schema_version")) is int and candidate["schema_version"] == 1:
        summary["schema_version"] = 1

    agent_statuses = _summary_mapping(candidate.get("agent_statuses"))
    if agent_statuses is not None:
        safe_statuses: dict[str, str] = {}
        for domain in _STATE_SUMMARY_DOMAINS:
            raw_status = agent_statuses.get(domain)
            status_record = _summary_mapping(raw_status)
            status = status_record.get("status") if status_record is not None else raw_status
            if isinstance(status, str) and status in _STATE_SUMMARY_AGENT_STATUSES:
                safe_statuses[domain] = status
        if safe_statuses:
            summary["agent_statuses"] = safe_statuses

    analysis_statuses = _summary_mapping(candidate.get("analysis_statuses"))
    safe_analyses: dict[str, str] = {}
    for domain in _STATE_SUMMARY_DOMAINS[:5]:
        raw_analysis = (
            analysis_statuses.get(domain)
            if analysis_statuses is not None
            else candidate.get(f"{domain}_analysis")
        )
        analysis_record = _summary_mapping(raw_analysis)
        status = analysis_record.get("status") if analysis_record is not None else raw_analysis
        if isinstance(status, str) and status in _STATE_SUMMARY_ANALYSIS_STATUSES:
            safe_analyses[domain] = status
    if safe_analyses:
        summary["analysis_statuses"] = safe_analyses

    for source, destination in _STATE_SUMMARY_COUNTS.items():
        raw = candidate.get(source)
        if isinstance(raw, (list, tuple)):
            summary[destination] = min(len(raw), 1_000_000)
        elif type(candidate.get(destination)) is int and 0 <= candidate[destination] <= 1_000_000:
            summary[destination] = candidate[destination]

    for source, destination in (
        ("query_plan", "plan_present"),
        ("manager_decision", "decision_present"),
        ("pending_interrupt", "interrupt_pending"),
    ):
        if source in candidate:
            summary[destination] = candidate[source] is not None
        elif type(candidate.get(destination)) is bool:
            summary[destination] = candidate[destination]

    if not summary:
        return {}
    assert_secret_free(summary, path="langsmith.state_summary")
    return {"state_summary": summary}


def project_langsmith_run_ops(
    runs: list[dict[str, Any]], *, project_name: str | None = None,
    include_state_summary: bool = False,
    include_debug: bool = False,
) -> list[dict[str, Any]]:
    """Allow graph structure and optional safe progress through the buffered-run boundary.

    The pinned LangSmith SDK hides I/O and metadata but not arbitrary errors,
    events, tags or future fields. Projection is intentionally fail-closed for
    structural identifiers and discards every unrecognized run field.
    """

    projected: list[dict[str, Any]] = []
    for run in runs:
        if not isinstance(run, dict):
            raise ObservabilityConfigurationError("invalid LangSmith run operation")
        safe: dict[str, Any] = {}
        for field in ("id", "trace_id", "parent_run_id"):
            if field not in run or run[field] is None:
                continue
            try:
                safe[field] = UUID(str(run[field]))
            except (TypeError, ValueError, AttributeError) as exc:
                raise ObservabilityConfigurationError(
                    "invalid LangSmith run identifier"
                ) from exc
        if "id" not in safe:
            raise ObservabilityConfigurationError("missing LangSmith run identifier")
        if "dotted_order" in run and run["dotted_order"] is not None:
            order = run["dotted_order"]
            if not isinstance(order, str) or not _LANGSMITH_DOTTED_ORDER.fullmatch(order):
                raise ObservabilityConfigurationError("invalid LangSmith dotted order")
            safe["dotted_order"] = order
        for field in ("start_time", "end_time"):
            if field in run and run[field] is not None:
                value = run[field]
                if hasattr(value, "isoformat"):
                    value = value.isoformat()
                if not isinstance(value, str) or not _LANGSMITH_TIMESTAMP.fullmatch(value):
                    raise ObservabilityConfigurationError(
                        "invalid LangSmith run timestamp"
                    )
                safe[field] = value
        if "name" in run and run["name"] is not None:
            safe["name"] = (
                run["name"]
                if isinstance(run["name"], str)
                and run["name"] in _LANGSMITH_NODE_NAMES
                else "redacted_run"
            )
            if include_debug and safe["name"] == "redacted_run":
                name = run["name"]
                if isinstance(name, str) and (name in {
                    "ChatOpenAI", "RunnableSequence", "RunnableParallel<raw>",
                    "sales_subgraph",
                    "RunnableAssign<parsed,parsing_error>", "RunnableParallel<parsed,parsing_error>",
                    "RunnableLambda", "PydanticOutputParser", "JsonOutputParser",
                } or re.fullmatch(r"tool\.(?:sales|marketing|supply|price|finance|reference)\.[a-z_]{1,64}(?:\.hydrate)?", name)):
                    safe["name"] = name
        if project_name is not None and "session_name" in run:
            safe["session_name"] = _safe_langsmith_project(project_name)
        if "run_type" in run and run["run_type"] is not None:
            safe["run_type"] = (
                run["run_type"]
                if isinstance(run["run_type"], str)
                and run["run_type"] in _LANGSMITH_RUN_TYPES
                else "chain"
            )
        if "inputs" in run:
            safe["inputs"] = sanitize_debug_payload(run["inputs"], allow_prepared_messages=safe.get("run_type") == "llm") if include_debug else {}
        if "outputs" in run:
            safe["outputs"] = (
                project_langsmith_state_outputs(run["outputs"])
                if include_state_summary and safe.get("name") in _LANGSMITH_NODE_NAMES
                else {}
            )
            if include_debug:
                safe["outputs"] = sanitize_debug_payload(run["outputs"])
        if "error" in run and run["error"] is not None:
            safe["error"] = "redacted_error"
        if "events" in run:
            safe["events"] = []
        if "tags" in run:
            safe["tags"] = []
        if "extra" in run:
            safe["extra"] = {}
        assert_secret_free(safe, path="langsmith.export")
        projected.append(safe)
    return projected


def _numeric_telemetry(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value >= 0
    )


def _safe_metric_map(value: Any) -> bool:
    if not isinstance(value, str) or len(value) > 4096:
        return False
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError):
        return False
    return (
        isinstance(decoded, dict)
        and 0 < len(decoded) <= 32
        and all(
            isinstance(key, str)
            and key in _METRIC_NAMES
            and _numeric_telemetry(metric)
            for key, metric in decoded.items()
        )
    )


def _safe_export_attribute(key: str, value: Any) -> bool:
    if key in _JSON_METRIC_ATTRIBUTES:
        return _safe_metric_map(value)
    if key == "langfuse.observation.type":
        return isinstance(value, str) and value in _OBSERVATION_TYPES
    if key == "langfuse.observation.level":
        return isinstance(value, str) and value in _OBSERVATION_LEVELS
    if key == "langfuse.environment":
        return isinstance(value, str) and value in _EXPORT_ENVIRONMENTS
    if key == "langfuse.observation.model.name":
        allowed_models = {
            name.strip()
            for name in os.environ.get("LANGFUSE_MODEL_ALLOWLIST", "").split(",")
            if name.strip()
        }
        return (
            isinstance(value, str)
            and bool(_MODEL_NAME.fullmatch(value))
            and value in allowed_models
        )
    return any(
        key == prefix + metric_name
        for prefix in _NUMERIC_ATTRIBUTE_PREFIXES
        for metric_name in _METRIC_NAMES
    ) and _numeric_telemetry(value)


def mask_langfuse_otel_spans(*, params: Any) -> Any:
    """Export only validated usage/cost metrics and approved model names."""

    from langfuse.types import MaskOtelSpansResult, OtelSpanPatch

    patches = {}
    for identifier, span in params.spans.items():
        delete_attributes: list[str] = []
        for key, value in span.attributes.items():
            try:
                assert_secret_free({key: value}, path="trace")
            except SensitiveStateError:
                delete_attributes.append(key)
                continue
            if not _safe_export_attribute(key, value):
                delete_attributes.append(key)
        if delete_attributes:
            patches[identifier] = OtelSpanPatch(
                delete_attributes=tuple(delete_attributes)
            )
    return MaskOtelSpansResult(span_patches=patches) if patches else None


def should_export_safe_langfuse_span(span: Any) -> bool:
    """Drop spans whose unpatchable OTel fields could carry user data."""

    from langfuse.span_filter import is_default_export_span

    if not is_default_export_span(span):
        return False
    scope = getattr(span, "instrumentation_scope", None)
    if scope is None or scope.name != "langfuse-sdk":
        return False
    name = getattr(span, "name", None)
    if not isinstance(name, str) or not _SAFE_SPAN_NAME.fullmatch(name):
        return False
    if getattr(span, "events", ()) or getattr(span, "links", ()):
        return False
    status = getattr(span, "status", None)
    if status is not None and getattr(status, "description", None):
        return False
    resource = getattr(span, "resource", None)
    attributes = getattr(resource, "attributes", None)
    if not isinstance(attributes, Mapping) or not set(attributes) <= _RESOURCE_KEYS:
        return False
    if any(
        not isinstance(value, str)
        or len(value) > 128
        or (value and not _MODEL_NAME.fullmatch(value))
        for value in attributes.values()
    ):
        return False
    try:
        assert_secret_free(
            {
                "name": name,
                "scope_name": scope.name,
                "scope_version": getattr(scope, "version", None),
                "resource": dict(attributes),
            },
            path="trace",
        )
    except SensitiveStateError:
        return False
    return True


def _create_langfuse_callback() -> Any:
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler

    client = Langfuse(
        mask_otel_spans=mask_langfuse_otel_spans,
        should_export_span=should_export_safe_langfuse_span,
    )
    if (
        getattr(getattr(client, "_resources", None), "mask_otel_spans", None)
        is not mask_langfuse_otel_spans
        or getattr(getattr(client, "_resources", None), "should_export_span", None)
        is not should_export_safe_langfuse_span
    ):
        # Langfuse reuses an existing process-wide client for the same key.
        # Never attach our callback to one initialized without both safety hooks.
        raise ObservabilityConfigurationError(
            "Langfuse client was initialized without the required export safety hooks"
        )
    return CallbackHandler(public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"))


class _ProjectedLangSmithClient(Client):
    """Track pinned SDK projection workers so shutdown can drain them."""

    def __init__(self, **kwargs: Any) -> None:
        self._projection_futures: set[Future[Any]] = set()
        self._projection_futures_lock = Lock()
        super().__init__(**kwargs)

    def _flush_run_ops_buffer(self) -> None:
        # The pinned SDK submits this batch to a background pool but its flush()
        # does not wait for that pool when compression is not yet initialized.
        # Keep our own strong references to pending projection/send workers.
        if not self._run_ops_buffer:
            return
        batch = list(self._run_ops_buffer)
        self._run_ops_buffer.clear()
        self._run_ops_buffer_last_flush_time = time()

        from langsmith._internal._background_thread import (
            LANGSMITH_CLIENT_THREAD_POOL,
            _process_buffered_run_ops_batch,
        )

        try:
            future = LANGSMITH_CLIENT_THREAD_POOL.submit(
                _process_buffered_run_ops_batch, self, batch
            )
        except RuntimeError:
            _process_buffered_run_ops_batch(self, batch)
            return
        with self._projection_futures_lock:
            self._projection_futures.add(future)
        future.add_done_callback(self._forget_projection_future)
        if self._futures is not None:
            self._futures.add(future)

    def _forget_projection_future(self, future: Future[Any]) -> None:
        with self._projection_futures_lock:
            self._projection_futures.discard(future)

    def wait_for_projection(self, *, timeout: float) -> None:
        with self._projection_futures_lock:
            pending = tuple(self._projection_futures)
        if not pending:
            return
        _, unfinished = wait(pending, timeout=timeout)
        if unfinished:
            raise TimeoutError("LangSmith projection drain timed out")


def _create_langsmith_callback(
    config: ObservabilityConfig,
    environment: Mapping[str, str],
) -> Any:
    from langchain_core.tracers.langchain import LangChainTracer
    client = _ProjectedLangSmithClient(
        api_url=EU_LANGSMITH_API_ENDPOINT,
        api_key=environment.get("LANGSMITH_API_KEY"),
        workspace_id=environment.get("LANGSMITH_WORKSPACE_ID") or None,
        hide_inputs=sanitize_debug_inputs if config.langsmith_debug else True,
        hide_outputs=(
            sanitize_debug_payload if config.langsmith_debug else
            project_langsmith_state_outputs
            if config.langsmith_state_summary else True
        ),
        hide_metadata=True,
        omit_traced_runtime_info=True,
        process_buffered_run_ops=partial(project_langsmith_run_ops, **{
            "project_name": config.langsmith_project,
            "include_state_summary": config.langsmith_state_summary,
            **({"include_debug": True} if config.langsmith_debug else {}),
        }),
        run_ops_buffer_size=1,
    )
    return LangChainTracer(project_name=config.langsmith_project, client=client)


def flush_langsmith_traces(compiled_graph: Any, *, timeout: float = 30.0) -> None:
    """Drain pinned SDK projection workers and its export queue at shutdown."""

    from langchain_core.tracers.langchain import LangChainTracer

    if timeout < 0:
        raise ValueError("trace flush timeout must not be negative")
    config = getattr(compiled_graph, "config", None) or {}
    for callback in config.get("callbacks", ()):
        if isinstance(callback, LangChainTracer):
            client = callback.client
            started = monotonic()
            client.flush(timeout=timeout)
            if isinstance(client, _ProjectedLangSmithClient):
                client.wait_for_projection(
                    timeout=max(0.0, timeout - (monotonic() - started))
                )
                client.flush(timeout=max(0.0, timeout - (monotonic() - started)))


def configure_graph_observability(
    compiled_graph: Any,
    *,
    environment: Mapping[str, str] | None = None,
    langsmith_callback_factory: Callable[[ObservabilityConfig, Mapping[str, str]], Any] | None = None,
    langfuse_callback_factory: Callable[[], Any] | None = None,
) -> Any:
    """Attach masked tracing callbacks and safe correlation metadata.

    The explicit LangSmith callback prevents LangChain from adding its implicit
    environment-driven tracer with a client that only hides I/O and metadata.
    The callback client projects every buffered run operation before export.
    """

    env = os.environ if environment is None else environment
    config = ObservabilityConfig.from_environment(env)
    config.assert_safe_export_configuration()
    for warning in config.warnings():
        logger.warning("commerce_agent_observability_configuration: %s", warning)

    callbacks: list[Any] = []
    if config.langsmith_ready:
        factory = langsmith_callback_factory or _create_langsmith_callback
        try:
            callbacks.append(factory(config, env))
        except Exception as exc:
            # A safety hook failure must not silently turn into unprojected
            # tracing on a later invocation.
            raise ObservabilityConfigurationError(
                "safe LangSmith callback initialization failed"
            ) from exc
    langfuse_attached = False
    if config.langfuse_ready:
        factory = langfuse_callback_factory or _create_langfuse_callback
        try:
            callbacks.append(factory())
            langfuse_attached = True
        except Exception:
            logger.exception("commerce_agent_langfuse_callback_initialization_failed")

    mode = config.mode(langfuse_attached=langfuse_attached)
    runtime_config: dict[str, Any] = {
        "metadata": {
            "service": "commerce-agent",
            "graph_id": "commerce_agent",
            "observability_mode": mode,
            "langsmith_project": config.langsmith_project,
            "environment": config.langfuse_environment,
        }
    }
    if callbacks:
        runtime_config["callbacks"] = callbacks
    return compiled_graph.with_config(runtime_config)


__all__ = [
    "DEFAULT_LANGFUSE_BASE_URL",
    "EU_LANGSMITH_API_ENDPOINT",
    "ObservabilityConfig",
    "ObservabilityConfigurationError",
    "configure_graph_observability",
    "flush_langsmith_traces",
    "mask_langfuse_otel_spans",
    "project_langsmith_state_outputs",
    "project_langsmith_run_ops",
    "should_export_safe_langfuse_span",
]
