"""Opt-in read-only M3 Sales invocation, local checkpoints and EU state readback.

No accounts/bootstrap/schema setup/workers/actions. The actor must already
exist in the chosen development backend and is independently checked by M2.
"""
import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
import re
import stat
import time
from uuid import uuid4

import httpx
from langsmith import Client

if __package__:
    from .check_langsmith_state_summary import _is_local_development_database_uri, _pin_eu_environment, _safe_readback
else:
    from check_langsmith_state_summary import _is_local_development_database_uri, _pin_eu_environment, _safe_readback
from commerce_agent.observability import EU_LANGSMITH_API_ENDPOINT, project_langsmith_state_outputs
from commerce_agent.debug_privacy import sanitize_debug_payload
from commerce_agent.adapters.privacy import assert_no_direct_identifiers
from commerce_agent.safety import assert_secret_free


def _debug_readback(runs, *, request_id, expected=None):
    """Read-only proof: compare business outputs, never print their raw payloads."""
    roots = [run for run in runs if run.parent_run_id is None]
    if len(roots) != 1:
        return None
    for run in runs:
        payload = {"inputs": run.inputs or {}, "outputs": run.outputs or {},
                   "extra": {k: v for k, v in (run.extra or {}).items() if k != "metadata"},
                   "error": run.error, "events": run.events or [], "tags": run.tags or []}
        if (run.error not in (None, "redacted_error") or run.events or run.tags
            or getattr(run, "attachments", None) or payload["extra"]):
            raise ValueError("unexpected debug side channel")
        assert_secret_free(payload)
        assert_no_direct_identifiers(payload)
        # Only server-owned structural metadata may be added by LangSmith.
        metadata = (run.extra or {}).get("metadata", {})
        if set(metadata) - {"ls_run_depth"}:
            raise ValueError("unexpected debug metadata")
        if request_id in json.dumps(payload, default=str):
            raise ValueError("identity in debug readback")
        if sanitize_debug_payload(payload["inputs"], allow_prepared_messages=run.run_type == "llm") != payload["inputs"]:
            raise ValueError("unsafe debug inputs")
        if sanitize_debug_payload(payload["outputs"]) != payload["outputs"]:
            raise ValueError("unsafe debug outputs")
    root = roots[0]
    if expected is not None and root.outputs != expected:
        return None
    return root


def _private_token(path):
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    with os.fdopen(fd, "r", encoding="ascii") as handle:
        info = os.fstat(handle.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) & 0o077 or not 32 <= info.st_size <= 512):
            raise ValueError("invalid private credential file")
        token = handle.read(513).strip()
    if not re.fullmatch(r"[a-f0-9]{64}", token):
        raise ValueError("invalid private credential")
    return token


class DiagnosticSalesModel:
    """Local-only rejection taxonomy; never retain a provider response/error body."""

    def __init__(self, model):
        self.model = model
        self.diagnostic = {}

    async def analyze(self, *, instructions, prepared):
        from commerce_agent.sales.publication import validate_sales_publication, SalesNarrativePolicyError
        from commerce_agent.adapters.privacy import PrivacyError
        from commerce_agent.safety import SensitiveStateError
        from openai import APIStatusError
        from pydantic import ValidationError

        stage = "model"
        try:
            candidate = await self.model.analyze(instructions=instructions, prepared=prepared)
            stage = "publication"
            return validate_sales_publication(candidate, prepared)
        except Exception as exc:
            # Exact application strings map to static labels. Unknown/provider
            # messages, values, request IDs and response bodies are suppressed.
            reasons = {
                "Sales model response did not complete safely": "unsafe_completion",
                "Sales model returned disallowed response content": "disallowed_content",
                "Sales grounded summary cannot contain provider prose": "mixed_grounded_summary",
                "Sales grounded observation cannot contain provider prose": "mixed_grounded_observation",
                "Sales native proposition evidence is missing": "missing_proposition_evidence",
                "Sales legacy guardrail failed": "legacy_guardrail",
                "Sales authoritative table changed": "changed_table",
                "Sales authoritative facts changed": "changed_facts",
                "Sales finding has unknown provenance": "unknown_provenance",
                "Sales quantitative prose needs a bound native proposition": "unbound_quantity",
                "Sales directional prose needs a bound native comparison": "unbound_direction",
                "Sales causal assertion has no verified causal evidence": "unverified_cause",
                "Sales foreign-domain statement requires a separate verified check": "foreign_domain_prose",
                "Sales finding has no verified foreign-domain evidence": "foreign_domain_finding",
                "Sales native proposition cannot mix with provider context": "mixed_claim_context",
                "Sales native proposition is not a causal/action/impact proof": "wrong_claim_role",
                "Sales claim has no native fact": "unknown_claim_fact",
                "Sales claim lacks metric/scope/unit/period/provenance": "unbound_claim_fact",
                "Sales claim has an invalid period": "invalid_claim_period",
                "Sales comparison mixes metrics/scopes/units/periods/provenance": "mixed_claim_scope",
                "Sales comparison needs a previous fact": "missing_previous_fact",
                "Sales amount must reference only one fact": "mixed_amount_fact",
                "Sales payload exceeds budget": "payload_budget",
                "Sales finding label/classification has no matching native proof": "unbound_finding_semantics",
                "Sales finding has no verified causal classification": "unverified_classification",
                "Sales action is not a typed requested check": "unbound_requested_action",
                "Sales recommendation is not a typed requested check": "unbound_requested_recommendation",
                "Sales finding check scope differs from its observation": "mismatched_check_scope",
                "Sales check has no native scope": "unknown_check_scope",
                "Sales qualitative observation has no sourced signal": "unbound_signal",
                "Sales finding impact has no verified business proof": "unbound_impact",
                "Sales summary has no sourced proposition": "unbound_summary",
                "Sales gap dimension is not a closed label": "unbound_gap_dimension",
                "Sales delegated request is not a typed check": "unbound_delegated_check",
                "Sales delegated request scope has no native proof": "unbound_delegated_scope",
            }
            self.diagnostic = {"stage": stage, "reason": reasons.get(str(exc), "suppressed")}
            if isinstance(exc, SensitiveStateError):
                self.diagnostic["reason"] = "sensitive_content"
            elif isinstance(exc, PrivacyError):
                self.diagnostic["reason"] = "private_content"
            elif isinstance(exc, ValidationError):
                self.diagnostic["reason"] = "invalid_schema"
            elif isinstance(exc, TimeoutError):
                self.diagnostic["reason"] = "timeout"
            if isinstance(exc, APIStatusError):
                self.diagnostic["http_status"] = exc.status_code
            if isinstance(exc, SalesNarrativePolicyError) and exc.role in {
                "summary", "recommendation", "factor", "observation", "impact", "cause", "action"}:
                self.diagnostic["field"] = exc.role
            raise


def _validate_analysis_budget(analysis):
    from commerce_agent.sales.policy import MAX_RESULT_BYTES, assert_bounded_safe
    assert_bounded_safe(analysis.model_dump(mode="json"), max_bytes=MAX_RESULT_BYTES)


async def _run_sales(args, token, profile):
    from langchain_core.messages import HumanMessage
    from commerce_agent.adapters.business_tools import BusinessToolClient, BusinessToolsSettings
    from commerce_agent.context import CommerceRuntimeContext
    from commerce_agent.graph import build_graph
    from commerce_agent.ingress import GuardedGraph
    from commerce_agent.persistence import PersistenceSettings, open_postgres_checkpointer
    from commerce_agent.sales.model import LangChainSalesModel, SalesModelSettings
    from commerce_agent.sales.runtime import SalesDependencies

    settings = BusinessToolsSettings(api_url="http://127.0.0.1:8000", service_token=token)
    context = CommerceRuntimeContext(tenant_id="ai-ecom-development", user_id=str(args.actor_id),
        roles=("viewer",), request_id=str(uuid4()), model_profile=profile["profile"])
    model = DiagnosticSalesModel(LangChainSalesModel(SalesModelSettings(
        api_key=os.environ.get("OPENAI_API_KEY", ""), model=profile["model"], profile=profile["profile"])))
    dependencies = SalesDependencies(client_factory=lambda: BusinessToolClient(settings), model=model,
        model_profile=profile["profile"], workflow_version=os.environ.get("AGENT_WORKFLOW_VERSION", "v1"))
    trace_id, thread_id = uuid4(), str(uuid4())
    config = {"run_id": trace_id, "configurable": {"thread_id": thread_id}}
    async with open_postgres_checkpointer(PersistenceSettings.from_environment()) as checkpointer:
        graph = GuardedGraph(build_graph(checkpointer=checkpointer, sales_dependencies=dependencies))
        started = time.monotonic()
        result = await graph.ainvoke({"schema_version": 1,
            "messages": [HumanMessage(content="Отчет продаж")]}, config=config, context=context)
        snapshot = await graph.aget_state(config)
        history = [entry async for entry in graph.aget_state_history(config)]
        assert not snapshot.next and snapshot.values["sales_analysis"] == result["sales_analysis"]
        assert result["execution_receipts"] == [] and result["action_tasks"] == []
        # This is a full public result, not the compact provider input. Do not
        # silently truncate valid native rows to fit the prepared-input budget.
        _validate_analysis_budget(result["sales_analysis"])
        graph.flush_traces(timeout=20)
        elapsed = time.monotonic() - started
    analysis = result["sales_analysis"]
    published = analysis.result
    if published is not None:
        assert analysis.summary.startswith("# Отчёт Sales")
        assert "## Доказательность и ограничения" in analysis.summary
    report = {"trace_id": str(trace_id), "thread_id": thread_id,
        "schema_version": result["schema_version"], "checkpoint_count": len(history),
        "elapsed_seconds": round(elapsed, 3), "model": profile["model"],
        "state_summary": project_langsmith_state_outputs(result)["state_summary"],
        "sales_status": analysis.status,
        "fact_count": len(published.facts) if published else 0,
        "finding_count": len(published.findings) if published else 0,
        "error_codes": [error.code for error in result["errors"]],
        "summary": analysis.summary,
        "local_diagnostic": model.diagnostic}
    if getattr(args, "debug", False):
        report["_expected_debug_outputs"] = sanitize_debug_payload(result)
    return report, context.request_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tools-token-file", type=Path, required=True)
    parser.add_argument("--actor-id", type=int, required=True)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--debug", action="store_true", help="development-only sanitized business export; requires --trace")
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)  # no provider/DB exception bodies in diagnostics
    try:
        if args.debug and not args.trace:
            raise ValueError("debug requires explicit tracing")
        if (args.actor_id < 1 or os.environ.get("COMMERCE_AGENT_ENVIRONMENT") != "development"
            or not _is_local_development_database_uri(os.environ.get("COMMERCE_AGENT_DATABASE_URI", ""), os.environ)):
            raise ValueError("local development checkpoint target required")
        token = _private_token(args.tools_token_file)
        project = os.environ.get("LANGSMITH_PROJECT", "ai-ecom-commerce-agent-development")
        _pin_eu_environment(project, tracing=False)
        os.environ["COMMERCE_AGENT_LANGSMITH_DEBUG"] = "true" if args.debug else "false"
        response = httpx.get("http://127.0.0.1:8000/v1/model-profile", timeout=5, trust_env=False)
        response.raise_for_status()
        profile = response.json()
        if profile.get("provider") != "openai" or profile.get("profile") != "openai_default":
            raise ValueError("unsupported effective backend model profile")
        client = None
        if args.trace:
            client = Client(api_url=EU_LANGSMITH_API_ENDPOINT, api_key=os.environ.get("LANGSMITH_API_KEY"),
                workspace_id=os.environ.get("LANGSMITH_WORKSPACE_ID") or None,
                hide_inputs=True, hide_outputs=True, hide_metadata=True)
            next(client.list_projects(limit=1), None)
            _pin_eu_environment(project, tracing=True)
        report, request_id = asyncio.run(_run_sales(args, token, profile))
        expected_debug = report.pop("_expected_debug_outputs", None)
        if client is not None:
            for _ in range(15):
                runs = list(client.list_runs(project_name=project, trace_id=report["trace_id"]))
                root = (_debug_readback(runs, request_id=request_id, expected=expected_debug) if args.debug
                        else _safe_readback(runs, request_id=request_id)) if runs else None
                if root is not None and (args.debug or (root.outputs or {}).get("state_summary") == report["state_summary"]):
                    report["langsmith_readback"] = "verified"
                    report["trace_url"] = client.get_run_url(run=root, project_name=project)
                    if args.debug:
                        llm_runs = [run for run in runs if run.run_type == "llm" and run.outputs]
                        tool_runs = [run for run in runs if run.name.startswith("tool.") and run.outputs]
                        if not llm_runs or not tool_runs:
                            raise ValueError("missing detailed LLM/tool export")
                        report["debug_export"] = {"llm_run_count": len(llm_runs), "tool_run_count": len(tool_runs),
                                                  "checkpoint_outputs_match": True}
                    break
                time.sleep(1)
            else:
                report["langsmith_readback"] = "not_verified"
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 0 if report["fact_count"] and report["sales_status"] in {"partial", "completed"} and (
            client is None or report.get("langsmith_readback") == "verified") else 1
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__,
                          "detail": "raw provider/network/credential details suppressed"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
