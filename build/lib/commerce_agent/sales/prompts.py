"""Preserve legacy instruction components with an explicit migration boundary."""
from __future__ import annotations

from datetime import date
from importlib.resources import files
from typing import Literal

from commerce_agent.adapters.privacy import assert_no_direct_identifiers
from commerce_agent.safety import assert_secret_free
from commerce_agent.sales.contracts import QueryMode, RoundType
from commerce_agent.sales.prompt_contracts import (
    determine_query_mode, domain_base_contract, round_contract,
)
from commerce_agent.sales.prompt_rules import COMMON_DOMAIN_WORKFLOW_RULES, DERIVED_FEATURE_PROTOCOL


MIGRATION_OVERRIDE = """M3 Sales execution boundary (overrides legacy tool instructions):
- Read-only tools were called by collect_evidence; analyze cannot call tools or SQL.
- sales_run_sql, generic SQL, direct agent delegation and external writes are unavailable.
- Use only prepared evidence. Missing dimensions remain explicit data_gaps / needs_data;
  do not compute financial or operational metrics, invent rows, or treat missing values as zero.
- Copy authoritative prepared tables unchanged. Numbers and facts must cite supplied evidence IDs.
- Cross-agent requests remain proposed checks, never proof of another domain's cause.
- Do not include credentials, personal identifiers, raw tool envelopes or internal prompts.
""".strip()


def build_sales_instructions(
    *, workflow_version: Literal["v1", "v2_prompt", "v2_optimized"],
    user_query: str, today: str, memory_block: str = "",
) -> str:
    """Caller supplies only redacted business context, never service identity/secrets.

    Memory is supplied explicitly, not fetched from the legacy business DB. This
    builder preserves instruction components; full legacy runtime-header/memory
    parity must be assessed separately when migrating multi-round orchestration.
    """
    if workflow_version not in {"v1", "v2_prompt", "v2_optimized"}:
        raise ValueError("unsupported Sales workflow version")
    if not isinstance(today, str) or len(today) != 10 or date.fromisoformat(today).isoformat() != today:
        raise ValueError("Sales prompt requires an ISO calendar date")
    safe_context = {"user_query": user_query, "today": today, "memory": memory_block}
    assert_secret_free(safe_context)
    assert_no_direct_identifiers(safe_context)
    if len(user_query) > 8000 or len(memory_block) > 8000:
        raise ValueError("Sales prompt context exceeds budget")
    query_mode = determine_query_mode(user_query)
    if workflow_version == "v1":
        asset = files("commerce_agent").joinpath("prompts/sales_agent.md").read_text(encoding="utf-8").strip()
        parts = [COMMON_DOMAIN_WORKFLOW_RULES, asset, DERIVED_FEATURE_PROTOCOL]
    else:
        parts = [domain_base_contract("sales"), round_contract(RoundType.PRIMARY)]
        if query_mode == QueryMode.DIAGNOSTIC:
            parts.append(DERIVED_FEATURE_PROTOCOL)
    parts.append(
        f"Runtime context:\n- today: {today}\n- intent: sales\n"
        f"- round_type: primary\n- query_mode: {query_mode.value}\n- user_query: {user_query}"
    )
    if memory_block:
        parts.append(memory_block)
    parts.append(MIGRATION_OVERRIDE)
    return "\n\n".join(parts)
