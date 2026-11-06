"""Legacy Supply instruction components plus explicit read-only boundary."""
from datetime import date
from importlib.resources import files
from typing import Literal

from commerce_agent.adapters.privacy import assert_no_direct_identifiers
from commerce_agent.safety import assert_secret_free
from commerce_agent.sales.contracts import QueryMode, RoundType
from commerce_agent.sales.prompt_contracts import determine_query_mode, domain_base_contract, round_contract
from commerce_agent.sales.prompt_rules import COMMON_DOMAIN_WORKFLOW_RULES, DERIVED_FEATURE_PROTOCOL


MIGRATION_OVERRIDE = """M3 Supply execution boundary (overrides legacy tool instructions):
- Read-only tools belong to collect_evidence; analyze cannot call tools or SQL.
- supply_run_sql, generic SQL, direct agent delegation and external writes are unavailable.
- Use only prepared evidence; preserve service-computed stock coverage, turnover, valuation and source periods.
- Missing stock is not zero; simulation is not observed freshness; exact turnover is not a proxy.
- Never calculate financial or operational metrics or invent values. Preserve all native SKU matrix rows.
- Proposed cross-agent checks require a verified SKU and period/snapshot, not claimed causes.
- Do not place orders, change supplies, or include credentials/personal identifiers.
""".strip()


def build_supply_instructions(
    *, workflow_version: Literal["v1", "v2_prompt", "v2_optimized"],
    user_query: str, today: str, memory_block: str = "",
) -> str:
    """Baseline only: no model, tool, memory-service or graph invocation."""
    if workflow_version not in {"v1", "v2_prompt", "v2_optimized"}:
        raise ValueError("unsupported Supply workflow version")
    if not isinstance(today, str) or len(today) != 10 or date.fromisoformat(today).isoformat() != today:
        raise ValueError("Supply prompt requires an ISO calendar date")
    context = {"user_query": user_query, "today": today, "memory": memory_block}
    assert_secret_free(context)
    assert_no_direct_identifiers(context)
    if len(user_query) > 8000 or len(memory_block) > 8000:
        raise ValueError("Supply prompt context exceeds budget")
    mode = determine_query_mode(user_query)
    if workflow_version == "v1":
        asset = files("commerce_agent").joinpath("prompts/supply_agent.md").read_text(encoding="utf-8").strip()
        parts = [COMMON_DOMAIN_WORKFLOW_RULES, asset, DERIVED_FEATURE_PROTOCOL]
    else:
        parts = [domain_base_contract("supply"), round_contract(RoundType.PRIMARY)]
        if mode == QueryMode.DIAGNOSTIC:
            parts.append(DERIVED_FEATURE_PROTOCOL)
    parts.append(f"Runtime context:\n- today: {today}\n- intent: supply\n"
                 f"- round_type: primary\n- query_mode: {mode.value}\n- user_query: {user_query}")
    if memory_block:
        parts.append(memory_block)
    parts.append(MIGRATION_OVERRIDE)
    return "\n\n".join(parts)
