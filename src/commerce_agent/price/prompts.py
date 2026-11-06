"""Effective legacy Price components plus explicit native read-only boundary."""
from datetime import date
from importlib.resources import files
from typing import Literal

from commerce_agent.adapters.privacy import assert_no_direct_identifiers
from commerce_agent.safety import assert_secret_free
from commerce_agent.sales.contracts import QueryMode, RoundType
from commerce_agent.sales.prompt_contracts import determine_query_mode, domain_base_contract, round_contract
from commerce_agent.sales.prompt_rules import COMMON_DOMAIN_WORKFLOW_RULES, DERIVED_FEATURE_PROTOCOL


MIGRATION_OVERRIDE = """M3 Price execution boundary (overrides legacy tool instructions):
- Read-only tools belong to collect_evidence; analyze cannot call tools, SQL, other agents or writes.
- Canonical seller/display/wallet/effective/club price layers and their independent native dates never merge.
- Keep customer_price_effective and customer_price_basis as native outputs; do not recompute min or price formulas.
- Broad/raw medians and statuses cannot override strict trusted canonical competitor positioning.
- Native decision prices and expected orders/profit/margin are scenarios, not achieved effects or final approval.
- Relative decision_freshness is not proof of current SKU freshness; old/future/missing source dates remain limitations.
- Missing COGS is not zero or a certified floor. Low-confidence/stale scenarios may be displayed, not executed.
- Daily price/sales association is noncausal; observed transaction averages are not exact-day seller snapshots.
- Latest-only readers cannot satisfy historical requests. Keep full returned matrices and warnings.
- Only verified closed observations and scoped SKU/nm_id/period neighbor requests; no invented causes or numbers.
- No credentials or personal records.
""".strip()


def build_price_instructions(
    *, workflow_version: Literal["v1", "v2_prompt", "v2_optimized"],
    user_query: str, today: str, memory_block: str = "",
) -> str:
    """Baseline instruction components, not full runner/query-plan/memory parity."""
    if workflow_version not in {"v1", "v2_prompt", "v2_optimized"}:
        raise ValueError("unsupported Price workflow version")
    if not isinstance(today, str) or len(today) != 10 or date.fromisoformat(today).isoformat() != today:
        raise ValueError("Price prompt requires an ISO calendar date")
    context = {"user_query": user_query, "today": today, "memory": memory_block}
    assert_secret_free(context)
    assert_no_direct_identifiers(context)
    if not isinstance(user_query, str) or not isinstance(memory_block, str) or len(user_query) > 8000 or len(memory_block) > 8000:
        raise ValueError("Price prompt context exceeds budget")
    mode = determine_query_mode(user_query)
    if workflow_version == "v1":
        asset = files("commerce_agent").joinpath("prompts/price_agent.md").read_text(encoding="utf-8").strip()
        parts = [COMMON_DOMAIN_WORKFLOW_RULES, asset, DERIVED_FEATURE_PROTOCOL]
    else:
        parts = [domain_base_contract("price"), round_contract(RoundType.PRIMARY)]
        if mode == QueryMode.DIAGNOSTIC:
            parts.append(DERIVED_FEATURE_PROTOCOL)
    parts.append(f"Runtime context:\n- today: {today}\n- intent: price\n"
                 f"- round_type: primary\n- query_mode: {mode.value}\n- user_query: {user_query}")
    if memory_block:
        parts.append(memory_block)
    parts.append(MIGRATION_OVERRIDE)
    return "\n\n".join(parts)
