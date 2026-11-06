"""Evidence-bound presentation, not financial calculations or provider prose.

Models select fact indices; native values, units, scope and periods are rendered
locally. Only comparisons of adjacent, equally sized native periods are allowed.
"""
from datetime import date, timedelta
from decimal import Decimal
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from commerce_agent.sales.analytics import analytical_claims


class SalesClaimSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["amount", "comparison", "analytic"]
    fact_index: int = Field(ge=0, le=2000, strict=True)
    previous_fact_index: int | None = Field(default=None, ge=0, le=2000, strict=True)


_METRICS = {
    "current_revenue": ("Текущая выручка", "RUB"), "previous_revenue": ("Предыдущая выручка", "RUB"),
    "current_sales": ("Текущие продажи", "units"), "previous_sales": ("Предыдущие продажи", "units"),
}


def _native_fact(prepared, index):
    if not 0 <= index < len(prepared["facts"]):
        raise ValueError("Sales claim has no native fact")
    fact = prepared["facts"][index]
    if (fact["metric"] not in _METRICS
        or fact.get("status", "confirmed") != "confirmed"
        or fact.get("unit") != _METRICS[fact["metric"]][1]
        or not isinstance(fact.get("value"), (int, float))
        or isinstance(fact["value"], bool) or not math.isfinite(fact["value"])
        or not all(fact.get(key) for key in ("sku", "unit", "period", "source_table", "evidence_ref"))
        or fact["evidence_ref"] not in {item["evidence_ref"] for item in prepared["evidence"]}):
        raise ValueError("Sales claim lacks metric/scope/unit/period/provenance")
    start, end = (date.fromisoformat(value) for value in fact["period"].split("/"))
    if end < start:
        raise ValueError("Sales claim has an invalid period")
    return fact, start, end


def render_sales_claim(selection: SalesClaimSelection, prepared: dict) -> str:
    """Exact text proves one native proposition; no scalar matching/LLM math."""
    if selection.kind == "analytic":
        claims = analytical_claims(prepared)
        if selection.previous_fact_index is not None or not 0 <= selection.fact_index < len(claims):
            raise ValueError("Sales analytic claim has no native proof")
        return claims[selection.fact_index]["text"]
    fact, start, end = _native_fact(prepared, selection.fact_index)
    label = _METRICS[fact["metric"]][0]
    if selection.kind == "amount":
        if selection.previous_fact_index is not None:
            raise ValueError("Sales amount must reference only one fact")
        return (f"{label} SKU {fact['sku']} за период {fact['period']}: "
                f"{fact['value']} {fact['unit']}.")
    if selection.previous_fact_index is None:
        raise ValueError("Sales comparison needs a previous fact")
    previous, previous_start, previous_end = _native_fact(prepared, selection.previous_fact_index)
    if (fact["metric"] not in {"current_revenue", "current_sales"}
        or previous["metric"] != fact["metric"].replace("current_", "previous_", 1)
        or any(fact[key] != previous[key] for key in ("sku", "unit", "source_table", "evidence_ref"))
        or previous_end + timedelta(days=1) != start
        or end - start != previous_end - previous_start):
        raise ValueError("Sales comparison mixes metrics/scopes/units/periods/provenance")
    # Presentation comparison only: no delta, rate, totals or business formula.
    current_value, previous_value = Decimal(str(fact["value"])), Decimal(str(previous["value"]))
    units = fact["metric"] == "current_sales"
    direction = "выше" if current_value > previous_value else "ниже" if current_value < previous_value else "равны" if units else "равна"
    before = "предыдущих" if units else "предыдущей"
    return (f"{label} SKU {fact['sku']} за период {fact['period']} {direction} "
            f"{before} за период {previous['period']}.")


def native_claim_proofs(prepared: dict) -> dict[str, set[str]]:
    """Rebuild proofs from authoritative evidence at each publication boundary."""
    return _native_claim_proofs(prepared)


def _native_claim_proofs(prepared: dict, analytic=None) -> dict[str, set[str]]:
    texts = {}
    pairs = {}
    for index, fact in enumerate(prepared["facts"]):
        try:
            text = render_sales_claim(SalesClaimSelection(kind="amount", fact_index=index), prepared)
            texts.setdefault(text, set()).add(fact["evidence_ref"])
        except ValueError:
            continue
        key = tuple(fact.get(field) for field in ("sku", "unit", "source_table", "evidence_ref"))
        pairs.setdefault(key, {}).setdefault(fact["metric"], []).append(index)
    for group in pairs.values():
        metric = "current_revenue" if group.get("current_revenue") else "current_sales"
        current, previous = group.get(metric, []), group.get(metric.replace("current_", "previous_"), [])
        # Ambiguous duplicate facts/periods do not produce an implicit comparison.
        if len(current) != 1 or len(previous) != 1:
            continue
        selection = SalesClaimSelection(kind="comparison", fact_index=current[0], previous_fact_index=previous[0])
        try:
            text = render_sales_claim(selection, prepared)
        except ValueError:
            continue
        texts.setdefault(text, set()).add(prepared["facts"][current[0]]["evidence_ref"])
        fact, before = prepared["facts"][current[0]], prepared["facts"][previous[0]]
        units = metric == "current_sales"
        direction = "выше" if fact["value"] > before["value"] else "ниже" if fact["value"] < before["value"] else "равны" if units else "равна"
        previous_label = "предыдущих" if units else "предыдущей"
        alias = f"{_METRICS[metric][0]} SKU {fact['sku']} {direction} {previous_label} за сопоставимые периоды."
        texts.setdefault(alias, set()).add(fact["evidence_ref"])
    for claim in analytical_claims(prepared) if analytic is None else analytic:
        texts.setdefault(claim["text"], set()).update(claim["evidence_refs"])
    return texts


def native_claim_sentences(prepared: dict) -> set[str]:
    return set(native_claim_proofs(prepared))


def native_claim_catalog(prepared: dict) -> list[SalesClaimSelection]:
    """Enumerate valid native propositions before asking a model to select.

    Relational constraints (same SKU/unit/period/source) cannot be expressed by
    independent index bounds. The provider never assembles an arbitrary pair.
    """
    return _native_claim_catalog(prepared)


def _native_claim_catalog(prepared: dict, analytic=None) -> list[SalesClaimSelection]:
    claims, groups = [], {}
    for index, fact in enumerate(prepared["facts"]):
        selection = SalesClaimSelection(kind="amount", fact_index=index)
        try:
            render_sales_claim(selection, prepared)
        except ValueError:
            continue
        claims.append(selection)
        key = tuple(fact.get(field) for field in ("sku", "unit", "source_table", "evidence_ref"))
        groups.setdefault(key, {}).setdefault(fact["metric"], []).append(index)
    for group in groups.values():
        for metric in ("current_revenue", "current_sales"):
            current, previous = group.get(metric, []), group.get(metric.replace("current_", "previous_"), [])
            if len(current) != 1 or len(previous) != 1:
                continue
            selection = SalesClaimSelection(kind="comparison", fact_index=current[0], previous_fact_index=previous[0])
            try:
                render_sales_claim(selection, prepared)
            except ValueError:
                continue
            claims.append(selection)
    analytic = analytical_claims(prepared) if analytic is None else analytic
    selections = [SalesClaimSelection(kind="analytic", fact_index=index) for index, _ in enumerate(analytic)]
    # Present meaningful native dynamics before scalar amounts when the
    # business API actually provides portfolio comparisons. Fact indices and
    # public contracts do not change; catalog positions are invocation-local.
    return selections + claims if any(c["sku"] is None for c in analytic) else claims + selections


def claim_scope(selection, prepared):
    if selection.kind == "analytic":
        item = analytical_claims(prepared)[selection.fact_index]
        return item["sku"], item["period"]
    fact = prepared["facts"][selection.fact_index]
    return fact["sku"], fact["period"]


def claim_evidence(selection, prepared):
    if selection.kind == "analytic":
        return analytical_claims(prepared)[selection.fact_index]["evidence_refs"]
    return [prepared["facts"][selection.fact_index]["evidence_ref"]]


class _SalesClaimIndex:
    """A synchronous call-local view, never a persisted validation capability.

    Public entry points always create a new view from the current prepared
    data. In particular it must not survive model callbacks or checkpoints.
    Keep historical proof aliases separate from the ordered model catalog.
    """

    def __init__(self, prepared: dict):
        self.analytic = analytical_claims(prepared)
        self.catalog = tuple(_native_claim_catalog(prepared, self.analytic))
        self.native_proofs = _native_claim_proofs(prepared, self.analytic)
        self.texts = {}
        self.scopes = {}
        self.observations = {}
        for selection in self.catalog:
            if selection.kind == "analytic":
                claim = self.analytic[selection.fact_index]
                text, scope = claim["text"], (claim["sku"], claim["period"])
            else:
                text = render_sales_claim(selection, prepared)
                fact = prepared["facts"][selection.fact_index]
                scope = fact["sku"], fact["period"]
            self.texts[selection] = text
            self.scopes[selection] = scope
            # Equal text can prove multiple selections/scopes: never last-wins.
            self.observations.setdefault(text, []).append(selection)
