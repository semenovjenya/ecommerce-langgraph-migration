"""Private Marketing v6 contract, exact read identity and replay validation."""
from copy import deepcopy
from hashlib import sha256
import re
from typing import Literal
from pydantic import Field
from commerce_agent.marketing import compact, feedback
from commerce_agent.marketing.policy import assert_bounded_safe
from commerce_agent.marketing.research_evidence import observation_index

VERSION = 6
Operation = Literal["ad-history", "stock-history", "exact-search", "sales", "supply", "price"]


class RequestedCheck(compact.Closed):
    tool: Operation
    purpose: str = Field(min_length=8, max_length=600)


class Observation(compact.Closed):
    observation_ref: str
    native_comparison_ref: str | None
    asserted_relation: Literal["higher", "lower", "unchanged", "mixed", "not_comparable"]


class Hypothesis(compact.PriceHypothesis):
    expected_observation: str = Field(min_length=8, max_length=800)
    observations: list[Observation] = Field(max_length=12)
    requested_checks: list[RequestedCheck] = Field(max_length=5)


class Decision(compact.Closed):
    hypotheses: list[Hypothesis] = Field(min_length=1, max_length=4)
    retired_hypotheses: list[compact.Retired] = Field(max_length=4)
    finished: bool


def owner_scope(context):
    return sha256(compact.canonical([context.tenant_id, context.user_id]).encode()).hexdigest()


def new_journal(context, *, wire_version=1):
    if type(wire_version) is not int or wire_version not in (1, 2, 3):
        raise ValueError("research_wire_version")
    return dict(version=VERSION, projection_version=1, wire_version=wire_version,
        owner=owner_scope(context), events=[])


def assert_owner(journal, context):
    if journal["owner"] != owner_scope(context):
        raise ValueError("research_owner_mismatch")


def read_key(operation, prepared, owner):
    """All v6 operations have fixed typed arguments; no generic SQL/options."""
    return sha256(compact.canonical(dict(owner=owner, operation=operation, source_version=1,
        targets=prepared["request"]["targets"], period=prepared["period"],
        previous_period=prepared["previous_period"], namespace="vendor_code")).encode()).hexdigest()


def previous_decision(journal):
    return next((e["decision"] for e in reversed(journal["events"]) if e["status"] == "accepted"), None)


def rounds_used(journal):
    return sum(bool(e.get("plan")) for e in journal["events"])


def normalize(raw, prepared, reads, journal):
    decision = Decision.model_validate(raw)
    assert_bounded_safe(decision.model_dump(mode="json"), max_bytes=65536)
    if prepared["request"]["scope"] != "single_sku" or len(prepared["request"]["targets"]) != 1:
        raise ValueError("research_scope_invalid")
    if len(journal["events"]) >= 3:
        raise ValueError("research_call_budget")
    old_decision = previous_decision(journal)
    if old_decision and old_decision["finished"]:
        raise ValueError("research_already_finished")
    old = {h["id"] for h in (old_decision or {}).get("hypotheses", [])}
    active = {h.id for h in decision.hypotheses}
    retired = {h.id for h in decision.retired_hypotheses}
    all_retired = {h["id"] for e in journal["events"] if e["status"] == "accepted" for h in e["decision"]["retired_hypotheses"]}
    if (len(active) != len(decision.hypotheses) or len(retired) != len(decision.retired_hypotheses)
            or active & retired or old-active != retired or active & all_retired):
        raise ValueError("research_lifecycle_invalid")
    index = observation_index(prepared, reads)
    existing = {r["read_key"]: r for r in reads}
    bindings, plan = {}, {}
    prior_bindings = {}
    for event in journal["events"]:
        if event["status"] == "accepted":
            for key, identities in event["bindings"].items():
                prior_bindings.setdefault(key, set()).update(identities)
    for h in decision.hypotheses:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,39}", h.id):
            raise ValueError("research_hypothesis_id_invalid")
        prose = [h.explanation, h.observation, h.revision, h.limitation, h.expected_observation]
        if any(re.search(r"\d", text) for text in prose):
            raise ValueError("research_prose_contains_numbers")
        if len(set(h.evidence)) != len(h.evidence) or not set(h.evidence) <= {"baseline", "context", *[f"r{i}" for i in range(len(reads))]}:
            raise ValueError("research_evidence_invalid")
        for c in h.requested_checks:
            if c.tool not in compact.available_tools(prepared):
                raise ValueError("research_tool_disabled")
            key = read_key(c.tool, prepared, journal["owner"])
            bindings.setdefault(key, [])
            if h.id not in bindings[key]:
                bindings[key].append(h.id)
            if key not in existing:
                if key not in plan:
                    plan[key] = dict(operation=c.tool, target=prepared["request"]["targets"][0],
                        hypothesis_id=h.id, purpose=c.purpose, read_key=key, hypothesis_ids=[], purposes={})
                plan[key]["hypothesis_ids"] = bindings[key]
                plan[key]["purposes"][h.id] = c.purpose
        usable = []
        for o in h.observations:
            item = index.get(o.observation_ref)
            if item is None or item["receipt"] not in h.evidence:
                raise ValueError("research_observation_unbound")
            if o.native_comparison_ref != item["native_comparison_ref"] or o.asserted_relation != item["relation"]:
                raise ValueError("research_comparison_mismatch")
            if journal["wire_version"] != 3 and item["layer"] != h.price_focus:
                raise ValueError("research_layer_mismatch")
            if item["usable"] and item["receipt"].startswith("r"):
                read = reads[int(item["receipt"][1:])]
                # Receipt can be reused by a new hypothesis only by explicitly
                # selecting that already completed typed check (zero extra reads).
                related = (h.id in read["hypothesis_ids"] or h.id in bindings.get(read["read_key"], [])
                    or h.id in prior_bindings.get(read["read_key"], set()))
                if related:
                    usable.append(item)
        if journal["wire_version"] == 3:
            layers = {index[o.observation_ref]["layer"] for o in h.observations}
            if h.price_focus != (next(iter(layers)) if len(layers) == 1 else "none"):
                raise ValueError("research_layer_mismatch")
        if h.assessment in {"strengthened", "weakened"} and not usable:
            raise ValueError("research_support_without_observation")
        if not reads and not decision.finished and h.assessment != "untested":
            raise ValueError("research_initial_assessment")
        if decision.finished and h.assessment == "untested":
            raise ValueError("research_final_unassessed")
    if any(re.search(r"\d", h.reason) for h in decision.retired_hypotheses):
        raise ValueError("research_prose_contains_numbers")
    if len(existing)+len(plan) > 5:
        raise ValueError("research_read_budget")
    if plan and (rounds_used(journal) >= 2 or len(journal["events"]) >= 2):
        raise ValueError("research_round_budget")
    if decision.finished != (not plan):
        raise ValueError("research_completion_invalid")
    return dict(status="accepted", reads_before=len(reads), decision=decision.model_dump(mode="json"),
        plan=list(plan.values()), bindings=bindings)


def validate_journal(prepared, reads, journal):
    from commerce_agent.marketing.historical_sources import receipt_byte_limit
    assert_bounded_safe(reads, max_bytes=receipt_byte_limit(prepared))
    allowed = {"operation", "target", "hypothesis_id", "purpose", "read_key", "hypothesis_ids", "purposes", "arguments", "status", "envelope", "reason"}
    if any(not isinstance(read, dict) or set(read)-allowed for read in reads):
        raise ValueError("research_receipt_fields_invalid")
    if (set(journal) != {"version", "projection_version", "wire_version", "owner", "events"}
            or any(type(journal[k]) is not int for k in ("version", "projection_version", "wire_version")) or journal["version"] != 6 or journal["projection_version"] != 1 or journal["wire_version"] not in (1, 2, 3)
            or not re.fullmatch(r"[a-f0-9]{64}", journal["owner"]) or len(journal["events"]) > 3 or len(reads) > 5):
        raise ValueError("research_journal_version_or_budget")
    replay = {**journal, "events": []}
    cursor = 0
    for position, event in enumerate(journal["events"]):
        prior = previous_decision(replay)
        if prior and prior["finished"] or replay["events"] and replay["events"][-1].get("error") in TERMINAL_ERRORS:
            raise ValueError("research_event_after_terminal")
        if event["reads_before"] != cursor:
            raise ValueError("research_journal_read_boundary")
        if event["status"] == "rejected":
            fields = {"status", "reads_before", "error"}
            if journal["wire_version"] == 3:
                from commerce_agent.marketing.research_owned import validate_issues, safe_draft
                fields.update(("issues", "draft"))
                validate_issues(event.get("issues"))
                if event.get("draft") is not None and safe_draft(event["draft"], replay) != event["draft"]:
                    raise ValueError("research_journal_draft_invalid")
            elif journal["wire_version"] == 2:
                from commerce_agent.marketing.research_phases import validate_issues
                fields.add("issues")
                validate_issues(event.get("issues"))
            if set(event) != fields or event["error"] not in REJECTION_CODES:
                raise ValueError("research_journal_error_invalid")
        else:
            if journal["wire_version"] == 3:
                from commerce_agent.marketing.research_owned import normalize_wire
                expected = normalize_wire(event["wire_response"], prepared, reads[:cursor], replay)
            elif journal["wire_version"] == 2:
                from commerce_agent.marketing.research_phases import normalize_wire
                expected = normalize_wire(event["wire_response"], prepared, reads[:cursor], replay)
            else:
                expected = normalize(event["decision"], prepared, reads[:cursor], replay)
            if compact.canonical(expected) != compact.canonical(event):
                raise ValueError("research_journal_plan_changed")
            for check in event["plan"]:
                if cursor == len(reads) and position == len(journal["events"])-1:
                    break  # saved boundary before pending read
                if cursor >= len(reads) or any(reads[cursor].get(k) != v for k, v in check.items()):
                    raise ValueError("research_journal_receipt_changed")
                feedback.validate_read(reads[cursor], prepared, details=True)
                cursor += 1
        replay["events"].append(event)
    if cursor != len(reads):
        raise ValueError("research_unbound_reads")
    assert_bounded_safe(journal, max_bytes=262144)


TERMINAL_ERRORS = {"provider_failed", "provider_timeout", "response_not_completed", "response_refused_or_tool"}

REJECTION_CODES = {"model_invalid", "provider_failed", "provider_timeout", "response_not_completed",
    "response_blocks_invalid", "response_refused_or_tool", "final_answer_missing", "duplicate_json_key",
    "nonfinite_json", "final_json_invalid", "final_json_not_object", "ambiguous_final"}


def pending(journal, reads):
    if not journal["events"] or journal["events"][-1]["status"] != "accepted":
        return []
    done = {r["read_key"] for r in reads}
    return [c for c in journal["events"][-1]["plan"] if c["read_key"] not in done]


def contains_research(value):
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="python")
    if isinstance(value, dict):
        return value.get("marketing_investigation_version") == 6 or "marketing_investigation_v6" in value or any(contains_research(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(contains_research(v) for v in value)
    return isinstance(value, str) and value.startswith("marketing-investigation-v6:")
