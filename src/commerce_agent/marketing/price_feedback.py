"""Bounded Marketing to Price evidence validation."""
from commerce_agent.models import DomainAnalysis
from commerce_agent.price.history import build_catalog, canonical
from commerce_agent.price.publication import validate_price_publication
from commerce_agent.marketing.feedback import check_arguments
from commerce_agent.marketing.policy import assert_bounded_safe


def validate_price_read(read, prepared):
    from commerce_agent.marketing.compact import ExecutionCheck
    if prepared.get("marketing_price_feedback_version") != 1:
        raise ValueError("Price feedback reader disabled")
    check = ExecutionCheck.model_validate({key: read[key] for key in ExecutionCheck.model_fields})
    if check.target != prepared["request"]["targets"][0] or read["arguments"] != check_arguments(check, prepared):
        raise ValueError("Price feedback scope changed")
    if read["status"] not in {"ok", "failed", "timeout", "unavailable"}:
        raise ValueError("Price feedback status invalid")
    if read["status"] != "ok":
        if read.get("envelope") is not None:
            raise ValueError("Failed Price check contains evidence")
        return
    envelope = read["envelope"]
    data = envelope["data"]
    if (envelope["source"] != "price_subgraph" or data["target"] != check.target
            or data["recursive_delegation"] is not False
            or envelope["period"] != {key: read["arguments"][key] for key in ("date_from", "date_to")}):
        raise ValueError("Price neighbor source mismatch")
    native = data["history"]
    if (native["request"]["targets"] != [check.target] or native["today"] != prepared["today"]
            or native["request"]["date_from"] != read["arguments"]["date_from"]
            or native["request"]["date_to"] != read["arguments"]["date_to"]):
        raise ValueError("Price history request mismatch")
    catalog = build_catalog(native)
    ids = {claim.get("nm_id") for claim in prepared["claims"] if claim.get("sku") == check.target and claim.get("nm_id")}
    identity = native["history_envelope"]["data"]
    if identity["identity_status"] == "resolved" and ids != {identity["nm_id"]}:
        raise ValueError("Price history native identity mismatch")
    analysis = DomainAnalysis.model_validate(data["analysis"])
    if analysis.domain.value != "price" or analysis.status not in {"completed", "partial"} or analysis.result is None:
        raise ValueError("Price neighbor failed")
    validate_price_publication(analysis.result, {**native, **catalog})
    if (canonical(data["evidence"]) != canonical(catalog["evidence"])
            or set(analysis.evidence_refs) != {item["evidence_ref"] for item in catalog["evidence"]}):
        raise ValueError("Price neighbor evidence mismatch")
    assert_bounded_safe(read, max_bytes=524288)


def comparable_price_observation(read, focus=None):
    native = read["envelope"]["data"]["history"]
    build_catalog(native)
    return any(item["direction"] != "insufficient_data" and (
        focus is None or focus == ("market_summary" if item["metric"] == "market_price_median" else item["layer"]))
        for item in native["history_envelope"]["data"]["comparisons"])


def contains_price_feedback(value):
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="python")
    if isinstance(value, dict):
        return (value.get("marketing_investigation_version") == 5
            or value.get("marketing_price_feedback_version") is not None
            or any(contains_price_feedback(item) for item in value.values()))
    if isinstance(value, (list, tuple)):
        return any(contains_price_feedback(item) for item in value)
    return isinstance(value, str) and value.startswith(("marketing-investigation-v5:", "marketing-investigation-v6:price:"))
