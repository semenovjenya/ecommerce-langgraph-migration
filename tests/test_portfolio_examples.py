"""Public synthetic checks independent of the private seller regression corpus."""
from copy import deepcopy

import pytest

from benchmarks.evidence_payload import datasets, measure
from examples.checkpoint_demo import run_demo
from commerce_agent.marketing.compact import canonical, pack, unpack


@pytest.mark.parametrize("case", list(datasets()))
def test_lossless_payload_preserves_original_and_types(case):
    original = datasets()[case]
    untouched = deepcopy(original)
    restored = unpack(pack(original))
    assert canonical(restored) == canonical(original)
    assert original == untouched


def test_null_zero_false_and_missing_are_distinct():
    data = [{"value": None}, {"value": 0}, {"value": False}, {"value": "0"}, {"missing": True}]
    assert canonical(unpack(pack(data))) == canonical(data)


def test_benchmark_does_not_claim_heterogeneous_control_savings():
    results = {row["case"]: row for row in measure()["cases"]}
    assert results["heterogeneous_control"]["reduction_percent"] == 0
    assert results["repeated_daily_rows"]["compact_bytes"] < results["repeated_daily_rows"]["original_bytes"]


def test_demo_resumes_after_real_checkpoint_without_external_writes():
    result = run_demo()
    assert result["paused_before"] == ["synthesis"]
    assert result["resumed"] is True
    assert result["external_writes"] == 0
    assert set(result["domain_statuses"]) == {"sales", "marketing", "supply", "price", "finance"}
