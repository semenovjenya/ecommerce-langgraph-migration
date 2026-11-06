"""Version-compatible provenance only; native financial values never change."""
import pytest

from commerce_agent.finance.evidence import analysis_policy_version


@pytest.mark.parametrize("value", [True, False, "1", 1.0, 2, None])
def test_missing_side_reader_marker_is_strict(value):
    with pytest.raises(ValueError, match="missing-side provenance"):
        analysis_policy_version({"analysis_policy_version": 2,
            "missing_side_provenance_policy_version": value})


def test_unmarked_historical_policies_remain_readable_without_silent_upgrade():
    old = {"analysis_policy_version": 2}
    assert analysis_policy_version(old) == 2
    assert old == {"analysis_policy_version": 2}
    assert analysis_policy_version({}) == 1
    assert analysis_policy_version({"analysis_policy_version": 2,
        "missing_side_provenance_policy_version": 1}) == 2
    with pytest.raises(ValueError, match="missing-side provenance"):
        analysis_policy_version({"missing_side_provenance_policy_version": 1})
