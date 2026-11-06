"""Deterministic byte benchmark of the actual compact evidence implementation.

This measures JSON payload bytes, not token billing, latency or model quality.
Run from the repository root: uv run python -m benchmarks.evidence_payload
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from commerce_agent.marketing.compact import canonical, pack, unpack


def datasets() -> dict:
    return {
        "repeated_daily_rows": [{
            "source": "synthetic_daily_sales", "currency": "RUB", "sku": f"DEMO-{i % 10:03}",
            "period": f"2026-01-{i % 28 + 1:02}", "orders": i % 17,
            "revenue": (i % 17) * 100, "margin": None if i % 7 == 0 else 0,
            "is_complete": True,
        } for i in range(1000)],
        "nested_evidence": {"rows": [{
            "source": "synthetic_campaign", "sku": f"DEMO-{i % 20:03}",
            "metrics": {"impressions": i * 10, "clicks": i % 11, "orders": i % 3},
            "scope": {"currency": "RUB", "attribution": "observational"},
            "warnings": [],
        } for i in range(300)], "limitations": ["Synthetic data; no causal conclusion"]},
        "heterogeneous_control": [{"a": 1}, {"b": 0}, {"a": None}, False, "0"],
    }


def measure() -> dict:
    results = []
    for name, original in datasets().items():
        packed = pack(original)
        restored = unpack(packed)
        assert canonical(restored) == canonical(original), name
        before = len(canonical(original).encode("utf-8"))
        after = len(canonical(packed).encode("utf-8"))
        results.append({"case": name, "original_bytes": before, "compact_bytes": after,
                        "reduction_percent": round(100 * (before - after) / before, 2),
                        "roundtrip_exact": True})
    return {"measurement": "UTF-8 bytes of canonical JSON", "data": "deterministic synthetic fixtures",
            "scope": "evidence section only; not full prompt, token cost, latency or analytical quality",
            "cases": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rendered = json.dumps(measure(), indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
