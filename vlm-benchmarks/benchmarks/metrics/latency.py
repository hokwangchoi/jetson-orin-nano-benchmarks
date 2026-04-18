"""Latency aggregation helpers.

Keeps percentiles around so we can report median/p95 not just means —
edge inference has long tails when the OS preempts the GPU.
"""

from statistics import mean, median
from typing import Any, Dict, List


def _percentile(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    idx = min(int(round(p / 100 * (len(sorted_vals) - 1))), len(sorted_vals) - 1)
    return sorted_vals[idx]


def _summarize_field(lats: List[Dict], field: str) -> Dict[str, float]:
    vals = sorted(x[field] for x in lats if x.get(field) is not None)
    if not vals:
        return {"mean": 0, "median": 0, "p95": 0, "min": 0, "max": 0}
    return {
        "mean":   round(mean(vals), 3),
        "median": round(median(vals), 3),
        "p95":    round(_percentile(vals, 95), 3),
        "min":    round(vals[0], 3),
        "max":    round(vals[-1], 3),
    }


def summarize_latencies(latencies: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not latencies:
        return {}
    return {
        "n":       len(latencies),
        "ttft_ms": _summarize_field(latencies, "ttft_ms"),
        "tpot_ms": _summarize_field(latencies, "tpot_ms"),
        "tps":     _summarize_field(latencies, "tps"),
        "e2e_ms":  _summarize_field(latencies, "e2e_ms"),
        "total_tokens": sum(x.get("n_tokens", 0) for x in latencies),
    }
