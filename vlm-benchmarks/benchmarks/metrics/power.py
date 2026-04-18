"""
Power summary over a tegrastats sample stream.

Calculates:
- Mean / peak power for each rail (VDD_IN is the one to care about).
- Total energy in joules (trapezoidal integration over time).
- Energy per token (fed by the harness, which knows total_tokens).
"""

from statistics import mean
from typing import Any, Dict, List


RAILS = ("vdd_in", "vdd_cpu_gpu_cv", "vdd_soc")


def _trapezoid_j(samples: List[Dict[str, Any]], field: str) -> float:
    """Integrate instantaneous power (mW) over time (s) → joules."""
    pts = [(s["t"], s[field]) for s in samples if field in s and "t" in s]
    if len(pts) < 2:
        return 0.0
    energy_mj_s = 0.0
    for (t0, p0), (t1, p1) in zip(pts, pts[1:]):
        dt = t1 - t0
        if dt <= 0 or dt > 5:  # guard against pauses
            continue
        energy_mj_s += 0.5 * (p0 + p1) * dt
    return energy_mj_s / 1000.0  # mW·s → J


def summarize_power(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not samples:
        return {}

    out: Dict[str, Any] = {}
    for rail in RAILS:
        field = f"{rail}_mw"
        vals = [s[field] for s in samples if field in s]
        if not vals:
            continue
        out[rail] = {
            "mean_mw": round(mean(vals), 1),
            "peak_mw": max(vals),
            "energy_j": round(_trapezoid_j(samples, field), 3),
        }
    return out


def energy_per_token(power_summary: Dict[str, Any], total_tokens: int,
                     rail: str = "vdd_in") -> float:
    """Joules per generated token on the given rail."""
    if not power_summary or total_tokens <= 0 or rail not in power_summary:
        return 0.0
    return round(power_summary[rail]["energy_j"] / total_tokens, 4)
