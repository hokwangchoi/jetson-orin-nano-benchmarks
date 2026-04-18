"""
tegrastats parser + background recorder.

Example tegrastats line (MAXN_SUPER, interval=1000):
  04-17-2026 10:15:23 RAM 4521/7620MB (lfb 2x4MB) SWAP 0/3810MB (cached 0MB)
  CPU [3%@1510,2%@1510,1%@1510,0%@1510,0%@1510,0%@1510] EMC_FREQ 0%@3199 GR3D_FREQ 0%@[1020]
  cpu@47.5C soc2@48C soc0@48.9C gpu@46.4C tj@49.3C soc1@47.5C
  VDD_IN 4567mW/4567mW VDD_CPU_GPU_CV 989mW/989mW VDD_SOC 1234mW/1234mW

We extract a flat dict per line and buffer them for the harness.
"""

import re
import subprocess
import threading
import time
from statistics import mean, median
from typing import Any, Dict, List, Optional


# Regexes — kept loose enough to survive minor tegrastats format changes
_RE_RAM       = re.compile(r"RAM\s+(\d+)/(\d+)MB")
_RE_SWAP      = re.compile(r"SWAP\s+(\d+)/(\d+)MB")
_RE_CPU       = re.compile(r"CPU\s+\[([^\]]+)\]")
_RE_CPU_CORE  = re.compile(r"(\d+)%@(\d+)")
_RE_GR3D      = re.compile(r"GR3D_FREQ\s+(\d+)%@\[?(\d+)\]?")
_RE_EMC       = re.compile(r"EMC_FREQ\s+(\d+)%@?(\d+)?")
_RE_TEMP      = re.compile(r"(\w+)@([\d.]+)C")
_RE_POWER     = re.compile(r"(VDD_\w+)\s+(\d+)mW/(\d+)mW")


def parse_line(line: str) -> Optional[Dict[str, Any]]:
    if "RAM" not in line:
        return None
    rec: Dict[str, Any] = {"t": time.time(), "raw": line.strip()}

    m = _RE_RAM.search(line)
    if m:
        rec["ram_used_mb"] = int(m.group(1))
        rec["ram_total_mb"] = int(m.group(2))

    m = _RE_SWAP.search(line)
    if m:
        rec["swap_used_mb"] = int(m.group(1))

    m = _RE_CPU.search(line)
    if m:
        cores = _RE_CPU_CORE.findall(m.group(1))
        rec["cpu_pct_per_core"] = [int(u) for u, _ in cores]
        rec["cpu_pct_mean"] = (sum(int(u) for u, _ in cores) / len(cores)) if cores else 0

    m = _RE_GR3D.search(line)
    if m:
        rec["gpu_pct"] = int(m.group(1))
        rec["gpu_mhz"] = int(m.group(2))

    m = _RE_EMC.search(line)
    if m:
        rec["emc_pct"] = int(m.group(1))

    # Temperatures, keep only gpu/cpu/tj
    for name, val in _RE_TEMP.findall(line):
        key = name.lower()
        if key in ("gpu", "cpu", "tj"):
            rec[f"temp_{key}_c"] = float(val)

    # Power rails — record both instantaneous and running-avg
    for rail, inst, avg in _RE_POWER.findall(line):
        rec[f"{rail.lower()}_mw"] = int(inst)
        rec[f"{rail.lower()}_avg_mw"] = int(avg)

    return rec


class TegrastatsRecorder:
    def __init__(self, interval_ms: int = 1000):
        self.interval_ms = interval_ms
        self._samples: List[Dict[str, Any]] = []
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self):
        self._proc = subprocess.Popen(
            ["tegrastats", "--interval", str(self.interval_ms)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        self._thread = threading.Thread(target=self._consume, daemon=True)
        self._thread.start()

    def _consume(self):
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            rec = parse_line(line)
            if rec:
                self._samples.append(rec)

    def stop(self):
        self._stop.set()
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread:
            self._thread.join(timeout=2)

    def samples(self) -> List[Dict[str, Any]]:
        return list(self._samples)

    def summary(self) -> Dict[str, Any]:
        """Aggregate numeric fields over the recording window."""
        if not self._samples:
            return {}

        def agg(field):
            vals = [s[field] for s in self._samples if field in s]
            if not vals:
                return None
            return {
                "min": min(vals),
                "max": max(vals),
                "mean": round(mean(vals), 2),
                "median": round(median(vals), 2),
            }

        out = {"n_samples": len(self._samples)}
        for f in ("ram_used_mb", "cpu_pct_mean", "gpu_pct",
                  "temp_gpu_c", "temp_tj_c"):
            a = agg(f)
            if a:
                out[f] = a
        return out
