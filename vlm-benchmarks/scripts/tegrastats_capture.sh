#!/usr/bin/env bash
# tegrastats_capture.sh — sample tegrastats at 100ms while running a benchmark,
# emit a clean CSV with GPU%, memory, CPU%, power per timestamp.
#
# Usage:
#   ./scripts/tegrastats_capture.sh <runtime_label> <benchmark_script> [args...]
#
# Examples:
#   ./scripts/tegrastats_capture.sh trt       ./scripts/bench_trt.sh
#   ./scripts/tegrastats_capture.sh vllm      ./scripts/bench_vllm.sh
#   ./scripts/tegrastats_capture.sh llamacpp  ./scripts/bench_llamacpp.sh
#
# Output: assets/results/tegrastats/<runtime_label>_<stamp>.csv
#         Columns: t_ms, ram_mb_used, ram_mb_total, swap_mb_used, iram_mb,
#                  cpu_avg_pct, gpu_pct, gpu_freq_mhz, emc_freq_mhz,
#                  power_soc_mw, power_cpu_gpu_cv_mw, power_in_mw, board_temp_c

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "usage: $0 <runtime_label> <benchmark_script> [args...]" >&2
    echo "       runtime_label is e.g. trt, vllm, llamacpp — used in output filename" >&2
    exit 1
fi

LABEL="$1"; shift
BENCH_CMD=("$@")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT_DIR="${REPO_ROOT}/assets/results/tegrastats"
mkdir -p "$OUT_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
RAW="$OUT_DIR/${LABEL}_${STAMP}.raw"
CSV="$OUT_DIR/${LABEL}_${STAMP}.csv"

echo "=== tegrastats capture for ${LABEL} ==="
echo "raw:  $RAW"
echo "csv:  $CSV"
echo ""

# Start tegrastats in the background at 100ms sampling
sudo tegrastats --interval 100 --logfile "$RAW" &
TEGRA_PID=$!

# Clean up on any exit path
cleanup() {
    if kill -0 "$TEGRA_PID" 2>/dev/null; then
        sudo kill "$TEGRA_PID" 2>/dev/null || true
        wait "$TEGRA_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

sleep 1  # let tegrastats get a few baseline samples

echo "=== Running benchmark: ${BENCH_CMD[*]} ==="
T_START_MS=$(($(date +%s%N) / 1000000))
"${BENCH_CMD[@]}"
BENCH_RC=$?
T_END_MS=$(($(date +%s%N) / 1000000))

sleep 1  # let tegrastats catch the tail
cleanup
trap - EXIT

echo ""
echo "=== Benchmark duration: $((T_END_MS - T_START_MS)) ms (rc=$BENCH_RC) ==="

# Parse raw tegrastats into CSV
# Sample line format (Jetson Orin Nano, L4T 36.x):
#   04-19-2026 05:12:34 RAM 2145/7433MB (lfb 2x4MB) SWAP 0/3716MB (cached 0MB) CPU [14%@1510,10%@1510,8%@1510,12%@1510,17%@1510,13%@1510] EMC_FREQ 0%@2133 GR3D_FREQ 45%@[1020] VIC_FREQ 0% APE 200 CV0@41.5C SOC0@40.5C CPU@43C TJ@44.5C SOC1@42C SOC2@42.5C CV1@41C CV2@42C VDD_IN 6842mW/5980mW VDD_CPU_GPU_CV 1983mW/1623mW VDD_SOC 1437mW/1317mW
#
# We pull: RAM used/total, SWAP used, CPU avg, GR3D_FREQ (GPU %), GR3D clock,
# EMC clock, VDD_IN current draw, VDD_CPU_GPU_CV current, VDD_SOC current,
# hottest temperature.

echo ""
echo "=== Parsing raw log → CSV ==="
python3 - "$RAW" "$CSV" "$T_START_MS" <<'PY'
import re, sys

raw_path, csv_path, t_start_ms_str = sys.argv[1:]
t_start_ms = int(t_start_ms_str)

patterns = {
    'ram':       re.compile(r'RAM\s+(\d+)/(\d+)MB'),
    'swap':      re.compile(r'SWAP\s+(\d+)/(\d+)MB'),
    'cpu':       re.compile(r'CPU\s+\[([^\]]+)\]'),
    'gr3d':      re.compile(r'GR3D_FREQ\s+(\d+)%@\[?(\d+)'),
    'emc':       re.compile(r'EMC_FREQ\s+(\d+)%@(\d+)'),
    'vdd_in':    re.compile(r'VDD_IN\s+(\d+)mW/(\d+)mW'),
    'vdd_cgc':   re.compile(r'VDD_CPU_GPU_CV\s+(\d+)mW/(\d+)mW'),
    'vdd_soc':   re.compile(r'VDD_SOC\s+(\d+)mW/(\d+)mW'),
    'temps':     re.compile(r'(CPU|SOC\d|TJ|CV\d)@([\d.]+)C'),
}

rows = []
first_sample_ms = None

with open(raw_path, 'r', errors='replace') as f:
    for line in f:
        line = line.rstrip('\n')
        if not line.strip():
            continue

        m = patterns['ram'].search(line)
        if not m:
            continue
        ram_used, ram_total = int(m.group(1)), int(m.group(2))

        m = patterns['swap'].search(line)
        swap_used = int(m.group(1)) if m else 0

        m = patterns['cpu'].search(line)
        if m:
            cpu_pcts = [int(p.split('%')[0]) for p in m.group(1).split(',')]
            cpu_avg = sum(cpu_pcts) / len(cpu_pcts) if cpu_pcts else 0.0
        else:
            cpu_avg = 0.0

        m = patterns['gr3d'].search(line)
        gpu_pct, gpu_freq = (int(m.group(1)), int(m.group(2))) if m else (0, 0)

        m = patterns['emc'].search(line)
        emc_freq = int(m.group(2)) if m else 0

        m = patterns['vdd_in'].search(line)
        power_in = int(m.group(1)) if m else 0
        m = patterns['vdd_cgc'].search(line)
        power_cgc = int(m.group(1)) if m else 0
        m = patterns['vdd_soc'].search(line)
        power_soc = int(m.group(1)) if m else 0

        # Hottest temperature across all sensors
        temps = patterns['temps'].findall(line)
        tmax = max((float(v) for _, v in temps), default=0.0)

        # Sample rows are synthesized at 100ms cadence; use sample index for t_ms.
        idx = len(rows)
        t_ms = idx * 100
        if first_sample_ms is None:
            first_sample_ms = t_ms

        rows.append((t_ms, ram_used, ram_total, swap_used,
                     round(cpu_avg, 1), gpu_pct, gpu_freq, emc_freq,
                     power_soc, power_cgc, power_in, tmax))

with open(csv_path, 'w') as f:
    f.write("t_ms,ram_mb_used,ram_mb_total,swap_mb_used,"
            "cpu_avg_pct,gpu_pct,gpu_freq_mhz,emc_freq_mhz,"
            "power_soc_mw,power_cpu_gpu_cv_mw,power_in_mw,board_temp_c\n")
    for r in rows:
        f.write(",".join(str(x) for x in r) + "\n")

print(f"  parsed {len(rows)} samples → {csv_path}")
PY

echo ""
echo "=== Done ==="
echo "CSV:  $CSV"
echo "Raw:  $RAW   (keep for debugging, can delete once CSV looks right)"
