#!/usr/bin/env python3
"""
plot_tegrastats.py — plot tegrastats CSVs captured by tegrastats_capture.sh.

Usage:
    python3 scripts/plot_tegrastats.py \
        --csv assets/results/tegrastats/trt_*.csv:trt \
        --csv assets/results/tegrastats/vllm_*.csv:vllm \
        --csv assets/results/tegrastats/llamacpp_*.csv:llamacpp \
        --out assets/results/tegrastats/comparison.png

Each --csv argument is <path>:<label>. Produces a 3-panel figure:
    row 1: GPU utilization % vs time, all runtimes overlaid
    row 2: RAM used (MB) vs time, all runtimes overlaid
    row 3: power draw (VDD_IN, mW) vs time, all runtimes overlaid
"""

import argparse
import csv
import os
import sys

try:
    import matplotlib
    matplotlib.use("Agg")  # no display needed
    import matplotlib.pyplot as plt
except ImportError:
    print("error: matplotlib required. pip install matplotlib", file=sys.stderr)
    sys.exit(1)


def read_csv(path):
    """Return dict of column arrays from a tegrastats CSV."""
    cols = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                cols.setdefault(k, []).append(float(v))
    return cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        action="append",
        required=True,
        metavar="PATH:LABEL",
        help="path to CSV and its display label, separated by colon",
    )
    ap.add_argument("--out", required=True, help="output PNG path")
    ap.add_argument(
        "--trim-to-active",
        action="store_true",
        help="trim leading/trailing baseline by cropping to the time when GPU>10%%",
    )
    args = ap.parse_args()

    runs = []
    for spec in args.csv:
        if ":" not in spec:
            print(f"bad --csv spec {spec!r}, expected PATH:LABEL", file=sys.stderr)
            sys.exit(1)
        path, label = spec.rsplit(":", 1)
        if not os.path.exists(path):
            print(f"missing: {path}", file=sys.stderr)
            sys.exit(1)
        runs.append((label, path, read_csv(path)))

    # Optional trim
    if args.trim_to_active:
        for label, path, cols in runs:
            t = cols["t_ms"]
            g = cols["gpu_pct"]
            active = [i for i, v in enumerate(g) if v > 10]
            if active:
                a, b = max(0, active[0] - 5), min(len(t), active[-1] + 5)
                t0 = t[a]
                for k in cols:
                    cols[k] = cols[k][a:b]
                cols["t_ms"] = [x - t0 for x in cols["t_ms"]]

    colors = {"trt": "#76b900", "vllm": "#4477ff", "llamacpp": "#ff8800"}

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    for label, _, cols in runs:
        c = colors.get(label, None)
        t_s = [x / 1000.0 for x in cols["t_ms"]]
        axes[0].plot(t_s, cols["gpu_pct"], label=label, linewidth=1.3, color=c)
        axes[1].plot(t_s, cols["ram_mb_used"], label=label, linewidth=1.3, color=c)
        axes[2].plot(t_s, cols["power_in_mw"], label=label, linewidth=1.3, color=c)

    axes[0].set_ylabel("GPU util %")
    axes[0].set_ylim(0, 105)
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper right", fontsize=9)

    axes[1].set_ylabel("RAM used (MB)")
    axes[1].grid(alpha=0.25)

    axes[2].set_ylabel("VDD_IN (mW)")
    axes[2].set_xlabel("time (s)")
    axes[2].grid(alpha=0.25)

    fig.suptitle(
        "Runtime utilization during one bench_trt.sh run (text then image)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
