#!/usr/bin/env python3
"""
VLM Benchmark Harness — Jetson Orin Nano
Runtime-agnostic: picks a client by --runtime and drives it against a workload.

Collects latency (TTFT, TPOT, TPS) in-process, while a background thread
tails tegrastats for memory / CPU / GPU / power. Writes one JSON per run
to assets/results/raw/.

Usage:
    python3 -m benchmarks.harness \
        --runtime llamacpp \
        --workload benchmarks/workloads/text.json \
        --served-name Cosmos-Reason2-2B-Q4_K_M
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from benchmarks.metrics.resource import TegrastatsRecorder
from benchmarks.metrics.latency import summarize_latencies
from benchmarks.metrics.power import summarize_power

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "assets" / "results" / "raw"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def make_client(runtime: str, args):
    """
    Both llama.cpp (via llama-server) and TRT-Edge-LLM (via an OpenAI
    shim we'll add later) expose the same OpenAI /v1/chat/completions
    interface, so one client handles both.
    """
    if runtime in ("llamacpp", "trtedgellm"):
        from benchmarks.clients.openai_client import OpenAIClient
        return OpenAIClient(base_url=args.server_url, model=args.served_name)
    else:
        raise ValueError(f"unknown runtime: {runtime}")


def load_workload(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def run_single(client, sample: dict, max_tokens: int) -> dict:
    """Run one prompt, return a latency dict."""
    return client.infer(
        messages=sample["messages"],
        max_tokens=max_tokens,
        temperature=0.0,
    )


def run_sequential(client, workload, repeats: int, max_tokens: int):
    """Run prompts one at a time, repeats×samples total."""
    latencies = []
    for rep in range(repeats):
        for i, sample in enumerate(workload["samples"]):
            print(f"  [{rep+1}/{repeats}] sample {i+1}/{len(workload['samples'])} "
                  f"({sample.get('id', '?')})...", flush=True)
            lat = run_single(client, sample, max_tokens)
            lat["sample_id"] = sample.get("id", str(i))
            lat["repeat"] = rep
            latencies.append(lat)
    return latencies


def run_concurrent(client, workload, concurrency: int, max_tokens: int):
    """Fire N identical requests in parallel."""
    sample = workload["samples"][0]
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(run_single, client, sample, max_tokens)
                   for _ in range(concurrency)]
        results = []
        for f in as_completed(futures):
            results.append(f.result())
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", required=True, choices=["llamacpp", "trtedgellm"])
    ap.add_argument("--workload", required=True, type=Path)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--concurrency", type=str, default=None,
                    help="Comma-separated list, e.g. 1,2,4.")
    ap.add_argument("--tag", type=str, default="main",
                    help="Short label for this run (goes into the filename).")
    ap.add_argument("--server-url", type=str, default="http://localhost:8000")
    ap.add_argument("--served-name", type=str, default="Cosmos-Reason2-2B-Q4_K_M")
    ap.add_argument("--tegrastats-interval-ms", type=int, default=500)
    args = ap.parse_args()

    workload = load_workload(args.workload)

    client = make_client(args.runtime, args)

    # Quick ping so we fail fast if the runtime isn't ready
    print(f"[harness] probing {args.runtime} at {args.server_url}...")
    client.ping()

    # Run
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{args.runtime}_{workload['name']}_{args.tag}_{ts}"
    out_file = RESULTS_DIR / f"{run_id}.json"

    print(f"[harness] run_id={run_id}")
    print(f"[harness] starting tegrastats recorder (interval={args.tegrastats_interval_ms}ms)")
    recorder = TegrastatsRecorder(interval_ms=args.tegrastats_interval_ms)
    recorder.start()

    t_start = time.perf_counter()

    sections = []
    if args.concurrency:
        levels = [int(x) for x in args.concurrency.split(",")]
        for n in levels:
            if args.runtime == "trtedgellm" and n > 1:
                print(f"[harness] skipping concurrency={n} for trtedgellm (single-stream)")
                continue
            print(f"[harness] concurrency={n}")
            lats = run_concurrent(client, workload, n, args.max_tokens)
            sections.append({"concurrency": n, "latencies": lats})
    else:
        lats = run_sequential(client, workload, args.repeats, args.max_tokens)
        sections.append({"concurrency": 1, "latencies": lats})

    t_end = time.perf_counter()
    recorder.stop()

    samples = recorder.samples()

    result = {
        "run_id": run_id,
        "runtime": args.runtime,
        "workload": workload["name"],
        "tag": args.tag,
        "timestamp": ts,
        "wall_time_s": round(t_end - t_start, 3),
        "config": {
            "repeats": args.repeats,
            "max_tokens": args.max_tokens,
            "concurrency": args.concurrency,
            "served_name": args.served_name,
        },
        "sections": sections,
        "latency_summary": {
            str(s["concurrency"]): summarize_latencies(s["latencies"])
            for s in sections
        },
        "resource_summary": recorder.summary(),
        "power_summary": summarize_power(samples),
        "tegrastats_samples": samples,  # full timeline for plotting
    }

    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[harness] wrote {out_file}")

    # Print a quick summary
    for s in sections:
        summary = summarize_latencies(s["latencies"])
        print(f"[harness] concurrency={s['concurrency']}: "
              f"TTFT={summary['ttft_ms']['median']:.1f}ms  "
              f"TPOT={summary['tpot_ms']['median']:.2f}ms  "
              f"TPS={summary['tps']['median']:.1f}")


if __name__ == "__main__":
    main()
