#!/usr/bin/env python3
"""
bench_vllm.py — streaming benchmark against a running vLLM server.

Captures proper TTFT (time-to-first-token), TPOT (time-per-output-token),
and TPS by reading server-sent-events rather than the final response.
Runs text-only and text+image workloads, writes JSON to assets/results/.

Usage:
    python3 benchmarks/bench_vllm.py \
        --url http://localhost:8000 \
        --model embedl/Cosmos-Reason2-2B-W4A16 \
        --image assets/images/crossing.jpg   # optional

    # Text-only quick pass:
    python3 benchmarks/bench_vllm.py --text-only

See benchmarks/harness.py for the full runtime-agnostic harness (drives
llama.cpp, vLLM, and TRT-Edge-LLM through a common OpenAI interface,
also captures tegrastats + power). This script is the faster single-
runtime measurement tool kept for convenience.
"""

import argparse
import base64
import json
import statistics
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "assets" / "results"

TEXT_PROMPT = (
    "Describe a pedestrian crossing scene in an autonomous driving context. "
    "Include vehicles, pedestrians, signals, and potential hazards."
)
IMAGE_PROMPT = "Describe this image in 3 sentences. Focus only on what you can actually see."


def post_stream(url: str, payload: dict, timeout: float = 120.0):
    """Yield (timestamp_perf_counter, content_delta) for each streamed token."""
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.strip()
            if not line or not line.startswith(b"data: "):
                continue
            data = line[6:]
            if data == b"[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = obj.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta", {}) or {}
            content = delta.get("content")
            if content:
                yield time.perf_counter(), content


def bench_one(url: str, payload: dict) -> dict:
    """Run one request, return TTFT/TPOT/TPS."""
    start = time.perf_counter()
    first_token = None
    last_token = None
    n_tokens = 0
    try:
        for ts, _ in post_stream(f"{url}/v1/chat/completions", payload):
            if first_token is None:
                first_token = ts
            last_token = ts
            n_tokens += 1
    except (urllib.error.URLError, TimeoutError) as e:
        return {"error": str(e)}
    end = time.perf_counter()

    if first_token is None:
        return {"error": "no tokens received"}

    ttft_ms = (first_token - start) * 1000.0
    decode_s = (last_token - first_token) if n_tokens > 1 else 0.0
    tpot_ms = (decode_s / (n_tokens - 1) * 1000.0) if n_tokens > 1 else 0.0
    tps = n_tokens / (end - start)
    return {
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "tps": tps,
        "output_tokens": n_tokens,
        "wall_s": end - start,
    }


def bench_workload(url: str, model: str, messages: list, max_tokens: int,
                   n_runs: int, n_warmup: int, temperature: float,
                   label: str) -> list:
    base_payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    # Warmup
    for i in range(n_warmup):
        warm = dict(base_payload)
        warm["max_tokens"] = 10
        warm["messages"] = [{"role": "user", "content": "hi"}]
        bench_one(url, warm)

    results = []
    print(f"\n=== {label} ({n_runs} runs, max_tokens={max_tokens}) ===")
    for i in range(n_runs):
        r = bench_one(url, base_payload)
        r["run"] = i + 1
        if "error" in r:
            print(f"  Run {i+1}: ERROR {r['error']}")
        else:
            print(f"  Run {i+1}: TTFT={r['ttft_ms']:6.1f}ms "
                  f"TPOT={r['tpot_ms']:5.1f}ms "
                  f"TPS={r['tps']:5.2f} "
                  f"out={r['output_tokens']}t "
                  f"wall={r['wall_s']:.2f}s")
        results.append(r)
    return results


def summarize(results: list, label: str) -> dict:
    good = [r for r in results if "error" not in r]
    if not good:
        print(f"\n{label}: no successful runs")
        return {}
    summary = {}
    for metric in ("ttft_ms", "tpot_ms", "tps", "output_tokens", "wall_s"):
        vals = [r[metric] for r in good]
        summary[metric] = {
            "median": statistics.median(vals),
            "mean": statistics.mean(vals),
            "min": min(vals),
            "max": max(vals),
            "n": len(vals),
        }
    print(f"\n{label} — median of {len(good)} runs:")
    print(f"  TTFT : {summary['ttft_ms']['median']:6.1f} ms")
    print(f"  TPOT : {summary['tpot_ms']['median']:6.1f} ms")
    print(f"  TPS  : {summary['tps']['median']:6.2f} tok/s")
    return summary


def encode_image_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--model", default="embedl/Cosmos-Reason2-2B-W4A16")
    p.add_argument("--image", type=Path, default=None,
                   help="path to a test image for text+image benchmark")
    p.add_argument("--text-only", action="store_true")
    p.add_argument("--n-runs", type=int, default=5)
    p.add_argument("--n-warmup", type=int, default=3)
    p.add_argument("--max-tokens-text", type=int, default=128)
    p.add_argument("--max-tokens-image", type=int, default=64,
                   help="kept low because image tokens eat context budget "
                        "at max_model_len=256")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--output", type=Path, default=None,
                   help="JSON output path (default: auto-named in assets/results/)")
    p.add_argument("--runtime-label", default="vllm_cosmos_w4a16",
                   help="short tag used in output filename")
    args = p.parse_args()

    # Sanity-check server
    try:
        with urllib.request.urlopen(f"{args.url}/v1/models", timeout=5) as r:
            models = json.loads(r.read())
        ids = [m["id"] for m in models.get("data", [])]
        if args.model not in ids:
            print(f"warning: model '{args.model}' not found on server. "
                  f"Available: {ids}")
    except Exception as e:
        print(f"error: cannot reach vLLM at {args.url}: {e}")
        sys.exit(1)

    text_msgs = [{"role": "user", "content": TEXT_PROMPT}]
    text_results = bench_workload(
        args.url, args.model, text_msgs, args.max_tokens_text,
        args.n_runs, args.n_warmup, args.temperature, "TEXT-ONLY"
    )
    text_summary = summarize(text_results, "TEXT-ONLY")

    image_results = []
    image_summary = {}
    if args.image and args.image.exists() and not args.text_only:
        img_b64 = encode_image_b64(args.image)
        image_msgs = [{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                {"type": "text", "text": IMAGE_PROMPT},
            ],
        }]
        image_results = bench_workload(
            args.url, args.model, image_msgs, args.max_tokens_image,
            args.n_runs, args.n_warmup, args.temperature, "TEXT+IMAGE"
        )
        image_summary = summarize(image_results, "TEXT+IMAGE")

    # Write JSON
    if args.output is None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = RESULTS_DIR / f"{args.runtime_label}_{ts}_streaming.json"
    out = {
        "runtime_label": args.runtime_label,
        "model": args.model,
        "url": args.url,
        "n_runs": args.n_runs,
        "n_warmup": args.n_warmup,
        "temperature": args.temperature,
        "text_prompt": TEXT_PROMPT,
        "image_prompt": IMAGE_PROMPT,
        "image_path": str(args.image) if args.image else None,
        "max_tokens_text": args.max_tokens_text,
        "max_tokens_image": args.max_tokens_image,
        "text_results": text_results,
        "text_summary": text_summary,
        "image_results": image_results,
        "image_summary": image_summary,
        "timestamp": datetime.now().isoformat(),
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
