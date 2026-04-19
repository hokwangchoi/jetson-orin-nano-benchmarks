#!/usr/bin/env bash
# bench_vllm.sh — benchmark a running vLLM server.
#
# Prerequisite: 03_run_vllm_server.sh is running in another terminal and
# the server is ready (watch its log for "Application startup complete").
#
# Runs 5 streaming runs each for text and text+image workloads.
# Output: assets/results/vllm/vllm_cosmos_w4a16_<timestamp>_streaming.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_DIR="${REPO_ROOT}/assets/results/vllm"
IMAGE_PATH="${IMAGE_PATH:-${REPO_ROOT}/assets/images/bus.jpg}"

mkdir -p "${RESULTS_DIR}"

TS="$(date +%Y%m%d_%H%M%S)"
OUTPUT="${RESULTS_DIR}/vllm_cosmos_w4a16_${TS}_streaming.json"

URL="${VLLM_URL:-http://localhost:8000}"
MODEL="${VLLM_MODEL:-embedl/Cosmos-Reason2-2B-W4A16}"

echo "[bench_vllm] server: ${URL}"
echo "[bench_vllm] model:  ${MODEL}"
echo "[bench_vllm] image:  ${IMAGE_PATH}"
echo "[bench_vllm] output: ${OUTPUT}"
echo ""

python3 "${REPO_ROOT}/benchmarks/bench_vllm.py" \
    --url "${URL}" \
    --model "${MODEL}" \
    --image "${IMAGE_PATH}" \
    --output "${OUTPUT}" \
    --runtime-label "vllm_cosmos_w4a16"

echo ""
echo "[bench_vllm] done."
echo "[bench_vllm] results: ${OUTPUT}"
