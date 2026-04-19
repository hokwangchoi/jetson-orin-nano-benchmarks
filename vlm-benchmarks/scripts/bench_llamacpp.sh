#!/usr/bin/env bash
# bench_llamacpp.sh — benchmark a running llama.cpp server.
#
# Prerequisite: 11_run_llamacpp_server.sh is running in another terminal
# and the server is ready (watch for "HTTP server listening" in its log).
#
# llama.cpp's server exposes an OpenAI-compatible /v1/chat/completions
# endpoint, so the same Python streaming benchmark used for vLLM drives
# it unchanged — only the model label and output location differ.
#
# Runs 5 streaming runs each for text and text+image workloads.
# Output: assets/results/llamacpp/llamacpp_cosmos_q4km_<timestamp>_streaming.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_DIR="${REPO_ROOT}/assets/results/llamacpp"
IMAGE_PATH="${IMAGE_PATH:-${REPO_ROOT}/assets/images/bus.jpg}"

# Source the server env for the SERVED_NAME used in API requests
ENV_FILE="${REPO_ROOT}/device/configs/llamacpp_server.env"
[ -f "${ENV_FILE}" ] && { set -a; source "${ENV_FILE}"; set +a; }

mkdir -p "${RESULTS_DIR}"

TS="$(date +%Y%m%d_%H%M%S)"
OUTPUT="${RESULTS_DIR}/llamacpp_cosmos_q4km_${TS}_streaming.json"

URL="${LLAMACPP_URL:-http://localhost:8000}"
MODEL="${SERVED_NAME:-Cosmos-Reason2-2B-Q4_K_M}"

echo "[bench_llamacpp] server: ${URL}"
echo "[bench_llamacpp] model:  ${MODEL}"
echo "[bench_llamacpp] image:  ${IMAGE_PATH}"
echo "[bench_llamacpp] output: ${OUTPUT}"
echo ""

python3 "${REPO_ROOT}/benchmarks/bench_vllm.py" \
    --url "${URL}" \
    --model "${MODEL}" \
    --image "${IMAGE_PATH}" \
    --output "${OUTPUT}" \
    --runtime-label "llamacpp_cosmos_q4km"

echo ""
echo "[bench_llamacpp] done."
echo "[bench_llamacpp] results: ${OUTPUT}"
