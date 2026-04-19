#!/usr/bin/env bash
# bench_trt.sh — run the 5-iteration text and image TRT-Edge-LLM benchmarks.
#
# Prerequisite: 40_build_cosmos_trt_engines.sh has been run and
# $WORKSPACE/$MODEL_NAME/engine-1024/llm/llm.engine exists.
#
# Produces profile JSONs compatible with the aggregation downstream. Uses
# --warmup 2 to match the post-warmup regime measured for llama.cpp and vLLM.
#
# Output JSONs land under vlm-benchmarks/assets/results/trt/ with a timestamp
# suffix so multiple runs can be kept.

set -euo pipefail

WORKSPACE="${WORKSPACE:-$HOME/tensorrt-edgellm-workspace}"
MODEL_NAME="${MODEL_NAME:-Cosmos-Reason2-2B-split}"
EDGELLM_ROOT="${EDGELLM_ROOT:-$HOME/TensorRT-Edge-LLM}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
INPUT_DIR="${REPO_ROOT}/device/inputs/trt"
RESULTS_DIR="${REPO_ROOT}/assets/results/trt"

export EDGELLM_PLUGIN_PATH="$EDGELLM_ROOT/build/libNvInfer_edgellm_plugin.so"

ENGINE_LLM_DIR="$WORKSPACE/$MODEL_NAME/engine-1024/llm"
ENGINE_VISUAL_DIR="$WORKSPACE/$MODEL_NAME/engine/visual"

if [[ ! -f "$ENGINE_LLM_DIR/llm.engine" || ! -f "$ENGINE_VISUAL_DIR/visual.engine" ]]; then
    echo "error: engines not built. run 40_build_cosmos_trt_engines.sh first" >&2
    exit 1
fi

mkdir -p "$RESULTS_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"

echo "=== Pre-run hygiene check ==="
echo "CMA:      $(grep -E '^CmaTotal:|^CmaFree:' /proc/meminfo | tr '\n' ' ')"
echo "Power:    $(sudo nvpmodel -q | head -1)"
echo "Swap:     $(free -h | awk '/Swap:/ {print $2, $3, $4}')"
echo ""

echo "=== Text benchmark (5 runs, warmup 2) ==="
TEXT_OUT="$RESULTS_DIR/trt_bench_text_5x_${STAMP}.json"
TEXT_PROFILE="$RESULTS_DIR/trt_profile_text_5x_${STAMP}.json"
TEXT_LOG="$RESULTS_DIR/trt_bench_text_5x_${STAMP}.log"
"$EDGELLM_ROOT/build/examples/llm/llm_inference" \
    --engineDir "$ENGINE_LLM_DIR" \
    --multimodalEngineDir "$ENGINE_VISUAL_DIR" \
    --inputFile "$INPUT_DIR/bench_text_5x.json" \
    --outputFile "$TEXT_OUT" \
    --profileOutputFile "$TEXT_PROFILE" \
    --warmup 2 \
    --dumpProfile \
    2>&1 | tee "$TEXT_LOG" | grep -E "Performance Summary|LLM Prefill|LLM Generation|Tokens/Second|Average Time per Token|Peak Unified|Response for request"

echo ""
echo "=== Image benchmark (5 runs, warmup 2) ==="
IMAGE_OUT="$RESULTS_DIR/trt_bench_image_5x_${STAMP}.json"
IMAGE_PROFILE="$RESULTS_DIR/trt_profile_image_5x_${STAMP}.json"
IMAGE_LOG="$RESULTS_DIR/trt_bench_image_5x_${STAMP}.log"
"$EDGELLM_ROOT/build/examples/llm/llm_inference" \
    --engineDir "$ENGINE_LLM_DIR" \
    --multimodalEngineDir "$ENGINE_VISUAL_DIR" \
    --inputFile "$INPUT_DIR/bench_image_5x.json" \
    --outputFile "$IMAGE_OUT" \
    --profileOutputFile "$IMAGE_PROFILE" \
    --warmup 2 \
    --dumpProfile \
    2>&1 | tee "$IMAGE_LOG" | grep -E "Performance Summary|LLM Prefill|LLM Generation|Tokens/Second|Average Time per Token|Peak Unified|Response for request|Multimodal"

echo ""
echo "=== Done ==="
echo "Results:"
echo "  text:  $TEXT_PROFILE"
echo "  image: $IMAGE_PROFILE"
echo ""
echo "Full logs (with per-token details and any warnings):"
echo "  text:  $TEXT_LOG"
echo "  image: $IMAGE_LOG"
