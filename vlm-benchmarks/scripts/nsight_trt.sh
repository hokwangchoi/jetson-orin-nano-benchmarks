#!/usr/bin/env bash
# nsight_trt.sh — capture a single TRT Edge-LLM image inference under nsys.
#
# Wraps llm_inference with nsys profile. Single request, --warmup 0 (TRT
# engines are pre-compiled so there is no lazy compilation; first request
# is essentially steady-state). Timeline shows the full inference path:
# ViT → LLM prefill → decode. Same input JSON as sanity_image.json
# (bus.jpg, 64 output tokens) so the workload matches the benchmark numbers.
#
# Output:
#   assets/results/nsight/trt_image_<stamp>.nsys-rep  — open in Nsight Systems GUI
#   assets/results/nsight/trt_image_<stamp>.qdstrm    — intermediate, can be deleted

set -euo pipefail

WORKSPACE="${WORKSPACE:-$HOME/tensorrt-edgellm-workspace}"
MODEL_NAME="${MODEL_NAME:-Cosmos-Reason2-2B-split}"
EDGELLM_ROOT="${EDGELLM_ROOT:-$HOME/TensorRT-Edge-LLM}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
INPUT_DIR="${REPO_ROOT}/device/inputs/trt"
OUT_DIR="${REPO_ROOT}/assets/results/nsight"
mkdir -p "$OUT_DIR"

export EDGELLM_PLUGIN_PATH="$EDGELLM_ROOT/build/libNvInfer_edgellm_plugin.so"
ENGINE_LLM_DIR="$WORKSPACE/$MODEL_NAME/engine-1024/llm"
ENGINE_VISUAL_DIR="$WORKSPACE/$MODEL_NAME/engine/visual"

if ! command -v nsys &>/dev/null; then
    echo "error: nsys not found. install via \`sudo apt install nsight-systems-cli\` or it ships with JetPack's nsight-systems package." >&2
    exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
REPORT="$OUT_DIR/trt_image_${STAMP}"

echo "=== Pre-run hygiene check ==="
echo "CMA:   $(grep -E '^CmaTotal:|^CmaFree:' /proc/meminfo | tr '\n' ' ')"
echo "Power: $(sudo nvpmodel -q | head -1)"
echo ""

echo "=== Capturing TRT image inference → $REPORT.nsys-rep ==="
nsys profile \
    --trace=cuda,nvtx,osrt \
    --sample=none \
    --cpuctxsw=none \
    --output="$REPORT" \
    --force-overwrite=true \
    "$EDGELLM_ROOT/build/examples/llm/llm_inference" \
        --engineDir "$ENGINE_LLM_DIR" \
        --multimodalEngineDir "$ENGINE_VISUAL_DIR" \
        --inputFile "$INPUT_DIR/sanity_image.json" \
        --outputFile /tmp/nsys_trt_out.json \
        --warmup 0

echo ""
echo "=== Done ==="
echo "Report: $REPORT.nsys-rep"
echo "Open with Nsight Systems GUI on your laptop (scp the .nsys-rep file over)."
