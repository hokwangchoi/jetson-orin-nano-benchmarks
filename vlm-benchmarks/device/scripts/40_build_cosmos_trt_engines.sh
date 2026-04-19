#!/usr/bin/env bash
# 40_build_cosmos_trt_engines.sh — build TRT-Edge-LLM engines for Cosmos-2B.
#
# Prerequisites (all done once, manually, see trt_cosmos_patches/README.md):
#   1. cma=950M in /boot/extlinux/extlinux.conf (reboot required after setting)
#   2. ~/TensorRT-Edge-LLM cloned, source-patched, built
#   3. Cosmos-Reason2-2B ONNX exported to the workspace
#   4. split_lm_head.py run, producing Cosmos-Reason2-2B-split/onnx/llm/
#
# Builds two engines:
#   engine-1024/llm/    — LLM engine at 1024 input len / 1024 KV cache
#                         (matches vLLM's max-model-len for apples-to-apples)
#   engine/visual/      — visual encoder engine (no surgery needed)
#
# Build time: ~10 minutes total on Orin Nano 8 GB (Super mode). The llm_build
# step is the slow one; visual_build is ~5 min.

set -euo pipefail

WORKSPACE="${WORKSPACE:-$HOME/tensorrt-edgellm-workspace}"
MODEL_NAME="${MODEL_NAME:-Cosmos-Reason2-2B-split}"
EDGELLM_ROOT="${EDGELLM_ROOT:-$HOME/TensorRT-Edge-LLM}"

export EDGELLM_PLUGIN_PATH="$EDGELLM_ROOT/build/libNvInfer_edgellm_plugin.so"

ONNX_LLM_DIR="$WORKSPACE/$MODEL_NAME/onnx/llm"
ONNX_VISUAL_DIR="$WORKSPACE/$MODEL_NAME/onnx/visual"
ENGINE_LLM_DIR="$WORKSPACE/$MODEL_NAME/engine-1024/llm"
ENGINE_VISUAL_PARENT="$WORKSPACE/$MODEL_NAME/engine"   # visual_build appends /visual/

# Sanity checks
if [[ ! -d "$ONNX_LLM_DIR" ]]; then
    echo "error: LLM ONNX not found at $ONNX_LLM_DIR" >&2
    echo "run split_lm_head.py first" >&2
    exit 1
fi
if [[ ! -d "$ONNX_VISUAL_DIR" ]]; then
    # Fall back to the unpatched visual ONNX — we don't patch it
    ONNX_VISUAL_DIR="$WORKSPACE/Cosmos-Reason2-2B/onnx/visual"
fi
if [[ ! -d "$ONNX_VISUAL_DIR" ]]; then
    echo "error: visual ONNX not found (tried both patched and unpatched paths)" >&2
    exit 1
fi
if [[ ! -f "$EDGELLM_PLUGIN_PATH" ]]; then
    echo "error: edge-llm plugin lib not found at $EDGELLM_PLUGIN_PATH" >&2
    exit 1
fi

# Pre-build hygiene reminder
CMA_KB=$(grep -E "^CmaTotal:" /proc/meminfo | awk '{print $2}')
if [[ "$CMA_KB" -lt 900000 ]]; then
    echo "warning: CmaTotal is ${CMA_KB} kB (expected ~983040)." >&2
    echo "check /boot/extlinux/extlinux.conf has cma=950M and you rebooted." >&2
    echo "continuing anyway, but build will likely fail." >&2
fi

echo "=== Building LLM engine (split LM head, maxInputLen=1024) ==="
mkdir -p "$ENGINE_LLM_DIR"
LOG_LLM="/tmp/build_trt_llm_$(date +%Y%m%d_%H%M%S).log"
"$EDGELLM_ROOT/build/examples/llm/llm_build" \
    --onnxDir "$ONNX_LLM_DIR" \
    --engineDir "$ENGINE_LLM_DIR" \
    --maxBatchSize 1 \
    --maxInputLen 1024 \
    --maxKVCacheCapacity 1024 \
    --debug 2>&1 | tee "$LOG_LLM"

if ! grep -q "LLM engine built successfully" "$LOG_LLM"; then
    echo "error: LLM engine build did not complete successfully, see $LOG_LLM" >&2
    exit 1
fi
echo "  -> $ENGINE_LLM_DIR/llm.engine ($(du -h "$ENGINE_LLM_DIR/llm.engine" | cut -f1))"

echo ""
echo "=== Building visual encoder engine ==="
LOG_VISUAL="/tmp/build_trt_visual_$(date +%Y%m%d_%H%M%S).log"
"$EDGELLM_ROOT/build/examples/multimodal/visual_build" \
    --onnxDir "$ONNX_VISUAL_DIR" \
    --engineDir "$ENGINE_VISUAL_PARENT" \
    --debug 2>&1 | tee "$LOG_VISUAL"

if ! grep -q "Visual engine built successfully" "$LOG_VISUAL"; then
    echo "error: visual engine build did not complete, see $LOG_VISUAL" >&2
    exit 1
fi
echo "  -> $ENGINE_VISUAL_PARENT/visual/visual.engine ($(du -h "$ENGINE_VISUAL_PARENT/visual/visual.engine" | cut -f1))"

echo ""
echo "=== Done ==="
echo "LLM engine:    $ENGINE_LLM_DIR/llm.engine"
echo "Visual engine: $ENGINE_VISUAL_PARENT/visual/visual.engine"
echo "Build logs:"
echo "  $LOG_LLM"
echo "  $LOG_VISUAL"
echo ""
echo "Next: run 41_sanity_cosmos_trt.sh"
