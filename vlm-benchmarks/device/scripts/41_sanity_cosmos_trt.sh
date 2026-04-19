#!/usr/bin/env bash
# 41_sanity_cosmos_trt.sh — verify the built TRT engines produce sane output.
#
# Runs two single-prompt inferences (one text, one image). Purpose is to
# confirm that after all the graph surgery the engine isn't producing
# gibberish. Not a benchmark; use 42_bench_cosmos_trt.sh for numbers.
#
# Expected: the text prompt returns a plausible answer ("Paris" for
# "capital of France"), and the image prompt describes the bus scene.

set -euo pipefail

WORKSPACE="${WORKSPACE:-$HOME/tensorrt-edgellm-workspace}"
MODEL_NAME="${MODEL_NAME:-Cosmos-Reason2-2B-split}"
EDGELLM_ROOT="${EDGELLM_ROOT:-$HOME/TensorRT-Edge-LLM}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="$SCRIPT_DIR/../inputs/trt"

export EDGELLM_PLUGIN_PATH="$EDGELLM_ROOT/build/libNvInfer_edgellm_plugin.so"

ENGINE_LLM_DIR="$WORKSPACE/$MODEL_NAME/engine-1024/llm"
ENGINE_VISUAL_DIR="$WORKSPACE/$MODEL_NAME/engine/visual"

if [[ ! -f "$ENGINE_LLM_DIR/llm.engine" ]]; then
    echo "error: LLM engine not found at $ENGINE_LLM_DIR/llm.engine" >&2
    echo "run 40_build_cosmos_trt_engines.sh first" >&2
    exit 1
fi
if [[ ! -f "$ENGINE_VISUAL_DIR/visual.engine" ]]; then
    echo "error: visual engine not found at $ENGINE_VISUAL_DIR/visual.engine" >&2
    exit 1
fi

echo "=== Text-only sanity check ==="
"$EDGELLM_ROOT/build/examples/llm/llm_inference" \
    --engineDir "$ENGINE_LLM_DIR" \
    --multimodalEngineDir "$ENGINE_VISUAL_DIR" \
    --inputFile "$INPUT_DIR/sanity_text.json" \
    --outputFile /tmp/trt_sanity_text_out.json \
    --dumpOutput \
    2>&1 | tee /tmp/trt_sanity_text.log | tail -25

echo ""
echo "=== Image sanity check (bus.jpg) ==="
"$EDGELLM_ROOT/build/examples/llm/llm_inference" \
    --engineDir "$ENGINE_LLM_DIR" \
    --multimodalEngineDir "$ENGINE_VISUAL_DIR" \
    --inputFile "$INPUT_DIR/sanity_image.json" \
    --outputFile /tmp/trt_sanity_image_out.json \
    --dumpOutput \
    2>&1 | tee /tmp/trt_sanity_image.log | tail -25

echo ""
echo "=== Done ==="
echo "Text response:   /tmp/trt_sanity_text_out.json"
echo "Image response:  /tmp/trt_sanity_image_out.json"
echo ""
echo "Eyeball both outputs for coherence. If they look sane, proceed to"
echo "42_bench_cosmos_trt.sh for timing measurements."
