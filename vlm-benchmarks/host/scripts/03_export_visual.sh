#!/usr/bin/env bash
# 03_export_visual.sh — export Cosmos-Reason2-2B ViT encoder to ONNX.
#
# The visual encoder runs fp16 (no INT4 quantization on the ViT path —
# TRT-Edge-LLM's visual_build handles kernel tuning at engine build
# time on-device). Export pulls the ViT submodule from the raw BF16
# HuggingFace checkpoint and casts inputs to fp16.
#
# ~3 minutes on A40. Output: visual.onnx, consumed by
# device/scripts/40_build_cosmos_trt_engines.sh on the Jetson.

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
MODEL="${MODEL:-nvidia/Cosmos-Reason2-2B}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE}/cosmos-2b-onnx/visual}"

if [ -f "${OUTPUT_DIR}/visual.onnx" ]; then
    echo "[export-visual] output exists at ${OUTPUT_DIR}, skipping."
    echo "[export-visual] remove the directory to force rerun."
    exit 0
fi

mkdir -p "${OUTPUT_DIR}"

echo "[export-visual] model:  ${MODEL}"
echo "[export-visual] output: ${OUTPUT_DIR}"
echo ""

python -m tensorrt_edge_llm.examples.visual.export_visual \
    --model_path  "${MODEL}" \
    --output_dir  "${OUTPUT_DIR}"

echo ""
echo "[export-visual] done. Next: ./scripts/04_package_for_jetson.sh"
