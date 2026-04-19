#!/usr/bin/env bash
# 02_export_llm.sh — export quantized Cosmos-Reason2-2B to ONNX.
#
# Reads the AWQ W4A16 checkpoint from 01_quantize_llm.sh, writes
# onnx_model.onnx + onnx_model.data (external weights) ready for
# TRT-Edge-LLM's llm_build on the Jetson.
#
# ~10 minutes on A40. Do NOT split the LM head here — that's done on
# the Jetson in device/trt_cosmos_patches/split_lm_head.py, after scp.
# (Keeps the host export reproducible from the stock quantized
# checkpoint; the split is a Jetson-specific platform workaround.)

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
QUANT_DIR="${QUANT_DIR:-${WORKSPACE}/cosmos-2b-quantized}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE}/cosmos-2b-onnx/llm}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-1}"
MAX_INPUT_LEN="${MAX_INPUT_LEN:-1024}"

if [ ! -f "${QUANT_DIR}/config.json" ]; then
    echo "[export-llm] quantized checkpoint not found at ${QUANT_DIR}"
    echo "[export-llm] run 01_quantize_llm.sh first."
    exit 1
fi

if [ -f "${OUTPUT_DIR}/onnx_model.onnx" ]; then
    echo "[export-llm] output exists at ${OUTPUT_DIR}, skipping."
    echo "[export-llm] remove the directory to force rerun."
    exit 0
fi

mkdir -p "${OUTPUT_DIR}"

echo "[export-llm] quant dir:     ${QUANT_DIR}"
echo "[export-llm] output dir:    ${OUTPUT_DIR}"
echo "[export-llm] max batch:     ${MAX_BATCH_SIZE}"
echo "[export-llm] max input len: ${MAX_INPUT_LEN}"
echo ""

python -m tensorrt_edge_llm.examples.llm.export_llm \
    --checkpoint_dir  "${QUANT_DIR}" \
    --output_dir      "${OUTPUT_DIR}" \
    --max_batch_size  "${MAX_BATCH_SIZE}" \
    --max_input_len   "${MAX_INPUT_LEN}"

echo ""
echo "[export-llm] done. Next: ./scripts/03_export_visual.sh"
