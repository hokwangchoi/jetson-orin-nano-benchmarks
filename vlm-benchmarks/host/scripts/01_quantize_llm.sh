#!/usr/bin/env bash
# 01_quantize_llm.sh — AWQ W4A16 quantization of Cosmos-Reason2-2B.
#
# Runs on an x86 host with a datacenter GPU (A40/L40S/A100/H100).
# ~4 minutes on A40. Output is a quantized checkpoint dir that feeds
# 02_export_llm.sh.
#
# Reconstruction note: see host/README.md. If you have the original
# script from when Phase 0 was first done, use that — this is a
# scaffold, not a historical artifact.

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
MODEL="${MODEL:-nvidia/Cosmos-Reason2-2B}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE}/cosmos-2b-quantized}"
CALIB_SIZE="${CALIB_SIZE:-512}"
CALIB_DATASET="${CALIB_DATASET:-cnn_dailymail}"

if [ -f "${OUTPUT_DIR}/config.json" ]; then
    echo "[quantize] output exists at ${OUTPUT_DIR}, skipping."
    echo "[quantize] remove the directory to force rerun."
    exit 0
fi

mkdir -p "${OUTPUT_DIR}"

# TRT-Edge-LLM ships its quantizer under examples/llm/. If you installed
# the host tools with `pip install -e ./python`, the module path below
# should resolve. If it doesn't, fall back to invoking the example
# script directly (see the TRT-Edge-LLM repo).
echo "[quantize] model:   ${MODEL}"
echo "[quantize] output:  ${OUTPUT_DIR}"
echo "[quantize] calib:   ${CALIB_DATASET} (${CALIB_SIZE} samples)"
echo ""

python -m tensorrt_edge_llm.examples.llm.quantize_llm \
    --model_path     "${MODEL}" \
    --quant_format   int4_awq \
    --calib_dataset  "${CALIB_DATASET}" \
    --calib_size     "${CALIB_SIZE}" \
    --output_dir     "${OUTPUT_DIR}"

echo ""
echo "[quantize] done. Next: ./scripts/02_export_llm.sh"
