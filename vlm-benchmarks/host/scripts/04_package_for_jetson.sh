#!/usr/bin/env bash
# 04_package_for_jetson.sh — tar the ONNX artifacts and scp to Jetson.
#
# Creates cosmos-2b-onnx.tar.gz (~2 GB) with both LLM and visual ONNX
# outputs and transfers to the Jetson's TRT-Edge-LLM workspace.
#
# Set JETSON_HOST and JETSON_USER before running, or override per-call:
#   JETSON_HOST=jetson.local JETSON_USER=hc ./scripts/04_package_for_jetson.sh

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
ONNX_DIR="${ONNX_DIR:-${WORKSPACE}/cosmos-2b-onnx}"
TARBALL="${TARBALL:-${WORKSPACE}/cosmos-2b-onnx.tar.gz}"
JETSON_HOST="${JETSON_HOST:-}"
JETSON_USER="${JETSON_USER:-${USER}}"
JETSON_DEST="${JETSON_DEST:-~/tensorrt-edgellm-workspace/Cosmos-Reason2-2B/}"

if [ ! -d "${ONNX_DIR}/llm" ] || [ ! -d "${ONNX_DIR}/visual" ]; then
    echo "[package] expected ${ONNX_DIR}/{llm,visual}/ — run 02 and 03 first."
    exit 1
fi

echo "[package] creating tarball: ${TARBALL}"
tar -czf "${TARBALL}" -C "$(dirname "${ONNX_DIR}")" "$(basename "${ONNX_DIR}")"

SIZE_MB=$(du -m "${TARBALL}" | cut -f1)
echo "[package] tarball size: ${SIZE_MB} MB"
echo ""

if [ -z "${JETSON_HOST}" ]; then
    echo "[package] JETSON_HOST not set — skipping scp."
    echo "[package] transfer manually with:"
    echo "          scp ${TARBALL} <user>@<jetson>:${JETSON_DEST}"
    exit 0
fi

echo "[package] scp ${TARBALL} ${JETSON_USER}@${JETSON_HOST}:${JETSON_DEST}"
scp "${TARBALL}" "${JETSON_USER}@${JETSON_HOST}:${JETSON_DEST}"

echo ""
echo "[package] done. On the Jetson, untar and proceed to"
echo "          device/trt_cosmos_patches/split_lm_head.py (see §9)"
echo "          then device/scripts/40_build_cosmos_trt_engines.sh."
