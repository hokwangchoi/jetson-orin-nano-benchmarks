#!/usr/bin/env bash
# 11_run_llamacpp_server.sh — launch llama.cpp's OpenAI-compatible server
# with the Cosmos-Reason2-2B Q4_K_M model + vision projector.
#
# Prerequisite: 10_prepare_llamacpp.sh has been run at least once so
# $MODEL_DIR contains the quantized GGUF and mmproj GGUF.

set -euo pipefail

CONFIG="${CONFIG:-$(dirname "$0")/../configs/llamacpp_server.env}"
[ -f "$CONFIG" ] && { echo "[llama.cpp] loading $CONFIG"; set -a; source "$CONFIG"; set +a; }

# Defaults
MODEL_DIR="${MODEL_DIR:-$HOME/models/cosmos-reason2-2b}"
MODEL_FILE="${MODEL_FILE:-Cosmos-Reason2-2B-Q4_K_M.gguf}"
MMPROJ_FILE="${MMPROJ_FILE:-mmproj-Cosmos-Reason2-2B-BF16.gguf}"
SERVED_NAME="${SERVED_NAME:-Cosmos-Reason2-2B-Q4_K_M}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
CTX_SIZE="${CTX_SIZE:-4096}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
N_PARALLEL="${N_PARALLEL:-1}"
IMAGE="${IMAGE:-ghcr.io/nvidia-ai-iot/llama_cpp:latest-jetson-orin}"

# Verify prep ran
for f in "$MODEL_FILE" "$MMPROJ_FILE"; do
    if [ ! -s "$MODEL_DIR/$f" ]; then
        echo "[llama.cpp] missing $MODEL_DIR/$f — run 10_prepare_llamacpp.sh first"
        exit 1
    fi
done

# Free page cache before load
sudo sysctl -w vm.drop_caches=3 >/dev/null

echo "[llama.cpp] serving $SERVED_NAME from $MODEL_DIR"
echo "            ctx=$CTX_SIZE ngl=$N_GPU_LAYERS parallel=$N_PARALLEL"
echo "            listen on :$PORT"

sudo docker run --rm -it \
    --name llamacpp-serve \
    --network host \
    --runtime=nvidia \
    -v "$MODEL_DIR":/models:ro \
    "$IMAGE" \
    llama-server \
        --model       "/models/$MODEL_FILE" \
        --mmproj      "/models/$MMPROJ_FILE" \
        --alias       "$SERVED_NAME" \
        --host        "$HOST" \
        --port        "$PORT" \
        --ctx-size    "$CTX_SIZE" \
        --n-gpu-layers "$N_GPU_LAYERS" \
        --parallel    "$N_PARALLEL" \
        --flash-attn  on \
        --jinja
