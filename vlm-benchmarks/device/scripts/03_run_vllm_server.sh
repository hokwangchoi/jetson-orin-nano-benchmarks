#!/usr/bin/env bash
# 03_run_vllm_server.sh — launch vLLM Jetson container serving the
# W4A16 Cosmos-Reason2-2B checkpoint on :8000.
#
# Loads config from ../configs/vllm_server.env if present; otherwise uses
# Orin Nano 8 GB defaults.

set -euo pipefail

CONFIG="${CONFIG:-$(dirname "$0")/../configs/vllm_server.env}"
[ -f "$CONFIG" ] && { echo "[vllm] loading $CONFIG"; set -a; source "$CONFIG"; set +a; }

# Defaults (Orin Nano 8 GB safe)
MODEL_PATH="${MODEL_PATH:-$HOME/vlm-artifacts/vllm/Cosmos-Reason2-2B-W4A16}"
SERVED_NAME="${SERVED_NAME:-Cosmos-Reason2-2B-W4A16}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.75}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
IMAGE="${IMAGE:-ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin}"

[ -d "$MODEL_PATH" ] || { echo "missing model at $MODEL_PATH — run Phase 0 first"; exit 1; }

# Free page cache before load
sudo sysctl -w vm.drop_caches=3 >/dev/null

echo "[vllm] starting server:"
echo "       model=$MODEL_PATH"
echo "       max_model_len=$MAX_MODEL_LEN gpu_mem_util=$GPU_MEM_UTIL max_num_seqs=$MAX_NUM_SEQS"
echo "       listen on :$PORT"

sudo docker run --rm -it \
    --name vllm-serve \
    --network host \
    --shm-size=8g \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --runtime=nvidia \
    -v "$MODEL_PATH":/models/served:ro \
    -v "$HOME/.cache/vllm":/root/.cache/vllm \
    -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
    "$IMAGE" \
    vllm serve /models/served \
        --served-model-name "$SERVED_NAME" \
        --max-model-len      "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --max-num-seqs       "$MAX_NUM_SEQS" \
        --reasoning-parser qwen3 \
        --media-io-kwargs '{"video": {"num_frames": -1}}' \
        --enable-prefix-caching \
        --port "$PORT"
