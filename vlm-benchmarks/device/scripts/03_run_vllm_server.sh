#!/usr/bin/env bash
# 03_run_vllm_server.sh — launch vLLM Jetson container serving a W4A16
# VLM checkpoint on :8000.
#
# MODEL_PATH in the config may be either a Hugging Face model ID
# (e.g. embedl/Cosmos-Reason2-2B-W4A16) or a local directory. This
# script detects which and mounts / passes env accordingly.

set -euo pipefail

CONFIG="${CONFIG:-$(dirname "$0")/../configs/vllm_server.env}"
[ -f "$CONFIG" ] && { echo "[vllm] loading $CONFIG"; set -a; source "$CONFIG"; set +a; }

# Defaults (Orin Nano 8 GB safe)
MODEL_PATH="${MODEL_PATH:-embedl/Cosmos-Reason2-2B-W4A16}"
SERVED_NAME="${SERVED_NAME:-Cosmos-Reason2-2B-W4A16}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.6}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
IMAGE="${IMAGE:-ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin}"

# Detect HF ID vs local path. HF IDs look like "org/model", local paths
# start with / or ~ or . and exist on disk.
EXPANDED_PATH="${MODEL_PATH/#\~/$HOME}"
if [ -d "$EXPANDED_PATH" ]; then
    MODE="local"
    SERVE_ARG="/models/served"
    MOUNT_ARGS=(-v "$EXPANDED_PATH":/models/served:ro)
elif [[ "$MODEL_PATH" =~ ^[^/]+/[^/]+$ ]]; then
    MODE="hf"
    SERVE_ARG="$MODEL_PATH"
    MOUNT_ARGS=()
    : "${HF_TOKEN:?HF_TOKEN not set — accept the upstream license at https://huggingface.co/nvidia/Cosmos-Reason2-2B and export HF_TOKEN}"
else
    echo "[vllm] MODEL_PATH='$MODEL_PATH' is neither a local directory nor 'org/name'"
    exit 1
fi

# Free page cache before load
sudo sysctl -w vm.drop_caches=3 >/dev/null

echo "[vllm] mode=$MODE  model=$MODEL_PATH"
echo "       max_model_len=$MAX_MODEL_LEN gpu_mem_util=$GPU_MEM_UTIL max_num_seqs=$MAX_NUM_SEQS"
echo "       listen on :$PORT"

sudo docker run --rm -it \
    --name vllm-serve \
    --network host \
    --shm-size=8g \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --runtime=nvidia \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e HF_HOME=/root/.cache/huggingface \
    -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
    -v "$HOME/.cache/vllm":/root/.cache/vllm \
    "${MOUNT_ARGS[@]}" \
    "$IMAGE" \
    vllm serve "$SERVE_ARG" \
        --served-model-name "$SERVED_NAME" \
        --max-model-len      "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --max-num-seqs       "$MAX_NUM_SEQS" \
        --reasoning-parser qwen3 \
        --media-io-kwargs '{"video": {"num_frames": -1}}' \
        --enable-prefix-caching \
        --port "$PORT"
