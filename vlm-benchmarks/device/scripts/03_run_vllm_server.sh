#!/usr/bin/env bash
# Launch vLLM serving Cosmos-Reason2-2B-W4A16 on Jetson Orin Nano.
# Based on Embedl's published reference command, with two Orin-Nano-specific
# additions (--mm-processor-kwargs, --limit-mm-per-prompt) that are required
# for the server to start on 8 GB unified memory; see vllm_server.env.
#
# Reference: https://huggingface.co/embedl/Cosmos-Reason2-2B-W4A16
#
# Pre-launch hygiene (run once before this script, from another shell):
#   sudo systemctl isolate multi-user.target   # kill desktop
#   sudo swapoff -a
#   sudo sysctl -w vm.drop_caches=3
#   sudo nvpmodel -m 2                         # MAXN_SUPER
#   sudo jetson_clocks

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/../configs/vllm_server.env"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "error: config file not found at $CONFIG_FILE" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "$CONFIG_FILE"

# Auto-detect HF model ID vs absolute local path.
if [[ "$MODEL" == /* && -d "$MODEL" ]]; then
    MODEL_MOUNT=(-v "$MODEL:/model")
    MODEL_ARG="/model"
else
    MODEL_MOUNT=()
    MODEL_ARG="$MODEL"
fi

# Pick up HF token from the standard cache location if present.
HF_TOKEN_VALUE=""
if [[ -f "${HOME}/.cache/huggingface/token" ]]; then
    HF_TOKEN_VALUE="$(cat "${HOME}/.cache/huggingface/token")"
fi

# Warn if likely OOM.
if command -v free >/dev/null 2>&1; then
    FREE_MB=$(free -m | awk '/^Mem:/{print $7}')
    if [[ "${FREE_MB:-0}" -lt 5000 ]]; then
        echo "warning: only ${FREE_MB} MB available memory." >&2
        echo "         Run: sudo systemctl isolate multi-user.target && sudo swapoff -a" >&2
        echo "         then SSH in from another machine and re-run this script." >&2
        echo "" >&2
    fi
fi

# Remove stale container from previous runs
sudo docker rm -f vllm-serve 2>/dev/null || true

echo "[vllm] launching server"
echo "[vllm]   model:              $MODEL_ARG"
echo "[vllm]   image:              $VLLM_IMAGE"
echo "[vllm]   max-model-len:      $MAX_MODEL_LEN"
echo "[vllm]   gpu-mem-util:       $GPU_MEM_UTIL"
echo "[vllm]   max-num-seqs:       $MAX_NUM_SEQS"
echo "[vllm]   mm-processor-kwargs: $MM_PROCESSOR_KWARGS"
echo "[vllm]   limit-mm-per-prompt: $LIMIT_MM_PER_PROMPT"
echo ""

exec sudo docker run --rm --name vllm-serve \
    --network host \
    --shm-size=8g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --runtime=nvidia \
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
    -e "HF_TOKEN=${HF_TOKEN_VALUE}" \
    -e "HF_HOME=/root/.cache/huggingface" \
    "${MODEL_MOUNT[@]}" \
    "$VLLM_IMAGE" \
    vllm serve "$MODEL_ARG" \
        --host "$HOST" --port "$PORT" \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --mm-processor-kwargs "$MM_PROCESSOR_KWARGS" \
        --limit-mm-per-prompt "$LIMIT_MM_PER_PROMPT"
