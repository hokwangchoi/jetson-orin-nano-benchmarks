#!/usr/bin/env bash
# Launch vLLM serving Cosmos-Reason2-2B-W4A16 on Jetson Orin Nano.
# Reproduces Embedl's published reference command verbatim, with the
# minimal additions needed for HF auth and local model caching.
#
# Reference: https://huggingface.co/embedl/Cosmos-Reason2-2B-W4A16
#
# Pre-launch hygiene (run once before this script, from another shell):
#   sudo systemctl isolate multi-user.target   # kill desktop
#   sudo swapoff -a
#   sudo sysctl -w vm.drop_caches=3
#   sudo nvpmodel -m 2                         # MAXN_SUPER
#   sudo jetson_clocks
#
# The --gpu-memory-utilization 0.75 in Embedl's reference needs a headless
# system; with the desktop running you won't have enough free memory.

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
# HF IDs look like "org/name"; local paths start with "/".
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

# Warn if likely OOM. 0.75 GPU util on an 8GB unified-memory Orin Nano
# needs the desktop killed; otherwise vLLM will abort at startup.
if command -v free >/dev/null 2>&1; then
    FREE_MB=$(free -m | awk '/^Mem:/{print $7}')
    if [[ "${FREE_MB:-0}" -lt 5000 ]]; then
        echo "warning: only ${FREE_MB} MB available memory." >&2
        echo "         Embedl's reference setup needs >= 5 GB free." >&2
        echo "         Run: sudo systemctl isolate multi-user.target && sudo swapoff -a" >&2
        echo "         then SSH in from another machine and re-run this script." >&2
        echo "" >&2
    fi
fi

# Remove stale container from previous runs
sudo docker rm -f vllm-serve 2>/dev/null || true

echo "[vllm] launching server (Embedl reference config)"
echo "[vllm]   model:            $MODEL_ARG"
echo "[vllm]   image:            $VLLM_IMAGE"
echo "[vllm]   max-model-len:    $MAX_MODEL_LEN"
echo "[vllm]   gpu-mem-util:     $GPU_MEM_UTIL"
echo "[vllm]   max-num-seqs:     $MAX_NUM_SEQS"
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
        --max-num-seqs "$MAX_NUM_SEQS"
