#!/usr/bin/env bash
# 20_run_vllm_cosmos.sh — launch vLLM serving Cosmos-Reason2-2B on
# Orin Nano 8GB. Uses Embedl's W4A16 checkpoint because the raw
# nvidia/Cosmos-Reason2-2B BF16 model does not fit: 5 GB of weights
# plus PyTorch allocator overhead exceeds 8 GB.
#
# Requires JetPack 6.2.2+ (L4T 36.5.0+). On 6.2.1 vLLM crashes with
# an NVML assertion even at startup.
#
# Serve settings match NVIDIA's documented Orin Super Nano recipe:
# https://huggingface.co/blog/nvidia/cosmos-on-jetson (Option C)
# with model substitution (Embedl W4A16 vs NVIDIA FP8 from NGC).

set -euo pipefail

CONFIG="${CONFIG:-$(dirname "$0")/../configs/vllm_cosmos.env}"
[ -f "$CONFIG" ] && { echo "[vllm] loading $CONFIG"; set -a; source "$CONFIG"; set +a; }

# Defaults — proven-working values on Orin Nano 8GB, JetPack 6.2.2
MODEL_ID="${MODEL_ID:-embedl/Cosmos-Reason2-2B-W4A16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-256}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-256}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.6}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_PIXELS="${MAX_PIXELS:-150528}"  # ~388x388 cap on ViT image input
NUM_FRAMES="${NUM_FRAMES:-2}"       # video frame cap
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
IMAGE="${IMAGE:-ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"

# HF_TOKEN optional — Embedl's model is public. Still pass if set so the
# container can reach other gated models without re-login.
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HF_CACHE/token" ]; then
    HF_TOKEN="$(cat "$HF_CACHE/token")"
fi

# Verify JetPack / L4T revision
rev="$(head -n1 /etc/nv_tegra_release | sed -n 's/.*REVISION: \([0-9.]*\).*/\1/p')"
case "$rev" in
    4.7|4.6|4.5|4.4|4.3)
        echo "[vllm] L4T $rev detected — vLLM is known to fail on this revision"
        echo "[vllm] see device/README.md for JetPack upgrade instructions"
        exit 1
        ;;
    5.0|5.*|6.*)
        echo "[vllm] L4T $rev — supported"
        ;;
    *)
        echo "[vllm] L4T $rev — unknown, proceeding"
        ;;
esac

# Free page cache before load
sudo sysctl -w vm.drop_caches=3 >/dev/null
echo "[vllm] memory before serve:"
free -h | awk 'NR<=2'

echo
echo "[vllm] serving $MODEL_ID on :$PORT"
echo "       max-model-len=$MAX_MODEL_LEN gpu-memory-utilization=$GPU_MEM_UTIL"
echo "       max-pixels=$MAX_PIXELS num-frames=$NUM_FRAMES"

sudo docker run --rm -it \
    --name vllm-serve \
    --network host \
    --shm-size=8g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --runtime=nvidia \
    -v "$HF_CACHE":/root/.cache/huggingface \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e HF_HOME=/root/.cache/huggingface \
    "$IMAGE" \
    vllm serve "$MODEL_ID" \
        --host "$HOST" \
        --port "$PORT" \
        --trust-remote-code \
        --enforce-eager \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --enable-chunked-prefill \
        --limit-mm-per-prompt "{\"image\":1,\"video\":1}" \
        --mm-processor-kwargs "{\"num_frames\":$NUM_FRAMES,\"max_pixels\":$MAX_PIXELS}"
