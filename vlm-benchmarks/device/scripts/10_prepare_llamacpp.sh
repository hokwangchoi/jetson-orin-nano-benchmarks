#!/usr/bin/env bash
# 10_prepare_llamacpp.sh — one-time setup for the llama.cpp runtime path.
#
# Downloads robertzty/Cosmos-Reason2-2B-GGUF (BF16 split + mmproj), merges
# the split main model, and quantizes it to Q4_K_M (~1.1 GB). Takes ~10 min.
# Safe to rerun — skips steps whose output files already exist.
#
# Output layout (host):
#   $HOME/models/cosmos-reason2-2b/
#     Cosmos-Reason2-2B-BF16-split-00001-of-00002.gguf
#     Cosmos-Reason2-2B-BF16-split-00002-of-00002.gguf
#     mmproj-Cosmos-Reason2-2B-BF16.gguf
#     Cosmos-Reason2-2B-BF16.gguf        (merged, deletable after quantize)
#     Cosmos-Reason2-2B-Q4_K_M.gguf      (what the server loads)

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-$HOME/models/cosmos-reason2-2b}"
IMAGE="${IMAGE:-ghcr.io/nvidia-ai-iot/llama_cpp:latest-jetson-orin}"
HF_REPO="robertzty/Cosmos-Reason2-2B-GGUF"

mkdir -p "$MODEL_DIR"
cd "$MODEL_DIR"

# Files to fetch (all public; HF_TOKEN only used if set)
FILES=(
    "Cosmos-Reason2-2B-BF16-split-00001-of-00002.gguf"
    "Cosmos-Reason2-2B-BF16-split-00002-of-00002.gguf"
    "mmproj-Cosmos-Reason2-2B-BF16.gguf"
)

echo "[prep] downloading to $MODEL_DIR (3 files, ~4.6 GB total)"
for f in "${FILES[@]}"; do
    if [ -s "$f" ]; then
        echo "[prep] $f already present, skipping"
        continue
    fi
    url="https://huggingface.co/${HF_REPO}/resolve/main/${f}"
    auth_args=()
    [ -n "${HF_TOKEN:-}" ] && auth_args=(--header="Authorization: Bearer $HF_TOKEN")
    wget -c "${auth_args[@]}" -O "$f.tmp" "$url"
    mv "$f.tmp" "$f"
done

# Merge split files into a single BF16 GGUF
MERGED="Cosmos-Reason2-2B-BF16.gguf"
if [ -s "$MERGED" ]; then
    echo "[prep] merged file already present, skipping"
else
    echo "[prep] merging split files → $MERGED"
    sudo docker run --rm -v "$MODEL_DIR":/models "$IMAGE" \
        llama-gguf-split --merge \
            /models/Cosmos-Reason2-2B-BF16-split-00001-of-00002.gguf \
            /models/"$MERGED"
fi

# Quantize BF16 → Q4_K_M (~1.1 GB target)
Q4="Cosmos-Reason2-2B-Q4_K_M.gguf"
if [ -s "$Q4" ]; then
    echo "[prep] quantized file already present, skipping"
else
    echo "[prep] quantizing BF16 → Q4_K_M (takes ~5-10 min on Orin Nano)"
    sudo docker run --rm -v "$MODEL_DIR":/models "$IMAGE" \
        llama-quantize /models/"$MERGED" /models/"$Q4" Q4_K_M
fi

# Fix ownership (docker wrote as root)
sudo chown -R "$USER:$USER" "$MODEL_DIR"

echo
echo "[prep] done. artifacts in $MODEL_DIR:"
ls -lh "$MODEL_DIR"
echo
echo "[prep] next: ./vlm-benchmarks/device/scripts/11_run_llamacpp_server.sh"
