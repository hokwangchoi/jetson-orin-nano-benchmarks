#!/usr/bin/env bash
# nsight_vllm.sh — capture one vLLM image inference by attaching nsys to the
# running vLLM engine process, after 2 image warmup requests.
#
# Prerequisite: vLLM server is running (./device/scripts/03_run_vllm_server.sh
# in another terminal) and has completed startup. Script finds the engine PID
# on the host namespace (vLLM runs in docker but the engine process is
# visible via pgrep from the host).
#
# Strategy: 2 warmup image requests → nsys attaches to engine PID with a
# duration window → 1 captured image request → nsys exits when window ends.
# The warmup triggers vLLM's lazy ViT CUDA graph compilation so the captured
# trace shows steady-state inference, matching the benchmark numbers.
#
# Output: assets/results/nsight/vllm_image_<stamp>.nsys-rep

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT_DIR="${REPO_ROOT}/assets/results/nsight"
mkdir -p "$OUT_DIR"

SERVER_URL="${SERVER_URL:-http://localhost:8000}"
MODEL="${MODEL:-embedl/Cosmos-Reason2-2B-W4A16}"
IMAGE_PATH="${IMAGE_PATH:-$REPO_ROOT/assets/images/bus.jpg}"
PROMPT="${PROMPT:-What do you see in this image?}"
MAX_TOKENS="${MAX_TOKENS:-64}"
CAPTURE_DURATION_S="${CAPTURE_DURATION_S:-8}"

if ! command -v nsys &>/dev/null; then
    echo "error: nsys not found." >&2
    exit 1
fi

# Find the engine PID — vLLM spawns VllmWorker / EngineCore processes
ENGINE_PID="$(pgrep -f 'EngineCore|VllmWorker' | head -1)"
if [ -z "$ENGINE_PID" ]; then
    echo "error: no vLLM engine process found. is the server running?" >&2
    exit 1
fi
echo "=== Attaching to vLLM engine PID $ENGINE_PID ==="

# Warmup: same image + similar prompt as the captured request. This triggers
# vLLM's lazy ViT CUDA graph capture (cudagraph_mm_encoder: False means the
# graph is built on first image per unique input shape — ~500ms–3s spike on
# first run). Two warmup requests handle both the lazy compile and any JIT
# path that fires on second-request. Without this, the captured trace would
# be dominated by a one-time compilation spike instead of showing the
# steady-state inference we care about.
echo "=== Sending 2 warmup image requests (not captured) ==="
IMG_B64_WARMUP="$(base64 -w0 "$IMAGE_PATH")"
for i in 1 2; do
    echo "  warmup $i/2..."
    curl -s -X POST "$SERVER_URL/v1/chat/completions" \
         -H "Content-Type: application/json" \
         -d "{
           \"model\":\"$MODEL\",
           \"max_tokens\":16,
           \"temperature\":0.0,
           \"messages\":[{\"role\":\"user\",\"content\":[
             {\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/jpeg;base64,$IMG_B64_WARMUP\"}},
             {\"type\":\"text\",\"text\":\"warmup\"}
           ]}]
         }" >/dev/null
done

STAMP="$(date +%Y%m%d_%H%M%S)"
REPORT="$OUT_DIR/vllm_image_${STAMP}"

echo ""
echo "=== Starting nsys capture (will run ${CAPTURE_DURATION_S}s) ==="
# nsys attach: capture from now until we stop it
nsys profile \
    --trace=cuda,nvtx,osrt \
    --sample=none \
    --cpuctxsw=none \
    --output="$REPORT" \
    --force-overwrite=true \
    --duration="$CAPTURE_DURATION_S" \
    --attach="$ENGINE_PID" &
NSYS_PID=$!

# Give nsys 2s to initialize before firing the request
sleep 2

echo "=== Firing image request ==="
# Base64-encode image (vLLM expects data URL)
IMG_B64="$(base64 -w0 "$IMAGE_PATH")"
curl -s -X POST "$SERVER_URL/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -d "{
       \"model\":\"$MODEL\",
       \"max_tokens\":$MAX_TOKENS,
       \"temperature\":0.0,
       \"messages\":[{\"role\":\"user\",\"content\":[
         {\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/jpeg;base64,$IMG_B64\"}},
         {\"type\":\"text\",\"text\":\"$PROMPT\"}
       ]}]
     }" \
     | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'][:200])"

# Wait for nsys to finish the capture window
wait $NSYS_PID

echo ""
echo "=== Done ==="
echo "Report: $REPORT.nsys-rep"
