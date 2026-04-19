#!/usr/bin/env bash
# nsight_llamacpp.sh — capture one llama.cpp image inference by attaching nsys
# to the running llama-server process, after 2 image warmup requests.
#
# Prerequisite: llama.cpp server is running (./device/scripts/11_run_llamacpp_server.sh
# in another terminal).
#
# Same shape as nsight_vllm.sh: 2 warmup image requests (discarded) → nsys
# attaches to server PID with a duration window → 1 captured image request →
# nsys exits. The warmup drains mmproj/vision-encoder compilation so the
# captured trace shows steady-state decode.
#
# Output: assets/results/nsight/llamacpp_image_<stamp>.nsys-rep

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT_DIR="${REPO_ROOT}/assets/results/nsight"
mkdir -p "$OUT_DIR"

SERVER_URL="${SERVER_URL:-http://localhost:8000}"
MODEL="${MODEL:-cosmos-reason2-2b}"
IMAGE_PATH="${IMAGE_PATH:-$REPO_ROOT/assets/images/bus.jpg}"
PROMPT="${PROMPT:-What do you see in this image?}"
MAX_TOKENS="${MAX_TOKENS:-64}"
CAPTURE_DURATION_S="${CAPTURE_DURATION_S:-8}"

if ! command -v nsys &>/dev/null; then
    echo "error: nsys not found." >&2
    exit 1
fi

SERVER_PID="$(pgrep -f 'llama-server|llama_server' | head -1)"
if [ -z "$SERVER_PID" ]; then
    echo "error: no llama-server process found. is the server running?" >&2
    exit 1
fi
echo "=== Attaching to llama-server PID $SERVER_PID ==="

# Build JSON payloads once as temp files — avoids ARG_MAX limits from
# stuffing ~480KB of base64 image data into a single curl -d argument
# (base64 of a 358KB JPEG blows past the shell command line limit).
PAYLOAD_DIR="$(mktemp -d)"
trap 'rm -rf "$PAYLOAD_DIR"' EXIT

python3 - "$IMAGE_PATH" "$MODEL" "$PROMPT" "$MAX_TOKENS" "$PAYLOAD_DIR" <<'PY'
import base64, json, sys, os
image_path, model, prompt, max_tokens, out_dir = sys.argv[1:]
with open(image_path, 'rb') as f:
    b64 = base64.b64encode(f.read()).decode('ascii')

def req(prompt_text, tokens):
    return {
        "model": model,
        "max_tokens": int(tokens),
        "temperature": 0.0,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": prompt_text}
        ]}]
    }

with open(os.path.join(out_dir, 'warmup.json'), 'w') as f:
    json.dump(req("warmup", 16), f)
with open(os.path.join(out_dir, 'capture.json'), 'w') as f:
    json.dump(req(prompt, max_tokens), f)
PY

# Warmup: image request triggers mmproj/vision-encoder path on first run.
# Run 1 of the fresh-boot benchmark showed 2287 ms TTFT vs 114 ms for runs
# 2-5 — that 2 second spike is the mmproj backend allocating and compiling
# kernels on the vision side. Two warmup requests drain that overhead so the
# captured trace shows steady-state inference, matching the published numbers.
echo "=== Sending 2 warmup image requests (not captured) ==="
for i in 1 2; do
    echo "  warmup $i/2..."
    curl -s -X POST "$SERVER_URL/v1/chat/completions" \
         -H "Content-Type: application/json" \
         -d "@${PAYLOAD_DIR}/warmup.json" >/dev/null
done

STAMP="$(date +%Y%m%d_%H%M%S)"
REPORT="$OUT_DIR/llamacpp_image_${STAMP}"

echo ""
echo "=== Starting nsys capture (will run ${CAPTURE_DURATION_S}s) ==="
nsys profile \
    --trace=cuda,nvtx,osrt \
    --sample=none \
    --cpuctxsw=none \
    --output="$REPORT" \
    --force-overwrite=true \
    --duration="$CAPTURE_DURATION_S" \
    --attach="$SERVER_PID" &
NSYS_PID=$!

sleep 2

echo "=== Firing image request ==="
curl -s -X POST "$SERVER_URL/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -d "@${PAYLOAD_DIR}/capture.json" \
     | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'][:200])"

wait $NSYS_PID

echo ""
echo "=== Done ==="
echo "Report: $REPORT.nsys-rep"
