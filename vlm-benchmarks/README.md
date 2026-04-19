# VLM Benchmarks on Jetson Orin Nano

Running a 2B-parameter vision-language model on 8 GB of unified memory
— and measuring what really matters when you take it off the data-center
rack and put it on an edge device.

The short version: **three inference runtimes on the same model, same
hardware, all three working**. Getting vLLM serving required a
community-quantized W4A16 checkpoint and a vision-encoder profile cap.
Getting TRT-Edge-LLM building took more: a kernel-level CMA change, an
ONNX graph rewrite to split the LM head MatMul, and careful CMA-pressure
watching during the build. As far as I can tell this is the first public
report of Cosmos-Reason2-2B running on TRT-Edge-LLM on an Orin Nano —
NVIDIA's own supported matrix pairs this combination with Thor, not
Orin. See the [blog post](./index.html) for the full story.

## Why VLMs on the edge?

Perception on a robot or autonomous vehicle used to mean a stack of
narrow models: YOLO for boxes, DeepLab for segmentation, SORT for tracks,
maybe a Transformer trajectory predictor. Each answered one kind of
question. None could tell you *why* the scene looks the way it does.

Vision-Language Models collapse that stack. A single VLM can caption a
frame, answer a grounded question about it, reason about object
relationships over time, and return structured outputs. The promise for
physical AI is obvious:

- **Autonomous driving perception**: "Is it safe to turn right?" with
  evidence from the scene. Failure-case understanding for offline data
  curation and active learning.
- **Robotics planning**: VLA (vision-language-action) models use a VLM
  as the "brain" that decomposes a natural-language goal into motor
  primitives.
- **Safety monitoring**: catching out-of-distribution scenes a narrow
  classifier would miss — construction zones, debris, novel agents.
- **Teleoperation fallback**: when the autonomy stack disengages, a VLM
  can describe the state of the world to a remote operator in one round
  trip.

Almost every VLM demo runs on a pair of H100s. The actual edge platforms
that ship on robots and cars — Jetson Orin Nano, Orin NX, DRIVE Orin —
have 8 to 16 GB of memory shared between CPU and GPU, 20 to 100 TOPS of
compute, and power budgets under 30 W. What does it take to close that
gap?

## The model

[`nvidia/Cosmos-Reason2-2B`](https://huggingface.co/nvidia/Cosmos-Reason2-2B)
— NVIDIA's 2B physical-AI reasoning VLM, post-trained from Qwen3-VL-2B
with SFT and RL on embodied-reasoning datasets. Architecturally it is
a Qwen3-VL stack: a ViT vision encoder, an MLP projector, and a dense
Transformer decoder. It understands spatial relationships, temporal
ordering, and basic physics, and outputs chain-of-thought reasoning in
`<think>...</think><answer>...</answer>` format.

llama.cpp runs the raw HuggingFace checkpoint via Q4_K_M on-device
quantization. vLLM needs a pre-quantized checkpoint because the raw
BF16 model doesn't fit on 8 GB; we use
[`embedl/Cosmos-Reason2-2B-W4A16`](https://huggingface.co/embedl/Cosmos-Reason2-2B-W4A16),
a community port that's about 1.1 GB on disk.

## The three runtimes

| Aspect                | llama.cpp                        | vLLM                             | TensorRT Edge-LLM                |
|-----------------------|----------------------------------|----------------------------------|----------------------------------|
| Language              | C (ggml core), C++ wrapper       | Python + PyTorch + CUDA kernels  | C++ runtime, zero Python in path |
| Kernels               | Hand-tuned ggml-cuda             | FlashAttention + Marlin W4A16 GEMM | TensorRT fused + hand-tuned      |
| Graph execution       | Interpretive (op-by-op)          | CUDA graphs (decode + mixed prefill/decode, batch 1–2) | CUDA graph captured at build     |
| Batching              | Slot pool (`--parallel N`)       | PagedAttention + chunked prefill | Static batch size at build time  |
| KV cache              | Contiguous per slot              | Paged, virtualized               | Preallocated contiguous          |
| Cold start            | ~5–10 s (mmap GGUF)              | ~2–3 min (weight load + warmup)  | ~5–10 s (engine load)            |
| Model used here       | Cosmos-Reason2-2B Q4_K_M         | Cosmos-Reason2-2B W4A16 (Embedl) | Cosmos-Reason2-2B W4A16 (split LM head — see notes) |

Each end-point answers the same benchmark questions, but the runtimes
come from very different design philosophies. llama.cpp is the generic,
hackable local runtime. vLLM is the production serving engine with
PagedAttention and continuous batching. TRT Edge-LLM is the embedded,
single-app C++ runtime for automotive and robotics.

## The L4T 36.4.7 blocker

Halfway into the work, I hit an NvMap memory-allocation bug in
JetPack 6.2.1 (L4T 36.4.7) that blocks CUDA workloads asking for large
contiguous allocations — see the blog post for the full forensics.

**The same bug surfaces differently on each runtime:**

- **llama.cpp** — unaffected. Its small-chunk allocation pattern
  sidesteps the problem entirely.
- **vLLM** — PyTorch's CUDA caching allocator asserts on an NVML
  sanity check. Process dies on startup in 6.2.1; in 6.2.2 a variant
  still triggers if you load the raw BF16 model (hits the bug during
  multimodal profiling).
- **TRT Edge-LLM** — the Myelin autotuner's 1 GB scratch-buffer request
  during engine build gets rejected, with the exact error signature
  `NvMapMemAllocInternalTagged: 1075072515 error 12`. Node turned out
  to be the LM head output projection (vocab=151936), not RoPE as
  initially diagnosed. Not fixed by 6.2.2 — required a separate
  workaround (see resolution notes).

JetPack 6.2.2 / L4T 36.5.0 resolves the startup-level path. vLLM now
serves Cosmos-2B if you use a pre-quantized W4A16 checkpoint and cap
the vision encoder profile. TRT Edge-LLM needs additional kernel-level
work (CMA pool expansion) plus an ONNX graph rewrite to split the LM
head — both documented in `device/trt_cosmos_patches/README.md`.

## Current runtime status

| Runtime | Model | Status |
|---------|-------|--------|
| llama.cpp | Cosmos-Reason2-2B Q4_K_M | ✅ serving, benchmarked (TPS = 38) |
| vLLM | Cosmos-Reason2-2B W4A16 (Embedl) | ✅ serving, benchmarked (TPS = 56) |
| TRT Edge-LLM | Cosmos-Reason2-2B W4A16 | ✅ engine built + benchmarked (TPS = 60), after CMA config + ONNX graph surgery — see [`notes/trt_edgellm_cosmos_resolution.md`](./notes/trt_edgellm_cosmos_resolution.md) |

Asymmetry: llama.cpp runs at context 4096, vLLM and TRT Edge-LLM at
context 1024 (the fit-in-memory ceiling on 8 GB). TRT Edge-LLM's path
also required kernel-level setup that isn't in NVIDIA's standard
tutorial — see the runtime's README in `device/trt_cosmos_patches/`.

## What each runtime measures

- **Latency**: TTFT (time to first token — includes ViT + prefill),
  TPOT (time per output token), TPS (tokens/sec).
- **Throughput under concurrency**: 1, 2, 4 concurrent requests via
  llama.cpp's slot pool. TRT Edge-LLM is single-stream. vLLM supports
  continuous batching but KV cache headroom on 8 GB limits
  `max-num-seqs` to 1 here.
- **Memory**: peak resident, steady-state, KV cache growth with context
  length. Read from `tegrastats` plus per-process `/proc/PID/status`.
- **Power**: `VDD_IN`, `VDD_CPU_GPU_CV`, `VDD_SOC` rails from
  `tegrastats`. Energy per token is the edge-relevant metric.

## Hardware setup

```bash
# MAXN_SUPER and max clocks (every benchmark run starts with this)
sudo nvpmodel -m 2
sudo jetson_clocks

# Text-mode boot (no GUI) — frees ~100-200 MB GPU memory
sudo systemctl set-default multi-user.target
sudo reboot

# Swap: on for production runs, off for memory-measurement sessions
# (zram is default 3.7 GB; add 8 GB swapfile for kernel build or heavy tactics)
```

## Phase plan (current)

| Phase | Where     | What                                                           | Status |
|-------|-----------|----------------------------------------------------------------|--------|
| 0     | Host      | AWQ quantize + ONNX export for TRT-Edge-LLM (on A40 pod, ~30 min). See `host/README.md`. | ✅ |
| 1a    | Orin Nano | Verify JetPack ≥ 6.2.2 (L4T ≥ 36.5.0). Upgrade via apt if on 6.2.1. See `device/README.md`. | ✅ |
| 1b    | Orin Nano | Build TRT-Edge-LLM C++ runtime + plugin library. | ✅ |
| 1c    | Orin Nano | Download Cosmos-Reason2-2B GGUF, merge splits, quantize to Q4_K_M. | ✅ |
| 2a    | Orin Nano | llama.cpp — launch server, run text + image benchmarks, concurrency sweep. | ✅ |
| 2b    | Orin Nano | vLLM — launch server with Embedl W4A16, run benchmark suite. | ✅ |
| 2c    | Orin Nano | TRT Edge-LLM — kernel CMA setup (`cma=950M`), ONNX graph surgery (split LM head), engine build, text + image benchmarks. | ✅ |
| 3     | Orin Nano | Capture Nsight profiles for the same prompt on each runtime. | ⏳ |
| 4     | Anywhere  | Plots, roofline analysis, writeup. | ⏳ |

## Reproducing

Start with `device/README.md` for the JetPack upgrade and hardware
setup, then work through the phase scripts in order. Each script is
idempotent and checks for existing outputs before re-running.

```bash
cd vlm-benchmarks
cat device/README.md                     # JetPack upgrade, MAXN_SUPER, swap
cat device/scripts/10_prepare_llamacpp.sh
cat device/scripts/11_run_llamacpp_server.sh
cat device/scripts/03_run_vllm_server.sh

# TRT Edge-LLM path (requires the CMA + ONNX surgery workarounds)
cat device/trt_cosmos_patches/README.md  # operator-facing recipe
cat notes/trt_edgellm_cosmos_resolution.md  # full investigation narrative
python3 device/trt_cosmos_patches/split_lm_head.py  # one-time graph rewrite
./device/scripts/40_build_cosmos_trt_engines.sh     # build LLM + visual engines
./device/scripts/41_sanity_cosmos_trt.sh            # verify correct outputs
./device/scripts/42_bench_cosmos_trt.sh             # 5-run TTFT/TPOT/TPS

# Streaming benchmark against any OpenAI-compatible endpoint (llama.cpp or vLLM):
python3 benchmarks/bench_vllm.py --url http://localhost:8000 \
    --model embedl/Cosmos-Reason2-2B-W4A16
```

## Repository layout

```
vlm-benchmarks/
├── README.md                  # this file
├── index.html                 # blog post
├── host/                      # Phase 0 — x86/A40 host
│   ├── scripts/               # setup, quantize, export, package
│   └── calibration/           # shared calibration set
├── device/                    # Phase 1-2 — Jetson
│   ├── README.md              # JetPack upgrade + hardware setup
│   ├── scripts/               # 03_* vLLM, 10_*/11_* llama.cpp, 40_-42_* TRT
│   ├── configs/               # per-runtime parameters
│   ├── inputs/trt/            # TRT benchmark prompt files
│   ├── results/trt/           # TRT per-run timing JSONs, logs
│   └── trt_cosmos_patches/    # CMA recipe, split_lm_head.py, source patch
├── benchmarks/                # Phase 2-3 — runtime-agnostic harness
│   ├── harness.py             # orchestrator (tegrastats + power + latency)
│   ├── bench_vllm.py          # streaming TTFT/TPOT benchmark
│   ├── clients/               # one thin client per runtime
│   ├── metrics/               # latency, resource, power parsers
│   ├── workloads/             # text / image / video prompt sets
│   └── profiling/             # Nsight + roofline
├── analysis/                  # Phase 4 — post-processing
│   └── plots/                 # matplotlib scripts
├── notes/                     # investigation writeups
│   └── trt_edgellm_cosmos_resolution.md
├── assets/                    # inputs + outputs
│   ├── images/ videos/        # test inputs
│   └── results/               # raw JSON, memory snapshots, CSV
└── scripts/                   # v1 placeholder (kept as historical reference)
```

## References

- Jetson AI Lab, "TensorRT Edge-LLM on Jetson" —
  https://www.jetson-ai-lab.com/tutorials/tensorrt-edge-llm/
- NVIDIA, "Deploying Open Source Vision Language Models (VLM) on Jetson" —
  https://huggingface.co/blog/nvidia/cosmos-on-jetson
- NVIDIA, Cosmos-Reason2-2B model card —
  https://huggingface.co/nvidia/Cosmos-Reason2-2B
- Embedl, W4A16 port of Cosmos-Reason2-2B —
  https://huggingface.co/embedl/Cosmos-Reason2-2B-W4A16
- `robertzty/Cosmos-Reason2-2B-GGUF` (BF16 GGUF + mmproj for llama.cpp) —
  https://huggingface.co/robertzty/Cosmos-Reason2-2B-GGUF
- llama.cpp — https://github.com/ggml-org/llama.cpp
- vLLM — https://github.com/vllm-project/vllm
- NVIDIA, TensorRT Edge-LLM — https://github.com/NVIDIA/TensorRT-Edge-LLM
- NVIDIA forum, "unable to allocate CUDA0 buffer after updating Ubuntu packages" —
  https://forums.developer.nvidia.com/t/unable-to-allocate-cuda0-buffer-after-updating-ubuntu-packages/347862
- JetPack 6.2.2 release notes (L4T 36.5.0) —
  https://docs.nvidia.com/jetson/archives/r36.5/ReleaseNotes/
