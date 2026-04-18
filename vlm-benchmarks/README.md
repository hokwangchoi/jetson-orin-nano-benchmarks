# VLM Benchmarks on Jetson Orin Nano

Running a 2B-parameter vision-language model on 8 GB of unified memory —
and measuring what really matters when you take it off the data-center
rack and put it on an edge device.

## Why VLMs on the edge?

Perception on a robot or autonomous vehicle used to mean a stack of
narrow models: YOLO for boxes, DeepLab for segmentation, SORT for tracks,
maybe BEV-former or a Transformer trajectory predictor. Each was trained
to answer one kind of question about the world. None of them could tell
you *why* the scene looks the way it does, or what's about to happen.

Vision-Language Models collapse that stack. A single VLM can caption a
frame, answer a grounded question about it, reason about object
relationships over time, and return structured outputs (bounding boxes,
keypoints, trajectories, JSON). The promise for physical AI is obvious:

- **Autonomous driving perception**: "Is it safe to turn right?" with
  evidence drawn from the scene, not just a detection feed. Failure-case
  understanding for offline data curation and active learning.
- **Robotics planning**: VLA (vision-language-action) models like π0,
  OpenVLA, and GR00T-style policies use a VLM as the "brain" that
  decomposes a natural-language goal into motor primitives.
- **Safety monitoring**: catching out-of-distribution scenes a narrow
  classifier would miss — construction zones, debris, novel agents.
- **Teleoperation fallback**: when the autonomy stack disengages, a VLM
  can describe the state of the world to a remote operator in one round
  trip.

The catch is that almost every VLM demo you see runs on a pair of H100s.
The actual edge platforms that ship on robots and cars — Jetson Orin
Nano, Orin NX, DRIVE Orin — have 8 to 16 GB of memory shared between CPU
and GPU, 20 to 100 TOPS of compute, and power budgets under 30 W. What
does it take to close that gap?

## The model

[`nvidia/Cosmos-Reason2-2B`](https://huggingface.co/nvidia/Cosmos-Reason2-2B)
— NVIDIA's 2B physical-AI reasoning VLM, post-trained from Qwen3-VL-2B
with SFT and RL on embodied-reasoning datasets. Architecturally it is
a Qwen3-VL stack: a ViT vision encoder, an MLP projector, and a dense
Transformer decoder. It understands spatial relationships, temporal
ordering, and basic physics, and outputs chain-of-thought reasoning in
`<think>...</think><answer>...</answer>` format.

We run it at W4A16: INT4 weights (AWQ), FP16 activations, vision encoder
kept at FP16. The LLM weights drop from ~4.4 GB (BF16) to ~1.1 GB. That
is the only configuration that fits comfortably on Orin Nano 8 GB with
headroom for KV cache, activations, and the OS.

## The three-part story

### 1. Feasibility: can it fit?

Before measuring anything, account for every byte. On an 8 GB unified
memory system with no PCIe-and-discrete-GPU split, every allocation
contests with the OS, the desktop, and the runtime itself. Details in
[`FEASIBILITY.md`](./FEASIBILITY.md) (filled in as measurements come in):

- Weight footprint per component (LLM INT4, ViT FP16, projector FP16).
- KV cache math: `2 × n_layers × n_kv_heads × head_dim × 2B × seq_len`,
  and why PagedAttention exists.
- The VLM inference pipeline:
  1. Vision encoding (ViT forward pass over image patches → visual tokens)
  2. Projector (MLP into LLM embedding space)
  3. Prefill (LLM processes `[visual tokens, text tokens]`, compute-bound)
  4. Decode (autoregressive generation, memory-bandwidth-bound)
- Why W4A16 and not W4A4: activations at FP16 because Orin's Ampere
  tensor cores don't accelerate INT8 activations the way Hopper/Blackwell
  do — lower bits on activations cost accuracy with no speedup.

### 2. Comparison: two runtimes, same model

The same W4A16 Cosmos-Reason2-2B weights run through two production
inference stacks with very different designs:

| Aspect                | vLLM                             | TensorRT Edge-LLM                |
|-----------------------|----------------------------------|----------------------------------|
| Language              | Python orchestrator              | C++ runtime, zero Python in path |
| Kernels               | Triton-compiled, PagedAttention  | TensorRT fused, hand-tuned       |
| Batching              | Continuous / in-flight           | Static batch size at build time  |
| KV cache              | Paged, block-virtual addressing  | Preallocated contiguous          |
| Graph capture         | Optional CUDA graphs             | Required; captured at build      |
| Quantization toolchain | `llm-compressor` → compressed-tensors | `ModelOpt` → ONNX w/ quant-metadata |
| Multi-request         | High throughput under concurrency | Single-stream optimized          |
| Cold start            | ~60–120 s (graph compile)        | ~5–10 s (engine load)            |
| Intended use          | Multi-tenant serving             | Embedded, single-app             |

Both end up as INT4 AWQ on disk, but they are *not* bit-equivalent. The
calibration data, group sizes, and quantized-GEMM kernel implementations
differ by runtime. To keep the comparison honest, we run our own
quantization on the rented x86 host using the same calibration dataset
for both pipelines. See
[`host/README.md`](./host/README.md).

### 3. Profiling: what is each one actually doing?

This is the section that separates "I ran a benchmark" from "I
understand my system." Metrics:

- **Latency**: TTFT (time to first token — includes ViT + prefill),
  TPOT (time per output token — decode-dominated), TPS (tokens/sec).
- **Throughput under concurrency**: 1, 2, 4 concurrent requests. Tests
  PagedAttention's whole value proposition.
- **Memory**: peak resident, steady-state, KV cache growth with context
  length. Read from `tegrastats` plus per-process `/proc/PID/status`.
- **CPU utilization**: per-core from `tegrastats`. Python interpreter
  overhead in vLLM vs near-zero in Edge-LLM should show up here.
- **GPU utilization**: `GR3D_FREQ` from `tegrastats`. Low numbers during
  decode are expected — decode is memory-bandwidth-bound on Orin.
- **Power**: `VDD_IN`, `VDD_CPU_GPU_CV`, `VDD_SOC` rails from
  `tegrastats`. Energy per token is the edge-relevant metric.
- **Kernel traces**: Nsight Systems captures for prefill and decode.
  Kernel gaps tell you where Python overhead lives; kernel mix tells you
  who did the fusion work.

## Phase plan

| Phase | Where     | What                                                           | Status |
|-------|-----------|----------------------------------------------------------------|--------|
| 0     | x86 host  | Quantize with `ModelOpt` (TRT-Edge-LLM) and `llm-compressor` (vLLM) using the same calibration set. Export ONNX for TRT-Edge-LLM. | ☐ |
| 1a    | Orin Nano | Build TRT-Edge-LLM C++ runtime + engines.                      | ☐ |
| 1b    | Orin Nano | Launch vLLM Jetson container.                                  | ☐ |
| 2     | Orin Nano | Run the harness against both — text, image, video workloads; latency, memory, CPU/GPU, power. | ☐ |
| 3     | Orin Nano | Capture Nsight profiles for the same prompt on each runtime.   | ☐ |
| 4     | Anywhere  | Plots, roofline analysis, writeup.                             | ☐ |

## Reproducing

Start here:

```bash
cd vlm-benchmarks
cat host/README.md      # Phase 0: x86 host quantization + ONNX export
cat device/README.md    # Phase 1: Jetson runtime build + serving
cat benchmarks/README.md # Phase 2-3: harness and profiling
```

Each phase's README lists exact commands, expected timings, and known
failure modes.

## Repository layout (this folder)

```
vlm-benchmarks/
├── README.md                  # this file
├── index.html                 # blog post (draft)
├── host/                      # Phase 0 — x86 GPU
│   ├── scripts/               # setup, quantize (×2), package
│   └── calibration/           # shared calibration set
├── device/                    # Phase 1 — Jetson
│   ├── scripts/               # build runtime, engines, vLLM server
│   └── configs/               # runtime parameters
├── benchmarks/                # Phase 2-3 — runtime-agnostic harness
│   ├── harness.py             # orchestrator
│   ├── clients/               # one thin client per runtime
│   ├── metrics/               # latency, resource, power parsers
│   ├── workloads/             # text / image / video prompt sets
│   └── profiling/             # Nsight + roofline
├── analysis/                  # Phase 4 — post-processing
│   └── plots/                 # all the matplotlib scripts
├── assets/                    # inputs + outputs
│   ├── images/ videos/        # test inputs (add your own)
│   └── results/               # raw JSON, aggregated CSV, Nsight traces, plots
└── scripts/
    └── benchmark_vlm.py       # v1 placeholder (kept as historical reference)
```

## References

- Jetson AI Lab, "TensorRT Edge-LLM on Jetson" —
  https://www.jetson-ai-lab.com/tutorials/tensorrt-edge-llm/
- Jetson AI Lab, "Cosmos Reason 2 8B" —
  https://www.jetson-ai-lab.com/models/cosmos-reason2-8b/
- NVIDIA, Cosmos-Reason2-2B model card —
  https://huggingface.co/nvidia/Cosmos-Reason2-2B
- Embedl, Cosmos-Reason2-2B-W4A16 (reference numbers on Orin Nano Super) —
  https://huggingface.co/embedl/Cosmos-Reason2-2B-W4A16
- vLLM, `llm-compressor` —
  https://github.com/vllm-project/llm-compressor
- NVIDIA, TensorRT Edge-LLM —
  https://github.com/NVIDIA/TensorRT-Edge-LLM
