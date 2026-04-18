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

We run it at INT4 weights — the only configuration that fits comfortably
on Orin Nano 8 GB with headroom for KV cache, activations, and the OS.
The two runtimes take different routes to INT4:

- **llama.cpp**: Q4_K_M (K-quants), applied on-device via `llama-quantize`
  from the BF16 GGUF checkpoint. Group-wise quantization with a mix of 4-
  and 5-bit blocks. ~1.1 GB model file.
- **TRT Edge-LLM**: W4A16 AWQ (Activation-aware Weight Quantization) with
  FP16 activations, produced on an x86 host with `ModelOpt` using a
  calibration dataset. ~1.1 GB exported weights.

Both target INT4 on average; the calibration methods, group sizes, and
GEMM kernels are different. We note the asymmetry in the writeup and
check output quality on a held-out prompt set to confirm neither
quantization wrecked the model. Vision encoder stays at FP16/BF16 in
both cases; lower bits on activations cost accuracy without speeding up
Ampere tensor cores.

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

The same 2B Cosmos-Reason weights run through two C++-native inference
stacks with very different designs:

| Aspect                | llama.cpp                        | TensorRT Edge-LLM                |
|-----------------------|----------------------------------|----------------------------------|
| Language              | C (ggml core), C++ wrapper       | C++ runtime, zero Python in path |
| Kernels               | Hand-tuned ggml-cuda kernels     | TensorRT fused, hand-tuned       |
| Graph execution       | Interpretive (op-by-op)          | Captured as CUDA graph at build  |
| Batching              | Slot pool (`--parallel N`)       | Static batch size at build time  |
| KV cache              | Contiguous per slot              | Preallocated contiguous          |
| Weight format         | GGUF (mmap-friendly, single file)| TRT engine (`.engine`)            |
| Quantization toolchain | `llama-quantize` (K-quants)     | `ModelOpt` → ONNX w/ quant-metadata |
| Cold start            | ~5–10 s (mmap load)              | ~5–10 s (engine load)            |
| Intended use          | Edge-native, hackable local runtime | Embedded, single-app          |

Both end up at INT4 average weight precision, but they are *not*
bit-equivalent: K-quants and AWQ are different quantization algorithms
that were chosen by their respective ecosystems for good reasons.
llama.cpp's K-quants pack mixed precision into small blocks and don't
require activation calibration; AWQ scales weights based on activation
outlier magnitudes from a calibration set. See
[`host/README.md`](./host/README.md) for the AWQ pipeline (TRT-Edge-LLM
only; llama.cpp's quantization runs on-device and is driven by
[`device/scripts/10_prepare_llamacpp.sh`](./device/scripts/10_prepare_llamacpp.sh)).

### 3. Profiling: what is each one actually doing?

This is the section that separates "I ran a benchmark" from "I
understand my system." Metrics:

- **Latency**: TTFT (time to first token — includes ViT + prefill),
  TPOT (time per output token — decode-dominated), TPS (tokens/sec).
- **Throughput under concurrency**: 1, 2, 4 concurrent requests via
  llama.cpp's slot pool. TRT Edge-LLM is single-stream, so concurrent
  numbers only apply to llama.cpp.
- **Memory**: peak resident, steady-state, KV cache growth with context
  length. Read from `tegrastats` plus per-process `/proc/PID/status`.
- **CPU utilization**: per-core from `tegrastats`. Both runtimes are
  C++-native, so CPU usage should be low and similar — if it isn't, that
  tells us something interesting about where the orchestration overhead
  actually lives.
- **GPU utilization**: `GR3D_FREQ` from `tegrastats`. Low numbers during
  decode are expected — decode is memory-bandwidth-bound on Orin.
- **Power**: `VDD_IN`, `VDD_CPU_GPU_CV`, `VDD_SOC` rails from
  `tegrastats`. Energy per token is the edge-relevant metric.
- **Kernel traces**: Nsight Systems captures for prefill and decode.
  Kernel mix tells you how much fusion the TRT compiler actually did
  over the hand-written ggml baseline.

## Phase plan

| Phase | Where     | What                                                           | Status |
|-------|-----------|----------------------------------------------------------------|--------|
| 0     | x86 host  | Quantize with `ModelOpt` (TRT-Edge-LLM only, AWQ W4A16) and export ONNX. llama.cpp's Q4_K_M is produced on-device in phase 1b. | ☐ |
| 1a    | Orin Nano | Build TRT-Edge-LLM C++ runtime + engines.                      | ☐ |
| 1b    | Orin Nano | Download Cosmos-Reason2-2B GGUF, merge splits, quantize to Q4_K_M. | ☐ |
| 1c    | Orin Nano | Launch llama-server (OpenAI-compatible, port 8000).            | ☐ |
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
│   ├── scripts/               # build TRT runtime/engines, prep + run llama-server
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
- `robertzty/Cosmos-Reason2-2B-GGUF` (BF16 GGUF + mmproj for llama.cpp) —
  https://huggingface.co/robertzty/Cosmos-Reason2-2B-GGUF
- llama.cpp —
  https://github.com/ggml-org/llama.cpp
- NVIDIA, TensorRT Edge-LLM —
  https://github.com/NVIDIA/TensorRT-Edge-LLM
