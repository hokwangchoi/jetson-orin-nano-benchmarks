# Jetson Orin Nano Benchmarks

Deep learning inference benchmarks on the NVIDIA Jetson Orin Nano 8GB — from PyTorch to TensorRT, vision models to VLMs.

## Why This Exists

Edge AI deployment requires understanding real-world performance. Vendor specs tell you theoretical TOPS; this repo tells you actual FPS, latency, memory, power consumption, and GPU utilization on production workloads.

Each benchmark includes:
- Reproducible scripts
- Raw results data
- Detailed blog post explaining methodology and findings

## Hardware

| Spec | Value |
|------|-------|
| **Device** | Jetson Orin Nano 8GB Developer Kit (Super) |
| **GPU** | 1024 CUDA cores, 32 Tensor Cores (Ampere, SM 8.7) |
| **Memory** | 8GB LPDDR5 unified (CPU + GPU), 68 GB/s |
| **AI Performance** | 40 TOPS / 67 TOPS (Super mode, MAXN_SUPER) |
| **JetPack** | 6.2.2 (L4T 36.5.0), upgraded from 6.2.1 mid-project — see VLM post |

## Benchmarks

### [Vision Models](./vision-benchmarks/) — ✅ complete

YOLOv8 object detection across the full optimization pipeline:
- PyTorch → ONNX → TensorRT
- FP32, FP16, INT8 precision comparison
- GPU-only latency via CUDA events + Nsight Systems profiling

**[Read the blog post →](https://hokwangchoi.com/blog/vision-benchmarks/)**

### [Vision Language Models](./vlm-benchmarks/) — 🚧 in progress

A 2B-parameter VLM on 8GB unified memory, across three inference
runtimes. Intended as a straight comparison on the same model
(Cosmos-Reason2-2B); ended up as a case study in platform fragility.

- **Model**: Cosmos-Reason2-2B (Qwen3-VL-2B post-trained for physical reasoning)
- **Runtimes**: llama.cpp (ggml C runtime) vs vLLM (PyTorch + PagedAttention)
  vs TensorRT Edge-LLM (C++, fused TRT kernels)
- **What happened**: JetPack 6.2.1 shipped with an NvMap memory-allocation
  bug that blocked vLLM and TRT-Edge-LLM in different ways. NVIDIA
  released a partial fix (JetPack 6.2.2). vLLM now works with a
  community-quantized W4A16 checkpoint (Embedl) and NVIDIA's
  memory-constrained serve config. TRT-Edge-LLM on Cosmos-2B is still
  blocked.
- **Metrics**: TTFT, TPOT, TPS, peak/steady-state memory
- **Angle**: robotics and AV perception workloads — what does it
  actually take to deploy a production inference runtime on a 2026 edge
  platform?

**[Read the blog post →](./vlm-benchmarks/index.html)**

## Quick Start

```bash
# Clone
git clone https://github.com/hokwangchoi/jetson-orin-nano-benchmarks.git
cd jetson-orin-nano-benchmarks

# Set power mode (MAXN SUPER, max clocks)
sudo nvpmodel -m 2
sudo jetson_clocks

# Run vision benchmarks
cd vision-benchmarks/scripts
pip3 install ultralytics
python3 benchmark_yolov8.py

# VLM benchmarks — see vlm-benchmarks/README.md and vlm-benchmarks/device/README.md.
# Requires JetPack 6.2.2 or later. Upgrade instructions in device/README.md.
```

## Results at a Glance

### YOLOv8n (640×640, batch=1, MAXN_SUPER)

| Runtime  | Precision | GPU latency | Throughput | Tensor-core util |
|----------|-----------|------------:|-----------:|-----------------:|
| PyTorch  | FP32      |           — |          — |                — |
| TensorRT | FP32      |      8.37ms |   120 FPS  |                — |
| TensorRT | FP16      |      4.43ms |   226 FPS  |                — |
| TensorRT | INT8      |      3.49ms |   287 FPS  |                — |

### VLM Inference (Orin Nano 8GB, MAXN_SUPER, JetPack 6.2.2)

| Runtime          | Model                | Quant    | Context | TTFT (text) | TTFT (img) | TPOT | TPS |
|------------------|----------------------|----------|--------:|------------:|-----------:|-----:|----:|
| llama.cpp        | Cosmos-Reason2-2B    | Q4_K_M   |    4096 |       58 ms |     306 ms |27 ms |  38 |
| vLLM             | Cosmos-Reason2-2B (Embedl port) | W4A16 AWQ |  256 |           — |      — | — |  16 |
| TRT Edge-LLM     | Cosmos-Reason2-2B    | W4A16 AWQ|     256 |   *blocked* |  *blocked* |    — |   — |

*vLLM TPS is wall-clock (includes prefill). Streaming TTFT/TPOT captured
via `benchmarks/bench_vllm.py`.*
*TRT Edge-LLM + Cosmos-2B: blocked on NvMap/Myelin bug, see
[`vlm-benchmarks/notes/trt_edgellm_cosmos_blocker.md`](./vlm-benchmarks/notes/trt_edgellm_cosmos_blocker.md).
Retry planned when NVIDIA patches the upstream issue.*

## Repository Structure

```
.
├── README.md
├── LICENSE
├── vision-benchmarks/           # complete
│   ├── index.html               # blog post
│   ├── assets/                  # figures, results
│   └── scripts/
│       └── benchmark_yolov8.py
└── vlm-benchmarks/              # in progress
    ├── README.md                # story, methodology, phase plan
    ├── index.html               # blog post
    ├── host/                    # x86 host — quantization + ONNX export
    ├── device/                  # Jetson — runtime build + serving
    │   ├── README.md            # JetPack upgrade, hardware setup
    │   ├── scripts/             # 10_* llama.cpp, 20_* vLLM
    │   └── configs/             # per-runtime env files
    ├── benchmarks/              # runtime-agnostic harness + streaming bench
    │   ├── harness.py           # full orchestrator (tegrastats, power, latency)
    │   └── bench_vllm.py        # streaming TTFT/TPOT benchmark
    ├── notes/                   # investigation writeups (blockers, retries)
    ├── analysis/                # plots + final writeup
    ├── assets/                  # inputs, results, profiles
    │   └── results/             # raw JSON results, memory snapshots
    └── scripts/                 # v1 placeholder (superseded by benchmarks/)
```

## License

MIT — see [LICENSE](./LICENSE)

## Author

[Hokwang Choi](https://hokwangchoi.com) · [@hokwangchoi](https://github.com/hokwangchoi)
