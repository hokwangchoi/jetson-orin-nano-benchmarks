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
| **Device** | Jetson Orin Nano 8GB Developer Kit |
| **GPU** | 1024 CUDA cores, 32 Tensor Cores (Ampere, SM 8.7) |
| **Memory** | 8GB LPDDR5 unified (CPU + GPU), 68 GB/s |
| **AI Performance** | 40 TOPS / 67 TOPS (Super mode, MAXN_SUPER) |
| **JetPack** | 6.2 (L4T 36.4.7) |

## Benchmarks

### [Vision Models](./vision-benchmarks/) — ✅ complete

YOLOv8 object detection across the full optimization pipeline:
- PyTorch → ONNX → TensorRT
- FP32, FP16, INT8 precision comparison
- GPU-only latency via CUDA events + Nsight Systems profiling

**[Read the blog post →](https://hokwangchoi.com/blog/vision-benchmarks/)**

### [Vision Language Models](./vlm-benchmarks/) — 🚧 in progress

Running a 2B-parameter VLM on 8GB unified memory, comparing two inference
runtimes on the exact same quantized model:
- **Model**: Cosmos-Reason2-2B (Qwen3-VL-2B post-trained for physical reasoning)
- **Quantization**: W4A16 AWQ (INT4 weights, FP16 activations)
- **Runtimes**: vLLM (Python, PagedAttention, continuous batching) vs
  TensorRT Edge-LLM (C++, fused TensorRT kernels, CUDA graphs)
- **Metrics**: TTFT, TPOT, TPS, CPU + GPU utilization, VDD power rails,
  peak/steady-state memory, Nsight kernel traces
- **Angle**: robotics and AV perception workloads — what does it take to put
  a reasoning VLM on a deployed edge platform?

**[Read the WIP →](./vlm-benchmarks/)**

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

# VLM benchmarks (see vlm-benchmarks/README.md — requires x86 host for
# quantization, then Jetson for inference)
```

## Results at a Glance

### YOLOv8n (640×640, batch=1, MAXN_SUPER)

| Runtime  | Precision | GPU latency | Throughput | Tensor-core util |
|----------|-----------|------------:|-----------:|-----------------:|
| PyTorch  | FP32      |           — |          — |                — |
| TensorRT | FP32      |      8.37ms |   120 FPS  |                — |
| TensorRT | FP16      |      4.43ms |   226 FPS  |                — |
| TensorRT | INT8      |      3.49ms |   287 FPS  |                — |

### VLM Inference (pending)

| Runtime          | Model                  | TTFT | TPOT | TPS | Mem | CPU | GPU |
|------------------|------------------------|-----:|-----:|----:|----:|----:|----:|
| vLLM             | Cosmos-Reason2-2B-W4A16 |    — |    — |   — |   — |   — |   — |
| TRT Edge-LLM     | Cosmos-Reason2-2B-W4A16 |    — |    — |   — |   — |   — |   — |

*Results filled in as benchmarks complete.*

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
    ├── index.html               # blog post (draft)
    ├── host/                    # x86 host — quantization + ONNX export
    ├── device/                  # Jetson — runtime build + serving
    ├── benchmarks/              # runtime-agnostic harness
    ├── analysis/                # plots + final writeup
    ├── assets/                  # inputs, results, profiles
    └── scripts/                 # v1 placeholder (superseded by benchmarks/)
```

## License

MIT — see [LICENSE](./LICENSE)

## Author

[Hokwang Choi](https://hokwangchoi.com) · [@hokwangchoi](https://github.com/hokwangchoi)
