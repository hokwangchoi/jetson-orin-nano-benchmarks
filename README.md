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

### [Vision Language Models](./vlm-benchmarks/) — ✅ complete

A 2B-parameter VLM on 8GB unified memory, across three inference
runtimes. Same model (Cosmos-Reason2-2B), same quantization class
(INT4 weights, FP16 activations), same hardware.

- **Model**: Cosmos-Reason2-2B (Qwen3-VL-2B post-trained for physical reasoning)
- **Runtimes**: llama.cpp (ggml C runtime) vs vLLM (PyTorch + PagedAttention)
  vs TensorRT Edge-LLM (C++, fused TRT kernels)
- **What happened**: JetPack 6.2.1 shipped with a tightened NvMap
  contiguous-allocation ceiling that blocked vLLM and TRT-Edge-LLM in
  different ways. JetPack 6.2.2 relaxed the PyTorch initialization
  path. vLLM now works with a community-quantized W4A16 checkpoint
  (Embedl) plus a vision-encoder profile cap that keeps startup
  allocations under the ceiling. TRT-Edge-LLM took more: a kernel-level
  CMA parameter plus an ONNX graph rewrite (splitting the LM head
  MatMul) to get past Myelin's 1 GiB tactic scratch request. As far as
  I can tell this is the first public report of Cosmos-Reason2-2B
  running on TRT-Edge-LLM on an Orin Nano.
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

| Runtime  | Precision | GPU latency | Throughput | TOPS util |
|----------|-----------|------------:|-----------:|----------:|
| PyTorch  | FP32      |     20.88ms |   47.9 FPS |      4.2% |
| TensorRT | FP32      |      8.36ms |  119.5 FPS |     10.4% |
| TensorRT | FP16      |      4.43ms |  225.3 FPS |      9.8% |
| TensorRT | INT8      |      3.49ms |  286.2 FPS |      6.2% |

*TOPS util is achieved TOPS (GFLOPs / latency) divided by peak TOPS for the precision. GPU latency is CUDA-event (PyTorch) or `trtexec` GPU Compute Time (TensorRT), both exclude H2D/D2H transfers.*

### VLM Inference (Orin Nano 8GB, MAXN_SUPER, JetPack 6.2.2)

| Runtime          | Model                | Quant    | Context | TTFT (text) | TTFT (img) | TPOT | TPS | Peak mem |
|------------------|----------------------|----------|--------:|------------:|-----------:|-----:|----:|---------:|
| llama.cpp        | Cosmos-Reason2-2B    | Q4_K_M   |    4096 |       58 ms |     306 ms |27 ms |  38 |        — |
| vLLM             | Cosmos-Reason2-2B (Embedl port) | W4A16 AWQ |  1024 |       61 ms |      77 ms |17 ms |  56 |   6.9 GB |
| TRT Edge-LLM     | Cosmos-Reason2-2B    | W4A16 AWQ|    1024 |       29 ms |     420 ms |17 ms |  60 |   4.3 GB |

*All values are medians of 5 streaming runs captured by `benchmarks/bench_vllm.py`
(driven by `scripts/bench_vllm.sh` and `scripts/bench_llamacpp.sh`) and
`scripts/bench_trt.sh` (TRT Edge-LLM).
Text runs: 128 output tokens. Image runs: full-size `bus.jpg`, up to 128
output tokens. vLLM image TTFT is the steady-state value (runs 2–5); the
first image request in any batch takes 0.5–3 s because the ViT path is
not CUDA-graph-captured and compiles lazily per unique input shape. TRT
Edge-LLM image TTFT does not have the same warmup effect — its visual
engine always runs (~208 ms) per request.*

*TRT Edge-LLM + Cosmos-2B on Orin Nano required workarounds that go
beyond the standard tutorial path: a `cma=950M` kernel parameter plus an
ONNX graph rewrite splitting the LM head MatMul to avoid Myelin's 1 GiB
tactic scratch. See [`vlm-benchmarks/notes/trt_edgellm_cosmos_resolution.md`](./vlm-benchmarks/notes/trt_edgellm_cosmos_resolution.md)
for the full investigation and [`vlm-benchmarks/device/trt_cosmos_patches/`](./vlm-benchmarks/device/trt_cosmos_patches/)
for the patches and scripts.*

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
└── vlm-benchmarks/              # complete
    ├── README.md                # story, methodology, phase plan
    ├── index.html               # blog post
    ├── device/                  # Jetson — runtime setup + serving
    │   ├── README.md            # JetPack upgrade, hardware setup
    │   ├── scripts/             # 03_* vLLM, 10_-11_* llama.cpp, 40_-41_* TRT
    │   ├── configs/             # per-runtime env files
    │   ├── inputs/trt/          # TRT-specific benchmark prompts
    │   └── trt_cosmos_patches/  # CMA recipe, split_lm_head.py, source patch
    ├── benchmarks/              # runtime-agnostic Python library
    │   ├── harness.py           # full orchestrator (tegrastats, power, latency)
    │   └── bench_vllm.py        # streaming TTFT/TPOT benchmark
    ├── scripts/                 # benchmark runners (bench_vllm/llamacpp/trt.sh)
    ├── notes/                   # investigation writeups
    └── assets/                  # inputs + outputs
        ├── images/              # test inputs
        └── results/             # per-runtime result JSONs
            ├── llamacpp/
            ├── vllm/
            └── trt/
```

## License

MIT — see [LICENSE](./LICENSE)

## Author

[Hokwang Choi](https://hokwangchoi.com) · [@hokwangchoi](https://github.com/hokwangchoi)
