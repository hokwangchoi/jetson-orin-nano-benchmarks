# Jetson Orin Nano Benchmarks

Deep learning inference benchmarks on the NVIDIA Jetson Orin Nano 8GB — from PyTorch to TensorRT, vision models to VLMs.

## Why This Exists

Edge AI deployment requires understanding real-world performance. Vendor specs tell you theoretical TOPS; this repo tells you actual FPS, latency, memory, and power consumption on production workloads.

Each benchmark includes:
- Reproducible scripts
- Raw results data
- Detailed blog post explaining methodology and findings

## Hardware

| Spec | Value |
|------|-------|
| **Device** | Jetson Orin Nano 8GB Developer Kit |
| **GPU** | 1024 CUDA cores, 32 Tensor Cores (Ampere) |
| **Memory** | 8GB LPDDR5 unified (CPU + GPU) |
| **AI Performance** | 40 TOPS / 67 TOPS (Super mode) |
| **JetPack** | 6.2 |

## Benchmarks

### [Vision Models](./vision-benchmarks/)

YOLOv8 object detection across the full optimization pipeline:
- PyTorch → ONNX → TensorRT
- FP32, FP16, INT8 precision comparison
- Power efficiency analysis across power modes

**[Read the blog post →](https://hokwangchoi.com/blog/vision-benchmarks/)**

### [Vision Language Models](./vlm-benchmarks/)

Deploying VLMs on 8GB unified memory:
- Qwen2.5-VL-3B, Cosmos Reason 2B
- Quantization strategies (INT4, FP8)
- Time-to-first-token, tokens/sec benchmarks

**[Read the blog post →](./vlm-benchmarks/index.html)**

## Quick Start

```bash
# Clone
git clone https://github.com/hokwangchoi/jetson-orin-nano-benchmarks.git
cd jetson-orin-nano-benchmarks

# Set power mode
sudo nvpmodel -m 0
sudo jetson_clocks

# Run vision benchmarks
cd vision-benchmarks/scripts
pip3 install ultralytics
python3 benchmark_yolov8.py

# Run VLM benchmarks
cd ../../vlm-benchmarks/scripts
python3 benchmark_vlm.py
```

## Results at a Glance

### YOLOv8n (640×640, batch=1)

| Runtime | Precision | Latency | FPS |
|---------|-----------|---------|-----|
| PyTorch | FP32 | — | — |
| TensorRT | FP32 | — | — |
| TensorRT | FP16 | — | — |
| TensorRT | INT8 | — | — |

### VLM Inference

| Model | Quantization | TTFT | Tokens/s | Memory |
|-------|--------------|------|----------|--------|
| Qwen2.5-VL-3B | INT4 | — | — | — |
| Cosmos Reason 2B | FP8 | — | — | — |

*Results will be filled after running benchmarks.*

## Repository Structure

```
.
├── README.md
├── LICENSE
├── vision-benchmarks/
│   ├── index.html          # Blog post
│   ├── assets/             # Screenshots, figures
│   │   └── results/        # JSON benchmark data
│   └── scripts/
│       └── benchmark_yolov8.py
└── vlm-benchmarks/
    ├── index.html          # Blog post
    ├── assets/
    │   └── results/
    └── scripts/
        └── benchmark_vlm.py
```

## License

MIT — see [LICENSE](./LICENSE)

## Author

[Hokwang Choi](https://hokwangchoi.com) · [@hokwangchoi](https://github.com/hokwangchoi)
