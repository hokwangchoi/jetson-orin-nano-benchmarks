# Device-side workflow (Phase 1)

Everything in this folder runs **on the Jetson Orin Nano 8 GB**, assuming
you already completed Phase 0 (`../host/`) and transferred the artifact
bundle.

## Prerequisites

- JetPack 6.2 (L4T 36.4.7), CUDA 12.6, TensorRT 10.x — all included.
- MAXN_SUPER power mode, max clocks:
  ```bash
  sudo nvpmodel -m 2
  sudo jetson_clocks
  ```
- Artifacts extracted:
  ```bash
  mkdir -p ~/vlm-artifacts
  tar -xzf ~/artifacts.tar.gz -C ~/vlm-artifacts/
  ```
- For vLLM path: Docker + NVIDIA Container Toolkit (pre-installed with
  JetPack 6.2 but confirm `docker info | grep nvidia`).

## Run order

```bash
cd vlm-benchmarks/device/scripts

./01_build_trtedgellm_runtime.sh  # C++ runtime build on Jetson (~30-60 min, first time)
./02_build_engines.sh             # llm_build + visual_build (~5-15 min)
./03_run_vllm_server.sh           # launches vLLM container, serves on :8000
./04_smoketest.sh                 # one prompt to each runtime, sanity check
```

The TRT-Edge-LLM runtime is built once. Engines (`.engine`) files must be
rebuilt if you change `--maxInputLen` or `--maxKVCacheCapacity`.

## Memory budget cheat sheet

On 8 GB unified memory you are fighting for headroom. Reasonable defaults:

| Knob                       | vLLM                 | TRT Edge-LLM           |
|----------------------------|----------------------|------------------------|
| `max-model-len` / input    | 8192                 | `--maxInputLen 1024`   |
| `gpu-memory-utilization`   | 0.75                 | n/a (static alloc)     |
| `max-num-seqs`             | 2                    | `--maxBatchSize 1`     |
| KV cache capacity          | paged (auto)         | `--maxKVCacheCapacity 4096` |

If you hit OOM during engine build:
```bash
sudo sysctl -w vm.drop_caches=3    # free page cache
# then drop --maxInputLen 512 --maxKVCacheCapacity 1024
```

## Profiling gotchas

- **Disable ZRAM** for memory measurements:
  ```bash
  sudo swapoff -a
  ```
  otherwise peak memory numbers get confusing.
- **tegrastats sampling interval**: default 1000 ms is fine for
  throughput runs; use `--interval 100` for capturing TTFT spikes.
- **nsys on Jetson**: TRT-Edge-LLM is a C++ binary so `nsys profile
  ./llm_inference ...` just works. vLLM wants `nsys profile -o trace
  --trace=cuda,osrt,nvtx python ...` and you need to attach by PID to
  the container's vllm-worker process — see
  `../benchmarks/profiling/nsight_capture.sh`.
