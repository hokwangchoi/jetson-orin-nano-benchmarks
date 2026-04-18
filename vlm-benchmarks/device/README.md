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
- TRT-Edge-LLM artifacts extracted (from Phase 0 host):
  ```bash
  mkdir -p ~/vlm-artifacts
  tar -xzf ~/artifacts.tar.gz -C ~/vlm-artifacts/
  ```
- Docker + NVIDIA Container Toolkit (pre-installed with JetPack 6.2 but
  confirm `docker info | grep nvidia`). Required for the llama.cpp path
  which runs from NVIDIA's official Jetson container.

## Run order

```bash
cd vlm-benchmarks/device/scripts

./01_build_trtedgellm_runtime.sh  # TRT C++ runtime build on Jetson (~30-60 min, first time)
./02_build_engines.sh             # llm_build + visual_build (~5-15 min)
./10_prepare_llamacpp.sh          # download GGUF, merge splits, quantize to Q4_K_M (~15 min, one-time)
./11_run_llamacpp_server.sh       # launches llama-server on :8000
./04_smoketest.sh                 # one prompt to each runtime, sanity check
```

The TRT-Edge-LLM runtime is built once. Engines (`.engine`) files must be
rebuilt if you change `--maxInputLen` or `--maxKVCacheCapacity`. The
llama.cpp prep is also one-time — the quantized GGUF lives at
`~/models/cosmos-reason2-2b/` and is reused across server runs.

## Memory budget cheat sheet

On 8 GB unified memory you are fighting for headroom. Reasonable defaults:

| Knob                       | llama.cpp            | TRT Edge-LLM           |
|----------------------------|----------------------|------------------------|
| Context / input            | `--ctx-size 4096`    | `--maxInputLen 1024`   |
| GPU layer offload          | `--n-gpu-layers 99`  | n/a (all on GPU)       |
| Parallel slots             | `--parallel 1`       | `--maxBatchSize 1`     |
| KV cache capacity          | per slot, contiguous | `--maxKVCacheCapacity 4096` |

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
- **nsys on Jetson**: both TRT-Edge-LLM and llama-server are C++
  binaries running inside containers. Profile with `nsys profile
  --trace=cuda,osrt,nvtx` and attach by PID — see
  `../benchmarks/profiling/nsight_capture.sh`.
