# Device-side workflow (Phase 1)

Everything in this folder runs **on the Jetson Orin Nano 8 GB**.

## Prerequisites

- **JetPack 6.2.2 (L4T 36.5.0) or later.** If you're on 6.2.1
  (L4T 36.4.7), upgrade first — see [JetPack upgrade](#jetpack-upgrade)
  below. On 6.2.1, vLLM fails to start (PyTorch CUDA allocator asserts
  during NVML sanity check) and TRT Edge-LLM engine builds fail at the
  LM head Myelin tactic selection.
- CUDA 12.6, TensorRT 10.3 — included with JetPack 6.2.x.
- MAXN_SUPER power mode and pinned clocks:
  ```bash
  sudo nvpmodel -m 2
  sudo jetson_clocks
  ```
- Docker + NVIDIA Container Toolkit (pre-installed with JetPack,
  confirm `docker info | grep nvidia`). Required for the llama.cpp and
  vLLM paths which run from official Jetson containers.

## JetPack upgrade

If `head -n1 /etc/nv_tegra_release` shows `REVISION: 4.7`, you need this.

```bash
# Back up the apt source
sudo cp /etc/apt/sources.list.d/nvidia-l4t-apt-source.list{,.bak}

# Change r36.4 → r36.5 in all deb lines
sudo sed -i 's|r36.4|r36.5|g' /etc/apt/sources.list.d/nvidia-l4t-apt-source.list

# Refresh + upgrade. Use dist-upgrade, not upgrade — kernel package must change.
sudo apt update
sudo apt dist-upgrade -y

# Reboot to load the new kernel
sudo reboot

# After reboot, verify
uname -r                       # should be 5.15.185-tegra (was 5.15.148)
head -n1 /etc/nv_tegra_release # should show REVISION: 5.0
```

The upgrade downloads ~500 MB, takes ~20 minutes total (including
reboot). Two config-file prompts during the install — answer `Y`
(take the maintainer's version) for both: `nv-oem-config-post.sh` and
`nvidia-l4t-apt-source.list`. Other prompts for files you haven't
edited can be answered `Y` as well.

## Memory hygiene

On 8 GB unified memory you fight for headroom. Reasonable defaults:

| Knob                       | llama.cpp            | vLLM                       |
|----------------------------|----------------------|----------------------------|
| Context / input            | `--ctx-size 4096`    | `--max-model-len 256`      |
| GPU memory utilization     | n/a (auto)           | `0.6` (leaves room for OS) |
| Parallel slots             | `--parallel 1`       | `--max-num-seqs 1`         |
| KV cache                   | per slot, contiguous | paged, via PagedAttention  |
| CUDA graphs                | n/a                  | disabled (`--enforce-eager`) |

**Swap policy.** The default Jetson zram swap is ~3.7 GB, which is
enough for normal use. For heavy tactic searches during TRT engine
builds, add 8 GB of disk-backed swap on top:

```bash
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

Turn swap **off** during memory benchmarks to get clean peak-memory
numbers (`sudo swapoff -a`), and **on** during production runs as a
safety net for peak transients.

## Run order

```bash
cd vlm-benchmarks/device/scripts

# llama.cpp path
./10_prepare_llamacpp.sh              # download GGUF, merge splits, quantize (~15 min, one-time)
./11_run_llamacpp_server.sh           # launches llama-server on :8000

# vLLM path (requires JetPack 6.2.2+)
./03_run_vllm_server.sh               # launches vLLM serve on :8000
```

Only one server can listen on port 8000 at a time — stop the previous
one before starting the next. Each script is idempotent and checks for
existing outputs.

TRT Edge-LLM on Cosmos-2B required additional setup beyond what NVIDIA's
standard tutorial covers — a kernel-level `cma=950M` parameter and an
ONNX graph rewrite to split the LM head MatMul. The full operator-facing
recipe is in
[`trt_cosmos_patches/README.md`](./trt_cosmos_patches/README.md);
the investigation writeup is in
[`../notes/trt_edgellm_cosmos_resolution.md`](../notes/trt_edgellm_cosmos_resolution.md).
Build the engines with `./scripts/40_build_cosmos_trt_engines.sh`
and sanity-check with `./scripts/41_sanity_cosmos_trt.sh` (both here
in `device/scripts/`). Then run the actual benchmark with
`../scripts/bench_trt.sh` (from `vlm-benchmarks/scripts/`, alongside
the vLLM and llama.cpp wrappers).

## Profiling gotchas

- **Disable swap** for memory measurements (`sudo swapoff -a`);
  otherwise peak-memory numbers include paged-out pages.
- **tegrastats sampling interval**: default 1000 ms is fine for
  throughput runs; use `--interval 100` for capturing TTFT spikes.
- **nsys on Jetson**: all three runtimes run as C++ binaries inside
  containers. Profile with
  `nsys profile --trace=cuda,osrt,nvtx` and attach by PID. See
  `../benchmarks/profiling/nsight_capture.sh`.
- **Text-mode boot** frees ~100-200 MB of GPU memory that would
  otherwise be held by the GNOME display compositor. For benchmarking
  VLMs on an 8 GB board, the extra headroom matters:
  ```bash
  sudo systemctl set-default multi-user.target
  sudo reboot
  ```
  To restore GUI: `sudo systemctl set-default graphical.target && sudo reboot`.
