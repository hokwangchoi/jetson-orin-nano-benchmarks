# TRT-Edge-LLM + Cosmos-Reason2-2B on Jetson Orin Nano 8GB

This directory contains the patches and graph surgery required to build
Cosmos-Reason2-2B as a TensorRT Edge-LLM engine on Jetson Orin Nano 8 GB.

As of April 2026, this combination is not in NVIDIA's supported matrix. Their
official tutorial pairs Cosmos-Reason2 *8B* with Thor (128 GB unified memory)
and pairs Orin Nano 8 GB with Qwen3-4B-Instruct (text-only LLM). Getting
Cosmos-2B to build on Orin Nano required three interventions:

1. A kernel boot parameter to enlarge the CMA pool
2. An ONNX graph rewrite to split the LM head MatMul
3. (Documentary) a one-line Edge-LLM source patch, which did not bind but
   is kept here as a record of what was attempted

## TL;DR — apply the workaround to a fresh setup

1. **CMA kernel param.** Edit `/boot/extlinux/extlinux.conf` APPEND line to
   include `cma=950M`. Reboot. Verify with `grep Cma /proc/meminfo` showing
   `CmaTotal: 983040 kB`. See "Why 950M" below for the full story.

2. **Apply the Edge-LLM source patch** (optional, documentary):
   ```bash
   cd ~/TensorRT-Edge-LLM
   git apply /path/to/this/builderUtils_kTACTIC_DRAM.patch
   cd build && make -j2 llm_build
   ```

3. **Export the Cosmos-Reason2-2B ONNX** using the standard Edge-LLM export
   pipeline (AWQ INT4 quantization). The llm/ export lands at
   `~/tensorrt-edgellm-workspace/Cosmos-Reason2-2B/onnx/llm/`.

4. **Run the graph surgery**:
   ```bash
   python3 -m venv ~/onnx_surgery_venv
   source ~/onnx_surgery_venv/bin/activate
   pip install onnx numpy
   python3 split_lm_head.py   # writes to Cosmos-Reason2-2B-split/onnx/llm/
   deactivate
   ```

5. **Build the engines** with the supplied script:
   ```bash
   vlm-benchmarks/device/scripts/40_build_cosmos_trt_engines.sh
   ```

6. **Sanity check**:
   ```bash
   vlm-benchmarks/device/scripts/41_sanity_cosmos_trt.sh
   ```

7. **Benchmark**:
   ```bash
   vlm-benchmarks/scripts/bench_trt.sh
   ```

## Why 950M for CMA?

Several values were tried in order; all those above ~1 GiB failed at boot
with `cma: Failed to reserve N MiB`:

| cma=   | Result                                 | Reason                          |
|--------|----------------------------------------|---------------------------------|
| 3G     | Failed, no CMA reserved                | No 3 GiB contiguous below 4 GiB |
| 2G     | Failed                                 | Same                            |
| 2G@0x100000000-0x280000000 | Failed              | ARM64 CMA is hardcoded < 4 GiB  |
| 1500M  | Failed                                 | Same                            |
| 1200M  | Failed                                 | Same                            |
| 1G     | Failed                                 | Each low-RAM chunk is ~992 MiB  |
| 950M   | **Success** — CmaTotal = 983040 kB     | Fits in second low chunk        |
| (default) | 262144 kB (256 MiB)                 | Too small for Myelin            |

From `/proc/iomem`, low memory on Orin Nano 8 GB is two ~992 MiB chunks
(`0x80000000-0xbdffffff` and `0xc2000000-0xfffdffff`) with a 64 MiB firmware
carveout hole between them. The ARM64 kernel refuses to place CMA above
4 GiB (`exceeds limit (0x100000000)` in the error). So 950M is effectively
the ceiling on this hardware — 40 MiB of slack within one ~992 MiB chunk.

## What we measured (CMA trace during build)

Critical moment of the LM head tactic selection, with the split ONNX and
CMA=950M:

- **Before split surgery**: CmaFree dropped from 779 MB to 0 MB as tactic 0
  retried 12 times trying to grab 1075 MB contiguous. Pool never recovered
  in time for tactic 1 (593 MB) to run. Build failed.
- **After split surgery**: CmaFree dropped to ~0 MB briefly but each
  allocation succeeded on first try (no retry loop, no wedging). Pool
  released cleanly between tactics. Build completed.

See `vlm-benchmarks/notes/trt_edgellm_cosmos_resolution.md` for the full
investigation writeup, including data from the CMA watcher logs captured
during build attempts.

## What is *not* needed

- Swap. Enabling swap does not help because NvMap requires physically pinned
  contiguous memory — the kernel can't satisfy that from swap.
- Reducing `--maxInputLen` / `--maxKVCacheCapacity`. The failing allocation
  is vocab-driven (`2 × 151936 × 2048`), not sequence-driven.
- Re-exporting ONNX with `--trt_native_ops`. Produces ONNX the TRT 10.3
  parser can't resolve — different, dead-end error.
- Upgrading JetPack 6.2.1 → 6.2.2. Fixes the PyTorch/NVML path (which
  unblocks vLLM) but does not affect this Myelin path. Same bit-identical
  `1075072515` byte allocation request.

## Files in this directory

- `split_lm_head.py` — the ONNX graph surgery. Splits the final LM head
  MatMul along the vocab dimension and stitches it back with a Concat.
  Mathematically identical output; breaks the Myelin fusion boundary.
- `builderUtils_kTACTIC_DRAM.patch` — one-line source change to
  Edge-LLM's `cpp/builder/builderUtils.cpp` setting a smaller tactic-scratch
  cap. Documented as "does not bind" — kept as historical record of what
  was tried before the graph surgery.

## Reproducibility notes

- Hardware: Jetson Orin Nano 8 GB Developer Kit (Super mode, MAXN_SUPER)
- JetPack: 6.2.2 (L4T R36.5.0)
- TensorRT: 10.3.0.30+cuda12.5
- TRT-Edge-LLM: (whatever main was on 2026-04-18)
- ONNX tooling: onnx 1.21.0
- The CMA trace logs, build logs, and profile JSONs from the actual runs
  are preserved in `vlm-benchmarks/assets/results/trt/` and cited in the
  resolution notes.
