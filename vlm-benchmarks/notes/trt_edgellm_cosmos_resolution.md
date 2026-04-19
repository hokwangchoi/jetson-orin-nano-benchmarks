# TRT Edge-LLM + Cosmos-Reason2-2B on Orin Nano — the full story

Investigation notes April 2026. Original version of this document was
titled `trt_edgellm_cosmos_blocker.md` and concluded the path was blocked.
This version reflects what we found on re-investigation, and the workarounds
that got the build to complete.

## Result

Cosmos-Reason2-2B (W4A16 AWQ) now builds and runs end-to-end via
TensorRT Edge-LLM on Jetson Orin Nano 8 GB. As far as we can tell this
is the first public report of this combination working — NVIDIA's own
supported matrix pairs Cosmos-2 8B with Thor and pairs Orin Nano with
Qwen3-4B-Instruct (text-only LLM, not a VLM).

Getting there required three interventions, in order of impact:

1. Increase the kernel's CMA pool from 256 MiB default to 950 MiB
2. Rewrite the ONNX graph to split the LM head MatMul along the vocab dim
3. (Documentary) patch Edge-LLM's `setMemoryPoolLimit(kTACTIC_DRAM)` — did
   not bind at runtime but kept in the repo for completeness

The workaround is preserved in `device/trt_cosmos_patches/` along with the
CMA configuration notes. See that directory's README for the operator-facing
recipe. This document is the investigation narrative.

## The error we were chasing

Reproducibly, on both JetPack 6.2.1 (L4T 36.4.7) and JetPack 6.2.2
(L4T 36.5.0), `llm_build` fails during engine construction for the quantized
Cosmos-Reason2-2B ONNX:

```
[TensorRT] Timing Runner: {ForeignNode[/Unsqueeze.../Cast]} (Myelin[0x80000023])
[TensorRT] MemUsageChange: CPU 2380 MB, GPU 4513 MB
NvMapMemAllocInternalTagged: 1075072515 error 12     # repeats 12 times
[ERROR] CUDA error 2 for 622329856-byte allocation.
[ERROR] IBuilder::buildSerializedNetwork: Could not find any implementation
        for node {ForeignNode[/Unsqueeze.../Cast]}.
[ERROR] Failed to build LLM engine.
```

## What the node actually is

The blocker document originally identified this subgraph as the rotary
position embedding path. That turned out to be wrong. The shape TRT reports
just before failure — `Float(151936,151936,1)` — is the vocab dimension of
Qwen3's tokenizer. `/Unsqueeze.../Cast` is TRT shorthand for "the last
Unsqueeze node, through a bunch of intermediate ops, down to the final
Cast". Tracing the actual graph, the fused nodes are:

- `/GatherND` (pick last-token hidden state)
- `/lm_head/MatMul` (the 2048 × 151936 projection — this is the big one)
- `/Cast` (fp16 → fp32 for logits output)

The allocation sizes confirm it. 1,075,072,515 bytes ≈ `1.5 × 151936 × 2048 × 2`
(Myelin's workspace for the tactic — weight matrix plus intermediate buffers
plus bookkeeping). And 622,329,856 bytes is exactly `2 × 2048 × 151936`
(the fp16 LM head weight matrix). The matching size of `embedding.safetensors`
in the ONNX export (622,329,944 bytes, within a few bytes of the same number)
is the final confirmation.

Getting the node identity right mattered because it redirected the fix:
shrinking `maxInputLen` or `maxKVCacheCapacity` doesn't help, because the
allocation is vocab-driven, not sequence-driven. The original document had
tried that and correctly noted it didn't work; we now know why.

## What we tried that didn't work

| Attempt | Outcome |
|---|---|
| Reduce `--maxInputLen` / `--maxKVCacheCapacity` (512/1024 → 128/256 → 64/128) | No change. Allocation is vocab × hidden, not seq_len. |
| Drop caches, stop desktop/docker/snapd, ensure >6.8 GB available | No change. Bug is contiguous-size, not total-memory. |
| Set `kWORKSPACE` memory pool limit to 512 MB via `IBuilderConfig` | Accepted by API, ignored by Myelin autotuner. |
| Set `kTACTIC_DRAM` to 2 GB, then 768 MB | Accepted, does not bind the failing allocation. Myelin's tactic scratch is a separate pool from anything `IBuilderConfig` exposes. |
| Enable zram swap, then 8 GB disk swap during build | No change. NvMap requires physically pinned contiguous memory; swap can't satisfy that. |
| Re-export ONNX with `--trt_native_ops` | Produces ONNX whose `Attention` op the TRT 10.3 parser can't resolve. Different, dead-end error. |
| Upgrade JetPack 6.2.1 → 6.2.2 | Fixes the PyTorch/NVML NvMap path (which unblocks vLLM), does not fix the Myelin path. Same bit-identical error signature. |
| Insert `Identity` nodes as Myelin fusion barriers | TRT's constant-folding pass strips Identity nodes before Myelin sees the graph. No effect. |

## What actually worked

### CMA pool expansion

Default Jetson CMA is 256 MiB. Myelin's tactic 0 wants ~1.025 GiB contiguous,
and falls back to a 593 MiB tactic if tactic 0 fails cleanly. With 256 MiB
available, neither tactic fits — even the fallback. And worse: tactic 0's
12 retries progressively exhaust the small pool, so when the fallback is
attempted, no contiguous memory is free.

Setting `cma=950M` via `/boot/extlinux/extlinux.conf` raises the pool to
960 MiB. Tactic 0 still exceeds that ceiling (1025 > 960) and fails, but
it now fails cleanly on a size check rather than wedging the pool through
partial allocations. Tactic 1's 593 MiB then fits easily and succeeds on
first try.

Why exactly 950M: Jetson's low physical memory comes in two ~992 MiB chunks
(`0x80000000-0xbdffffff` and `0xc2000000-0xfffdffff`, separated by a 64 MiB
firmware carveout), and ARM64 hardcodes CMA placement below 4 GiB. We can't
place CMA in high memory (`cma=2G@0x100000000-0x280000000` is rejected with
"exceeds limit (0x100000000)") and no contiguous region above 992 MiB exists
below 4 GiB. 950M is effectively the ceiling — see the build script for
defensive check.

CMA trace during the failing attempt vs. the succeeding attempt, from the
watcher we ran alongside the build:

```
# Failing (default 256 MiB CMA, no split):
19:59:09  CmaFree 779 MB   ← idle
19:59:13  CmaFree  36 MB   ← tactic 0 retry 4 or so
19:59:14-18  CmaFree 0 MB  ← wedged
19:59:21  CmaFree 932 MB   ← fully released after build error propagates

# Succeeding (960 MB CMA, split ONNX):
CmaFree drops briefly to ~300 MB for each half of the split MatMul
(tactic 1 = 593 MB), then cleanly releases between stages. No zero state.
```

### LM head split surgery

With CMA at 960 MiB the fallback *could* in principle work, but tactic 0
still gets attempted first and its 1025 MiB request still fails 12 times
before Myelin gives up. During those 12 retries the pool is briefly
exhausted, and fallback allocation timing occasionally lands during that
window and fails too. Behavior is flaky.

The clean fix is to remove tactic 0 from consideration entirely, by
preventing Myelin from fusing the LM head into a single node that needs
such a large scratch. `split_lm_head.py` does this: splits the 2048 × 151936
MatMul into two 2048 × 75968 halves with a Concat reassembling the output.
Two smaller fusion candidates each well under 600 MiB scratch. Output is
mathematically identical — this is a graph rewrite, not a model change.

An earlier attempt used `Identity` nodes as fusion barriers. That didn't
work: TRT's graph optimizer constant-folds and removes Identity nodes
before Myelin runs its pattern matcher. Concat is a real data-rearranging
op and the optimizer leaves it in place.

### `setMemoryPoolLimit` patch — kept but doesn't bind

`cpp/builder/builderUtils.cpp:233` originally set `kTACTIC_DRAM` to 2 GiB
(`2ULL << 30`). We patched it to 768 MiB (`768ULL << 20`) intending to force
Myelin to skip the 1 GiB tactic. Confirmed experimentally that this does
not bind — same failure with exact same allocation sizes. Left patched in
the repo as documentary evidence of what was tried, with a comment in the
source explaining the limitation. Future NVIDIA TRT versions may wire this
pool through to Myelin; if so, the patch would start being useful and could
potentially replace the graph surgery.

## Measured cost of the workaround

The split MatMul introduces a Concat op that wasn't in the original graph.
For a single-batch inference, the Concat is one extra kernel launch
(microseconds) and one extra memory allocation for the intermediate tensor
(negligible at batch 1). First-light measurement (warmup 1, single run):

- Prefill: 211 ms for 512 tokens (0.41 ms/tok)
- Decode: 16.7 ms/tok, 59.9 tok/s
- Vision encoder: 208 ms
- Peak unified memory: 4330 MB

These numbers are competitive with vLLM on the same hardware
(56.5 tok/s, 17.4 ms/tok decode). The Concat overhead isn't visible in
the per-token numbers at this batch size. For larger batches or higher
throughput scenarios the overhead may become measurable, though the
fundamental 1 GiB Myelin tactic that blocked the original build scales
worse with batch size, so the split likely remains the faster path.

## Related

- `device/trt_cosmos_patches/README.md` — operator-facing recipe to apply
  the workaround to a fresh setup
- `device/trt_cosmos_patches/split_lm_head.py` — the ONNX surgery
- `device/trt_cosmos_patches/builderUtils_kTACTIC_DRAM.patch` — the
  documentary source patch
- `device/scripts/40_build_cosmos_trt_engines.sh` — end-to-end build
- `device/scripts/41_sanity_cosmos_trt.sh` — post-build correctness check
- `device/scripts/42_bench_cosmos_trt.sh` — benchmark runner
- `device/results/trt/` — profile JSONs and logs
- NVIDIA forum thread on related NvMap issues:
  <https://forums.developer.nvidia.com/t/unable-to-allocate-cuda0-buffer-after-updating-ubuntu-packages/347862>
- JetPack 6.2.2 release:
  <https://docs.nvidia.com/jetson/archives/r36.5/ReleaseNotes/>
- TRT-Edge-LLM repository:
  <https://github.com/NVIDIA/TensorRT-Edge-LLM>
