# TRT Edge-LLM + Cosmos-Reason2-2B: the Myelin blocker

Investigation notes from April 2026. Kept here as a reference for
future retry attempts and as a worked example of how a platform-level
memory bug surfaces in TensorRT's graph optimizer.

## The error

Reproducibly, on JetPack 6.2.1 (L4T 36.4.7) **and** on JetPack 6.2.2
(L4T 36.5.0), `llm_build` fails during engine construction for the
quantized ONNX export of Cosmos-Reason2-2B:

```
[INFO] [TensorRT] Timing Runner: {ForeignNode[/Unsqueeze.../Cast]} (Myelin[0x80000023])
[INFO] [TensorRT] MemUsageChange: CPU 2380 MB, GPU 3710 MB
NvMapMemAllocInternalTagged: 1075072515 error 12   # repeats 12 times
[ERROR] [TensorRT] CUDA error 2 for 622329856-byte allocation.
[ERROR] [TensorRT] IBuilder::buildSerializedNetwork: Could not find any
                   implementation for node {ForeignNode[/Unsqueeze.../Cast]}.
[ERROR] [llm_build.cpp:234:main] Failed to build LLM engine.
```

## What's happening

- TRT parses the ONNX graph. Edge-LLM's custom `AttentionPlugin` and
  `Int4GroupwiseGemmPlugin` nodes register cleanly — they have a single
  tactic, so the autotuner skips them.
- The rotary-position-embedding math (a fused
  Unsqueeze/GatherND/Cast subgraph) doesn't match either plugin, so TRT
  hands it to Myelin — its own fused-subgraph optimizer.
- Myelin runs tactic selection on the fused graph. Default scratch
  workspace for its first tactic is exactly `1,075,072,515` bytes
  (~1 GiB with some internal bookkeeping).
- The kernel driver's NvMap subsystem refuses the contiguous
  allocation. Tactic 0 fails 12 times, Myelin falls back to a smaller
  622 MB tactic, that also fails (memory context is wedged), no other
  tactic satisfies the op, build aborts.

The error size `1,075,072,515` is consistent across attempts and matches
reports from other users hitting NvMap issues on the same L4T revision.

## What doesn't fix it

| Attempt | Outcome |
|---|---|
| Reduce `--maxInputLen` / `--maxKVCacheCapacity` (512/1024 → 128/256) | No change. The allocation is for tactic-selection scratch, not KV cache. |
| Drop caches, stop desktop/docker/snapd, ensure >6.8 GB available | No change. Bug is contiguous-size, not total-memory. |
| Patch `builderUtils.cpp` to call `setMemoryPoolLimit(kWORKSPACE, 512 << 20)` | Accepted by API, ignored by Myelin autotuner. Myelin's pool is separate. |
| Patch to set `kTACTIC_DRAM, 2ULL << 30` (must be power of 2) | Accepted, still doesn't bind the failing allocation. |
| Re-export ONNX with `--trt_native_ops` flag | Produces ONNX with `op_type: "Attention"` that TRT 10.3's ONNX parser can't resolve. Different error, dead end. |
| Upgrade JetPack 6.2.1 → 6.2.2 (L4T 36.4.7 → 36.5.0) | Fixes PyTorch/NVML path (vLLM now works). Does not fix this Myelin path. Same exact error signature. |

## What would fix it

Likely, in order of probability:

1. **NVIDIA patches the Myelin-specific NvMap path** in a future L4T
   update. They've shown willingness to patch this area (L4T 36.5 did
   exactly that for one path).
2. **TensorRT 10.4+ lands on Jetson** via a future JetPack. Newer TRT
   has reworked Myelin; the autotuner behavior changes.
3. **TRT-Edge-LLM ships a workaround** in its export pipeline — either
   forcing a decomposition that avoids the Myelin fusion, or explicit
   tactic filtering in the builder.

## Verifying the bug is still present

After a JetPack upgrade, re-run engine build with the saved Cosmos-2B
ONNX. If the exact error signature changes — different allocation size,
different failed node, different error code — the bug has moved or been
partially patched, and it's worth a deeper attempt. If it's bit-identical,
skip and wait for the next release.

```bash
# Minimal reproduction
cd ~/TensorRT-Edge-LLM
export EDGELLM_PLUGIN_PATH=$(pwd)/build/libNvInfer_edgellm_plugin.so
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Cosmos-Reason2-2B

./build/examples/llm/llm_build \
    --onnxDir $WORKSPACE_DIR/$MODEL_NAME/onnx/llm \
    --engineDir $WORKSPACE_DIR/$MODEL_NAME/engine/llm \
    --maxBatchSize 1 \
    --maxInputLen 256 \
    --maxKVCacheCapacity 512 \
    --debug 2>&1 | tee /tmp/build_debug.log

# Check for the signature
grep "NvMapMemAllocInternalTagged: 1075072515" /tmp/build_debug.log
# If present, still blocked. If absent, something changed — investigate.
```

## For the retry workflow

When this is resolved, the steps to fill in TRT Edge-LLM's Cosmos-2B
benchmark numbers are:

1. Confirm JetPack version has the fix.
2. Restore the saved quantized ONNX at `~/tensorrt-edgellm-workspace/Cosmos-Reason2-2B/onnx/`.
3. Run `device/scripts/40_build_cosmos_engines.sh` (TBD script, to be
   written when the path unblocks).
4. Run `device/scripts/41_run_cosmos_trt_inference.sh`.
5. Rerun the harness against TRT's port-8000 endpoint with the same
   prompt set the llama.cpp and vLLM runs used.
6. Add the numbers to the blog post's results table, drop the "pending"
   note.

Total work at that point should be <1 hour.

## Related reading

- NVIDIA forum thread on the broader NvMap issue:
  https://forums.developer.nvidia.com/t/unable-to-allocate-cuda0-buffer-after-updating-ubuntu-packages/347862
- JetPack 6.2.2 release (L4T 36.5.0):
  https://docs.nvidia.com/jetson/archives/r36.5/ReleaseNotes/
- TRT Edge-LLM repository and changelog:
  https://github.com/NVIDIA/TensorRT-Edge-LLM
