# Memory snapshot while vLLM serving

Captured on Jetson Orin Nano 8GB (JetPack 6.2.2, L4T 36.5.0) while
`vllm serve embedl/Cosmos-Reason2-2B-W4A16` was active and had just
served a benchmark request.

```
$ free -h
               total        used        free      shared  buff/cache   available
Mem:           7.4Gi       6.5Gi       156Mi       3.0Mi       809Mi       750Mi
Swap:           11Gi       414Mi        11Gi
```

## Interpretation

| Component | Memory |
|---|---|
| vLLM process (model + KV cache + activations + CUDA context) | ~5 GB |
| OS + systemd + kernel | ~400 MB |
| Docker daemon, containerd | ~300 MB |
| Other (snapd was stopped) | ~200 MB |
| Buffers + cache | 809 MB |
| Free (truly unused) | 156 MB |
| **Effectively available to new processes** | **750 MB** |

414 MB swap in use indicates peak transients pushed memory slightly over
RAM ceiling and got paged out (zram + /swapfile combined). Server
remained stable throughout benchmark runs.

## Why this is the interesting number

vLLM's own sanity check refused to start when `free` dropped below
`0.65 × 7.43 = 4.83 GB` — the gate triggered on `5.2 GB free vs 5.2 GB
requested`. With `--gpu-memory-utilization 0.6` (4.46 GB asked) and
`--max-model-len 256` (minimal activation budget), we're sitting at
~5 GB resident with 750 MB genuine headroom.

This is the edge of what Orin Nano 8GB + vLLM + a 2B-class VLM allows.
Any more generous setting (longer context, CUDA graphs enabled,
higher utilization) and vLLM refuses to start. llama.cpp, by contrast,
sits at ~2 GB for the same model class with 4096 context — different
runtime, different allocator strategy, different memory model.

The memory pressure is not a bug; it is the runtime's design tradeoff
made visible on constrained hardware.

## Working configuration reference

```bash
vllm serve embedl/Cosmos-Reason2-2B-W4A16 \
    --host 0.0.0.0 --port 8000 \
    --trust-remote-code --enforce-eager \
    --max-model-len 256 --max-num-batched-tokens 256 \
    --gpu-memory-utilization 0.6 --max-num-seqs 1 \
    --enable-chunked-prefill \
    --limit-mm-per-prompt '{"image":1,"video":1}' \
    --mm-processor-kwargs '{"num_frames":2,"max_pixels":150528}'
```

Docker flags (outside the vllm command):
```
--shm-size=8g --ulimit memlock=-1 --ulimit stack=67108864
```
