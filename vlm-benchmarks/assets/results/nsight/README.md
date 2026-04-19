# Nsight Systems captures

One-shot TRT Edge-LLM image inference under `nsys`, used for the
Profiling section of the blog (§11). vLLM and llama.cpp both run
inside docker containers without `--pid=host`, so host-side `nsys`
capture doesn't see their processes — kernel-level view here is
TRT-only.

## Files

| File | Source | Notes |
|---|---|---|
| `trt_image_<stamp>.nsys-rep` | `scripts/nsight_trt.sh` wrapping `llm_inference --inputFile sanity_image.json --warmup 0` | Single image request, 128 generated tokens. Open in Nsight Systems GUI (`nsys-ui`). |
| `trt_overview.png` | Screenshot | Full 8 s timeline — engine load, ViT, prefill, decode |
| `trt_decode_transition.png` | Screenshot | Zoomed to prefill → decode boundary around 4.9–6.0 s |
| `trt_one_tpot.png` | Screenshot | Zoomed to 17 ms — one CUDA graph replay = one TPOT |

## Reproducing

```bash
# Must be in graphical target for nsys-ui, or just capture in multi-user
# target and scp the .nsys-rep to your laptop for viewing.
./scripts/nsight_trt.sh
nsys-ui assets/results/nsight/trt_image_<stamp>.nsys-rep
```
