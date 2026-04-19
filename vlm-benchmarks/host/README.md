# Phase 0 — Host-side quantization + ONNX export

The TRT-Edge-LLM path for Cosmos-Reason2-2B on Jetson Orin Nano needs a
quantized ONNX export produced on a datacenter GPU. Doing the AWQ
calibration and ONNX export on the Jetson itself is impractical:
ModelOpt's calibration pass loads the full BF16 weights (~5 GB) plus
activation histograms, and the Jetson doesn't have headroom for it.

The phase runs on any x86 machine with one Ampere/Hopper GPU with
≥40 GB of VRAM (A40, A100 40 GB, L40S, H100, etc). Took ~30 minutes
end-to-end on a rented A40 pod.

## Outputs

This phase produces the files that are `scp`'d to the Jetson and
consumed by `device/scripts/40_build_cosmos_trt_engines.sh`:

| Artifact | Approx. size | Used by |
|---|---:|---|
| `cosmos-2b-quantized/` — checkpoint in AWQ W4A16 form | ~1.3 GB | `export_llm` |
| `cosmos-2b-onnx/llm/onnx_model.onnx` + `onnx_model.data` | ~1.2 GB | `llm_build` on Jetson |
| `cosmos-2b-onnx/visual/visual.onnx` | ~0.6 GB | `visual_build` on Jetson |

Total transfer to Jetson: ~2 GB.

## Workflow

```
HuggingFace checkpoint (nvidia/Cosmos-Reason2-2B, BF16)
        │
        ▼
┌─────────────────────────────────────┐
│ 01_quantize_llm.sh                  │  AWQ W4A16 via ModelOpt, ~4 min
└───────────┬─────────────────────────┘
            ▼
┌─────────────────────────────────────┐
│ 02_export_llm.sh                    │  Quantized checkpoint → ONNX + data, ~10 min
│ 03_export_visual.sh                 │  ViT → visual.onnx, ~3 min
└───────────┬─────────────────────────┘
            ▼
┌─────────────────────────────────────┐
│ 04_package_for_jetson.sh            │  tar + scp to Jetson
└─────────────────────────────────────┘
```

## Running

```bash
cd vlm-benchmarks/host
./scripts/01_quantize_llm.sh
./scripts/02_export_llm.sh
./scripts/03_export_visual.sh
./scripts/04_package_for_jetson.sh  # edit JETSON_HOST first
```

Each script is idempotent and checks for existing outputs before
rerunning. All three output into `$WORKSPACE/cosmos-2b-{quantized,onnx}/`
by default (`$WORKSPACE` defaults to `/workspace`, override per your
pod's layout).

## Prerequisites

- CUDA 12.6+, Python 3.10+
- ~80 GB free disk space (HF checkpoint cache + calibration intermediates + ONNX)
- `pip install nvidia-modelopt onnx transformers` (plus TRT-Edge-LLM
  host tooling — see `00_setup_host.sh` if you need a clean-install
  reference)
- HuggingFace auth token with access to `nvidia/Cosmos-Reason2-2B`

## Calibration

The AWQ quantization calibrates on a small held-out set of inputs to
pick per-channel scaling. `01_quantize_llm.sh` uses 512 samples from
CNN/DailyMail by default. For a domain-specific deployment (e.g. driving
video understanding), swap in a representative sample set: put ~500
examples as `.jsonl` with `{"text": "..."}` records in
`calibration/your_dataset.jsonl` and point `--calib-dataset` at it.

## Notes on reproducibility

These scripts were originally run on an A40 pod; the exact CLI names and
flags for TRT-Edge-LLM's quantize/export tools have varied between
versions. If you're reading this against a newer TRT-Edge-LLM checkout,
cross-reference `TensorRT-Edge-LLM/examples/llm/` for the current
argument names. The overall shape — quantize, then export LLM ONNX,
then export visual ONNX — has been stable.
