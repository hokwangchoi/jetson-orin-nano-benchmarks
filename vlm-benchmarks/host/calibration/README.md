# Calibration data for AWQ quantization

AWQ's per-channel scaling is chosen by running a small held-out set of
inputs through the model and recording activation statistics. This
directory is where domain-specific calibration sets live.

Default: `01_quantize_llm.sh` uses 512 samples from CNN/DailyMail, which
is fine for generic instruction/reasoning tuning. For a narrower
deployment target — say, driving video VLA — swap in a representative
set:

```
calibration/
├── README.md        # this file
├── cnn_dm_512.jsonl # generic baseline (auto-downloaded at runtime)
└── your_data.jsonl  # optional domain-specific calibration
```

The `.jsonl` format is one JSON object per line, each with a `"text"`
field (for text-only calibration) or `"text"` + `"image_path"`
(for multimodal calibration). Point at your file with:

```bash
CALIB_DATASET=calibration/your_data.jsonl ./scripts/01_quantize_llm.sh
```

If you're calibrating on driving or robotics footage, 256–512 samples
covering representative scenes is enough. More than ~1000 gives
diminishing returns and slows the calibration pass linearly.
