# TRT-Edge-LLM + Cosmos-Reason2-2B — measurement results

Profile JSONs emitted by `llm_inference --profileOutputFile` during the
benchmark runs. File naming: `trt_profile_{scenario}_{runs}_{timestamp}.json`.

## Files

- `trt_profile_image_single_20260418.json` — first successful end-to-end
  image inference after the engine build landed. Single run, warmup 1.
  Preserved as the "first light" data point. Single-run variance is high;
  use the 5-run aggregates below for published numbers.

## Reading the profile JSON

Top-level keys:

- `prefill` — LLM prefill stage. `average_time_per_run_ms` is the TTFT
  contribution from the LLM side (not including vision encoder).
- `generation` — LLM decode. `average_time_per_token_ms` is TPOT,
  `tokens_per_second` is TPS (both aggregated across all measurement runs).
- `multimodal` — vision encoder. `total_image_tokens` is how many tokens
  the image contributed to the prompt.
- `peak_unified_memory_mb` — high-water mark for the full inference session
  (weights + activations + KV cache + CUDA context), unified memory.
- `stages[]` — per-stage timing breakdown with full distribution
  (min/median/mean/p95/p99/stddev). Stage IDs are `vision_encoder`,
  `llm_prefill`, `llm_generation`.

## Computing TTFT for image prompts

Total TTFT = `stages.vision_encoder.median_ms` + `stages.llm_prefill.median_ms`.

Total TTFT for text-only prompts = `stages.llm_prefill.median_ms` alone
(vision encoder isn't invoked).

## Reproduction

All of these were produced by running:

```bash
./scripts/42_bench_cosmos_trt.sh
```

from the `vlm-benchmarks/device/` directory after the engines were built
with `40_build_cosmos_trt_engines.sh`.
