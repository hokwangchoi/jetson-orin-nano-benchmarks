# tegrastats captures

100 ms-sampled system-level utilization traces over complete benchmark
runs. Used for the Profiling section of the blog (§11).

## Files

| File | Source | Notes |
|---|---|---|
| `trt_<stamp>.raw` | `sudo tegrastats --interval 100 --logfile <out>` | Raw tegrastats lines, one per 100 ms |
| `trt_<stamp>.csv` | Parsed by `scripts/tegrastats_capture.sh` | Columns: t_ms, ram_mb_used, ram_mb_total, swap_mb_used, cpu_avg_pct, gpu_pct, gpu_freq_mhz, emc_freq_mhz, power_soc_mw, power_cpu_gpu_cv_mw, power_in_mw, board_temp_c |
| `trt_utilization.png` | `scripts/plot_tegrastats.py` | 3-panel plot: GPU%, RAM, VDD_IN power vs time |

## Reproducing

```bash
./scripts/tegrastats_capture.sh trt ./scripts/bench_trt.sh
python3 scripts/plot_tegrastats.py \
    --csv assets/results/tegrastats/trt_*.csv:trt \
    --out assets/results/tegrastats/trt_utilization.png \
    --trim-to-active
```

## Coverage

Only TRT has a tegrastats trace committed here. vLLM and llama.cpp
would work the same way (tegrastats reads the SoC from the host, so
it sees GPU activity from containerized servers too) — not captured
yet, left as future work.
