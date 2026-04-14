#!/usr/bin/env python3
"""
YOLOv8 Benchmark for Jetson Orin Nano
Uses CUDA events and Nsight Systems for precise GPU-only latency.

Hardware: Jetson Orin Nano 8GB Developer Kit
Storage: Samsung PRO Plus 256GB microSD
"""

import subprocess
import time
import json
import os
import re
import threading
from datetime import datetime
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "assets" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_SIZE = (640, 640)
WARMUP_RUNS = 100
BENCHMARK_RUNS = 500
BATCH_SIZE = 1

# TensorRT benchmark config
TRT_WARMUP_MS = 1000
TRT_DURATION_S = 15
TRT_ITERATIONS = 5  # Run multiple times for stable averages

# Hardware specs for utilization calculation
ORIN_NANO_TOPS_INT8 = 40   # Theoretical peak INT8 TOPS (MAXN SUPER mode)
ORIN_NANO_TFLOPS_FP16 = 20  # Theoretical peak FP16 TFLOPS
ORIN_NANO_TFLOPS_FP32 = 10  # Theoretical peak FP32 TFLOPS (estimated)

# YOLOv8n model specs (640x640 input)
YOLOV8N_GFLOPS = 8.7  # Giga FLOPs per inference (from ultralytics)
YOLOV8N_PARAMS = 3.2e6  # 3.2M parameters


def get_model_flops():
    """
    Get YOLOv8n FLOPs. Returns GFLOPs.
    
    FLOPs = Floating Point Operations (total ops for one inference)
    - Multiply-accumulate (MAC) counts as 2 FLOPs
    - YOLOv8n @ 640x640: ~8.7 GFLOPs
    """
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        # model.info() returns string with GFLOPs
        info = model.info(verbose=False)
        # Return known value for YOLOv8n
        return YOLOV8N_GFLOPS
    except:
        return YOLOV8N_GFLOPS


def calculate_tops(latency_ms, gflops=YOLOV8N_GFLOPS, precision="FP32"):
    """
    Calculate achieved TOPS/TFLOPS from latency.
    
    TOPS = Tera Operations Per Second
    TFLOPS = Tera Floating-point Operations Per Second
    
    Formula: TOPS = (GFLOPs / latency_s) / 1000
    
    Args:
        latency_ms: GPU latency in milliseconds
        gflops: Model GFLOPs (default: YOLOv8n = 8.7)
        precision: FP32, FP16, or INT8
    
    Returns:
        dict with tops, theoretical_peak, and utilization percentage
    """
    latency_s = latency_ms / 1000
    achieved_tops = (gflops / latency_s) / 1000  # GFLOPs/s -> TOPS
    
    # Get theoretical peak for this precision
    if precision == "INT8":
        peak = ORIN_NANO_TOPS_INT8
        unit = "TOPS"
    elif precision == "FP16":
        peak = ORIN_NANO_TFLOPS_FP16
        unit = "TFLOPS"
    else:  # FP32
        peak = ORIN_NANO_TFLOPS_FP32
        unit = "TFLOPS"
    
    utilization = (achieved_tops / peak) * 100
    
    return {
        "achieved": round(achieved_tops, 3),
        "peak": peak,
        "utilization_pct": round(utilization, 1),
        "unit": unit
    }


class TegrastatsMonitor:
    """Background monitor for power."""
    
    def __init__(self):
        self.running = False
        self.samples = []
        self.thread = None
        self.process = None
    
    def start(self):
        self.running = True
        self.samples = []
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()
        time.sleep(0.3)
    
    def stop(self):
        self.running = False
        if self.process:
            self.process.terminate()
        if self.thread:
            self.thread.join(timeout=2)
        return self._parse_samples()
    
    def _monitor(self):
        self.process = subprocess.Popen(
            ["tegrastats", "--interval", "100"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        while self.running:
            line = self.process.stdout.readline()
            if line:
                self.samples.append(line.strip())
    
    def _parse_samples(self):
        powers = []
        for line in self.samples:
            match = re.search(r'VDD_IN\s+(\d+)mW', line)
            if match:
                powers.append(int(match.group(1)) / 1000)
        return {
            "power_w": round(np.mean(powers), 2) if powers else None,
            "power_std_w": round(np.std(powers), 2) if powers else None,
        }


def get_system_info():
    """Get Jetson system information."""
    info = {
        "device": "Jetson Orin Nano 8GB",
        "storage": "Samsung PRO Plus 256GB microSD"
    }
    
    try:
        with open("/etc/nv_tegra_release") as f:
            line = f.readline()
            match = re.search(r'R(\d+).*REVISION:\s*([\d.]+)', line)
            if match:
                info["l4t_version"] = f"L4T {match.group(1)}.{match.group(2)}"
    except:
        pass
    
    try:
        result = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True)
        match = re.search(r'NV Power Mode:\s*(\w+)', result.stdout)
        if match:
            info["power_mode"] = match.group(1)
    except:
        pass
    
    return info


def benchmark_pytorch_cuda_events(num_runs=BENCHMARK_RUNS, warmup=WARMUP_RUNS):
    """
    Benchmark PyTorch using CUDA events for precise GPU-only timing.
    CUDA events measure only kernel execution, excluding Python overhead.
    """
    print("\n[PyTorch FP32] Benchmarking with CUDA events...")
    
    import torch
    from ultralytics import YOLO
    
    model = YOLO("yolov8n.pt")
    model.model.eval()
    model.model.cuda()
    
    dummy = torch.randn(BATCH_SIZE, 3, *INPUT_SIZE).cuda()
    
    # Warmup
    print(f"  Warmup ({warmup} runs)...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model.model(dummy)
    torch.cuda.synchronize()
    
    # Benchmark with CUDA events (GPU-only timing)
    print(f"  Benchmarking ({num_runs} runs)...")
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    monitor = TegrastatsMonitor()
    monitor.start()
    
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_event.record()
            _ = model.model(dummy)
            end_event.record()
            torch.cuda.synchronize()
            latencies.append(start_event.elapsed_time(end_event))
    
    power_stats = monitor.stop()
    
    latency_ms = np.mean(latencies)
    latency_std = np.std(latencies)
    fps = 1000 / latency_ms
    tops_info = calculate_tops(latency_ms, precision="FP32")
    
    return {
        "runtime": "PyTorch",
        "precision": "FP32",
        "gpu_latency_ms": round(latency_ms, 2),
        "gpu_latency_std_ms": round(latency_std, 3),
        "fps": round(fps, 1),
        "fps_std": round(1000 / (latency_ms + latency_std) - fps, 1) if latency_std > 0 else 0,
        "power_w": power_stats["power_w"],
        "tops": tops_info["achieved"],
        "tops_utilization_pct": tops_info["utilization_pct"],
        "iterations": num_runs,
        "measurement": "CUDA events (GPU-only)"
    }


def benchmark_trtexec(engine_path):
    """
    Benchmark TensorRT using trtexec with multiple iterations for stable averages.
    Extracts GPU Compute Time (excludes H2D/D2H transfers).
    """
    precision = "FP16" if "fp16" in engine_path else "INT8" if "int8" in engine_path else "FP32"
    print(f"\n[TensorRT {precision}] Benchmarking ({TRT_ITERATIONS} runs × {TRT_DURATION_S}s)...")
    
    gpu_latencies = []
    host_latencies = []
    throughputs = []
    power_samples = []
    
    for i in range(TRT_ITERATIONS):
        print(f"  Run {i+1}/{TRT_ITERATIONS}...", end=" ", flush=True)
        
        monitor = TegrastatsMonitor()
        monitor.start()
        
        result = subprocess.run([
            "trtexec",
            f"--loadEngine={engine_path}",
            f"--warmUp={TRT_WARMUP_MS}",
            f"--duration={TRT_DURATION_S}",
            "--useSpinWait"
        ], capture_output=True, text=True)
        
        power_stats = monitor.stop()
        output = result.stdout + result.stderr
        
        # GPU Compute Time (kernel execution only)
        for line in output.split("\n"):
            if "GPU Compute Time:" in line:
                match = re.search(r'mean\s*=\s*([\d.]+)\s*ms', line)
                if match:
                    gpu_latencies.append(float(match.group(1)))
                    break
        
        # Host Latency
        for line in output.split("\n"):
            if "Latency:" in line and "mean" in line:
                match = re.search(r'mean\s*=\s*([\d.]+)\s*ms', line)
                if match:
                    host_latencies.append(float(match.group(1)))
                    break
        
        # Throughput
        for line in output.split("\n"):
            if "Throughput:" in line:
                match = re.search(r'Throughput:\s*([\d.]+)\s*qps', line)
                if match:
                    throughputs.append(float(match.group(1)))
                    break
        
        if power_stats["power_w"]:
            power_samples.append(power_stats["power_w"])
        
        print(f"GPU: {gpu_latencies[-1]:.2f}ms" if gpu_latencies else "failed")
    
    # Calculate TOPS
    mean_latency = np.mean(gpu_latencies) if gpu_latencies else None
    tops_info = calculate_tops(mean_latency, precision=precision) if mean_latency else None
    
    return {
        "runtime": "TensorRT",
        "precision": precision,
        "gpu_latency_ms": round(np.mean(gpu_latencies), 2) if gpu_latencies else None,
        "gpu_latency_std_ms": round(np.std(gpu_latencies), 3) if gpu_latencies else None,
        "host_latency_ms": round(np.mean(host_latencies), 2) if host_latencies else None,
        "fps": round(np.mean(throughputs), 1) if throughputs else None,
        "fps_std": round(np.std(throughputs), 1) if throughputs else None,
        "power_w": round(np.mean(power_samples), 2) if power_samples else None,
        "power_std_w": round(np.std(power_samples), 2) if power_samples else None,
        "tops": tops_info["achieved"] if tops_info else None,
        "tops_utilization_pct": tops_info["utilization_pct"] if tops_info else None,
        "iterations": len(gpu_latencies),
        "measurement": "trtexec GPU Compute Time (mean of means)"
    }


def profile_with_nsight(engine_path, output_name):
    """
    Run Nsight Systems profiling for detailed kernel analysis.
    Saves .nsys-rep file for GUI inspection.
    """
    precision = "FP16" if "fp16" in engine_path else "INT8" if "int8" in engine_path else "FP32"
    print(f"\n[Nsight Systems] Profiling TensorRT {precision}...")
    
    profile_dir = RESULTS_DIR / "nsight_profiles"
    profile_dir.mkdir(exist_ok=True)
    
    output_path = profile_dir / output_name
    
    result = subprocess.run([
        "nsys", "profile",
        "--trace=cuda,nvtx",
        "--force-overwrite=true",
        f"--output={output_path}",
        "trtexec",
        f"--loadEngine={engine_path}",
        "--warmUp=500",
        "--duration=5"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"  Profile saved: {output_path}.nsys-rep")
        return str(output_path) + ".nsys-rep"
    else:
        print(f"  Profiling failed: {result.stderr[:200]}")
        return None


def export_onnx():
    """
    Export to ONNX for:
    1. Visual inspection with Netron (https://netron.app)
    2. Conversion to TensorRT engines
    """
    if os.path.exists("yolov8n.onnx"):
        print("[ONNX] Already exists (use Netron to inspect)")
        return
    print("[ONNX] Exporting YOLOv8n for Netron inspection + TensorRT conversion...")
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    model.export(format="onnx", imgsz=INPUT_SIZE[0], opset=17)
    print("  View with: netron yolov8n.onnx (or https://netron.app)")


def build_tensorrt_engines():
    engines = [
        ("yolov8n_fp32.engine", ""),
        ("yolov8n_fp16.engine", "--fp16"),
        ("yolov8n_int8.engine", "--int8")
    ]
    for engine, flags in engines:
        if not os.path.exists(engine):
            print(f"[TensorRT] Building {engine}...")
            cmd = f"trtexec --onnx=yolov8n.onnx {flags} --saveEngine={engine}"
            subprocess.run(cmd.split(), capture_output=True, text=True)


def main():
    print("=" * 70)
    print("YOLOv8 Benchmark — Jetson Orin Nano")
    print("GPU-only latency via CUDA events and trtexec")
    print("=" * 70)
    
    sys_info = get_system_info()
    print(f"\nSystem: {sys_info}")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "system": sys_info,
        "config": {
            "model": "yolov8n",
            "model_gflops": YOLOV8N_GFLOPS,
            "model_params": YOLOV8N_PARAMS,
            "input_size": INPUT_SIZE,
            "batch_size": BATCH_SIZE,
            "pytorch_warmup_runs": WARMUP_RUNS,
            "pytorch_benchmark_runs": BENCHMARK_RUNS,
            "trt_iterations": TRT_ITERATIONS,
            "trt_duration_per_iteration_s": TRT_DURATION_S,
            "hardware_peak_tops_int8": ORIN_NANO_TOPS_INT8,
            "hardware_peak_tflops_fp16": ORIN_NANO_TFLOPS_FP16,
            "notes": "GPU latency excludes H2D/D2H transfers. TOPS = GFLOPs / latency_s / 1000. Utilization = achieved / peak."
        },
        "benchmarks": []
    }
    
    # Prepare
    print("\n" + "=" * 70)
    print("Preparing models...")
    print("=" * 70)
    export_onnx()
    build_tensorrt_engines()
    
    # Benchmarks
    print("\n" + "=" * 70)
    print("Running benchmarks (GPU-only latency)...")
    print("=" * 70)
    
    # PyTorch with CUDA events
    try:
        pytorch_result = benchmark_pytorch_cuda_events()
        results["benchmarks"].append(pytorch_result)
    except Exception as e:
        print(f"  PyTorch failed: {e}")
    
    # TensorRT (trtexec reports GPU Compute Time separately)
    for engine in ["yolov8n_fp32.engine", "yolov8n_fp16.engine", "yolov8n_int8.engine"]:
        if os.path.exists(engine):
            try:
                trt_result = benchmark_trtexec(engine)
                results["benchmarks"].append(trt_result)
            except Exception as e:
                print(f"  TensorRT {engine} failed: {e}")
    
    # Optional: Nsight profiles for detailed analysis
    print("\n" + "=" * 70)
    print("Generating Nsight Systems profiles...")
    print("=" * 70)
    
    nsight_profiles = []
    for engine, name in [("yolov8n_fp16.engine", "yolov8n_fp16"), 
                          ("yolov8n_int8.engine", "yolov8n_int8")]:
        if os.path.exists(engine):
            profile = profile_with_nsight(engine, name)
            if profile:
                nsight_profiles.append(profile)
    
    results["nsight_profiles"] = nsight_profiles
    
    # Save
    out_file = RESULTS_DIR / f"yolov8_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print("\n" + "=" * 100)
    print("Results Summary (GPU-only latency, mean ± std)")
    print("=" * 100)
    print(f"{'Runtime':<12} {'Precision':<8} {'GPU Latency':<16} {'FPS':<12} {'TOPS':<8} {'Util%':<8} {'Power':<8}")
    print("-" * 100)
    
    for b in results["benchmarks"]:
        runtime = b.get("runtime", "?")
        precision = b.get("precision", "?")
        
        lat_mean = b.get('gpu_latency_ms')
        lat_std = b.get('gpu_latency_std_ms', 0)
        lat = f"{lat_mean:.2f}±{lat_std:.2f}ms" if lat_mean else "—"
        
        fps_mean = b.get('fps')
        fps = f"{fps_mean:.1f}" if fps_mean else "—"
        
        tops = f"{b['tops']:.2f}" if b.get('tops') else "—"
        util = f"{b['tops_utilization_pct']:.1f}%" if b.get('tops_utilization_pct') else "—"
        
        pwr = f"{b['power_w']:.1f}W" if b.get('power_w') else "—"
        
        print(f"{runtime:<12} {precision:<8} {lat:<16} {fps:<12} {tops:<8} {util:<8} {pwr:<8}")
    
    print(f"\nResults: {out_file}")
    
    # Speedup vs TRT FP32
    print("\n" + "=" * 80)
    print("Speedup Analysis")
    print("=" * 80)
    
    # PyTorch vs TRT FP32
    pytorch_lat = next((b['gpu_latency_ms'] for b in results["benchmarks"] 
                        if b.get('runtime') == 'PyTorch'), None)
    trt_fp32_lat = next((b['gpu_latency_ms'] for b in results["benchmarks"] 
                         if b.get('runtime') == 'TensorRT' and b.get('precision') == 'FP32'), None)
    
    if pytorch_lat and trt_fp32_lat:
        speedup = pytorch_lat / trt_fp32_lat
        print(f"  TensorRT FP32 vs PyTorch: {speedup:.2f}x faster")
    
    # TRT FP16/INT8 vs TRT FP32
    if trt_fp32_lat:
        for b in results["benchmarks"]:
            if b.get('gpu_latency_ms') and b.get('runtime') == 'TensorRT' and b.get('precision') != 'FP32':
                speedup = trt_fp32_lat / b['gpu_latency_ms']
                print(f"  TensorRT {b['precision']} vs TensorRT FP32: {speedup:.2f}x faster")
    
    if nsight_profiles:
        print(f"\nNsight profiles saved for detailed kernel analysis.")
        print("Open with: nsys-ui <profile>.nsys-rep")
    
    # TOPS Analysis
    print("\n" + "=" * 100)
    print("TOPS/TFLOPS Analysis (Hardware Utilization)")
    print("=" * 100)
    print(f"Model: YOLOv8n @ 640x640 = {YOLOV8N_GFLOPS} GFLOPs per inference")
    print(f"Hardware: Orin Nano 8GB (MAXN SUPER) = {ORIN_NANO_TOPS_INT8} TOPS INT8 / {ORIN_NANO_TFLOPS_FP16} TFLOPS FP16")
    print()
    print("Why utilization is low:")
    print("  - Small model (8.7 GFLOPs) doesn't saturate GPU")
    print("  - Memory-bound operations limit Tensor Core usage")
    print("  - Kernel launch overhead dominates for small workloads")
    print("  - Higher utilization requires larger batches or models")
    print()
    print("TOPS formula: achieved_tops = model_gflops / latency_seconds / 1000")
    print("Utilization: achieved_tops / hardware_peak_tops × 100%")


if __name__ == "__main__":
    main()
