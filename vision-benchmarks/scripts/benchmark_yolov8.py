#!/usr/bin/env python3
"""
YOLOv8 Benchmark for Jetson Orin Nano
Measures latency, throughput, memory, and power across PyTorch and TensorRT.
"""

import subprocess
import time
import json
import os
import re
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "assets" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_tegrastats_reading():
    """Get single tegrastats reading."""
    try:
        result = subprocess.run(
            ["tegrastats", "--interval", "100"],
            capture_output=True, text=True, timeout=1
        )
        return result.stdout.strip()
    except:
        return ""


def parse_power(line):
    """Parse power in watts from tegrastats."""
    match = re.search(r'VDD_IN\s+(\d+)mW', line)
    if match:
        return int(match.group(1)) / 1000
    return None


def benchmark_trtexec(engine_path):
    """Benchmark TensorRT engine."""
    precision = "FP16" if "fp16" in engine_path else "INT8" if "int8" in engine_path else "FP32"
    print(f"[TensorRT {precision}] Benchmarking {engine_path}...")
    
    result = subprocess.run([
        "trtexec",
        f"--loadEngine={engine_path}",
        "--batch=1",
        "--warmUp=500",
        "--duration=10",
        "--avgRuns=100"
    ], capture_output=True, text=True)
    
    output = result.stdout + result.stderr
    
    # Parse latency
    latency = None
    for line in output.split("\n"):
        if "mean:" in line.lower():
            match = re.search(r'mean:\s*([\d.]+)\s*ms', line)
            if match:
                latency = float(match.group(1))
                break
    
    # Parse throughput
    fps = None
    for line in output.split("\n"):
        if "throughput:" in line.lower():
            match = re.search(r'throughput:\s*([\d.]+)', line)
            if match:
                fps = float(match.group(1))
                break
    
    # Get power
    stats = get_tegrastats_reading()
    power = parse_power(stats)
    
    return {
        "runtime": "TensorRT",
        "precision": precision,
        "latency_ms": round(latency, 2) if latency else None,
        "fps": round(fps, 2) if fps else None,
        "power_w": round(power, 1) if power else None
    }


def export_and_build():
    """Export ONNX and build TensorRT engines."""
    from ultralytics import YOLO
    
    # Download and export
    if not os.path.exists("yolov8n.onnx"):
        print("[Export] PyTorch → ONNX...")
        model = YOLO("yolov8n.pt")
        model.export(format="onnx", imgsz=640)
    
    # Build engines
    engines = [
        ("yolov8n_fp32.engine", ""),
        ("yolov8n_fp16.engine", "--fp16"),
        ("yolov8n_int8.engine", "--int8")
    ]
    
    for engine, flags in engines:
        if not os.path.exists(engine):
            print(f"[Build] {engine}...")
            cmd = f"trtexec --onnx=yolov8n.onnx {flags} --saveEngine={engine}"
            subprocess.run(cmd.split(), check=True)


def main():
    print("=" * 60)
    print("YOLOv8 Vision Benchmark — Jetson Orin Nano")
    print("=" * 60)
    print()
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "device": "Jetson Orin Nano 8GB",
        "model": "yolov8n",
        "input_size": [640, 640],
        "benchmarks": []
    }
    
    # Build if needed
    export_and_build()
    
    # Run benchmarks
    print("\n[Benchmark] Running...")
    for engine in ["yolov8n_fp32.engine", "yolov8n_fp16.engine", "yolov8n_int8.engine"]:
        if os.path.exists(engine):
            result = benchmark_trtexec(engine)
            results["benchmarks"].append(result)
            print(f"  {result['precision']}: {result['latency_ms']}ms, {result['fps']} FPS")
    
    # Save
    out_file = RESULTS_DIR / f"yolov8_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[Done] Results saved to {out_file}")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"{'Runtime':<12} {'Precision':<10} {'Latency':<12} {'FPS':<10}")
    print("-" * 60)
    for b in results["benchmarks"]:
        lat = f"{b['latency_ms']:.2f}ms" if b['latency_ms'] else "—"
        fps = f"{b['fps']:.1f}" if b['fps'] else "—"
        print(f"{b['runtime']:<12} {b['precision']:<10} {lat:<12} {fps:<10}")


if __name__ == "__main__":
    main()
