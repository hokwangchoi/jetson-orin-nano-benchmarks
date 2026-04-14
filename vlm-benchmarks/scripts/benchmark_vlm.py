#!/usr/bin/env python3
"""
VLM Benchmark for Jetson Orin Nano
Measures TTFT, tokens/sec, memory for vision-language models.
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


def get_memory_usage():
    """Get current memory usage in MB."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    available_kb = int(line.split()[1])
                    break
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total_kb = int(line.split()[1])
                    break
        used_mb = (total_kb - available_kb) // 1024
        return used_mb
    except:
        return None


def benchmark_llama_cpp(model_path, image_path, prompt, num_tokens=100):
    """
    Benchmark VLM using llama.cpp (if available).
    Returns TTFT and tokens/sec.
    """
    print(f"[llama.cpp] Benchmarking {os.path.basename(model_path)}...")
    
    # This is a placeholder - actual implementation depends on llama.cpp setup
    # For llava models:
    cmd = [
        "./llama-llava-cli",
        "-m", model_path,
        "--image", image_path,
        "-p", prompt,
        "-n", str(num_tokens),
        "--temp", "0"
    ]
    
    mem_before = get_memory_usage()
    start = time.perf_counter()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        end = time.perf_counter()
        
        # Parse timing from output
        # llama.cpp prints: "llama_print_timings: load time = X ms"
        output = result.stdout + result.stderr
        
        ttft = None
        tps = None
        
        for line in output.split("\n"):
            if "eval time" in line.lower():
                match = re.search(r'([\d.]+)\s*tokens per second', line)
                if match:
                    tps = float(match.group(1))
            if "prompt eval time" in line.lower():
                match = re.search(r'([\d.]+)\s*ms', line)
                if match:
                    ttft = float(match.group(1))
        
        mem_after = get_memory_usage()
        
        return {
            "ttft_ms": ttft,
            "tokens_per_sec": tps,
            "memory_mb": mem_after - mem_before if mem_before and mem_after else None,
            "total_time_s": round(end - start, 2)
        }
    except Exception as e:
        print(f"  Error: {e}")
        return None


def benchmark_transformers(model_name, image_path, prompt, num_tokens=100):
    """
    Benchmark VLM using HuggingFace Transformers.
    """
    print(f"[Transformers] Benchmarking {model_name}...")
    
    try:
        import torch
        from transformers import AutoProcessor, AutoModelForVision2Seq
        from PIL import Image
    except ImportError:
        print("  Transformers not installed. Run: pip3 install transformers accelerate")
        return None
    
    mem_before = get_memory_usage()
    
    # Load model
    print("  Loading model...")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_4bit=True  # Requires bitsandbytes
    )
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")
    
    # Warmup
    print("  Warming up...")
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=10)
    torch.cuda.synchronize()
    
    # Benchmark
    print("  Benchmarking...")
    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=num_tokens)
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    # Calculate metrics
    total_time = end - start
    num_generated = outputs.shape[1] - inputs["input_ids"].shape[1]
    tps = num_generated / total_time
    
    mem_after = get_memory_usage()
    
    return {
        "ttft_ms": None,  # Would need hooks to measure accurately
        "tokens_per_sec": round(tps, 2),
        "memory_mb": mem_after - mem_before if mem_before and mem_after else None,
        "total_time_s": round(total_time, 2),
        "tokens_generated": num_generated
    }


def main():
    print("=" * 60)
    print("VLM Benchmark — Jetson Orin Nano")
    print("=" * 60)
    print()
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "device": "Jetson Orin Nano 8GB",
        "benchmarks": []
    }
    
    # Test image (create a simple one if not exists)
    test_image = Path(__file__).parent / "test_image.jpg"
    if not test_image.exists():
        print("[Setup] Creating test image...")
        try:
            from PIL import Image
            import numpy as np
            img = Image.fromarray(np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8))
            img.save(test_image)
        except:
            print("  Could not create test image. Please provide test_image.jpg")
            return
    
    prompt = "Describe this image in detail."
    
    # Models to benchmark
    models = [
        # ("Qwen/Qwen2-VL-2B-Instruct", "qwen2-vl-2b"),
        # Add more models as needed
    ]
    
    print("\n[Info] VLM benchmarking requires model downloads.")
    print("       This may take a while on first run.\n")
    
    # Placeholder results
    print("[Note] Full VLM benchmarks require specific model setup.")
    print("       See vlm-benchmarks/README.md for instructions.\n")
    
    # Save placeholder results
    results["benchmarks"] = [
        {"model": "Qwen2.5-VL-3B", "precision": "INT4", "status": "pending"},
        {"model": "Cosmos-Reason-2B", "precision": "FP8", "status": "pending"}
    ]
    
    out_file = RESULTS_DIR / f"vlm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"[Done] Placeholder results saved to {out_file}")
    print("\nTo run actual benchmarks:")
    print("  1. Install: pip3 install transformers accelerate bitsandbytes")
    print("  2. Download models from HuggingFace")
    print("  3. Update model paths in this script")


if __name__ == "__main__":
    main()
