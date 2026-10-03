"""
   Benchmark the latency and memory footprint of a model's predict() call.

   Args:
       model: Model instance inheriting from BasePlantModel.
       image_path: Path to sample image for testing.
       warmup_runs: Initial unmeasured runs to prime GPU/CPU caches.
       test_runs: Number of timed executions for calculating average latency.
       top_k: Top K predictions parameter passed to predict().

   Returns:
       Dict containing average latency (ms), FPS, peak RAM (MB), and VRAM (MB).
   """

# core/benchmarking.py

import csv
import os
import time
from typing import Dict, Any, List

import psutil
import torch


def measure_inference_speed(
    model,
    image_path: str,
    warmup_runs: int = 2,
    test_runs: int = 10,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Benchmark latency, FPS, and memory footprint of a model's predict() call."""
    # 1. Warm-up runs
    for _ in range(warmup_runs):
        _ = model.predict(image_path, top_k=top_k)

    # 2. Setup memory monitoring
    process = psutil.Process()
    ram_before = process.memory_info().rss / (1024 * 1024)  # MB

    use_cuda = (
        torch.cuda.is_available()
        and hasattr(model, "device")
        and model.device.type == "cuda"
    )
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # 3. Time inference iterations
    latencies = []

    for _ in range(test_runs):
        if use_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            _ = model.predict(image_path, top_k=top_k)
            end_event.record()

            torch.cuda.synchronize()
            latencies.append(start_event.elapsed_time(end_event))  # ms
        else:
            start_time = time.perf_counter()
            _ = model.predict(image_path, top_k=top_k)
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000.0)  # ms

    # 4. Measure memory usage
    ram_after = process.memory_info().rss / (1024 * 1024)  # MB
    ram_used = max(0.0, ram_after - ram_before)

    vram_peak_mb = 0.0
    if use_cuda:
        vram_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB

    avg_latency_ms = sum(latencies) / len(latencies)
    fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0

    return {
        "model_name": model.name,
        "avg_latency_ms": round(avg_latency_ms, 2),
        "fps": round(fps, 2),
        "ram_used_mb": round(ram_used, 2),
        "vram_peak_mb": round(vram_peak_mb, 2),
    }


def print_benchmark_summary(benchmarks: List[Dict[str, Any]]) -> None:
    """Print CLI summary table for all benchmarked models."""
    print("\n" + "=" * 70)
    print(f"{'MODEL BENCHMARK SUMMARY':^70}")
    print("=" * 70)
    print(
        f"{'Model Name':<28} | {'Latency (ms)':<12} | {'FPS':<8} | {'VRAM Peak (MB)':<14}"
    )
    print("-" * 70)

    for b in benchmarks:
        print(
            f"{b['model_name']:<28} | "
            f"{b['avg_latency_ms']:<12.2f} | "
            f"{b['fps']:<8.2f} | "
            f"{b['vram_peak_mb']:<14.2f}"
        )
    print("=" * 70 + "\n")


def export_benchmark_results(
    benchmarks: List[Dict[str, Any]],
    output_dir: str = "./results",
    filename: str = "benchmark_summary.csv",
) -> str:
    """
    Export benchmark metrics to a CSV file.

    Args:
        benchmarks: List of benchmark dictionary records.
        output_dir: Directory where the CSV file should be saved.
        filename: Name of output CSV file.

    Returns:
        Full path to the exported CSV file.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    fieldnames = ["model_name", "avg_latency_ms", "fps", "ram_used_mb", "vram_peak_mb"]

    with open(output_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(benchmarks)

    print(f"Exported benchmark summary to: {output_path}")
    return output_path
