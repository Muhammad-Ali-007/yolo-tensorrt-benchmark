#!/usr/bin/env python3
"""Latency/throughput benchmark for every built engine.

Measures pure GPU execution time via CUDA events (excludes host<->device
copy and Python overhead) as well as wall-clock end-to-end time (includes
everything a real deployment would pay for). Writes both to
results/benchmark_results.csv.

Usage:
    python scripts/benchmark.py --config configs/models.yaml --iterations 500 --warmup 50
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pycuda.driver as cuda
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.trt_infer import TensorRTInfer  # noqa: E402


def percentile(values, p):
    return float(np.percentile(values, p))


def benchmark_engine(engine_path: str, input_size: int, iterations: int, warmup: int) -> dict:
    runner = TensorRTInfer(engine_path)
    dummy_input = np.random.rand(1, 3, input_size, input_size).astype(np.float32)

    for _ in range(warmup):
        runner.infer(dummy_input)

    gpu_start, gpu_end = cuda.Event(), cuda.Event()
    gpu_latencies_ms, wall_latencies_ms = [], []

    for _ in range(iterations):
        t0 = time.perf_counter()
        runner.infer(dummy_input, timing_event_pair=(gpu_start, gpu_end))
        wall_latencies_ms.append((time.perf_counter() - t0) * 1000)
        gpu_latencies_ms.append(gpu_start.time_till(gpu_end))

    return {
        "gpu_latency_ms_p50": percentile(gpu_latencies_ms, 50),
        "gpu_latency_ms_p95": percentile(gpu_latencies_ms, 95),
        "gpu_latency_ms_p99": percentile(gpu_latencies_ms, 99),
        "wall_latency_ms_p50": percentile(wall_latencies_ms, 50),
        "wall_latency_ms_p95": percentile(wall_latencies_ms, 95),
        "fps_wall": 1000.0 / percentile(wall_latencies_ms, 50),
        "engine_size_mb": Path(engine_path).stat().st_size / (1024 * 1024),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--iterations", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--out", default="results/benchmark_results.csv")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    input_size = cfg.get("input_size", 640)

    rows = []
    for model_cfg in cfg["models"]:
        name = model_cfg["name"]
        for precision, engine_path in model_cfg["engines"].items():
            if not Path(engine_path).exists():
                print(f"[{name}] {precision}: skipping — {engine_path} not built yet")
                continue

            print(f"[{name}] {precision}: running {args.iterations} iterations "
                  f"({args.warmup} warmup)...")
            stats = benchmark_engine(engine_path, input_size, args.iterations, args.warmup)
            stats.update({"model": name, "precision": precision})
            rows.append(stats)
            print(f"  p50 GPU latency: {stats['gpu_latency_ms_p50']:.2f} ms  "
                  f"FPS (wall): {stats['fps_wall']:.1f}")

    if not rows:
        print("No engines found to benchmark — run build_engine.py first.")
        return

    df = pd.DataFrame(rows)
    cols = ["model", "precision", "gpu_latency_ms_p50", "gpu_latency_ms_p95",
            "gpu_latency_ms_p99", "wall_latency_ms_p50", "wall_latency_ms_p95",
            "fps_wall", "engine_size_mb"]
    df = df[cols]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
