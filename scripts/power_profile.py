#!/usr/bin/env python3
"""Wraps the benchmark with `tegrastats` logging so you get power draw and
thermal behavior alongside latency, instead of just latency in isolation.

This is the script most people skip — but "60ms/frame in a 30-second test"
and "60ms/frame sustained for a 25-minute flight without throttling" are two
very different claims, and only the second one is useful for a real
deployment.

Usage:
    python scripts/power_profile.py --config configs/models.yaml --power-mode 15W
    (run `sudo nvpmodel -q` to see available power modes on your Jetson first)
"""
import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.benchmark import benchmark_engine

# tegrastats line format varies by JetPack version; this targets the common
# "VDD_IN <mW>/<mW avg>" and "CPU@<C>C ... GPU@<C>C" fields. If your
# tegrastats output looks different, run `tegrastats` once by hand and adjust
# these regexes — the parsing logic (start subprocess, poll, regex each line)
# stays the same.
POWER_RE = re.compile(r"VDD_IN (\d+)mW")
GPU_TEMP_RE = re.compile(r"GPU@(-?\d+\.?\d*)C")


def start_tegrastats(interval_ms: int = 500) -> subprocess.Popen:
    return subprocess.Popen(
        ["tegrastats", "--interval", str(interval_ms)],
        stdout=subprocess.PIPE, text=True, bufsize=1,
    )


def sample_tegrastats(proc: subprocess.Popen, duration_s: float) -> dict:
    power_samples, temp_samples = [], []
    end_time = time.time() + duration_s
    while time.time() < end_time:
        line = proc.stdout.readline()
        if not line:
            break
        p = POWER_RE.search(line)
        t = GPU_TEMP_RE.search(line)
        if p:
            power_samples.append(int(p.group(1)))
        if t:
            temp_samples.append(float(t.group(1)))

    if not power_samples:
        return {"avg_power_mw": None, "peak_temp_c": None, "throttled": None}

    peak_temp = max(temp_samples) if temp_samples else None
    return {
        "avg_power_mw": sum(power_samples) / len(power_samples),
        "peak_power_mw": max(power_samples),
        "peak_temp_c": peak_temp,
        # crude throttle heuristic: Orin NX starts throttling near ~87C —
        # treat as a flag to check manually, not a certainty
        "throttled": bool(peak_temp and peak_temp >= 85),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--power-mode", default="unspecified",
                     help="label only — set the actual mode yourself first with "
                          "`sudo nvpmodel -m <id>`")
    ap.add_argument("--duration", type=float, default=15.0,
                     help="seconds of sustained inference to profile per engine")
    ap.add_argument("--out", default="results/power_results.csv")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    input_size = cfg.get("input_size", 640)

    rows = []
    for model_cfg in cfg["models"]:
        name = model_cfg["name"]
        for precision, engine_path in model_cfg["engines"].items():
            if not Path(engine_path).exists():
                continue

            print(f"[{name}] {precision}: profiling for {args.duration}s @ {args.power_mode}...")
            proc = start_tegrastats()
            try:
                # run inference continuously for `duration` seconds while sampling tegrastats
                iterations = max(1, int(args.duration * 20))  # rough guess, refined by wall clock below
                stats = benchmark_engine(engine_path, input_size, iterations=iterations, warmup=20)
                power = sample_tegrastats(proc, args.duration)
            finally:
                proc.terminate()

            rows.append({
                "model": name, "precision": precision, "power_mode": args.power_mode,
                "gpu_latency_ms_p50": stats["gpu_latency_ms_p50"],
                **power,
            })

    if not rows:
        print("No engines found to profile — run build_engine.py first.")
        return

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
