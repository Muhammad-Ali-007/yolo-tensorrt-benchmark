#!/usr/bin/env python3
"""Stage 5: latency and memory for all 20 configurations.

Same measurement protocol for every format — PyTorch, ONNX Runtime CUDA and the
three TensorRT precisions all go through `model.predict()` and report
Ultralytics' own preprocess / inference / postprocess split. A comparison is
only meaningful if every row is timed the same way.

Tiles are pre-loaded into memory as numpy arrays before timing starts, so disk
I/O is never counted. The same seeded set of tiles is used for every
configuration.

Memory is reported as three separate, explicitly-named numbers rather than one
misleading "GPU memory" figure. This board has *unified* memory — there is no
discrete GPU pool, and system-wide RAM includes other processes:

  torch_cuda_peak_mb  - torch.cuda.max_memory_allocated(); meaningful for the
                        PyTorch rows, and does NOT capture TensorRT's or ONNX
                        Runtime's own allocations.
  proc_rss_peak_mb    - peak resident set size of this process (VmHWM).
  sys_ram_used_peak_mb- peak system-wide RAM in use, sampled during the run.
                        Includes every other process on the board.

Usage:
    python scripts/benchmark_obb.py --config configs/models_obb.yaml
"""
import argparse
import csv
import json
import os
import threading
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np
import psutil
import yaml
from ultralytics import YOLO

FIELDS = [
    "model", "format", "precision", "artifact", "file_size_mb",
    "warmup", "iterations", "imgsz", "batch",
    "preprocess_ms_p50", "preprocess_ms_p95",
    "inference_ms_p50", "inference_ms_p95", "inference_ms_p99",
    "postprocess_ms_p50", "postprocess_ms_p95",
    "total_ms_p50", "total_ms_p95", "total_ms_p99",
    "fps_p50", "p50_to_p99_spread_pct",
    "torch_cuda_peak_mb", "proc_rss_peak_mb", "sys_ram_used_peak_mb",
    "torch_device", "onnx_provider", "tensorrt_version",
    "nvpmodel", "jetson_clocks", "cpu_governor", "gpu_governor",
    "status", "error",
]


class RamSampler(threading.Thread):
    """Sample system-wide RAM use while the benchmark runs."""

    def __init__(self, interval: float = 0.1):
        super().__init__(daemon=True)
        self.interval, self.peak_used_mb, self._stop_evt = interval, 0.0, threading.Event()

    def run(self):
        total = psutil.virtual_memory().total
        while not self._stop_evt.is_set():
            used = (total - psutil.virtual_memory().available) / 1048576
            self.peak_used_mb = max(self.peak_used_mb, used)
            time.sleep(self.interval)

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=2)


def clock_state() -> dict:
    def read(p, default="unknown"):
        try:
            return Path(p).read_text().strip()
        except OSError:
            return default
    gov = "unknown"
    for d in Path("/sys/class/devfreq").glob("*gpu*"):
        gov = read(d / "governor")
    mode = "unknown"
    try:
        import subprocess
        out = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True, check=False).stdout
        for line in out.splitlines():
            if "Power Mode" in line:
                mode = line.split(":")[-1].strip()
    except Exception as e:  # noqa: BLE001
        mode = f"unknown ({type(e).__name__})"
    cpu_gov = read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    # jetson_clocks needs root to query; infer from the governors instead.
    fixed = "applied" if cpu_gov == "performance" else "NOT applied (DVFS active)"
    return {"nvpmodel": mode, "cpu_governor": cpu_gov,
            "gpu_governor": gov, "jetson_clocks": fixed}


def load_tiles(tile_dir: str, n: int, seed: int) -> list:
    paths = sorted(Path(tile_dir).glob("*.jpg"))
    rng = np.random.default_rng(seed)
    picked = [paths[i] for i in rng.choice(len(paths), n, replace=False)]
    return [cv2.imread(str(p)) for p in picked]


def pct(v, q):
    return float(np.percentile(v, q))


def bench_one(path: str, imgsz: int, tiles: list, warmup: int, iters: int) -> dict:
    import torch

    model = YOLO(path, task="obb")
    for i in range(warmup):
        model.predict(tiles[i % len(tiles)], imgsz=imgsz, device=0, verbose=False)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    proc = psutil.Process()
    sampler = RamSampler()
    sampler.start()

    pre, inf, post = [], [], []
    for i in range(iters):
        r = model.predict(tiles[i % len(tiles)], imgsz=imgsz, device=0, verbose=False)[0]
        pre.append(r.speed["preprocess"])
        inf.append(r.speed["inference"])
        post.append(r.speed["postprocess"])

    sampler.stop()
    total = [a + b + c for a, b, c in zip(pre, inf, post)]
    sess = getattr(model.predictor.model, "session", None)
    provider = sess.get_providers()[0] if sess else ""

    return {
        "preprocess_ms_p50": round(pct(pre, 50), 3), "preprocess_ms_p95": round(pct(pre, 95), 3),
        "inference_ms_p50": round(pct(inf, 50), 3), "inference_ms_p95": round(pct(inf, 95), 3),
        "inference_ms_p99": round(pct(inf, 99), 3),
        "postprocess_ms_p50": round(pct(post, 50), 3), "postprocess_ms_p95": round(pct(post, 95), 3),
        "total_ms_p50": round(pct(total, 50), 3), "total_ms_p95": round(pct(total, 95), 3),
        "total_ms_p99": round(pct(total, 99), 3),
        "fps_p50": round(1000.0 / pct(total, 50), 2),
        "p50_to_p99_spread_pct": round((pct(total, 99) - pct(total, 50)) / pct(total, 50) * 100, 2),
        "torch_cuda_peak_mb": round(torch.cuda.max_memory_allocated() / 1048576, 1)
        if torch.cuda.is_available() else "",
        "proc_rss_peak_mb": round(proc.memory_info().rss / 1048576, 1),
        "sys_ram_used_peak_mb": round(sampler.peak_used_mb, 1),
        "torch_device": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "cpu",
        "onnx_provider": provider,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/models_obb.yaml")
    ap.add_argument("--tiles-dir", default="data/DOTAv1-tiles/images/val")
    ap.add_argument("--n-tiles", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--formats", default="pytorch,onnx,trt-fp32,trt-fp16,trt-int8")
    ap.add_argument("--out", default="results/benchmark_results.csv")
    args = ap.parse_args()

    import tensorrt as trt

    cfg = yaml.safe_load(Path(args.config).read_text())
    imgsz, batch = cfg["imgsz"], cfg["batch"]
    clocks = clock_state()
    print(f"clock state: {json.dumps(clocks)}")
    print(f"loading {args.n_tiles} tiles into memory (seed {args.seed}) ...")
    tiles = load_tiles(args.tiles_dir, args.n_tiles, args.seed)

    out = Path(args.out)
    done = set()
    if out.exists():
        with open(out) as f:
            done = {(r["model"], r["format"]) for r in csv.DictReader(f) if r["status"] == "ok"}

    for m in cfg["models"]:
        arts = {
            "pytorch": (m["weights"], "fp32"), "onnx": (m["onnx"], "fp32"),
            "trt-fp32": (m["engines"]["fp32"], "fp32"),
            "trt-fp16": (m["engines"]["fp16"], "fp16"),
            "trt-int8": (m["engines"]["int8"], "int8"),
        }
        for fmt in args.formats.split(","):
            path, prec = arts[fmt]
            if (m["name"], fmt) in done:
                print(f"[{m['name']}] {fmt}: already measured, skipping")
                continue
            row = {"model": m["name"], "format": fmt, "precision": prec,
                   "artifact": path, "imgsz": imgsz, "batch": batch,
                   "warmup": args.warmup, "iterations": args.iterations,
                   "file_size_mb": round(Path(path).stat().st_size / 1048576, 2),
                   "tensorrt_version": trt.__version__, **clocks}
            try:
                print(f"[{m['name']}] {fmt}: {args.warmup} warmup + {args.iterations} iters ...")
                row.update(bench_one(path, imgsz, tiles, args.warmup, args.iterations))
                row["status"] = "ok"
                print(f"[{m['name']}] {fmt}: total p50={row['total_ms_p50']} ms "
                      f"({row['fps_p50']} FPS) inf p50={row['inference_ms_p50']} ms "
                      f"spread={row['p50_to_p99_spread_pct']}%")
            except Exception as e:  # noqa: BLE001
                row["status"], row["error"] = "FAILED", repr(e)[:300]
                print(f"[{m['name']}] {fmt}: FAILED {e}")

            out.parent.mkdir(parents=True, exist_ok=True)
            new = not out.exists()
            with open(out, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                if new:
                    w.writeheader()
                w.writerow({k: row.get(k, "") for k in FIELDS})

    print(f"\nrows appended to {out}")


if __name__ == "__main__":
    main()
