#!/usr/bin/env python3
"""Stage 3/4: OBB accuracy on the FULL DOTAv1 val tile set.

One row per model x format, written to results/accuracy_results.csv as each
configuration finishes, so an interrupted run is resumable and loses at most
the configuration in flight.

IMPORTANT - rect=False is not optional. Ultralytics' validator defaults to
rect=True, which batches images by aspect ratio and feeds the PyTorch model
non-square letterboxed inputs. The exported ONNX/TensorRT graphs have a fixed
1024x1024 input, so they always see square letterboxing. 4.5% of DOTA val tiles
are edge crops smaller than 1024x1024, and that asymmetry alone shifted PyTorch
mAP50 by ~3.3 points relative to the exported formats. Forcing rect=False makes
preprocessing identical across all formats, which is what the "same measurement
protocol" requirement means in practice.

Each configuration runs in its OWN SUBPROCESS. This board has 8GB of unified
memory, and evaluating all configurations in one process accumulates CUDA
contexts and caching-allocator arenas until an allocation fails. On Jetson that
failure surfaces as `NVML_SUCCESS == r INTERNAL ASSERT FAILED` from
CUDACachingAllocator rather than a clean OOM, because Tegra does not expose the
full NVML interface the allocator queries on its out-of-memory path. One process
per configuration guarantees every run starts from a clean allocator.

Usage:
    python scripts/evaluate_obb.py --config configs/models_obb.yaml
    python scripts/evaluate_obb.py --config configs/models_obb.yaml --formats trt-int8
    python scripts/evaluate_obb.py --single yolo11n-obb pytorch   # one config (internal)
"""
import argparse
import csv
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

# Must be set before CUDA initialises. On this 8GB unified-memory board the
# default allocator needs contiguous physical blocks, and rotated-IoU NMS
# allocates ~10 concurrent NxN temporaries (N up to ~8k candidates per DOTA
# tile, ~270MB each). That fragmentation surfaces as
# `NVML_SUCCESS == r INTERNAL ASSERT FAILED` instead of a clean OOM, because
# Tegra does not expose the NVML interface the allocator queries when it runs
# out. Expandable segments grow a virtual region instead, which fixes it.
# This changes allocation strategy only — never results.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import yaml
from ultralytics import YOLO

FIELDS = ["model", "format", "precision", "artifact", "mAP50", "mAP50_95",
          "tiles", "imgsz", "batch", "rect", "preprocess_ms", "inference_ms",
          "postprocess_ms", "eval_seconds", "provider", "status", "error"]


def done_keys(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    with open(csv_path) as f:
        return {(r["model"], r["format"]) for r in csv.DictReader(f)
                if r.get("status") == "ok"}


def append_row(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


def provider_of(model, imgsz: int) -> str:
    """Record what actually executed, so an 'ONNX' row can't silently be TRT.

    `val()` populates `model.validator`, not `model.predictor`, and the
    validator is not reachable as a plain attribute. So we run one throwaway
    image through `predict()`, which builds the *same* AutoBackend, and read
    the session off that. For ONNX this reports the real execution provider;
    onnxruntime-gpu also exposes TensorrtExecutionProvider, so this is the
    check that keeps the ONNX baseline honest.
    """
    try:
        import numpy as np

        model.predict(np.zeros((imgsz, imgsz, 3), dtype=np.uint8),
                      imgsz=imgsz, device=0, verbose=False)
        sess = getattr(model.predictor.model, "session", None)
        if sess is not None:
            return sess.get_providers()[0]
        return "torch/tensorrt (no onnxruntime session)"
    except Exception as e:  # noqa: BLE001
        return f"unknown ({type(e).__name__})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/models_obb.yaml")
    ap.add_argument("--formats", default="pytorch,onnx,trt-fp32,trt-fp16")
    ap.add_argument("--only", default=None)
    ap.add_argument("--out", default="results/accuracy_results.csv")
    ap.add_argument("--single", nargs=2, metavar=("MODEL", "FORMAT"),
                    help="evaluate exactly one configuration then exit "
                         "(used internally for process isolation)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    imgsz, batch, data = cfg["imgsz"], cfg["batch"], cfg["data"]
    out = Path(args.out)
    already = done_keys(out)
    formats = [f for f in args.formats.split(",") if f]

    n_tiles = len(list((Path("data/DOTAv1-tiles/images/val")).glob("*.jpg")))
    print(f"evaluating on {n_tiles} val tiles, imgsz={imgsz} batch={batch} rect=False\n")

    for m in cfg["models"]:
        name = m["name"]
        if args.only and name != args.only:
            continue
        if args.single and args.single[0] != name:
            continue
        artifacts = {
            "pytorch": (m["weights"], "fp32"),
            "onnx": (m["onnx"], "fp32"),
            "trt-fp32": (m["engines"]["fp32"], "fp32"),
            "trt-fp16": (m["engines"]["fp16"], "fp16"),
            "trt-int8": (m["engines"]["int8"], "int8"),
        }
        for fmt in formats:
            if args.single and args.single[1] != fmt:
                continue
            path, prec = artifacts[fmt]
            if (name, fmt) in already:
                print(f"[{name}] {fmt}: already in CSV, skipping")
                continue
            if not Path(path).exists():
                print(f"[{name}] {fmt}: MISSING {path}, skipping")
                continue

            if not args.single:
                # Fresh process per configuration — see module docstring.
                rc = subprocess.run(
                    [sys.executable, __file__, "--config", args.config,
                     "--out", str(out), "--formats", fmt, "--single", name, fmt], check=False,
                ).returncode
                if rc != 0:
                    print(f"[{name}] {fmt}: subprocess exited {rc}")
                continue

            print(f"[{name}] {fmt}: evaluating {path} ...")
            row = {"model": name, "format": fmt, "precision": prec,
                   "artifact": path, "tiles": n_tiles, "imgsz": imgsz,
                   "batch": batch, "rect": False}
            try:
                t0 = time.perf_counter()
                model = YOLO(path, task="obb")
                r = model.val(data=data, imgsz=imgsz, batch=batch, rect=False,
                              device=0, verbose=False, plots=False, save_json=False)
                row.update(
                    mAP50=round(float(r.box.map50), 5),
                    mAP50_95=round(float(r.box.map), 5),
                    preprocess_ms=round(r.speed["preprocess"], 3),
                    inference_ms=round(r.speed["inference"], 3),
                    postprocess_ms=round(r.speed["postprocess"], 3),
                    eval_seconds=round(time.perf_counter() - t0, 1),
                    provider=provider_of(model, imgsz), status="ok")
                print(f"[{name}] {fmt}: mAP50={row['mAP50']:.4f} "
                      f"mAP50-95={row['mAP50_95']:.4f} "
                      f"({row['eval_seconds']:.0f}s)")
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                row.update(status="FAILED", error=repr(e)[:300])
                print(f"[{name}] {fmt}: FAILED {e}")
            append_row(out, row)

    print(f"\nrows appended to {out}")


if __name__ == "__main__":
    main()
