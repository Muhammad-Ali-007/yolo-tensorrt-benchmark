#!/usr/bin/env python3
"""Stage 2: export each OBB model to ONNX and to TensorRT FP32/FP16 engines.

One build path only: every artifact comes from Ultralytics `model.export()`,
and the same `.engine` is later used for both accuracy and latency.

Resumable by design — an artifact that already exists is skipped, and a CSV
row is appended as each artifact finishes, so an interrupted run loses at most
the artifact in flight.

Note on precision args: ultralytics 8.4 replaced `half=`/`int8=` with
`quantize=` (32 / 16 / 8). The old args still work but emit a deprecation
warning, so this uses `quantize` directly.

Usage:
    python scripts/export_models.py --config configs/models_obb.yaml
    python scripts/export_models.py --config configs/models_obb.yaml --only yolo26n-obb
"""
import argparse
import csv
import shutil
import time
import traceback
from pathlib import Path

import yaml
from ultralytics import YOLO

CSV_FIELDS = ["model", "format", "precision", "path", "size_mb",
              "build_seconds", "status", "error"]


def append_row(csv_path: Path, row: dict) -> None:
    """Append one artifact's result immediately so runs are resumable."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def export_one(weights: str, dest: Path, fmt: str, imgsz: int, batch: int,
               quantize, extra: dict) -> float:
    """Export and move the artifact to its deterministic path.

    Ultralytics names exports after the weights file, so an FP32 and an FP16
    engine would both land on `<stem>.engine` and clobber each other. We move
    each one to its configured path straight after the build.
    """
    t0 = time.perf_counter()
    model = YOLO(weights)
    produced = Path(model.export(format=fmt, imgsz=imgsz, batch=batch,
                                 device=0, quantize=quantize, **extra))
    elapsed = time.perf_counter() - t0
    dest.parent.mkdir(parents=True, exist_ok=True)
    if produced.resolve() != dest.resolve():
        shutil.move(str(produced), str(dest))
    return elapsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/models_obb.yaml")
    ap.add_argument("--only", default=None, help="just this model name")
    ap.add_argument("--precisions", default="fp32,fp16",
                    help="engine precisions to build (int8 is Stage 4)")
    ap.add_argument("--skip-onnx", action="store_true")
    ap.add_argument("--out", default="results/export_results.csv")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    imgsz, batch = cfg["imgsz"], cfg["batch"]
    out = Path(args.out)
    precisions = [p for p in args.precisions.split(",") if p]
    quant = {"fp32": 32, "fp16": 16, "int8": 8}

    for m in cfg["models"]:
        name = m["name"]
        if args.only and name != args.only:
            continue

        targets = []
        if not args.skip_onnx:
            targets.append(("onnx", "fp32", Path(m["onnx"]), {}, None))
        for p in precisions:
            # INT8 needs calibration data. The exporter collects it from
            # data[args.split or "val"], so configs/dota_calib.yaml points its
            # 'val' key at the seeded TRAIN-tile sample — without this, INT8
            # would calibrate on the very tiles we score on.
            extra_args = {"data": cfg["calib_data"]} if p == "int8" else {}
            targets.append(("engine", p, Path(m["engines"][p]), extra_args, quant[p]))

        for fmt, prec, dest, extra, q in targets:
            if dest.exists():
                print(f"[{name}] {fmt}/{prec}: exists, skipping -> {dest}")
                continue
            print(f"[{name}] {fmt}/{prec}: building -> {dest}")
            try:
                secs = export_one(m["weights"], dest, fmt, imgsz, batch, q, extra)
                size = dest.stat().st_size / 1048576
                append_row(out, {"model": name, "format": fmt, "precision": prec,
                                 "path": str(dest), "size_mb": round(size, 2),
                                 "build_seconds": round(secs, 1), "status": "ok"})
                print(f"[{name}] {fmt}/{prec}: OK {size:.1f} MB in {secs:.0f}s")
            except Exception as e:  # noqa: BLE001 - record and continue
                traceback.print_exc()
                append_row(out, {"model": name, "format": fmt, "precision": prec,
                                 "path": str(dest), "status": "FAILED",
                                 "error": repr(e)[:300]})
                print(f"[{name}] {fmt}/{prec}: FAILED {e}")

    print(f"\nresults appended to {out}")


if __name__ == "__main__":
    main()
