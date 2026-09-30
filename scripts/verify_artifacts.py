#!/usr/bin/env python3
"""Stage 2 gate: every exported artifact must load and run one image.

Checks all 4 .pt, 4 .onnx and 8 engines. Also records the detection count per
format so a silently-broken export (loads fine, predicts nothing) is caught
rather than passing as "OK".

Usage:
    python scripts/verify_artifacts.py --config configs/models_obb.yaml
"""
import argparse
import csv
import os
from pathlib import Path

import yaml
from ultralytics import YOLO

FIELDS = ["model", "format", "precision", "path", "exists", "size_mb",
          "detections", "inference_ms", "status", "error"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/models_obb.yaml")
    ap.add_argument("--precisions", default="fp32,fp16")
    ap.add_argument("--out", default="results/artifact_verification.csv")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    imgsz = cfg["imgsz"]
    tile_dir = "data/DOTAv1-tiles/images/val"
    tile = os.path.join(tile_dir, min(os.listdir(tile_dir)))
    print(f"test tile: {tile}  imgsz={imgsz}\n")

    rows, failures = [], 0
    for m in cfg["models"]:
        targets = [("pytorch", "fp32", Path(m["weights"])),
                   ("onnx", "fp32", Path(m["onnx"]))]
        for p in args.precisions.split(","):
            targets.append(("tensorrt", p, Path(m["engines"][p])))

        for fmt, prec, path in targets:
            row = {"model": m["name"], "format": fmt, "precision": prec,
                   "path": str(path), "exists": path.exists()}
            if not path.exists():
                row.update(status="MISSING"); failures += 1
                rows.append(row); print(f"  MISSING {path}"); continue
            row["size_mb"] = round(path.stat().st_size / 1048576, 2)
            try:
                model = YOLO(str(path), task="obb")
                r = model.predict(tile, imgsz=imgsz, device=0, verbose=False)[0]
                row.update(detections=len(r.obb),
                           inference_ms=round(r.speed["inference"], 2),
                           status="ok")
                print(f"  OK  {m['name']:14s} {fmt:9s} {prec:5s} "
                      f"dets={len(r.obb):3d} inf={r.speed['inference']:7.2f} ms")
            except Exception as e:  # noqa: BLE001
                row.update(status="FAILED", error=repr(e)[:300]); failures += 1
                print(f"  FAILED {path}: {e}")
            rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    print(f"\n{len(rows)} artifacts checked, {failures} failure(s) -> {out}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
