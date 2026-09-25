#!/usr/bin/env python3
"""Export each configured model from PyTorch (.pt) to ONNX.

Usage:
    python scripts/export_onnx.py --config configs/models.yaml
"""
import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


def export_one(model_cfg: dict, input_size: int, onnx_dir: Path) -> None:
    name = model_cfg["name"]
    weights = model_cfg["weights"]
    out_path = Path(model_cfg["onnx"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[{name}] loading {weights} ...")
    # ultralytics will auto-download the base .pt if it doesn't exist locally
    model = YOLO(weights)

    print(f"[{name}] exporting to ONNX (imgsz={input_size}) -> {out_path}")
    exported = model.export(
        format="onnx",
        imgsz=input_size,
        opset=17,
        simplify=True,
        dynamic=False,  # fixed shape is simpler + faster to build as a TRT engine
    )

    exported_path = Path(exported)
    if exported_path.resolve() != out_path.resolve():
        exported_path.replace(out_path)

    print(f"[{name}] done -> {out_path}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to configs/models.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    input_size = cfg.get("input_size", 640)

    for model_cfg in cfg["models"]:
        export_one(model_cfg, input_size, Path("weights"))


if __name__ == "__main__":
    main()
