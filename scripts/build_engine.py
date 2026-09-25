#!/usr/bin/env python3
"""Build a TensorRT engine from an ONNX model at a given precision.

Usage:
    python scripts/build_engine.py --config configs/models.yaml --precision fp16
    python scripts/build_engine.py --config configs/models.yaml --precision int8 --calib-data data/calib
"""
import argparse
import sys
from pathlib import Path

import tensorrt as trt
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_int8 import ImageCalibrator

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def build_engine(onnx_path: str, engine_path: str, precision: str,
                  input_size: int, batch_size: int, calib_dir: str | None) -> None:
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError(f"failed to parse {onnx_path}")

    config = builder.create_builder_config()
    # TensorRT 10 removed IBuilderConfig.max_workspace_size in favour of the
    # memory-pool API. 2GB — adjust down if the Orin NX is under memory pressure.
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)

    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            print("  WARNING: platform reports no fast FP16 support, building anyway")
        config.set_flag(trt.BuilderFlag.FP16)

    elif precision == "int8":
        if not builder.platform_has_fast_int8:
            print("  WARNING: platform reports no fast INT8 support, building anyway")
        if not calib_dir:
            raise ValueError("--calib-data is required for --precision int8")
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)  # fall back to FP16 for layers INT8 can't handle well
        cache_path = str(Path(engine_path).with_suffix(".calib_cache"))
        config.int8_calibrator = ImageCalibrator(
            calib_dir, input_size, batch_size=1, cache_path=cache_path
        )

    elif precision != "fp32":
        raise ValueError(f"unknown precision {precision!r}")

    print(f"  building {precision.upper()} engine (this can take a few minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError(f"engine build failed for {onnx_path} at {precision}")

    Path(engine_path).parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    print(f"  -> {engine_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--precision", required=True, choices=["fp32", "fp16", "int8"])
    ap.add_argument("--calib-data", default=None, help="required for --precision int8")
    ap.add_argument("--only", default=None, help="build just this one model name")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    input_size = cfg.get("input_size", 640)
    batch_size = cfg.get("batch_size", 1)

    for model_cfg in cfg["models"]:
        name = model_cfg["name"]
        if args.only and name != args.only:
            continue

        onnx_path = model_cfg["onnx"]
        engine_path = model_cfg["engines"][args.precision]

        if not Path(onnx_path).exists():
            print(f"[{name}] skipping — {onnx_path} not found, run export_onnx.py first")
            continue

        print(f"[{name}] {args.precision.upper()}: {onnx_path} -> {engine_path}")
        build_engine(onnx_path, engine_path, args.precision, input_size, batch_size, args.calib_data)


if __name__ == "__main__":
    main()
