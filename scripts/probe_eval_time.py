#!/usr/bin/env python3
"""Stage 1 timing probe: measure per-tile cost to estimate full-val eval time.

This is a *probe*, not a result. It times `predict` over a sample of val tiles
to get the preprocess / inference / postprocess breakdown, so the Stage 1
estimate of "eval time x 20 configurations" is measured rather than guessed.
It does not shrink the evaluation set — the real Stage 3/4 runs use the
entire val tile set.

The PyTorch FP32 baseline is timed because it is expected to be the slowest of
the five formats, giving a conservative upper bound per configuration.

Usage:
    python scripts/probe_eval_time.py --weights weights/yolo11n-obb.pt --n 100
"""
import argparse
import json
import time
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/yolo11n-obb.pt")
    ap.add_argument("--tiles-dir", default="data/DOTAv1-tiles/images/val")
    ap.add_argument("--n", type=int, default=100, help="tiles to time")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--out", default="results/eval_time_probe.json")
    args = ap.parse_args()

    tiles = sorted(Path(args.tiles_dir).glob("*.jpg"))
    if len(tiles) < args.warmup + args.n:
        raise SystemExit(f"need {args.warmup + args.n} tiles, found {len(tiles)}")

    model = YOLO(args.weights)
    print(f"model {args.weights} task={model.task} imgsz={args.imgsz} device=cuda")

    for p in tiles[: args.warmup]:
        model.predict(str(p), imgsz=args.imgsz, device=0, verbose=False)

    sample = tiles[args.warmup : args.warmup + args.n]
    pre = inf = post = 0.0
    t0 = time.perf_counter()
    for p in sample:
        r = model.predict(str(p), imgsz=args.imgsz, device=0, verbose=False)[0]
        pre += r.speed["preprocess"]
        inf += r.speed["inference"]
        post += r.speed["postprocess"]
    wall_s = time.perf_counter() - t0

    n = len(sample)
    out = {
        "weights": args.weights,
        "imgsz": args.imgsz,
        "tiles_timed": n,
        "preprocess_ms": pre / n,
        "inference_ms": inf / n,
        "postprocess_ms": post / n,
        "sum_stages_ms": (pre + inf + post) / n,
        "wall_ms_per_tile": wall_s / n * 1000,
    }
    print(json.dumps(out, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
