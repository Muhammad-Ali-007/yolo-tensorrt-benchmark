#!/usr/bin/env python3
"""Stage 1: tile DOTAv1 into 1024x1024 crops and build the INT8 calibration set.

Uses Ultralytics' documented `split_dota` procedure at its default settings
(crop 1024, gap 200, rates=(1.0,), background tiles kept). The DOTAv1 *test*
split has no public labels and is never touched.

The calibration set is a seeded sample of TRAIN tiles. This matters: the
Ultralytics exporter defaults INT8 calibration to the `val` split, which would
calibrate on the exact images we score on. Sampling train keeps calibration
disjoint from evaluation by construction.

Usage:
    python scripts/prepare_dota.py --data-root data/DOTAv1 --tiles-dir data/DOTAv1-tiles
"""
import argparse
import json
import random
import shutil
import sys
from pathlib import Path

from ultralytics.data.split_dota import split_trainval

CROP_SIZE = 1024
GAP = 200
RATES = (1.0,)


def count_tiles(tiles_dir: Path, split: str) -> dict:
    """Count tiles for a split, split out by whether they carry labels."""
    im_dir = tiles_dir / "images" / split
    lb_dir = tiles_dir / "labels" / split
    images = sorted(im_dir.glob("*.jpg"))
    labelled = [p for p in images if (lb_dir / f"{p.stem}.txt").exists()]
    return {
        "total": len(images),
        "labelled": len(labelled),
        "background": len(images) - len(labelled),
    }


def build_calibration_set(tiles_dir: Path, calib_dir: Path, n: int, seed: int) -> dict:
    """Seeded sample of labelled TRAIN tiles, copied into their own YOLO tree.

    Labelled tiles only: INT8 calibration should see representative object
    activations, and background-only crops (open water, bare ground) would
    skew the per-tensor dynamic ranges low.
    """
    src_im = tiles_dir / "images" / "train"
    src_lb = tiles_dir / "labels" / "train"
    pool = sorted(p for p in src_im.glob("*.jpg") if (src_lb / f"{p.stem}.txt").exists())
    if len(pool) < n:
        raise SystemExit(f"only {len(pool)} labelled train tiles available, need {n}")

    rng = random.Random(seed)
    picked = rng.sample(pool, n)

    dst_im = calib_dir / "images" / "val"   # 'val' key: the exporter reads data['val']
    dst_lb = calib_dir / "labels" / "val"
    for d in (dst_im, dst_lb):
        d.mkdir(parents=True, exist_ok=True)
    for p in picked:
        shutil.copy2(p, dst_im / p.name)
        shutil.copy2(src_lb / f"{p.stem}.txt", dst_lb / f"{p.stem}.txt")

    # Disjointness is guaranteed by construction (train tiles never appear in
    # val), but assert it on stems so the claim is actually checked.
    val_stems = {p.stem for p in (tiles_dir / "images" / "val").glob("*.jpg")}
    overlap = {p.stem for p in picked} & val_stems
    if overlap:
        raise SystemExit(f"calibration/val overlap: {len(overlap)} tiles")

    return {"count": len(picked), "seed": seed, "pool_size": len(pool), "overlap_with_val": 0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/DOTAv1")
    ap.add_argument("--tiles-dir", default="data/DOTAv1-tiles")
    ap.add_argument("--calib-dir", default="data/DOTAv1-calib")
    ap.add_argument("--calib-n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-split", action="store_true", help="tiles already exist")
    ap.add_argument("--out", default="results/dota_stage1.json")
    args = ap.parse_args()

    data_root, tiles_dir = Path(args.data_root), Path(args.tiles_dir)
    calib_dir = Path(args.calib_dir)

    if not (data_root / "images" / "val").exists():
        raise SystemExit(f"missing {data_root}/images/val — is DOTAv1 extracted?")

    if not args.skip_split:
        print(f"tiling train+val: crop={CROP_SIZE} gap={GAP} rates={RATES} -> {tiles_dir}")
        print("(the test split is deliberately not tiled)")
        split_trainval(
            data_root=str(data_root), save_dir=str(tiles_dir),
            crop_size=CROP_SIZE, gap=GAP, rates=RATES,
        )

    stats = {
        "crop_size": CROP_SIZE, "gap": GAP, "rates": list(RATES),
        "train_tiles": count_tiles(tiles_dir, "train"),
        "val_tiles": count_tiles(tiles_dir, "val"),
    }
    print(json.dumps(stats, indent=2))

    print(f"\nbuilding calibration set: n={args.calib_n} seed={args.seed}")
    stats["calibration"] = build_calibration_set(
        tiles_dir, calib_dir, args.calib_n, args.seed
    )
    print(json.dumps(stats["calibration"], indent=2))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    sys.exit(main())
