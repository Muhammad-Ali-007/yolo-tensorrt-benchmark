#!/usr/bin/env bash
# Pulls a small subset of COCO val2017 to use for INT8 calibration and mAP
# evaluation. Full val2017 is ~1GB / 5000 images — most of that isn't needed
# just to get meaningful calibration + eval numbers, so this grabs the
# annotations and a configurable-size image subset.
#
# Usage: bash scripts/download_data.sh [N_IMAGES]
#   N_IMAGES defaults to 500 (plenty for calibration; evaluate_map.py will
#   note if you want the full 5000 for a "real" mAP number).

set -euo pipefail
N_IMAGES="${1:-500}"
DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "==> Downloading COCO val2017 annotations..."
if [ ! -f annotations_trainval2017.zip ]; then
  curl -L -o annotations_trainval2017.zip \
    http://images.cocodataset.org/annotations/annotations_trainval2017.zip
fi
unzip -n annotations_trainval2017.zip

echo "==> Downloading COCO val2017 images (full zip, ~1GB — this is the slow part)..."
if [ ! -f val2017.zip ]; then
  curl -L -o val2017.zip http://images.cocodataset.org/zips/val2017.zip
fi
unzip -n val2017.zip

echo "==> Building a ${N_IMAGES}-image subset in data/calib/ for INT8 calibration..."
mkdir -p calib
ls val2017 | shuf -n "${N_IMAGES}" --random-source=<(yes 42) | while read -r f; do
  cp "val2017/$f" "calib/$f"
done

echo "==> Done."
echo "    Full validation images: data/val2017/"
echo "    Annotations:            data/annotations/instances_val2017.json"
echo "    Calibration subset:     data/calib/ (${N_IMAGES} images)"
