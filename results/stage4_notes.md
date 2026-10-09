# Stage 4 — TensorRT INT8

4 INT8 engines built and evaluated on the same 5,278-tile val set as Stage 3.
`results/accuracy_results.csv` now holds all 20 configurations, 0 failures.

## Calibration split — verified, not assumed

The roadmap requires verifying which split ultralytics calibrates on, and
forcing the seeded train set. Checked three independent ways:

**1. Source.** `ultralytics/engine/exporter.py:946-949`:

```python
data = check_det_dataset(self.args.data, split=self.args.split)
dataset = build_yolo_dataset(cfg, data[self.args.split or "val"], ...)
```

`self.args.split` is None by default, so calibration reads the **`val` key**.
Pointed at `configs/dota_obb.yaml`, INT8 would have calibrated on the exact
tiles we score on.

**2. Forced to train tiles.** `configs/dota_calib.yaml` sets its `val` key to
`data/DOTAv1-calib/images/val`, which holds the seeded sample of **train**
tiles. `export_models.py` passes `data=configs/dota_calib.yaml` for INT8 only.

**3. Confirmed at runtime** in `results/logs/stage4_int8_gate.log`:

```
TensorRT: collecting INT8 calibration images from 'data=configs/dota_calib.yaml'
val: Scanning .../data/DOTAv1-calib/labels/val... 1000 images, 0 backgrounds, 1 corrupt
```

**Disjointness re-proved at two levels:**

| Check | Result |
|---|---|
| calibration tiles | 1,000 (seed 0) |
| calib ∩ eval tiles | **0** |
| calib ⊆ train tiles | **True** |
| calib source images | 606 |
| calib ∩ eval *source images* | **0** |

The source-image check matters: tiles are crops, so a val source image could in
principle contribute a calibration crop without any tile filename matching. It
does not — no val source image contributes to calibration.

**Calibration actually used 999 tiles**, not 1,000: one tile is rejected by the
same edge-coordinate quirk that drops 19 val tiles (see `stage3_notes.md`).
Still well above the 300-image threshold TensorRT warns below.

## Engines

| Model | FP32 MB | FP16 MB | INT8 MB | INT8 vs FP32 | INT8 build |
|---|---|---|---|---|---|
| yolo11n-obb | 12.35 | 8.44 | **5.28** | -57% | 623 s |
| yolo11s-obb | 39.29 | 22.16 | **13.80** | -65% | 992 s |
| yolo26n-obb | 11.80 | 8.64 | **5.40** | -54% | 621 s |
| yolo26s-obb | 39.53 | 22.42 | **15.50** | -61% | 1050 s |

INT8 builds averaged 822 s against ~380 s for FP16 — the calibration pass
roughly doubles build time. Total INT8 build time 54.8 min, 0 failures.

**INT8 here is mixed precision, not uniform INT8.** TensorRT logs
`keeping 85 Sigmoid layers in FP32 for INT8 accuracy`, so some layers stay in
FP32 by design. The results below should be read as "Ultralytics' INT8 export
path", not "every layer quantized to 8 bits".

## Accuracy cost of INT8

Against TensorRT FP32 on the same engine build path and the same tiles:

| Model | FP32 mAP50 | INT8 mAP50 | delta | % | mAP50-95 % |
|---|---|---|---|---|---|
| yolo11n-obb | 0.77058 | 0.72987 | -0.04071 | **-5.28%** | -6.22% |
| yolo11s-obb | 0.79359 | 0.74602 | -0.04757 | **-5.99%** | -8.68% |
| yolo26n-obb | 0.74511 | 0.71646 | -0.02865 | **-3.85%** | -6.33% |
| yolo26s-obb | 0.75305 | 0.74766 | -0.00539 | **-0.72%** | -3.59% |

### What this means

**INT8 is not free for rotated boxes, and FP16 is.** FP16 cost at most 0.037%
of mAP50 (Stage 3). INT8 costs between 0.72% and 5.99% — one to two orders of
magnitude more. Anyone reaching for INT8 on an OBB model should measure it
rather than assume the FP16 result carries over.

**The loss is larger at stricter IoU.** For every model the mAP50-95 drop
exceeds the mAP50 drop (e.g. yolo11s: -8.68% vs -5.99%). That is the signature
of *degraded localisation* rather than lost detections: boxes are still being
found, but their fit — including the angle, which is unique to OBB — is less
precise after quantization. This is exactly the failure mode INT8 on oriented
boxes was worth testing for.

**Sensitivity is model-dependent and not predictable from size.** yolo26s loses
0.72% while yolo11s loses 5.99% — an 8x difference between two models of
similar size evaluated identically. yolo26s-obb is the only configuration where
INT8 looks close to free on mAP50.

**It changes the accuracy ranking.** At FP32, yolo11s is the most accurate model
(0.79359) and yolo26s trails by 4 points. At INT8 they converge
(0.74602 vs 0.74766) and yolo26s is nominally ahead. A model chosen on FP32
accuracy may not be the right choice at INT8.

## Caveats

- Single calibration set (1,000 seeded train tiles, seed 0). Calibration-set
  size and composition are known to affect INT8 quality and were not ablated.
- One board, one TensorRT version (10.3.0). INT8 kernel selection is
  hardware- and version-specific.
- The engine is the same artifact used for latency in Stage 5, so the
  accuracy/speed trade-off is measured on identical builds.

## Data integrity note

The four INT8 rows were initially written misaligned. When the provider and
tile-count columns were backfilled at the end of Stage 3, `tiles` was replaced
by `tiles_tiled` / `tiles_evaluated` in the CSV, but `FIELDS` in
`scripts/evaluate_obb.py` still listed `tiles` in its old position — so
`csv.DictWriter` appended the INT8 rows in the old column order against the new
header, shifting every field after `mAP50_95` by one.

mAP50 and mAP50_95 precede the shift and were never affected. The remaining
fields were recovered by re-reading those four lines under the field order they
were actually written in, and `FIELDS` was corrected so later stages append
correctly. The misaligned file is preserved at
`results/logs/accuracy_results.csv.misaligned.bak`. Post-repair the CSV has
20 rows, all `status=ok`, `tiles_evaluated=5278`, `imgsz=1024`, `batch=1`.
