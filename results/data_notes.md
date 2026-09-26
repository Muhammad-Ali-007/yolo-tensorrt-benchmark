# Stage 1 — Data notes

DOTAv1, tiled for OBB benchmarking at `imgsz=1024`. Every number here comes
from `results/dota_stage1.json`, `results/eval_time_probe*.json` and
`results/eval_time_estimate.json`, produced by `scripts/prepare_dota.py` and
`scripts/probe_eval_time.py` on the Jetson.

## Source

| | |
|---|---|
| URL | `https://github.com/ultralytics/assets/releases/download/v0.0.0/DOTAv1.zip` |
| Size | 1,988,510,024 bytes (1.85 GiB), byte-exact on download, `unzip -t` OK |
| Extracted | `data/DOTAv1` — 2.0 GB |
| Labels | YOLO-OBB format (`cls x1 y1 x2 y2 x3 y3 x4 y4`, normalized) for train and val |

Raw split sizes as shipped (matches `DOTAv1.yaml`):

| Split | Images | Size | Labels? | Used? |
|---|---|---|---|---|
| train | 1,411 | 962 MB | yes | yes — tiled, calibration source |
| val | 458 | 318 MB | yes | yes — tiled, **the** evaluation set |
| **test** | **937** | **643 MB** | **no** | **no — bundled in the zip, left untouched** |

The zip does bundle the test split. It has no public labels, was not tiled,
and is not referenced by any config.

## Where data lives

All under the gitignored `data/` (`.gitignore` line 7, `data/*`):

- `data/DOTAv1/` — raw dataset (2.0 GB)
- `data/DOTAv1-tiles/` — tiled train + val (4.9 GB)
- `data/DOTAv1-calib/` — INT8 calibration sample (274 MB)

Ultralytics' global `datasets_dir` points at an unrelated project
(`~/Desktop/ied_project/datasets`) and was **deliberately not modified**.
`configs/dota_obb.yaml` and `configs/dota_calib.yaml` use absolute `path:`
values, which bypasses that setting entirely.

Free disk after Stage 1: 141 GB.

## Tiling

Ultralytics `split_dota.split_trainval` at its documented defaults:

| Setting | Value |
|---|---|
| crop_size | 1024 |
| gap | 200 (step = 824) |
| rates | `(1.0,)` — single scale |
| `allow_background_images` | `True` (library default) |

| Split | Tiles | With labels | Background |
|---|---|---|---|
| train | **15,749** | 9,734 | 6,015 |
| val | **5,297** | 3,066 | 2,231 |

Background tiles are crops containing no annotated object. The library saves
them as images with no label file; they are kept because that is the documented
procedure, and they matter for evaluation (a model can produce false positives
on them).

## INT8 calibration set

| | |
|---|---|
| Location | `data/DOTAv1-calib/` |
| Count | **1,000 tiles** |
| Seed | **0** (`random.Random(0).sample`) |
| Sampled from | the 9,734 **labelled train** tiles |
| Overlap with val | **0** — asserted, not assumed |

**Why train, not val.** The Ultralytics exporter collects calibration images
from `data[args.split or "val"]` (`ultralytics/engine/exporter.py`, in
`get_int8_calibration_dataloader`). Left alone, INT8 would calibrate on the
exact tiles we score on. `configs/dota_calib.yaml` therefore points its `val:`
key at this train-tile sample, so calibration data is disjoint from evaluation
data by construction. The script also asserts zero stem overlap with the val
tile set and fails loudly if that is ever violated.

**Why labelled tiles only.** Calibration should see representative object
activations. Background-only crops (open water, bare ground) would bias the
per-tensor dynamic ranges low.

## Eval-time estimate

Measured with `scripts/probe_eval_time.py` — PyTorch FP32, `imgsz=1024`,
batch 1, on real val tiles (100 tiles for yolo11n, 60 for the rest, after
warmup). This is a timing **probe**; it does not reduce the evaluation set.

| Model | preprocess | inference | postprocess | wall ms/tile |
|---|---|---|---|---|
| yolo11n-obb | 4.51 | 39.14 | 5.05 | 59.95 |
| yolo11s-obb | 4.24 | 64.34 | 4.69 | 83.94 |
| yolo26n-obb | 4.14 | 36.09 | **0.75** | 51.46 |
| yolo26s-obb | 4.24 | 67.30 | **0.76** | 82.83 |

Projected minutes per configuration over all 5,297 val tiles:

| Model | PyTorch FP32 | ONNX RT FP32 | TRT FP32 | TRT FP16 | TRT INT8 |
|---|---|---|---|---|---|
| yolo11n-obb | 5.3 | 5.8 | 3.9 | 2.9 | 2.8 |
| yolo11s-obb | 7.4 | 8.3 | 5.1 | 3.5 | 3.3 |
| yolo26n-obb | 4.5 | 5.0 | 3.3 | 2.4 | 2.2 |
| yolo26s-obb | 7.3 | 8.2 | 4.9 | 3.3 | 3.0 |

**Total for all 20 configurations: ~1.54 hours.**

Method: the PyTorch column is measured. Preprocess, postprocess and per-call
overhead are CPU-side and format-independent, so they are held fixed; only the
**inference** term is scaled, by 1.15x (ONNX RT), 0.60x (TRT FP32), 0.32x
(TRT FP16), 0.28x (TRT INT8) relative to measured PyTorch. Those multipliers
are assumptions and will be replaced by measurements in Stage 3/5. The
`predict()` per-call overhead (~10.5 ms/tile) is carried into every row, which
is conservative: `val()` uses a batched dataloader with workers and should be
lower.

**Conclusion: the full val set is practical.** No subsampling is needed or
proposed. Engine build time, not evaluation, will dominate the schedule.

## Finding: YOLO26 OBB is NMS-free

The yolo26 postprocess figures above (0.75 ms vs ~5 ms) are not a measurement
error. The released `yolo26*-obb.pt` checkpoints carry `one2one_cv2` /
`one2one_cv3` branches, so `Detect.end2end` evaluates **True** and the model
does not run rotated NMS.

This was verified on the loaded checkpoints:

```
yolo11n-obb:  head=OBB     one2one=False  end2end=False
yolo26n-obb:  head=OBB26   one2one=True   end2end=True
```

Consequence for the write-up: **the postprocess column is not comparable
between the yolo11 and yolo26 families.** yolo11 pays rotated NMS, yolo26 does
not. Comparisons of precision effects should use the *inference* column;
end-to-end latency differences between families partly reflect this
architectural difference rather than the export format.
