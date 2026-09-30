# Stage 2 — Weights and exports

All artifacts come from Ultralytics `model.export()` (one build path). The same
`.engine` will be reused for accuracy and latency. Evidence:
`results/export_results.csv`, `results/artifact_verification.csv`,
`results/weights_provenance.json`.

## Weights provenance

All 4 checkpoints are official Ultralytics assets, release tag **v8.4.0**,
from `https://github.com/ultralytics/assets/releases/download/v8.4.0/<name>`.
Each was loaded and confirmed `task=obb` with 15 class names matching
`DOTAv1.yaml` exactly. SHA-256 for each is in `results/weights_provenance.json`.

## Export settings

`imgsz=1024`, `batch=1`, `device=0`. Precision is set with `quantize=32|16`;
ultralytics 8.4 deprecated `half=`/`int8=` in favour of `quantize=`.

Ultralytics names every export after the weights stem, so an FP32 and an FP16
engine would both land on `<stem>.engine`. `scripts/export_models.py` moves each
artifact to its configured path immediately after the build.

## Artifacts built (0 failures, 39.5 min total)

| Model | ONNX | TRT FP32 | TRT FP16 |
|---|---|---|---|
| yolo11n-obb | 10.49 MB / 3.6 s | 12.35 MB / 190 s | 8.44 MB / 401 s |
| yolo11s-obb | 37.38 MB / 3.0 s | 39.29 MB / 215 s | 22.16 MB / 431 s |
| yolo26n-obb | 9.73 MB / 4.8 s | 11.80 MB / 145 s | 8.64 MB / 345 s |
| yolo26s-obb | 37.61 MB / 3.7 s | 39.53 MB / 175 s | 22.42 MB / 454 s |

FP16 engines are consistently ~45% smaller than FP32. FP16 builds take roughly
2x longer than FP32 because TensorRT evaluates more kernel tactics.

## Verification: 16/16 artifacts load and run

`scripts/verify_artifacts.py` loads every `.pt`, `.onnx` and `.engine` and runs
one real val tile. All 16 pass. Detection counts agree within each model across
formats (only yolo26n FP16 differs, 50 vs 51 — FP16 rounding at the confidence
threshold), so these are functional checks, not just "the file opened".

## ONNX Runtime really is CUDA

`ultralytics/nn/backends/onnx.py` (lines 70-78) selects
`[("CUDAExecutionProvider", ...), "CPUExecutionProvider"]` when the device is
CUDA, and never selects `TensorrtExecutionProvider`. Confirmed at runtime — the
backend logs `Using ONNX Runtime 1.23.0 with CUDAExecutionProvider`. The ONNX
baseline is therefore a genuine ONNX Runtime CUDA measurement, not a disguised
TensorRT one. Stage 3 runs will log the provider again as evidence.

## yolo26 OBB validates in every format

Roadmap Stage 2 asks this explicitly because the OBB26 head differs. Smoke
`val()` on a 24-tile scratchpad subset (**smoke test only — not a result, and
the real Stage 3 runs use the entire val tile set**):

| Format | yolo26n mAP50 | yolo11n mAP50 |
|---|---|---|
| PyTorch FP32 | 0.7166 | 0.7413 |
| ONNX RT FP32 | 0.7110 | 0.7924 |
| TRT FP32 | 0.7110 | 0.7925 |
| TRT FP16 | 0.7109 | 0.7897 |

Both families export and validate in all four formats without error.

## OPEN ISSUE for Stage 3: PyTorch disagrees with ONNX/TRT

Stage 3 requires that PyTorch, ONNX and TensorRT FP32 "agree closely", and this
smoke test suggests they may not:

- **yolo11n**: PyTorch 0.7413 vs ONNX 0.7924 / TRT-FP32 0.7925 — PyTorch is
  ~5.1 mAP50 points **lower**.
- **yolo26n**: PyTorch 0.7166 vs ONNX 0.7110 / TRT-FP32 0.7110 — PyTorch is
  ~0.6 points **higher**.

In both cases ONNX and TRT FP32 agree to four decimal places, which is the
expected result and suggests the export path itself is sound. It is the
**PyTorch** row that diverges, in opposite directions for the two families.

Caveat: this is 24 tiles. mAP over so few images is noisy and a handful of
detections can move it several points, so this is a flag, not a conclusion.

Likely causes to check in Stage 3, in order of suspicion:
1. The exported graphs bake in fixed NMS parameters (`conf`, `iou`, `max_det`)
   at export time, while the PyTorch path takes them from val defaults.
2. Preprocessing differences — the exported models have a fixed 1024x1024
   input; the PyTorch validator may letterbox or rect-batch differently.
3. Genuine small-sample noise, which the full 5,297-tile run would wash out.

This must be resolved on the full val set before Stage 3's results are trusted.
