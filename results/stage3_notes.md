# Stage 3 — Accuracy: PyTorch, ONNX, TensorRT FP32 and FP16

16 configurations (4 models x 4 formats), each evaluated on the DOTAv1 val tile
set at `imgsz=1024`, `batch=1`, `rect=False`. Zero failures. Total eval time
73.3 min. Every number traces to a row in `results/accuracy_results.csv`.

## Results

### mAP50

| Model | PyTorch FP32 | ONNX RT FP32 | TRT FP32 | TRT FP16 |
|---|---|---|---|---|
| yolo11n-obb | 0.77057 | 0.77054 | 0.77058 | 0.77031 |
| yolo11s-obb | 0.79364 | 0.79358 | 0.79359 | 0.79330 |
| yolo26n-obb | 0.74462 | 0.74465 | 0.74511 | 0.74509 |
| yolo26s-obb | 0.75327 | 0.75321 | 0.75305 | 0.75289 |

### mAP50-95

| Model | PyTorch FP32 | ONNX RT FP32 | TRT FP32 | TRT FP16 |
|---|---|---|---|---|
| yolo11n-obb | 0.60270 | 0.60268 | 0.60285 | 0.60290 |
| yolo11s-obb | 0.63259 | 0.63246 | 0.63253 | 0.63266 |
| yolo26n-obb | 0.58932 | 0.58938 | 0.58948 | 0.58910 |
| yolo26s-obb | 0.60523 | 0.60521 | 0.60518 | 0.60463 |

## FP32 cross-check — PASS

The roadmap requires PyTorch, ONNX and TensorRT FP32 to agree closely, and to
investigate any material gap before continuing.

| Model | max spread (mAP50) | as % of mean | verdict |
|---|---|---|---|
| yolo11n-obb | 0.00004 | 0.005% | PASS |
| yolo11s-obb | 0.00006 | 0.008% | PASS |
| yolo26n-obb | 0.00049 | 0.066% | PASS |
| yolo26s-obb | 0.00022 | 0.029% | PASS |

Worst disagreement across the three FP32 paths is **0.066% of the mean**. That
is the order of magnitude expected from floating-point kernel differences, not
from a broken export. The export path is sound and the three FP32 rows are
measuring the same model.

**This closes the open issue raised in `results/stage2_notes.md`,** where a
24-tile smoke test showed yolo11n PyTorch at 0.7413 against ONNX/TRT at 0.7924.
That gap was small-sample noise: on 24 tiles a handful of detections moves mAP
several points. On the full tile set the same comparison agrees to four decimal
places. No export defect existed.

## FP16 cost vs TensorRT FP32

| Model | TRT FP32 | TRT FP16 | delta | % |
|---|---|---|---|---|
| yolo11n-obb | 0.77058 | 0.77031 | -0.00027 | -0.035% |
| yolo11s-obb | 0.79359 | 0.79330 | -0.00029 | -0.037% |
| yolo26n-obb | 0.74511 | 0.74509 | -0.00002 | -0.003% |
| yolo26s-obb | 0.75305 | 0.75289 | -0.00016 | -0.021% |

FP16 is effectively free for rotated-box accuracy — worst case 0.037% of mAP50.
Whether INT8 behaves the same way is Stage 4's question, and should not be
assumed from this.

## Evaluated tile count: 5,278 of 5,297

Ultralytics' label scan rejects **19 tiles** when it builds the dataset cache:

```
results (found, missing, empty, corrupt, total) = (3066, 2231, 0, 19, 5297)
```

giving 5,278 usable tiles (3,066 labelled + 2,231 background), 55,653 instances.

The rejected tiles report `non-normalized or out of bounds coordinates`, e.g.
`[1.0859 1.0908]`. The cause is in Ultralytics' own `crop_and_save`: for windows
that overhang the image edge, the crop is clipped to the image bounds but the
labels are normalised by the clipped patch size, so a coordinate can land just
above 1.0. They come from 10 source images: P0259, P0261, P0837, P1440, P1456,
P1463, P1738, P2365, P2393, P2587.

This is **not** a reduction of the evaluation set to save time. It is the
library rejecting labels produced by its own documented tiling procedure, it
affects 0.36% of tiles, and the exclusion is identical for all 16 configurations
because every run reads the same `labels/val.cache`. Comparisons between formats
are therefore unaffected. It is recorded here rather than described as "the
entire val tile set".

## The allocator failure, and why it was not a results problem

The first Stage 3 attempt died with:

```
RuntimeError: NVML_SUCCESS == r INTERNAL ASSERT FAILED at
"/opt/pytorch/c10/cuda/CUDACachingAllocator.cpp":1165
```

Diagnosis (first hypothesis was wrong and is recorded for honesty):

1. **Wrong guess:** memory accumulating across configurations in one process.
   Running each configuration in its own subprocess did *not* fix it.
2. **Actual cause:** Ultralytics val uses `conf=0.001`, so dense DOTA tiles push
   up to ~8,200 candidate boxes into NMS (measured over a 300-tile sample;
   `max_nms=30000` is never the binding constraint). Rotated NMS calls
   `batch_probiou`, which holds roughly ten concurrent NxN temporaries — about
   270 MB each at N=8,200, so ~2.7 GB peak. On 8 GB of unified memory with
   ~3.9 GB free the allocator cannot find contiguous blocks. On Tegra it then
   queries NVML on its out-of-memory path, Tegra does not expose the full NVML
   interface, and the allocator asserts instead of raising a clean OOM — which
   is why the symptom looked like a PyTorch bug rather than memory pressure.

**Fix:** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, set inside
`scripts/evaluate_obb.py` before CUDA initialises so it is reproducible rather
than dependent on how the script is invoked. Expandable segments grow a virtual
region instead of requiring contiguous physical blocks.

This changes **allocation strategy only, never numerics**. The per-configuration
subprocess isolation was kept — it was not the fix, but it contains failures and
guarantees each run starts from a clean allocator.

## The ONNX baseline really is ONNX Runtime CUDA

All four ONNX rows record `provider = CUDAExecutionProvider`. This matters
because `onnxruntime-gpu` also exposes `TensorrtExecutionProvider`, so an
unguarded ONNX baseline can silently become a second TensorRT measurement.

Note on method: the provider column was initially written as
`unknown (AttributeError)`, because `val()` populates `model.validator` rather
than `model.predictor`. It was backfilled by loading each artifact through the
identical AutoBackend path and reading `predictor.model.session.get_providers()`.
The provider is a property of the artifact and device, not of a particular
evaluation run, so no mAP value was affected. `provider_of()` in
`scripts/evaluate_obb.py` is fixed for later stages. Independent corroboration:
the run log records `Using ONNX Runtime 1.23.0 with CUDAExecutionProvider` for
every ONNX configuration (`results/logs/stage3_eval.log`).

## Observation: the newer generation does not win here

yolo26 scores **below** yolo11 at both sizes on DOTAv1:

| Size | yolo11 mAP50 | yolo26 mAP50 | difference |
|---|---|---|---|
| nano | 0.77057 | 0.74462 | -2.6 points |
| small | 0.79364 | 0.75327 | -4.0 points |

This is an accuracy observation only. yolo26's NMS-free head makes its
postprocess roughly 6x cheaper (Stage 1: 0.75 ms vs ~5 ms), so the speed/accuracy
trade-off between the families is a Stage 5 question, not an accuracy one.
