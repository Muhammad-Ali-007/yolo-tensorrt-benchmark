# Stage 5 — Latency and memory

20 configurations, 0 failures. 20 warmup + 300 timed iterations each, batch 1,
`imgsz=1024`, on a fixed seeded set of 50 real val tiles pre-loaded into memory
so disk I/O is never timed. Every number traces to
`results/benchmark_results.csv`.

## Protocol

All five formats go through `model.predict()` and report Ultralytics' own
preprocess / inference / postprocess split. A custom CUDA-event runner for the
TensorRT rows would have measured something different from the baselines and
made the comparison meaningless, so it was not used — `src/trt_infer.py` is not
part of this measurement path.

## Latency (ms, p50) and throughput

| Model | Format | pre | inference | post | total | FPS |
|---|---|---|---|---|---|---|
| yolo11n-obb | PyTorch | 5.82 | 32.09 | 5.53 | 43.44 | 23.0 |
| yolo11n-obb | ONNX RT | 3.59 | 35.69 | 5.86 | 45.13 | 22.2 |
| yolo11n-obb | TRT FP32 | 3.61 | 19.36 | 5.77 | 28.75 | 34.8 |
| yolo11n-obb | TRT FP16 | 3.61 | 10.52 | 5.79 | 19.92 | 50.2 |
| yolo11n-obb | TRT INT8 | 3.62 | 10.14 | 5.75 | 19.50 | 51.3 |
| yolo11s-obb | PyTorch | 3.66 | 63.18 | 5.53 | 71.89 | 13.9 |
| yolo11s-obb | ONNX RT | 3.62 | 66.47 | 5.83 | 75.79 | 13.2 |
| yolo11s-obb | TRT FP32 | 3.61 | 38.00 | 5.81 | 46.69 | 21.4 |
| yolo11s-obb | TRT FP16 | 3.59 | 18.58 | 5.75 | 27.85 | 35.9 |
| yolo11s-obb | TRT INT8 | 3.60 | 15.58 | 5.80 | 24.98 | 40.0 |
| yolo26n-obb | PyTorch | 3.68 | 34.04 | 0.76 | 38.47 | 26.0 |
| yolo26n-obb | ONNX RT | 3.60 | 35.30 | 5.90 | 44.78 | 22.3 |
| yolo26n-obb | TRT FP32 | 3.61 | 19.33 | 0.94 | 23.86 | 41.9 |
| yolo26n-obb | TRT FP16 | 3.61 | 10.29 | 0.93 | **14.82** | **67.5** |
| yolo26n-obb | TRT INT8 | 3.60 | 10.61 | 5.83 | 20.05 | 49.9 |
| yolo26s-obb | PyTorch | 3.68 | 67.23 | 0.76 | 71.67 | 14.0 |
| yolo26s-obb | ONNX RT | 3.63 | 67.55 | 5.89 | 76.95 | 13.0 |
| yolo26s-obb | TRT FP32 | 3.62 | 42.12 | 0.95 | 46.71 | 21.4 |
| yolo26s-obb | TRT FP16 | 3.62 | 19.00 | 0.94 | 23.55 | 42.5 |
| yolo26s-obb | TRT INT8 | 3.59 | 17.13 | 5.82 | 26.54 | 37.7 |

## FINDING: yolo26's ONNX and INT8 exports silently lose the NMS-free head

The yolo26 postprocess column splits into two groups — ~0.9 ms for PyTorch,
TRT FP32 and TRT FP16, but ~5.8 ms for ONNX and TRT INT8, which is exactly what
yolo11 pays for rotated NMS. Inspecting what each backend actually emits for
`yolo26n-obb`:

| Format | raw output shape | decode path |
|---|---|---|
| PyTorch | `(1, 300, 7)` | end2end, NMS-free |
| **ONNX** | `(1, 20, 21504)` | **raw predictions -> full rotated NMS** |
| TRT FP16 | `(1, 300, 7)` | end2end, NMS-free |
| **TRT INT8** | `(1, 20, 21504)` | **raw predictions -> full rotated NMS** |

So the ONNX and INT8 exports do not preserve YOLO26's one2one (end2end) head;
they emit raw predictions and Ultralytics runs the standard rotated-NMS
postprocess over them. The effect is perfectly consistent across both yolo26
models (5.90 / 5.83 ms for n, 5.89 / 5.82 ms for s, against ~0.94 ms for the
formats that keep the head).

**Consequences, which must not be glossed over:**

1. **The yolo26 ONNX and INT8 rows are not architecturally like-for-like** with
   their PyTorch / TRT FP32 / TRT FP16 siblings. They measure the same weights
   through a different decode path.
2. It explains an otherwise bizarre result: **yolo26 INT8 is *slower end to end*
   than yolo26 FP16** (20.05 vs 14.82 ms for n; 26.54 vs 23.55 ms for s) even
   though INT8 inference is as fast or faster. The regression is entirely in
   postprocess, not in the engine.
3. The Stage 3/4 accuracy numbers for those rows are still valid — both decode
   paths produce near-identical mAP (yolo26n: ONNX 0.74465 vs PyTorch 0.74462) —
   but the *latency* comparison for those two rows reflects the decode path as
   much as the precision.

This was not assumed from the timings; it was confirmed by reading the backend
output shapes directly.

## TensorRT speedup over the baselines (total p50)

| Model | vs PyTorch (best TRT) | vs ONNX RT (best TRT) |
|---|---|---|
| yolo11n-obb | 2.23x (INT8) | 2.31x |
| yolo11s-obb | 2.88x (INT8) | 3.03x |
| yolo26n-obb | 2.60x (FP16) | 3.02x |
| yolo26s-obb | 3.04x (FP16) | 3.27x |

**ONNX Runtime CUDA is not a speedup over PyTorch here** — it is 1.4% to 7.4%
*slower* on every model. The useful baseline comparison is PyTorch; ONNX
Runtime mainly serves to show that the gain comes from TensorRT specifically,
not merely from leaving Python.

## INT8 latency: a poor trade on this hardware

| Model | FP16 total | INT8 total | latency gain | mAP50 cost (Stage 4) |
|---|---|---|---|---|
| yolo11n-obb | 19.92 | 19.50 | **+2.1%** | -5.28% |
| yolo11s-obb | 27.85 | 24.98 | **+10.3%** | -5.99% |
| yolo26n-obb | 14.82 | 20.05 | **-35.3%** (slower) | -3.85% |
| yolo26s-obb | 23.55 | 26.54 | **-12.7%** (slower) | -0.72% |

On this board INT8 buys at most 10% latency and costs 4-6 mAP50 points, and for
yolo26 it is slower overall because of the lost end2end head. **FP16 is the
better operating point for every model measured here.** That is the practical
conclusion of the project, and it is the opposite of the assumption that lower
precision is always faster.

## Memory

Three separate columns, because this board has *unified* memory and a single
"GPU memory" number would be misleading:

- `torch_cuda_peak_mb` — `torch.cuda.max_memory_allocated()`. Meaningful for the
  PyTorch rows. Does **not** capture TensorRT's or ONNX Runtime's own
  allocations, so it under-reports those rows and must not be read as their
  total footprint.
- `proc_rss_peak_mb` — peak RSS of the benchmark process.
- `sys_ram_used_peak_mb` — peak system-wide RAM, sampled at 10 Hz during the
  run. Includes every other process on the board, so it is an upper bound on
  the board's state, not this workload's cost.

Engine file sizes are in the same CSV (`file_size_mb`).

## Measurement trustworthiness

The roadmap's done-when asks that the p50-to-p99 spread be small enough to
trust. Reported honestly, it splits in two:

- **p50 -> p95: median 1.71%, max 5.16%, 18 of 20 configurations under 5%.**
  The measurements are stable.
- **p50 -> p99: up to 39%.** This is *not* proportional instability. The
  absolute p99-p50 gap is a near-constant ~5.85 ms across every configuration
  regardless of its duration (correlation of *relative* spread with p50 is
  -0.873), so it is a fixed-size stall that simply represents a larger fraction
  of a shorter run. With 300 iterations, p99 is the 3rd slowest iteration —
  roughly 3 iterations in 300 hit a ~6 ms host-side stall.

Per-stage jitter at p95 is small everywhere (preprocess 0.07-0.10 ms,
inference 0.03-0.97 ms, postprocess 0.16-0.37 ms), so the stalls are not in
GPU execution. **Inference-only p50->p99 spread, which is the column the
precision comparison rests on, has a median of 0.61% (INT8) to 4.34% (FP16).**

PyTorch rows show a smaller absolute p99 gap (median 1.88 ms) than the
accelerated rows (5.88 ms), consistent with longer GPU work absorbing host-side
stalls that are exposed when GPU work is short.

## Clocks — caveat

`jetson_clocks` requires root and **was not applied**; `nvpmodel` is MAXN. The
CPU governor is `schedutil` (dynamic) and the GPU governor is `nvhost_podgov`,
though the GPU was observed at `cur_freq == max_freq == 765 MHz`, i.e. already
at its ceiling. Every row records `nvpmodel`, `jetson_clocks`, `cpu_governor`
and `gpu_governor` so the caveat travels with the data.

The rare ~6 ms host-side stalls are the most likely place a fixed-clock run
would improve things, since short bursty workloads are exactly what a dynamic
CPU governor handles worst. Re-running Stage 5 after `sudo jetson_clocks` would
likely tighten the p99 tail; it should not move the p50 figures materially,
since those are already stable to within 1.71% at p95.

An unrelated process (`ied_project/serverJet.py`, ~1.4 GB resident) was running
throughout and was deliberately not touched. It contributes to both
`sys_ram_used_peak_mb` and the host-side scheduling noise.
