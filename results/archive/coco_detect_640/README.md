# Archived: COCO detection @ 640 (superseded)

These are **not** results for the current project scope. They are kept only
for reference and are deliberately not featured in the top-level README.

Produced 2026-09-25 on the Jetson Orin NX (MAXN) during the repo's earlier
scope: *horizontal-box* COCO detection at `imgsz=640`, YOLOv8n/s + YOLO11n/s,
TensorRT FP32/FP16 only, benchmarked with a custom pycuda CUDA-event runner.

The current project benchmarks **DOTAv1 OBB (rotated boxes) at imgsz=1024**
across 5 formats. Nothing here is comparable to it: different task, different
dataset, different input size, different measurement protocol.

Files:
- `benchmark_results.csv` — 8 rows (4 models x FP32/FP16), latency + engine size
- `latency_by_precision.png` — plot generated from that CSV
