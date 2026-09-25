# YOLO OBB → TensorRT Benchmark Suite (Jetson Orin NX)

Benchmarks Ultralytics' **DOTAv1-pretrained oriented-bounding-box (OBB)**
models across five deployment formats on a Jetson Orin NX, and reports what
each format actually costs in rotated-box accuracy, latency, file size, memory
and power.

> **Independent personal project.** Not affiliated with, and uses no data or
> code from, any funded/sponsored research project I've worked on. All models
> and datasets used here are public.

## Why this exists

Most YOLO benchmarks measure horizontal-box detection on a desktop GPU. This
one measures **rotated-box detection on edge hardware**, which is where aerial
and overhead imagery actually gets deployed — and it includes the parts people
usually skip: whether INT8 quantization damages *angle* regression, and whether
the board can sustain the throughput without thermal throttling.

Three questions per model:

1. **How much faster is TensorRT than the PyTorch and ONNX Runtime baselines?**
2. **What does each precision level (FP32 → FP16 → INT8) cost in OBB accuracy?**
3. **Can the Jetson sustain that throughput without throttling?**

## Scope

| | |
|---|---|
| **Models (4)** | `yolo11n-obb`, `yolo11s-obb`, `yolo26n-obb`, `yolo26s-obb` |
| **Formats (5)** | PyTorch FP32 *(baseline)*, ONNX Runtime GPU FP32 *(baseline)*, TensorRT FP32, TensorRT FP16, TensorRT INT8 |
| **Configurations** | 20 (4 models × 5 formats), `imgsz=1024`, batch 1 |
| **Data** | DOTAv1 train + val, tiled to 1024×1024 with Ultralytics' `split_dota` |
| **Evaluation** | the **entire** val tile set — no subsampling |
| **INT8 calibration** | seeded sample of ≥1000 **train** tiles, disjoint from val |

Official Ultralytics weights and public DOTAv1 data only. The DOTAv1 **test**
split has no public labels and is not used anywhere.

**Out of scope:** video / stream FPS, inference-time tiling or SAHI, training,
other datasets, and FP16 for the PyTorch/ONNX baselines (like-for-like
comparisons are made at FP32).

See [ROADMAP.md](ROADMAP.md) for staged progress and [CLAUDE.md](CLAUDE.md) for
how the repo is worked in.

## Hardware / software

Verified on-device and recorded in
**[results/environment.md](results/environment.md)** — including the two GPU
checks (PyTorch CUDA, and ONNX Runtime really executing on
`CUDAExecutionProvider` rather than silently falling back to CPU).

| | |
|---|---|
| Board | NVIDIA Jetson Orin NX, 8GB module |
| JetPack / L4T | 6.2 / R36.5.0, CUDA 12.6 |
| TensorRT | 10.3.0 (ships with JetPack) |
| ultralytics | 8.4.104 |
| onnxruntime-gpu | 1.23.0 |
| torch | 2.11.0 (NVIDIA Jetson build) |

Execution is **Jetson-only** — `tensorrt` and `pycuda` need the board.

## Quickstart

```bash
# 1. Environment. Do NOT use a bare `pip install -r requirements.txt` on a
#    Jetson — see the header of requirements.txt for why.
bash scripts/setup_venv.sh
source .venv/bin/activate
```

The remaining pipeline commands are added as each roadmap stage lands, so that
every documented command is one that has actually been run on the board.

## Results

**No results yet.** The pipeline for the new OBB scope has not been run.

Numbers appear here only once `results/accuracy_results.csv`,
`results/benchmark_results.csv` and `results/power_results.csv` exist, and
**every value in this section must trace to a row in one of those files**.
Estimated or hand-written numbers are not acceptable — an unverifiable
benchmark is worse than no benchmark.

Earlier COCO / horizontal-box / 640px results from this repo's previous scope
are kept in
[results/archive/coco_detect_640/](results/archive/coco_detect_640/) for
reference. They are **not** comparable to the current scope: different task,
dataset, input size and measurement protocol.

## License

MIT — see [LICENSE](LICENSE).
