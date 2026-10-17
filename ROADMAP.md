# Roadmap

Goal: benchmark how Ultralytics' **DOTAv1-pretrained OBB models** perform across export formats on a **Jetson Orin NX 8GB**. PyTorch and ONNX are the baselines; **TensorRT** (FP32 / FP16 / INT8) is what we're measuring against them. For each configuration we report rotated-box accuracy, latency, file size, memory and power.

- **Models (4):** yolo11n-obb, yolo11s-obb, yolo26n-obb, yolo26s-obb
- **Formats (5):** PyTorch FP32 (baseline), ONNX FP32 on ONNX Runtime GPU (baseline), TensorRT FP32, TensorRT FP16, TensorRT INT8
- **Configurations (20):** 4 models x 5 formats, `imgsz=1024`, batch 1
- **Data:** DOTAv1 train + val (public labels), tiled to 1024x1024 crops with Ultralytics' `split_dota`. The test split has no labels and is not used anywhere.
- **Evaluation:** the entire val tile set. No subsampling.
- **INT8 calibration:** a seeded sample of at least 1000 *train* tiles, disjoint from val.
- **Out of scope:** video / stream FPS, inference-time tiling or SAHI techniques, training, other datasets, FP16 for PyTorch/ONNX (like-for-like comparisons are made at FP32)

## Working rules

- Tick a box only when its evidence exists (a CSV row, a log, a committed file).
- Every number in the README traces to a row in `results/*.csv`. No hand-written numbers.
- One stage at a time. 
- Official Ultralytics weights and public DOTAv1 data only.
- Evaluation and calibration are never reduced to save time. If runtime is a problem, report it and let me decide.
- Long jobs run under `tmux` or `nohup` and append a CSV row as each configuration finishes, so a crash or a restart doesn't lose finished work.
- All five formats use the same measurement protocol.

---

## Stage 0: Reset and environment

- [x] Move the old COCO/640 detection results to `results/archive/coco_detect_640/` if not already done (kept for reference, not featured)
- [x] Update `README.md` and `CLAUDE.md` for the new scope
- [x] Set up the venv with `scripts/setup_venv.sh`; README quickstart points at it, not bare `pip install -r`
- [x] Verify the baselines actually use the GPU: `torch.cuda.is_available()` is True, and ONNX Runtime (a Jetson-compatible build) lists `CUDAExecutionProvider` and really runs on it, not a silent CPU fallback
- [x] Record versions (JetPack, TensorRT, ultralytics, torch, onnxruntime, Python) and power mode (`nvpmodel`, `jetson_clocks`) in `results/environment.md`

*Done when:* the environment file exists, both GPU checks are recorded, and the repo lints clean.

## Stage 1: Data

- [x] Check free disk; decide where data and tiles live (gitignored)
- [x] Download DOTAv1 (`DOTAv1.yaml`). If the download bundles the test split, note its size and leave it untouched; do not tile or use it
- [x] Tile train and val with Ultralytics' documented `split_dota` procedure; record the settings and the tile counts
- [x] Build the calibration set: >=1000 train tiles, fixed seed, disjoint from val; record the seed and count
- [x] Estimate eval time per configuration x 20, and report it

*Done when:* `results/data_notes.md` has the tile counts, the calibration seed and count, and the eval-time estimate. If the estimate is impractical, stop and ask; do not shrink the eval set.

## Stage 2: Weights and exports

- [x] Confirm each of the 4 `.pt` files is the official DOTAv1-pretrained checkpoint (source and tag noted)
- [x] Confirm the installed ultralytics exports and validates yolo26 OBB correctly in every format
- [x] Export ONNX (FP32, `imgsz=1024`, batch 1) for all 4 models
- [x] Build TensorRT FP32 and FP16 engines for all 4 models via Ultralytics `export()`
- [x] Record file sizes and any build failures

*Done when:* 4 `.pt`, 4 `.onnx` and 8 engines each load and run one image without error.

## Stage 3: Accuracy for PyTorch, ONNX, TensorRT FP32 and FP16

- [x] Run `val()` on the full val tile set for each of these 16 configurations
- [x] Write OBB mAP50 and mAP50-95 to `results/accuracy_results.csv`, one row per finished configuration
- [x] Sanity check: PyTorch, ONNX and TensorRT FP32 agree closely for each model. Investigate any material gap before continuing

*Done when:* the CSV has 16 rows and the FP32 cross-check is written up.

## Stage 4: TensorRT INT8

- [x] Verify which split the installed ultralytics uses for INT8 calibration; force the seeded train set
- [x] Build INT8 engines for all 4 models
- [x] Run `val()` on each and add the rows to the accuracy CSV
- [x] Compare the accuracy drop vs FP32 per model

*Done when:* the accuracy CSV has 20 rows and the calibration seed and count are recorded.

## Stage 5: Latency and memory

- [x] Warmup, then measure p50 / p95 for preprocess, inference and postprocess per configuration (batch 1, clocks fixed)
- [x] Record peak GPU / unified memory and file size
- [x] Note the runtime details per row (torch device, ONNX Runtime provider, TensorRT version)
- [x] Write `results/benchmark_results.csv`

*Done when:* 20 rows, and the p50-to-p99 spread is small enough to trust.

## Stage 6: Power and thermals

- [ ] Fix `power_profile.py` so tegrastats is sampled *while* inference runs
- [ ] Sustained run per configuration: average power, peak temperature, any throttling
- [ ] Write `results/power_results.csv`

*Done when:* 20 rows, and throttling is noted where it occurred.

## Stage 7: Analysis and write-up

- [ ] Plots: accuracy vs latency (Pareto), accuracy drop vs FP32 by precision, power by configuration
- [ ] README results tables filled only from the CSVs, including TensorRT speedup over the PyTorch and ONNX baselines at FP32
- [ ] Methods section: data split and tiling, calibration choice, timing method, versions, ONNX Runtime provider
- [ ] "What I found" section in my own words: which format wins, whether INT8 hurts rotated boxes, how the baselines compare, what surprised me
- [ ] Limitations section (one board, one dataset, single-image latency, etc.)
- [ ] README states how AI assistance was used
- [ ] Reproduce section: exact commands from a fresh clone

*Done when:* someone else could rerun it from the README.

## Stage 8: Release

- [ ] Final lint and a fresh-clone dry run
- [ ] Tag `v0.1`
- [ ] Update the GitHub repo description and topics to match the new scope
