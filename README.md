# YOLO → TensorRT Benchmark Suite (Jetson Orin NX)

Benchmarks PyTorch YOLO models exported through ONNX into TensorRT engines at
FP32 / FP16 / INT8, and measures the actual cost of each optimization step —
latency, throughput, accuracy, and (optionally) power/thermal draw — on a
Jetson Orin NX.

> **Independent personal project.** Not affiliated with, and uses no data or
> code from, any funded/sponsored research project I've worked on. All models
> and datasets used here are public.

## Why this exists

Everyone benchmarks YOLO on a desktop GPU. Fewer people show the actual
precision-vs-speed-vs-accuracy tradeoff *on the edge hardware it's meant to
run on*, and almost nobody shows what that costs in power draw or thermal
throttling over a sustained run — which is what actually matters for a
battery-powered or thermally-constrained deployment.

This repo answers three questions per model:
1. **How much faster is TensorRT than the raw PyTorch/ONNX baseline?**
2. **What does each precision level (FP32 → FP16 → INT8) cost in accuracy?**
3. **Can the Jetson actually sustain that throughput without throttling?**

## Hardware / software this was built for

- NVIDIA Jetson Orin NX 8GB, JetPack (fill in your version: `dpkg -l | grep nvidia-jetpack`)
- TensorRT (ships with JetPack — check `python3 -c "import tensorrt; print(tensorrt.__version__)"`)
- Python 3.8+ (JetPack default)

## Repo structure

```
.
├── configs/
│   └── models.yaml          # which model variants + precisions to benchmark
├── scripts/
│   ├── download_data.sh     # pulls a small public COCO val2017 subset for eval/calibration
│   ├── export_onnx.py       # PyTorch (.pt) -> ONNX
│   ├── build_engine.py      # ONNX -> TensorRT engine (fp32 / fp16 / int8)
│   ├── calibrate_int8.py    # INT8 entropy calibrator used by build_engine.py
│   ├── benchmark.py         # latency/throughput benchmark, writes results/benchmark_results.csv
│   ├── evaluate_map.py      # mAP@0.5:0.95 per engine, writes results/accuracy_results.csv
│   ├── power_profile.py     # wraps benchmark.py with tegrastats power/thermal logging
│   └── plot_results.py      # generates the charts in results/plots/
├── src/
│   └── trt_infer.py         # TensorRTInfer: engine loading, buffer allocation, inference
├── results/                 # benchmark_results.csv, accuracy_results.csv, plots/ (generated, not fabricated)
└── data/                    # downloaded dataset lives here (gitignored)
```

## Quickstart

```bash
pip install -r requirements.txt
bash scripts/download_data.sh                 # small COCO val2017 subset

# 1. Export each configured model to ONNX
python scripts/export_onnx.py --config configs/models.yaml

# 2. Build TensorRT engines at each precision
python scripts/build_engine.py --config configs/models.yaml --precision fp32
python scripts/build_engine.py --config configs/models.yaml --precision fp16
python scripts/build_engine.py --config configs/models.yaml --precision int8 --calib-data data/calib

# 3. Benchmark latency/throughput
python scripts/benchmark.py --config configs/models.yaml --iterations 500 --warmup 50

# 4. Evaluate accuracy per engine
python scripts/evaluate_map.py --config configs/models.yaml --data data/val2017

# 5. (optional) power/thermal profile while benchmarking
python scripts/power_profile.py --config configs/models.yaml --power-mode 15W

# 6. Plot everything
python scripts/plot_results.py
```

## Results

Not filled in yet — numbers below are placeholders until run on hardware.
Run the pipeline above and replace this table; **do not** hand-edit numbers
in without a corresponding `results/benchmark_results.csv` row backing them.

| Model | Precision | Latency (ms, p50) | Latency (ms, p95) | FPS | mAP@0.5:0.95 | Engine size |
|---|---|---|---|---|---|---|
| yolov8n | FP32 | – | – | – | – | – |
| yolov8n | FP16 | – | – | – | – | – |
| yolov8n | INT8 | – | – | – | – | – |
| yolo11n | FP32 | – | – | – | – | – |
| yolo11n | FP16 | – | – | – | – | – |
| yolo11n | INT8 | – | – | – | – | – |

Power/thermal (optional, `power_profile.py`):

| Model | Precision | Power mode | Avg power (mW) | Peak temp (°C) | Throttled? |
|---|---|---|---|---|---|
| – | – | – | – | – | – |

## Roadmap

- [ ] Fill in results table from actual Orin NX runs
- [ ] Add a second dataset option (VisDrone) for an aerial-detection angle
- [ ] Add DeepStream multi-stream variant for batched throughput numbers
- [ ] INT8 calibration dataset ablation (does calibration set size/composition matter?)

## License

MIT — see [LICENSE](LICENSE).
