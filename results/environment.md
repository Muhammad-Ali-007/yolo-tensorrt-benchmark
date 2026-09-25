# Environment

Recorded 2026-09-25 for the DOTAv1 OBB benchmark. Every measurement in
`results/*.csv` was produced on this machine with this software stack.

> **Board memory correction.** `ROADMAP.md` describes the target as an Orin NX
> **16GB**. This machine reports `MemTotal: 7789824 kB` (7.43 GiB), which is an
> Orin NX **8GB** module — a 16GB module reports ~15.3 GiB. Carveouts account
> for a few hundred MB, not 8 GB. Recorded as measured; flagged for the user.

## Hardware / platform

| Item | Value |
|---|---|
| Board | NVIDIA Jetson Orin NX, **8GB** module (Engineering Reference Developer Kit) |
| SoC | tegra234, compute capability 8.7 |
| L4T / JetPack | R36.5.0 (JetPack 6.2), GCID 43688277 |
| CUDA toolkit | 12.6 (V12.6.68) |
| Architecture | aarch64 |
| Total memory | **7.43 GiB** (`MemTotal: 7789824 kB`), unified CPU/GPU |

## Software versions

| Package | Version |
|---|---|
| Python | 3.10.12 |
| torch | 2.11.0 (NVIDIA Jetson build) |
| torchvision | 0.26.0 |
| TensorRT | 10.3.0 |
| ultralytics | 8.4.104 |
| onnxruntime-gpu | 1.23.0 |
| onnx | 1.22.0 |
| numpy | 1.26.4 |
| OpenCV | 4.8.0 |
| pycuda | 2026.1 |

TensorRT and OpenCV come from JetPack (`/usr/lib/python3.10/dist-packages`).
torch, torchvision, ultralytics, onnxruntime-gpu are pre-existing user-site
installs. Only `pycuda`, `pycocotools` and `ruff` were installed into the
project venv — see `scripts/setup_venv.sh`.

## Power mode and clocks

| Item | State |
|---|---|
| `nvpmodel` | **MAXN** (mode 0) |
| `jetson_clocks` | **NOT applied** — clocks are unlocked |
| CPU governor | `schedutil` (dynamic) |
| GPU devfreq governor | `nvhost_podgov` (dynamic) |
| GPU freq at capture | 765 MHz (cur == max) |

**Caveat for latency stages:** `jetson_clocks` requires root and was not run,
so CPU and GPU frequencies scale dynamically (DVFS). The roadmap's Stage 5
asks for latency "with clocks fixed". Locking clocks needs
`sudo jetson_clocks` before the timing runs; until that happens, run-to-run
variance will be higher than it needs to be. This is recorded rather than
worked around.

## GPU verification (Stage 0)

Both baselines were verified to really execute on the GPU, not assumed.
Raw output: `results/logs/gpu_verify.log`.

### PyTorch — PASS

```
torch.cuda.is_available(): True
device                   : Orin
capability               : (8, 7)
2048^3 matmul on GPU     : 20.65 ms/iter
matches CPU result       : True
tensor .device           : cuda:0
```

A real 2048x2048 matmul was executed on the GPU and its result checked against
the CPU computation, so this is not just a capability flag.

### ONNX Runtime — PASS (no silent CPU fallback)

```
onnxruntime        : 1.23.0   (package: onnxruntime-gpu)
get_device()       : GPU
available providers: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
session providers  : ['CUDAExecutionProvider', 'CPUExecutionProvider']

node executions by provider: {'CUDAExecutionProvider': 4375}
CUDA EP latency    : 18.16 ms
CPU  EP latency    : 84.96 ms
speedup            : 4.68x

VERDICT: CUDA nodes=4375 CPU nodes=0 -> REAL GPU EXECUTION
```

A provider list alone does not prove GPU execution, so ONNX Runtime profiling
was enabled and the per-node trace parsed: **4375 kernel executions were placed
on `CUDAExecutionProvider` and 0 on `CPUExecutionProvider`**. The 4.68x speedup
over a CPU-only session corroborates it.

Note: `TensorrtExecutionProvider` is also available. The roadmap benchmarks
ONNX Runtime as a **CUDA** baseline, so sessions must be created with
`providers=["CUDAExecutionProvider"]` explicitly — otherwise ORT could route
the graph through TensorRT and the "ONNX baseline" would silently become a
second TensorRT measurement.

The model used for this check was the pre-existing `weights/yolov8n.onnx`
(a leftover COCO/640 artifact). It is a runtime smoke test only and is not a
benchmark row; no DOTA/OBB artifacts existed yet at Stage 0.

## Memory contention at Stage 0 (note for later stages)

At the time of recording, an **unrelated process** held 1.18 GB:

```
PID 1397  elapsed 04:23:18  RSS 1207800 kB
/usr/bin/python /home/jetson/Desktop/ied_project/serverJet.py
```

Plus ~1.5 GB of VS Code server. On a 7.43 GiB unified-memory board this leaves
roughly 2.9 GiB available. It was **not** touched — it belongs to another
project. But it is worth knowing before Stages 2-5, because TensorRT engine
builds at `imgsz=1024` reserve a workspace pool, and the PyTorch baseline holds
model weights plus CUDA context in the same unified memory. If a build or a
`val()` run dies with an allocation failure, this is the first thing to check.
