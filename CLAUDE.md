# CLAUDE.md

Guidance for Claude Code when working in this repo.

## How I want to work (standing rules)

These are the user's process rules. They persist across sessions and take
precedence over convenience.

**Scope and staging**
- `ROADMAP.md` is the source of truth for scope. It supersedes any earlier
  plan or answer. If a plan differs from it, say so *before* acting.
- Work **one roadmap stage at a time**. Stop at the end of each stage; do not
  start the next until the user says so.
- At each stop, report: what was done, what evidence exists (files, CSV rows,
  logs), what decisions were made and why, and anything surprising or that
  failed. Explain non-obvious choices in plain language — the user needs to be
  able to explain and defend every part of this project themselves.

**Evidence and honesty**
- Tick a `ROADMAP.md` checkbox **only when its evidence exists**. Verification
  items (GPU checks, the calibration split, the FP32 cross-check) must be
  actually verified, never assumed.
- Every number in the README must trace to a CSV row. Never estimate or
  hand-write results.
- **Never reduce evaluation or calibration to save time.** If runtime is a
  problem, say so and let the user decide.

**Long-running jobs**
- Run long jobs under `tmux` or `nohup`.
- Append a CSV row as each configuration finishes, so runs are resumable.
- Tell the user *before* leaving anything running unattended.

**Boundaries — ask first**
- Do **not** commit or push. The user reviews and commits each stage.
- Ask before any download other than DOTAv1, and check free disk first.
  DOTAv1 train and val are approved. The test split is **not** used — don't
  tile or evaluate on it.
- Ask before deleting or overwriting anything the user didn't explicitly ask
  to change.
- Use only official Ultralytics weights and public DOTAv1 data. Don't
  reference, copy, or load any other checkpoint or dataset on this machine.

## What this is

A benchmark suite measuring how Ultralytics' **DOTAv1-pretrained OBB
(oriented bounding box) models** perform across five deployment formats on a
Jetson Orin NX: PyTorch FP32 and ONNX Runtime GPU FP32 as baselines, then
TensorRT FP32 / FP16 / INT8. Rotated-box accuracy, latency, file size, memory
and power, at `imgsz=1024`, batch 1.

`ROADMAP.md` is the source of truth for scope and staging. [README.md](README.md)
is the outward-facing description. This file is about *how to work in the repo*.

Scope in one line: **4 models** (`yolo11n-obb`, `yolo11s-obb`, `yolo26n-obb`,
`yolo26s-obb`) x **5 formats** = 20 configurations, evaluated on the **entire**
DOTAv1 val tile set.

## Where this actually runs

Development/editing can happen anywhere, but **execution is Jetson-only**.
`tensorrt`, `pycuda` and CUDA are not installable/runnable on a normal dev
machine. Off-device, the useful things to do are: edit code,
`python -m py_compile` for syntax checks, `ruff check scripts/ src/` for lint,
and reason about logic — not execute the pipeline.

Environment setup is **`bash scripts/setup_venv.sh`**, never a bare
`pip install -r requirements.txt`. The venv is `--system-site-packages` on
purpose: `tensorrt` and OpenCV come from JetPack, and `torch`, `torchvision`,
`ultralytics` and `onnxruntime-gpu` are pre-existing aarch64 CUDA builds that
cannot be reinstalled from PyPI. Installing the PyPI versions over them breaks
GPU support. See the header of `requirements.txt`.

The verified stack is recorded in `results/environment.md`: JetPack 6.2
(R36.5.0), CUDA 12.6, **TensorRT 10.3.0**, ultralytics 8.4.104,
onnxruntime-gpu 1.23.0, torch 2.11.0, Python 3.10.12, on an Orin NX **8GB**
module (note: the roadmap says 16GB; the board measures 7.43 GiB).

Two things that bite:
- **TensorRT 10 removed the legacy binding API.** `src/trt_infer.py` has
  already been ported to the tensor-name API (`get_tensor_name`,
  `set_tensor_address`, `execute_async_v3`), and `scripts/build_engine.py` uses
  `config.set_memory_pool_limit(...)` instead of the removed
  `max_workspace_size`. Don't "fix" these back toward the old API.
- **Ultralytics-exported `.engine` files are not raw engines.** They prepend a
  4-byte little-endian length plus a JSON metadata blob before the serialized
  engine (written at `ultralytics/utils/export/engine.py`, read back at
  `ultralytics/nn/backends/tensorrt.py`). Any code that deserializes one
  directly must skip that header.

## Common commands

```bash
bash scripts/setup_venv.sh                  # environment (idempotent)
source .venv/bin/activate

ruff check scripts/ src/                    # lint (works off-device too)
python -m py_compile scripts/*.py src/*.py  # syntax check (works off-device too)
```

Pipeline commands are added to README.md as each roadmap stage lands, so that
every documented command is one that has actually been run on the board.

## Ground rules for this repo specifically

- **Never hand-write numbers into the README's results table.** Every number
  there must trace back to a row in `results/benchmark_results.csv`,
  `results/accuracy_results.csv`, or `results/power_results.csv` produced by
  actually running the scripts on the Jetson. If asked to "fill in the
  results," the answer is to run the pipeline, not to estimate plausible
  numbers — an unverifiable benchmark is worse than no benchmark.
- **One engine build path.** Engines come from Ultralytics
  `model.export(format="engine", ...)`, and the *same* `.engine` is used for
  both accuracy and latency. Don't add a second custom ONNX->TRT path unless
  you can show its engines load and score identically.
- **All five formats use the same measurement protocol.** A comparison is only
  meaningful if PyTorch, ONNX Runtime and the three TensorRT precisions are
  timed and scored the same way.
- **The ONNX baseline must really be ONNX Runtime CUDA.** `onnxruntime-gpu`
  also exposes `TensorrtExecutionProvider`; create sessions with an explicit
  `providers=["CUDAExecutionProvider"]` or the "ONNX baseline" silently becomes
  a second TensorRT measurement.
- **Adding a model variant** means editing the model config only — every script
  iterates that list, nothing is hardcoded per-model. Don't add a new model by
  copy-pasting a script.
- **Don't commit weights/engines/data.** `.gitignore` already excludes `*.pt`,
  `*.onnx`, `*.engine`, and `data/*` — they're large and regeneratable. If a
  change seems to require committing one, stop and ask first.
- This project is independent of any funded/sponsored work — don't pull in
  data, weights, or code from elsewhere without knowing its provenance and
  that it's actually public. In particular, other projects exist on this
  machine (e.g. `~/Desktop/ied_project`); never read from or write to them.
