# CLAUDE.md

Guidance for Claude Code when working in this repo.

## What this is

A benchmark suite that exports YOLO models (PyTorch → ONNX → TensorRT) and
measures latency, throughput, accuracy, and power/thermal behavior on a
Jetson Orin NX. See [README.md](README.md) for the full pipeline and
quickstart — this file is about *how to work in the repo*, not what it does.

## Where this actually runs

Development/editing can happen anywhere, but **execution is Jetson-only**.
`tensorrt`, `pycuda`, and CUDA are not installable/runnable on a normal dev
machine (no ARM Jetson GPU) — don't try to `pip install tensorrt` off-device,
and don't expect `python scripts/benchmark.py` etc. to actually run outside
the Jetson. Off-device, the useful things Claude Code can do are: edit code,
`python -m py_compile` for syntax checks, `ruff check scripts/ src/` for
lint, and reason about logic — not execute the pipeline.

On the Jetson itself:
- `tensorrt` and its Python bindings ship with JetPack — install via
  `sudo apt install python3-libnvinfer python3-libnvinfer-dev`, **not** pip.
- `pycuda` is the one GPU dependency actually installed via pip
  (`pip install pycuda`), and needs the CUDA toolkit on `PATH`/`LD_LIBRARY_PATH`
  (JetPack sets this up; if `import pycuda.driver` fails, that's the first
  thing to check).
- Check the installed TensorRT version before touching `src/trt_infer.py`:
  `python3 -c "import tensorrt; print(tensorrt.__version__)"`. That file
  uses the pre-8.5 binding-index API (`get_binding_name`,
  `execute_async_v2`); TensorRT 10.x wants the tensor-name API
  (`get_tensor_name`, `execute_async_v3`) instead. If asked to fix or extend
  inference code and the version doesn't match, port it rather than papering
  over the mismatch.

## Common commands

```bash
ruff check scripts/ src/                                  # lint (works off-device too)
python -m py_compile scripts/*.py src/*.py                # syntax check (works off-device too)

bash scripts/download_data.sh                              # Jetson only, needs network + disk
python scripts/export_onnx.py --config configs/models.yaml
python scripts/build_engine.py --config configs/models.yaml --precision fp16
python scripts/benchmark.py --config configs/models.yaml --iterations 500 --warmup 50
python scripts/evaluate_map.py --config configs/models.yaml --data data/val2017
python scripts/plot_results.py
```

## Ground rules for this repo specifically

- **Never hand-write numbers into the README's results table.** Every number
  there must trace back to a row in `results/benchmark_results.csv`,
  `results/accuracy_results.csv`, or `results/power_results.csv` produced by
  actually running the scripts on the Jetson. If asked to "fill in the
  results," the answer is to run the pipeline, not to estimate plausible
  numbers — an unverifiable benchmark is worse than no benchmark.
- **Adding a model variant** means editing `configs/models.yaml` only — every
  script iterates that list, nothing is hardcoded per-model. Don't add a new
  model by copy-pasting a script.
- **Don't commit weights/engines/data.** `.gitignore` already excludes
  `*.pt`, `*.onnx`, `*.engine`, and `data/*` — they're large and
  regeneratable from the scripts. If a change seems to require committing
  one, stop and ask first.
- This project is independent of any funded/sponsored work — don't pull in
  data, weights, or code from elsewhere without knowing its provenance and
  that it's actually public.
