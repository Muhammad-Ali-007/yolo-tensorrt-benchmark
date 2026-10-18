# Stage 8 — Release checklist

Lint and the fresh-clone dry run are done and recorded below. Tagging and the
GitHub metadata are left for you — they require a commit and a push.

## Done: final lint

```
ruff check scripts/ src/     -> All checks passed
python -m py_compile ...     -> PASS
```

Result CSVs: `accuracy_results` 20 rows, `benchmark_results` 20 rows,
`power_results` 20 rows — all `status=ok`, 0 failures.

## Done: fresh-clone dry run

A clone was made from HEAD, the pending Stage 7 output overlaid (simulating the
post-commit state), and these checks run **inside the clone**:

| Check | Result |
|---|---|
| `ruff check scripts/ src/` | PASS |
| README regenerates byte-identical from committed CSVs | **PASS** |
| All 3 plots rebuild from the clone alone | PASS |
| Every Reproduce-section script accepts `--help` | 8/8 PASS |
| `bash -n scripts/setup_venv.sh` | PASS |
| Dataset configs present | 3/3 |

The README check is the important one: regenerating it in a clean clone produced
a **byte-identical** file, which proves every number in it derives from committed
CSV data rather than from anything local to the build machine.

## Fixed during the dry run: configs were not portable

`configs/dota_obb.yaml` and `configs/dota_calib.yaml` carried the original
machine's absolute paths:

```
path: /home/jetson/Desktop/yolo-tensorrt-benchmark/data/DOTAv1-tiles
```

In anyone else's clone those do not resolve, so the pipeline would fail at the
first `val()` — a direct failure of the stage's "someone else could rerun it"
condition.

They are absolute on purpose (Ultralytics resolves a relative dataset `path:`
against its machine-wide `datasets_dir`, which on the build machine pointed at
an unrelated project), so the fix is not to make them relative.
`scripts/prepare_dota.py` now **generates both yamls** from the running
checkout's location, and the README's Reproduce section says so. Verified by
running the generator in a clone at a different path — the paths followed the
clone. Class list preserved (15 DOTA classes).

## Remaining — for you to run

Stage 7 and the Stage 8 fix are uncommitted. Commit them first, then:

```bash
# 1. Review and commit the remaining work
git status
git add -A
git commit -m "Stage 7-8: analysis, write-up, plots, portable dataset configs"

# 2. Tag the release
git tag -a v0.1 -m "v0.1 — DOTAv1 OBB benchmark across 5 deployment formats on Jetson Orin NX"

# 3. Push commits and the tag
git push origin main
git push origin v0.1
```

### GitHub repo description

```
Benchmarking DOTAv1-pretrained YOLO OBB models across PyTorch, ONNX Runtime and
TensorRT FP32/FP16/INT8 on a Jetson Orin NX — rotated-box accuracy, latency,
memory and power, measured not estimated.
```

### GitHub topics

```
jetson  nvidia-jetson  jetson-orin-nx  tensorrt  onnxruntime  yolo
oriented-bounding-box  obb  dota  edge-ai  model-optimization
int8-quantization  benchmark  computer-vision
```

Applied from the CLI if you prefer:

```bash
gh repo edit --description "Benchmarking DOTAv1-pretrained YOLO OBB models across PyTorch, ONNX Runtime and TensorRT FP32/FP16/INT8 on a Jetson Orin NX — rotated-box accuracy, latency, memory and power, measured not estimated."
gh repo edit --add-topic jetson,nvidia-jetson,jetson-orin-nx,tensorrt,onnxruntime,yolo,oriented-bounding-box,obb,dota,edge-ai,model-optimization,int8-quantization,benchmark,computer-vision
```

## Two decisions still open

**1. Orphaned COCO-era scripts.** These are tracked but unreferenced by the new
pipeline, left from the repo's previous scope:

```
scripts/benchmark.py        scripts/calibrate_int8.py   scripts/download_data.sh
scripts/evaluate_map.py     scripts/export_onnx.py      scripts/plot_results.py
scripts/build_engine.py     configs/models.yaml         src/trt_infer.py
```

`build_engine.py` and `src/trt_infer.py` are referenced in the README only
historically (the TensorRT 10 port). None are part of the documented pipeline.
Leaving them makes the repo confusing to a newcomer; removing them is a
judgement call about whether the TensorRT-10 port work is worth keeping visible.
Not touched without your say-so.

**2. Log size.** `results/logs/` is 6.0 MB tracked, of which
`stage3_eval.log` alone is 4.85 MB — almost entirely ANSI progress-bar output.
The useful evidence in it is a handful of lines. Worth stripping the control
characters or trimming to the result lines before this is public.
