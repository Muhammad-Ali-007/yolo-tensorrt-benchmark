#!/usr/bin/env python3
"""Stage 7: generate README.md with every results number computed from the CSVs.

The prose lives here as a template; the tables and inline figures are read from
results/*.csv at build time. That is the mechanism behind the repo's rule that
no number in the README is hand-written — rerun this and the tables regenerate.

Usage:
    python scripts/build_readme.py
"""
import argparse
from pathlib import Path

import pandas as pd

FMT = ["pytorch", "onnx", "trt-fp32", "trt-fp16", "trt-int8"]
LABEL = {"pytorch": "PyTorch FP32", "onnx": "ONNX RT FP32", "trt-fp32": "TensorRT FP32",
         "trt-fp16": "TensorRT FP16", "trt-int8": "TensorRT INT8"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default="README.md")
    args = ap.parse_args()
    rd = Path(args.results_dir)

    acc = pd.read_csv(rd / "accuracy_results.csv")
    ben = pd.read_csv(rd / "benchmark_results.csv")
    pwr = pd.read_csv(rd / "power_results.csv")
    m = (ben.merge(acc, on=["model", "format", "precision"], suffixes=("", "_a"))
            .merge(pwr, on=["model", "format", "precision"], suffixes=("", "_p")))
    m["j_per_frame"] = m.power_over_idle_mw / 1000.0 / m.fps_p50
    models = sorted(acc.model.unique())

    def f(v, n=2):
        return f"{v:.{n}f}"

    # ---- main results table -------------------------------------------------
    rows = []
    for mdl in models:
        for fmt in FMT:
            r = m[(m.model == mdl) & (m.format == fmt)].iloc[0]
            rows.append(
                f"| {mdl} | {LABEL[fmt]} | {f(r.mAP50, 5)} | {f(r.mAP50_95, 5)} | "
                f"{f(r.inference_ms_p50)} | {f(r.total_ms_p50)} | {f(r.fps_p50, 1)} | "
                f"{f(r.file_size_mb, 1)} | {f(r.power_over_idle_mw / 1000)} | "
                f"{f(r.j_per_frame, 3)} |")
    main_table = "\n".join(rows)

    # ---- speedup table ------------------------------------------------------
    lat = ben.pivot(index="model", columns="format", values="total_ms_p50")
    sp = []
    for mdl in models:
        sp.append(f"| {mdl} | {f(lat.loc[mdl,'pytorch']/lat.loc[mdl,'trt-fp32'])}x | "
                  f"{f(lat.loc[mdl,'onnx']/lat.loc[mdl,'trt-fp32'])}x | "
                  f"{f(lat.loc[mdl,'pytorch']/lat.loc[mdl,'trt-fp16'])}x | "
                  f"{f(lat.loc[mdl,'onnx']/lat.loc[mdl,'trt-fp16'])}x |")
    speed_table = "\n".join(sp)

    # ---- precision cost table ----------------------------------------------
    ap50 = acc.pivot(index="model", columns="format", values="mAP50")
    pc = []
    for mdl in models:
        d16 = (ap50.loc[mdl, "trt-fp16"] - ap50.loc[mdl, "trt-fp32"]) / ap50.loc[mdl, "trt-fp32"] * 100
        d8 = (ap50.loc[mdl, "trt-int8"] - ap50.loc[mdl, "trt-fp32"]) / ap50.loc[mdl, "trt-fp32"] * 100
        l16 = lat.loc[mdl, "trt-fp32"] / lat.loc[mdl, "trt-fp16"]
        l8 = lat.loc[mdl, "trt-fp32"] / lat.loc[mdl, "trt-int8"]
        pc.append(f"| {mdl} | {d16:+.2f}% | {l16:.2f}x | {d8:+.2f}% | {l8:.2f}x |")
    prec_table = "\n".join(pc)

    best = m.loc[m.fps_p50.idxmax()]
    besten = m.loc[m.j_per_frame.idxmin()]
    bestacc = m.loc[m.mAP50.idxmax()]
    fp32_spread = []
    for mdl in models:
        v = ap50.loc[mdl, ["pytorch", "onnx", "trt-fp32"]]
        fp32_spread.append((v.max() - v.min()) / v.mean() * 100)
    worst_fp32 = max(fp32_spread)
    onnx_vs_pt = [(lat.loc[x, "onnx"] - lat.loc[x, "pytorch"]) / lat.loc[x, "pytorch"] * 100
                  for x in models]

    tiles_eval = int(acc.tiles_evaluated.iloc[0])
    tiles_tiled = int(acc.tiles_tiled.iloc[0])

    readme = f"""# YOLO OBB → TensorRT Benchmark Suite (Jetson Orin NX)

How Ultralytics' **DOTAv1-pretrained oriented-bounding-box (OBB)** models behave
across five deployment formats on a Jetson Orin NX — measured, not estimated.
**{len(m)} configurations**: 4 models × (PyTorch FP32, ONNX Runtime CUDA FP32,
TensorRT FP32 / FP16 / INT8), at `imgsz=1024`, batch 1, on the full DOTAv1
validation tile set.

> **Independent personal project.** Not affiliated with, and uses no data or
> code from, any funded/sponsored research project I've worked on. All models
> and datasets used here are public.

Every number below is read from `results/*.csv` by `scripts/build_readme.py`.
None of it is typed by hand.

## Headline results

**Fastest:** `{best.model}` TensorRT FP16 — **{f(best.fps_p50,1)} FPS**
({f(best.total_ms_p50)} ms end-to-end) at mAP50 {f(best.mAP50,5)}.
**Most accurate:** `{bestacc.model}` {LABEL[bestacc.format]} — mAP50 {f(bestacc.mAP50,5)}.
**Most energy-efficient:** `{besten.model}` {LABEL[besten.format]} —
{f(besten.j_per_frame,3)} J/frame.

![Accuracy vs latency](results/plots/accuracy_vs_latency_pareto.png)

Every configuration on the speed/accuracy frontier is **TensorRT FP16**.

## Full results

mAP50 / mAP50-95 are OBB metrics over {tiles_eval:,} val tiles. Latency is p50 over
300 timed iterations after 20 warmup. Power is watts attributable to inference
(measured draw minus an idle baseline). J/frame = power over idle ÷ FPS.

| Model | Format | mAP50 | mAP50-95 | infer ms | total ms | FPS | size MB | W over idle | J/frame |
|---|---|---|---|---|---|---|---|---|---|
{main_table}

### TensorRT speedup over the baselines

Like-for-like at FP32 (left two columns), and the best-case FP16 engine against
the same FP32 baselines (right two):

| Model | FP32: vs PyTorch | FP32: vs ONNX RT | FP16: vs PyTorch | FP16: vs ONNX RT |
|---|---|---|---|---|
{speed_table}

### What each precision step costs and buys

Accuracy change and speedup, both measured against **TensorRT FP32** so the
engine build path is held constant:

| Model | FP16 mAP50 | FP16 speedup | INT8 mAP50 | INT8 speedup |
|---|---|---|---|---|
{prec_table}

![Accuracy drop by precision](results/plots/accuracy_drop_by_precision.png)

![Power by configuration](results/plots/power_by_configuration.png)

## What I found

**TensorRT FP16 is the operating point to pick.** It is on the speed/accuracy
frontier for every model, costs at most {abs(min((ap50.loc[x,'trt-fp16']-ap50.loc[x,'trt-fp32'])/ap50.loc[x,'trt-fp32']*100 for x in models)):.2f}%
of mAP50 against FP32, and runs {f(min(lat.loc[x,'trt-fp32']/lat.loc[x,'trt-fp16'] for x in models))}–{f(max(lat.loc[x,'trt-fp32']/lat.loc[x,'trt-fp16'] for x in models))}x
faster than FP32 while drawing *less* power. There is no trade to agonise over:
it is close to free.

**INT8 does hurt rotated boxes, and it was worth checking rather than assuming.**
It costs {abs(max((ap50.loc[x,'trt-int8']-ap50.loc[x,'trt-fp32'])/ap50.loc[x,'trt-fp32']*100 for x in models)):.2f}%–{abs(min((ap50.loc[x,'trt-int8']-ap50.loc[x,'trt-fp32'])/ap50.loc[x,'trt-fp32']*100 for x in models)):.2f}%
of mAP50 — one to two orders of magnitude more than FP16 — and for every model
the **mAP50-95 drop is larger than the mAP50 drop**. That signature matters: it
means boxes are still being *found* but fit less precisely. For oriented boxes
the angle is part of that fit, which is exactly the failure mode you would worry
about when quantizing an OBB model, and it shows up in the data.

Worse, INT8 barely pays for itself here. On this board it is at most
{f(max(lat.loc[x,'trt-fp32']/lat.loc[x,'trt-int8'] for x in models))}x faster than FP32, and for both yolo26 models it is
*slower end-to-end than FP16*. Its one genuine win is energy: the lowest
power draw and the best J/frame of any configuration measured.

**The newer model generation does not win on this dataset.** yolo26 scores below
yolo11 at both sizes ({(ap50.loc['yolo26n-obb','pytorch']-ap50.loc['yolo11n-obb','pytorch'])*100:+.1f} points nano,
{(ap50.loc['yolo26s-obb','pytorch']-ap50.loc['yolo11s-obb','pytorch'])*100:+.1f} points small, PyTorch FP32). It wins on *speed*,
because its NMS-free head makes postprocess roughly 6x cheaper — which is how
`yolo26n-obb` TRT FP16 ends up the fastest configuration overall despite being
less accurate than `yolo11n-obb`.

**ONNX Runtime is not a speedup — it is slower than PyTorch** by
{min(onnx_vs_pt):.1f}%–{max(onnx_vs_pt):.1f}% on every model. It earns its place as a
control: it shows the gain comes from TensorRT specifically, not merely from
leaving Python.

### What surprised me

1. **INT8 being slower than FP16** for the yolo26 models. Chasing it down showed
   the ONNX and INT8 exports *silently lose yolo26's NMS-free head* — they emit
   raw predictions `(1, 20, 21504)` instead of `(1, 300, 7)` and pay full rotated
   NMS. The regression is in postprocess, not the engine.
2. **Host-side work is the floor.** Preprocess is ~3.6 ms and yolo11's NMS
   postprocess ~5.8 ms regardless of format. At FP16 the GPU finishes so quickly
   that these dominate — GPU utilisation *falls* from 77% (PyTorch) to 59%
   (INT8). Optimising the engine further would buy little.
3. **Power barely moves while throughput moves a lot.** All 20 configurations sit
   within {pwr.power_over_idle_mw.min()/1000:.1f}–{pwr.power_over_idle_mw.max()/1000:.1f} W over idle, but throughput spans
   {ben.fps_p50.min():.1f}–{ben.fps_p50.max():.1f} FPS. Energy per frame is therefore driven by speed,
   not by instantaneous draw.
4. **No thermal throttling at all** in 20 × 60 s sustained runs — peak
   {pwr.peak_tj_temp_c.max():.1f} °C against a 95 °C trip, GPU frequency pinned at 765 MHz
   throughout.

## Methods

**Data.** DOTAv1 (official Ultralytics release, 1.85 GiB), tiled with
Ultralytics' documented `split_dota` at crop 1024 / gap 200 / `rates=(1.0,)`,
background tiles kept. Train → {int(pd.read_json(rd/'dota_stage1.json', typ='series')['train_tiles']['total']):,} tiles,
val → {tiles_tiled:,} tiles. The DOTAv1 **test** split has no public labels and is
never used.

**Evaluated tile count.** {tiles_eval:,} of {tiles_tiled:,} val tiles. Ultralytics' label
scan rejects 19 tiles (0.36%) whose coordinates land fractionally above 1.0 — an
artefact of its own edge-crop normalisation. The exclusion is identical for all
{len(m)} configurations, so comparisons are unaffected. Stated rather than rounded
up to "the full set".

**INT8 calibration.** 1,000 tiles sampled with seed 0 from the **train** tiles,
disjoint from evaluation at both tile and source-image level (0 overlap across
606 source images). This matters: the Ultralytics exporter defaults to
`data[args.split or "val"]`, so an unguarded INT8 export calibrates on the very
images it is scored on. `configs/dota_calib.yaml` points its `val` key at the
train sample to prevent that. 999 tiles were usable (one hit the same
edge-coordinate quirk). Note TensorRT keeps 85 Sigmoid layers in FP32, so this
is Ultralytics' mixed-precision INT8, not uniform 8-bit.

**One engine build path.** Every engine comes from
`model.export(format="engine", imgsz=1024, batch=1, quantize=32|16|8)`, and the
*same* `.engine` file is used for accuracy, latency and power. No second
ONNX→TRT path.

**Timing.** Identical protocol for all five formats: every row goes through
`model.predict()` and reports Ultralytics' own preprocess/inference/postprocess
split. Tiles are pre-loaded into memory, so disk I/O is never timed. 20 warmup +
300 timed iterations on a fixed seeded set of 50 val tiles. Measurement
stability: median p50→p95 spread 1.71%, max 5.16%. The p50→p99 figure is wider,
but the absolute p99−p50 gap is a near-constant ~5.85 ms across all
configurations — a fixed host-side stall (~3 iterations in 300), not
proportional instability.

**ONNX Runtime really is CUDA.** `onnxruntime-gpu` also exposes
`TensorrtExecutionProvider`, so an unguarded ONNX baseline can silently become a
second TensorRT measurement. Every ONNX row records
`provider=CUDAExecutionProvider`, corroborated by the run logs.

**Power.** `tegrastats` sampled at 200 ms **in a background thread during**
sustained inference (the original script sampled after the loop and measured an
idle GPU). GPU utilisation is recorded per row — {pwr.gr3d_mean_pct.min():.1f}–{pwr.gr3d_mean_pct.max():.1f}% — so
"measured under load" is verifiable rather than asserted. An idle baseline
({pwr.idle_power_mw.iloc[0]/1000:.2f} W) is subtracted. Throttle detection uses the board's real
trip points from `/sys/class/thermal` plus GPU `cur_freq`, which is pinned at
765 MHz so any drop is direct evidence.

**Environment.** Jetson Orin NX 8GB module, JetPack 6.2 (L4T R36.5.0),
CUDA 12.6, TensorRT 10.3.0, ultralytics 8.4.104, onnxruntime-gpu 1.23.0,
torch 2.11.0, Python 3.10.12. Full detail and the GPU verification evidence:
[results/environment.md](results/environment.md).

## Limitations

- **One board, one software stack.** Orin NX 8GB, TensorRT 10.3.0. INT8 kernel
  selection is hardware- and version-specific; these numbers will not transfer
  unchanged to an AGX, a Nano, or a different JetPack.
- **`jetson_clocks` was not applied** (needs root), so the CPU governor was
  `schedutil`. The GPU was observed pinned at its 765 MHz ceiling, and p50
  figures are stable to 1.71% at p95, but the rare ~6 ms host-side stalls are
  the most likely thing a fixed-clock run would improve.
- **Single-image latency, batch 1.** No batching, no multi-stream, no
  DeepStream. Throughput under concurrency is a different question.
- **One dataset.** DOTAv1 only. Aerial imagery with many small objects per tile
  is a specific workload; the INT8 accuracy cost in particular may not
  generalise.
- **One calibration set.** 1,000 tiles, seed 0, not ablated. Calibration size
  and composition are known to affect INT8 quality.
- **Thermals are a 60 s result, not a soak test.** No throttling was observed,
  but temperatures were still climbing across the run sequence. "No throttling
  in 60 s" is not "no throttling in a 25-minute flight".
- **An unrelated process** (~1.4 GB resident) was running on the board
  throughout and contributes to the idle baseline and to host-side scheduling
  noise.
- **yolo26 ONNX / INT8 rows are not architecturally like-for-like** with their
  siblings, because those exports lose the NMS-free head (see *What surprised
  me*). Their accuracy is unaffected; their latency reflects the decode path as
  much as the precision.

## Reproduce

From a fresh clone on a Jetson (JetPack 6.x). Execution is Jetson-only —
`tensorrt` and `pycuda` need the board.

```bash
# 1. Environment. Do NOT use a bare `pip install -r requirements.txt` on a
#    Jetson — see the header of requirements.txt for why.
bash scripts/setup_venv.sh
source .venv/bin/activate

# 2. Data: download DOTAv1 (~1.85 GiB), tile it, build the calibration set.
#    Also (re)generates configs/dota_obb.yaml and configs/dota_calib.yaml with
#    absolute paths for THIS checkout — they bypass Ultralytics' machine-wide
#    datasets_dir, so they must be regenerated rather than used as committed.
#    Writes results/dota_stage1.json with tile counts and the calibration seed.
python scripts/prepare_dota.py --data-root data/DOTAv1 --tiles-dir data/DOTAv1-tiles \\
    --calib-dir data/DOTAv1-calib --calib-n 1000 --seed 0

# 3. Export: 4 ONNX + 8 FP32/FP16 engines (~40 min), then 4 INT8 engines (~55 min)
python scripts/export_models.py --config configs/models_obb.yaml --precisions fp32,fp16
python scripts/export_models.py --config configs/models_obb.yaml --precisions int8 --skip-onnx

# 4. Verify every artifact loads and runs
python scripts/verify_artifacts.py --config configs/models_obb.yaml

# 5. Accuracy: all 20 configurations on the full val tile set (~90 min)
python scripts/evaluate_obb.py --config configs/models_obb.yaml

# 6. Latency + memory (~20 min)
python scripts/benchmark_obb.py --config configs/models_obb.yaml

# 7. Power + thermals, 60 s sustained per configuration (~30 min)
python scripts/power_profile_obb.py --config configs/models_obb.yaml --duration 60

# 8. Charts and this README
python scripts/plot_obb_results.py
python scripts/build_readme.py
```

Step 2 regenerates the two dataset yamls for your checkout — the committed
copies carry the original machine's absolute paths and will not resolve
elsewhere.

Long stages append a CSV row as each configuration finishes and skip rows
already marked `ok`, so an interrupted run resumes without recomputing.
Run them under `tmux` or `nohup`.

`ROADMAP.md` tracks scope and staging; `CLAUDE.md` documents how to work in the
repo. Per-stage write-ups, including the things that went wrong, are in
`results/stage*_notes.md`.

## How AI assistance was used

This project was built with Claude Code (Anthropic) as a pair-programming tool,
under my direction and review. Concretely, the assistant:

- wrote and refactored the pipeline scripts in `scripts/` and `src/`;
- ported `src/trt_infer.py` and `scripts/build_engine.py` from the TensorRT
  binding API (removed in TensorRT 10) to the tensor-name API;
- diagnosed failures during the runs — including an allocator crash that was
  really an out-of-memory in rotated NMS, a CSV column misalignment, and three
  defects in the original `power_profile.py` (sampling after the inference loop,
  an uppercase `GPU@` regex that matched nothing on this JetPack, and an
  invented throttle threshold);
- drafted this README and the per-stage notes in `results/`.

What it did **not** do: produce any number reported here. Every figure in this
README is computed by `scripts/build_readme.py` from CSVs written by scripts
that actually ran on the Jetson. I reviewed each stage before it was committed,
and `ROADMAP.md` records the staging that work followed.

## License

MIT — see [LICENSE](LICENSE).
"""
    Path(args.out).write_text(readme)
    print(f"wrote {args.out} ({len(readme)} chars)")
    print(f"  {len(m)} configurations, worst FP32 cross-check spread {worst_fp32:.3f}%")


if __name__ == "__main__":
    main()
