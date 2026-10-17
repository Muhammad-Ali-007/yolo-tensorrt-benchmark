# Stage 6 — Power and thermals

20 configurations, 0 failures, 60 s of sustained inference each (~294 tegrastats
samples per configuration at 200 ms). Every number traces to
`results/power_results.csv`; energy-per-frame is derived in
`results/efficiency_summary.csv`.

## The fix: sampling during inference, not after it

`scripts/power_profile.py` (the original) called `benchmark_engine(...)` to
completion and *then* `sample_tegrastats(...)`, so it sampled an idle GPU and
reported idle power as if it were load power.

`scripts/power_profile_obb.py` runs the tegrastats reader in a background thread
for the entire sustained run. **`gr3d_mean_pct` (GPU utilisation) is recorded so
this is verifiable from the data rather than asserted:** it ranges
**58.8% to 92.4%** across the 20 runs. An idle GPU reads ~0%. Combined with the
idle baseline below, the samples are demonstrably taken under load.

Two further defects in the original, both silent:

- `GPU_TEMP_RE = re.compile(r"GPU@...")` — this JetPack emits **lowercase**
  `gpu@57.687C`, so the regex matched nothing and every temperature was `None`.
  Fixed with case-insensitive, word-boundary patterns.
- The `>= 85C` throttle heuristic was invented. Replaced with the board's real
  trip points, read from `/sys/class/thermal` (see below).

## Idle baseline

A 6 s idle measurement is taken once before profiling: **8,405 mW**. The board
draws roughly 8 W doing nothing, so raw `VDD_IN` substantially overstates what
inference costs. Both are reported — `avg_power_mw` (total board) and
`power_over_idle_mw` (attributable to the workload) — and the latter is the
honest number for comparing models.

## Results

| Model | Format | avg W | W over idle | GPU util % | peak tj C | throttled |
|---|---|---|---|---|---|---|
| yolo11n-obb | PyTorch | 17.11 | 8.70 | 77.4 | 70.3 | no |
| yolo11n-obb | ONNX RT | 17.59 | 9.18 | 79.6 | 72.7 | no |
| yolo11n-obb | TRT FP32 | 17.29 | 8.88 | 70.4 | 73.0 | no |
| yolo11n-obb | TRT FP16 | 15.57 | 7.17 | 59.9 | 72.2 | no |
| yolo11n-obb | TRT INT8 | 14.58 | 6.18 | 58.8 | 71.1 | no |
| yolo11s-obb | PyTorch | 18.96 | 10.55 | 87.1 | 74.7 | no |
| yolo11s-obb | ONNX RT | 19.01 | 10.60 | 88.2 | 75.9 | no |
| yolo11s-obb | TRT FP32 | 18.73 | 10.33 | 84.1 | 76.6 | no |
| yolo11s-obb | TRT FP16 | 17.51 | 9.10 | 71.7 | 75.8 | no |
| yolo11s-obb | TRT INT8 | 16.36 | 7.95 | 67.9 | 74.2 | no |
| yolo26n-obb | PyTorch | 18.22 | 9.81 | 85.8 | 75.2 | no |
| yolo26n-obb | ONNX RT | 17.43 | 9.02 | 80.7 | 74.8 | no |
| yolo26n-obb | TRT FP32 | 18.26 | 9.86 | 81.7 | 75.2 | no |
| yolo26n-obb | TRT FP16 | 16.68 | 8.27 | 71.2 | 74.4 | no |
| yolo26n-obb | TRT INT8 | 14.57 | 6.16 | 60.0 | 73.0 | no |
| yolo26s-obb | PyTorch | 18.92 | 10.52 | 92.4 | 75.3 | no |
| yolo26s-obb | ONNX RT | 18.89 | 10.48 | 86.6 | 76.3 | no |
| yolo26s-obb | TRT FP32 | 18.90 | 10.50 | 90.3 | 77.2 | no |
| yolo26s-obb | TRT FP16 | 18.64 | 10.24 | 81.9 | 77.3 | no |
| yolo26s-obb | TRT INT8 | 16.54 | 8.14 | 69.1 | 75.6 | no |

## Throttling: none observed, and here is what that claim rests on

**0 of 20 configurations throttled.** This is not inferred from a temperature
guess. GPU devfreq on this board is pinned — `min == max == 765 MHz` — so any
drop in `cur_freq` is direct evidence of thermal capping. `cur_freq` was sampled
continuously alongside tegrastats and **held at 765 MHz in every one of the 20
runs** (`gpu_freq_min_mhz == gpu_freq_max_mhz == 765`).

Trip points read from `/sys/class/thermal`, not assumed:

| Zone | Trip | Temp | Type |
|---|---|---|---|
| gpu-thermal | 2 | 70 C | passive |
| gpu-thermal | 0 | 99 C | passive |
| gpu-thermal | 1 | 104 C | critical |
| tj-thermal | 1 | 74 C | active |
| tj-thermal | 2 | 95 C | active |
| tj-thermal | 3 | 104 C | critical |

Peak temperature across all 20 sustained runs was **77.3 C** (tj and gpu),
against a 95 C active trip and 104 C critical. The 74 C tj trip was exceeded,
but it is an *active* trip (fan control), not a throttle, and no frequency drop
accompanied it.

**Caveat on generalising this.** Twenty 60-second runs is a sustained load but
not a long one, temperatures were still climbing through the sequence
(70.3 C early, 77.3 C late), and ambient conditions were not controlled. "No
throttling in 60 s" is not the same claim as "no throttling in a 25-minute
flight". A longer soak test would be needed for that, and was not run.

## Energy per frame — the metric that matters for a battery

Derived as `power_over_idle_W / fps_p50`, i.e. joules of *workload* energy per
inference. Top and bottom of the table:

| Config | FPS | W over idle | **J/frame** |
|---|---|---|---|
| yolo11n-obb TRT INT8 | 51.3 | 6.18 | **0.120** |
| yolo26n-obb TRT FP16 | 67.5 | 8.27 | **0.123** |
| yolo26n-obb TRT INT8 | 49.9 | 6.16 | 0.124 |
| yolo11n-obb TRT FP16 | 50.2 | 7.17 | 0.143 |
| ... | | | |
| yolo11n-obb PyTorch | 23.0 | 8.70 | 0.378 |
| yolo11s-obb PyTorch | 13.9 | 10.55 | 0.759 |
| yolo26s-obb ONNX RT | 13.0 | 10.48 | 0.807 |

**TensorRT FP16 on yolo26n is 3.1x more energy-efficient per frame than the same
model in PyTorch** (0.123 vs 0.378 J/frame), and 6.6x better than the worst
configuration measured.

## Observations

**Lower precision is monotonically more power-efficient.** For every model the
ladder holds — e.g. yolo11s: PyTorch 10.55 W -> FP32 10.33 -> FP16 9.10 ->
INT8 7.95 W over idle. TensorRT FP16 draws *less* power than FP32 while also
running faster, so the gain is compounded.

**Power differences between formats are modest; throughput differences are
not.** The spread in `power_over_idle_mw` across all 20 configurations is only
about 6.2-10.6 W, while throughput spans 13.0-67.5 FPS. Energy per frame is
therefore driven far more by speed than by instantaneous draw — which is why
J/frame, not watts, is the right column for a power-constrained deployment.

**GPU utilisation falls as precision drops** (yolo11n: 77.4% PyTorch -> 58.8%
INT8). The GPU finishes its work sooner but the CPU-side preprocess and
postprocess are unchanged, so the GPU idles more within each iteration. This is
consistent with Stage 5, where the ~3.6 ms preprocess and (for yolo11) ~5.8 ms
NMS postprocess are constant across every format — at FP16/INT8 the host-side
work, not the GPU, is the limiting factor.

**INT8's best case is energy, not latency.** Stage 5 showed INT8 buys at most
10% latency and is *slower* than FP16 for yolo26. But it is consistently the
lowest-power option, and yolo11n INT8 has the best J/frame of any configuration.
Against that it costs 4-6 mAP50 points (Stage 4). For a battery-limited
deployment that tolerates the accuracy loss, INT8 has a genuine case; for
anything else FP16 remains the better operating point.

## Conditions

`nvpmodel` MAXN; `jetson_clocks` **not applied** (requires root), CPU governor
`schedutil`. GPU frequency was nonetheless pinned at 765 MHz throughout. An
unrelated process (`ied_project/serverJet.py`, ~1.4 GB resident) was running and
contributes to both the idle baseline and the load figures; it was deliberately
not touched. Both clock fields are recorded on every row.
