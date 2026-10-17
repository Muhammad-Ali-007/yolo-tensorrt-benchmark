#!/usr/bin/env python3
"""Stage 6: power and thermal behaviour during sustained inference.

The original scripts/power_profile.py sampled tegrastats *after* the inference
loop had already finished, so it measured an idle GPU. Here the sampler runs in
a background thread for the whole duration of a sustained inference run, so
every sample is taken under load. `gr3d_mean_pct` (GPU utilisation) is recorded
specifically so that claim is verifiable from the data rather than asserted.

Also fixed: this JetPack emits lowercase sensor names (`gpu@57.687C`), so the
old uppercase `GPU@` regex matched nothing and temperatures were always None.

Throttle detection uses the board's real trip points rather than a guessed
threshold, read from /sys/class/thermal:
  gpu-thermal passive 70C and 99C, critical 104C
  tj-thermal  active 74C and 95C,  critical 104C
GPU devfreq on this board is pinned (min == max == 765 MHz), so any observed
drop in cur_freq is direct evidence of thermal capping.

Usage:
    python scripts/power_profile_obb.py --config configs/models_obb.yaml --duration 60
"""
import argparse
import csv
import os
import re
import subprocess
import threading
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

VDD_IN_RE = re.compile(r"VDD_IN (\d+)mW")
VDD_CGC_RE = re.compile(r"VDD_CPU_GPU_CV (\d+)mW")
VDD_SOC_RE = re.compile(r"VDD_SOC (\d+)mW")
GPU_T_RE = re.compile(r"\bgpu@(-?\d+\.?\d*)C", re.IGNORECASE)
CPU_T_RE = re.compile(r"\bcpu@(-?\d+\.?\d*)C", re.IGNORECASE)
TJ_T_RE = re.compile(r"\btj@(-?\d+\.?\d*)C", re.IGNORECASE)
GR3D_RE = re.compile(r"GR3D_FREQ (\d+)%")

GPU_FREQ_PATHS = list(Path("/sys/class/devfreq").glob("*gpu*"))
FIELDS = [
    "model", "format", "precision", "artifact", "duration_s", "iterations",
    "avg_power_mw", "peak_power_mw", "idle_power_mw", "power_over_idle_mw",
    "avg_cpu_gpu_cv_mw", "avg_soc_mw",
    "peak_gpu_temp_c", "peak_cpu_temp_c", "peak_tj_temp_c",
    "gr3d_mean_pct", "gr3d_max_pct",
    "gpu_freq_min_mhz", "gpu_freq_max_mhz", "throttled", "throttle_evidence",
    "samples", "nvpmodel", "jetson_clocks", "status", "error",
]


class TegraSampler(threading.Thread):
    """Read tegrastats continuously while inference runs."""

    def __init__(self, interval_ms: int = 200):
        super().__init__(daemon=True)
        self.interval_ms = interval_ms
        self.samples: list[dict] = []
        self.gpu_freqs: list[int] = []
        self._stop_evt = threading.Event()   # NOT _stop: Thread._stop is internal
        self.proc = None

    def run(self):
        self.proc = subprocess.Popen(
            ["tegrastats", "--interval", str(self.interval_ms)],
            stdout=subprocess.PIPE, text=True, bufsize=1,
        )
        for line in self.proc.stdout:
            if self._stop_evt.is_set():
                break
            s = {}
            for key, rx in (("vdd_in", VDD_IN_RE), ("cgc", VDD_CGC_RE), ("soc", VDD_SOC_RE),
                            ("gpu_t", GPU_T_RE), ("cpu_t", CPU_T_RE), ("tj_t", TJ_T_RE),
                            ("gr3d", GR3D_RE)):
                m = rx.search(line)
                if m:
                    s[key] = float(m.group(1))
            if s:
                self.samples.append(s)
            for d in GPU_FREQ_PATHS:
                try:
                    self.gpu_freqs.append(int((d / "cur_freq").read_text().strip()))
                except OSError:
                    pass

    def stop(self):
        self._stop_evt.set()
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.join(timeout=3)


def agg(samples, key, fn):
    vals = [s[key] for s in samples if key in s]
    return round(fn(vals), 1) if vals else ""


def measure_idle(seconds: float = 6.0) -> float:
    s = TegraSampler(); s.start(); time.sleep(seconds); s.stop()
    vals = [x["vdd_in"] for x in s.samples if "vdd_in" in x]
    return round(sum(vals) / len(vals), 1) if vals else 0.0


def profile_one(path: str, imgsz: int, tiles: list, duration: float, idle_mw: float) -> dict:
    model = YOLO(path, task="obb")
    for i in range(10):                       # warmup, not profiled
        model.predict(tiles[i % len(tiles)], imgsz=imgsz, device=0, verbose=False)

    sampler = TegraSampler()
    sampler.start()
    time.sleep(0.5)                           # let the first samples land

    iters, t_end = 0, time.perf_counter() + duration
    while time.perf_counter() < t_end:        # sustained load, sampler running
        model.predict(tiles[iters % len(tiles)], imgsz=imgsz, device=0, verbose=False)
        iters += 1
    sampler.stop()

    s = sampler.samples
    freqs = [f // 1_000_000 for f in sampler.gpu_freqs] or [0]
    peak_gpu = agg(s, "gpu_t", max)
    peak_tj = agg(s, "tj_t", max)
    fmin, fmax = min(freqs), max(freqs)

    evidence = []
    if fmax and fmin < fmax:
        evidence.append(f"gpu freq dropped {fmax}->{fmin} MHz")
    if peak_gpu != "" and peak_gpu >= 99:
        evidence.append(f"gpu {peak_gpu}C >= 99C passive trip")
    if peak_tj != "" and peak_tj >= 95:
        evidence.append(f"tj {peak_tj}C >= 95C active trip")
    avg_p = agg(s, "vdd_in", lambda v: sum(v) / len(v))

    return {
        "iterations": iters, "samples": len(s),
        "avg_power_mw": avg_p, "peak_power_mw": agg(s, "vdd_in", max),
        "idle_power_mw": idle_mw,
        "power_over_idle_mw": round(avg_p - idle_mw, 1) if avg_p != "" else "",
        "avg_cpu_gpu_cv_mw": agg(s, "cgc", lambda v: sum(v) / len(v)),
        "avg_soc_mw": agg(s, "soc", lambda v: sum(v) / len(v)),
        "peak_gpu_temp_c": peak_gpu, "peak_cpu_temp_c": agg(s, "cpu_t", max),
        "peak_tj_temp_c": peak_tj,
        "gr3d_mean_pct": agg(s, "gr3d", lambda v: sum(v) / len(v)),
        "gr3d_max_pct": agg(s, "gr3d", max),
        "gpu_freq_min_mhz": fmin, "gpu_freq_max_mhz": fmax,
        "throttled": bool(evidence),
        "throttle_evidence": "; ".join(evidence),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/models_obb.yaml")
    ap.add_argument("--tiles-dir", default="data/DOTAv1-tiles/images/val")
    ap.add_argument("--n-tiles", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--formats", default="pytorch,onnx,trt-fp32,trt-fp16,trt-int8")
    ap.add_argument("--out", default="results/power_results.csv")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    imgsz = cfg["imgsz"]
    paths = sorted(Path(args.tiles_dir).glob("*.jpg"))
    rng = np.random.default_rng(args.seed)
    tiles = [cv2.imread(str(paths[i])) for i in rng.choice(len(paths), args.n_tiles, replace=False)]

    def read(p):
        try:
            return Path(p).read_text().strip()
        except OSError:
            return "unknown"
    cpu_gov = read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    clocks = {"nvpmodel": "MAXN",
              "jetson_clocks": "applied" if cpu_gov == "performance"
              else "NOT applied (DVFS active)"}

    print("measuring idle baseline (6s, no inference) ...")
    idle = measure_idle()
    print(f"  idle VDD_IN = {idle} mW")

    out = Path(args.out)
    done = set()
    if out.exists():
        with open(out) as f:
            done = {(r["model"], r["format"]) for r in csv.DictReader(f) if r["status"] == "ok"}

    for m in cfg["models"]:
        arts = {"pytorch": (m["weights"], "fp32"), "onnx": (m["onnx"], "fp32"),
                "trt-fp32": (m["engines"]["fp32"], "fp32"),
                "trt-fp16": (m["engines"]["fp16"], "fp16"),
                "trt-int8": (m["engines"]["int8"], "int8")}
        for fmt in args.formats.split(","):
            path, prec = arts[fmt]
            if (m["name"], fmt) in done:
                print(f"[{m['name']}] {fmt}: already profiled, skipping")
                continue
            row = {"model": m["name"], "format": fmt, "precision": prec,
                   "artifact": path, "duration_s": args.duration, **clocks}
            try:
                print(f"[{m['name']}] {fmt}: {args.duration}s sustained ...")
                row.update(profile_one(path, imgsz, tiles, args.duration, idle))
                row["status"] = "ok"
                print(f"[{m['name']}] {fmt}: {row['avg_power_mw']} mW avg "
                      f"(+{row['power_over_idle_mw']} over idle), GPU {row['gr3d_mean_pct']}% util, "
                      f"peak tj {row['peak_tj_temp_c']}C, throttled={row['throttled']}")
            except Exception as e:  # noqa: BLE001
                row["status"], row["error"] = "FAILED", repr(e)[:300]
                print(f"[{m['name']}] {fmt}: FAILED {e}")

            out.parent.mkdir(parents=True, exist_ok=True)
            new = not out.exists()
            with open(out, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                if new:
                    w.writeheader()
                w.writerow({k: row.get(k, "") for k in FIELDS})

    print(f"\nrows appended to {out}")


if __name__ == "__main__":
    main()
