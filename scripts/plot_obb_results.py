#!/usr/bin/env python3
"""Stage 7: generate the three README charts from the result CSVs.

Charts are built from results/*.csv only — nothing here is hand-entered.

Palette: the dataviz reference palette, used unmodified.
  slot1 #2a78d6 blue, slot2 #eb6834 orange, slot3 #1baf7a aqua,
  slot4 #eda100 yellow, slot5 #e87ba4 magenta
Grouped bars use the *adjacent* pairlist, where all eight slots are documented
as passing. The Pareto chart is a scatter (all-pairs pairlist), where only the
first three slots validate — so rather than colour 20 points by a 4- or 5-value
category, it draws a single series and highlights the frontier, which is the
point of that chart anyway.

Usage:
    python scripts/plot_obb_results.py
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

S1, S2, S3, S4, S5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
SURFACE = "#fcfcfb"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8984"
GRID = "#e5e4e0"
FORMAT_ORDER = ["pytorch", "onnx", "trt-fp32", "trt-fp16", "trt-int8"]
FORMAT_LABEL = {"pytorch": "PyTorch FP32", "onnx": "ONNX RT FP32",
                "trt-fp32": "TRT FP32", "trt-fp16": "TRT FP16", "trt-int8": "TRT INT8"}


def style(ax):
    """Recessive grid and axes; ink tokens for text, never series colour."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)


# Stage 3 measured PyTorch / ONNX / TRT-FP32 agreeing to within 0.066% of their
# mean mAP50. Treating a 0.00005 mAP50 difference as "more accurate" would put
# noise on the frontier, so dominance uses a relative tolerance comfortably
# above that measured spread: a strictly faster point dominates unless it is
# more than TOL worse in mAP50.
PARETO_TOL = 0.001          # 0.1% of mAP50


def pareto_front(points, tol: float = PARETO_TOL):
    """Indices on the speed/accuracy frontier, with a noise tolerance on accuracy."""
    keep = []
    for i, (x_i, y_i) in enumerate(points):
        dominated = any(
            x_j < x_i and y_j >= y_i - tol * y_i
            for j, (x_j, y_j) in enumerate(points) if j != i
        )
        if not dominated:
            keep.append(i)
    return keep


def plot_pareto(acc, bench, out: Path):
    m = bench.merge(acc, on=["model", "format", "precision"], suffixes=("", "_a"))
    fig, ax = plt.subplots(figsize=(9.4, 6.2), facecolor=SURFACE)
    style(ax)

    pts = list(zip(m.total_ms_p50, m.mAP50))
    front = set(pareto_front(pts))

    off = [(i not in front) for i in range(len(m))]
    ax.scatter(m.total_ms_p50[off], m.mAP50[off], s=46, color=S1, alpha=0.30,
               edgecolors=SURFACE, linewidths=2, zorder=3, label="dominated")
    on = [(i in front) for i in range(len(m))]
    ax.scatter(m.total_ms_p50[on], m.mAP50[on], s=132, color=S1,
               edgecolors=SURFACE, linewidths=2, zorder=5, label="Pareto frontier")

    fr = m[on].sort_values("total_ms_p50")
    ax.plot(fr.total_ms_p50, fr.mAP50, color=S1, linewidth=2, alpha=0.55, zorder=4)
    for r in fr.itertuples():                      # selective direct labels: frontier only
        ax.annotate(f"{r.model.replace('-obb','')}\n{FORMAT_LABEL[r.format]}",
                    (r.total_ms_p50, r.mAP50), textcoords="offset points",
                    xytext=(9, -4), fontsize=8, color=INK, linespacing=1.3)

    ax.set_xlabel("End-to-end latency, p50 (ms)  —  lower is better", color=INK2, fontsize=10)
    ax.set_ylabel("OBB mAP50  —  higher is better", color=INK2, fontsize=10)
    ax.set_title("Accuracy vs latency: every point on the frontier is TensorRT FP16",
                 color=INK, fontsize=12.5, pad=14, loc="left")
    ax.annotate("Frontier uses a 0.1% mAP50 tolerance — Stage 3 measured the three FP32 "
                "paths agreeing to within 0.066%,\nso smaller differences are noise, not accuracy.",
                xy=(0, -0.145), xycoords="axes fraction", fontsize=8.3, color=INK3,
                linespacing=1.45)
    ax.annotate("better", xy=(0.035, 0.955), xycoords="axes fraction", color=INK3,
                fontsize=9, ha="left")
    ax.annotate("", xy=(0.02, 0.99), xytext=(0.09, 0.92), xycoords="axes fraction",
                arrowprops={"arrowstyle": "->", "color": INK3, "lw": 1.4})
    leg = ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK2)
    leg.set_title(f"{len(m)} configurations", prop={"size": 9})
    leg.get_title().set_color(INK3)
    ax.margins(x=0.16)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {out}")


def plot_accuracy_drop(acc, out: Path):
    p = acc.pivot(index="model", columns="format", values="mAP50")
    models = list(p.index)
    d16 = (p["trt-fp16"] - p["trt-fp32"]) / p["trt-fp32"] * 100
    d8 = (p["trt-int8"] - p["trt-fp32"]) / p["trt-fp32"] * 100

    fig, ax = plt.subplots(figsize=(9, 5.0), facecolor=SURFACE)
    style(ax)
    x = range(len(models))
    w = 0.36
    gap = 0.012                                    # surface gap between adjacent fills
    ax.bar([i - w/2 - gap for i in x], d16, w, color=S1, label="TRT FP16", zorder=3)
    ax.bar([i + w/2 + gap for i in x], d8, w, color=S2, label="TRT INT8", zorder=3)
    ax.axhline(0, color=INK3, linewidth=1.2, zorder=4)

    for i, (a, b) in enumerate(zip(d16, d8)):      # direct labels (2 series, so label both)
        ax.annotate(f"{a:+.2f}%", (i - w/2 - gap, a), ha="center", va="top",
                    xytext=(0, -5), textcoords="offset points", fontsize=8.5, color=INK)
        ax.annotate(f"{b:+.2f}%", (i + w/2 + gap, b), ha="center", va="top",
                    xytext=(0, -5), textcoords="offset points", fontsize=8.5, color=INK)

    ax.set_xticks(list(x))
    ax.set_xticklabels([m.replace("-obb", "") for m in models], color=INK2)
    ax.set_ylabel("Change in mAP50 vs TensorRT FP32 (%)", color=INK2, fontsize=10)
    ax.set_title("FP16 is essentially free; INT8 costs real rotated-box accuracy",
                 color=INK, fontsize=12.5, pad=14, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower left", labelcolor=INK2)
    ax.set_ylim(min(d8.min(), d16.min()) * 1.35, max(0.6, d16.max() * 2))
    fig.tight_layout()
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {out}")


def plot_power(power, out: Path):
    p = power.pivot(index="model", columns="format", values="power_over_idle_mw") / 1000.0
    models = list(p.index)
    colors = [S1, S2, S3, S4, S5]

    fig, ax = plt.subplots(figsize=(9.6, 5.2), facecolor=SURFACE)
    style(ax)
    n = len(FORMAT_ORDER)
    w = 0.16
    gap = 0.008
    for k, fmt in enumerate(FORMAT_ORDER):
        off = (k - (n - 1) / 2) * (w + gap)
        ax.bar([i + off for i in range(len(models))], p[fmt], w,
               color=colors[k], label=FORMAT_LABEL[fmt], zorder=3)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.replace("-obb", "") for m in models], color=INK2)
    ax.set_ylabel("Power attributable to inference (W over idle)", color=INK2, fontsize=10)
    ax.set_title("Power spans 1.7x across configurations; throughput spans 5.2x",
                 color=INK, fontsize=12.5, pad=14, loc="left")
    ax.legend(frameon=False, fontsize=9, ncol=5, loc="upper center",
              bbox_to_anchor=(0.5, -0.09), labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()
    rd = Path(args.results_dir)
    out = rd / "plots"
    out.mkdir(parents=True, exist_ok=True)

    acc = pd.read_csv(rd / "accuracy_results.csv")
    bench = pd.read_csv(rd / "benchmark_results.csv")
    power = pd.read_csv(rd / "power_results.csv")

    plot_pareto(acc, bench, out / "accuracy_vs_latency_pareto.png")
    plot_accuracy_drop(acc, out / "accuracy_drop_by_precision.png")
    plot_power(power, out / "power_by_configuration.png")


if __name__ == "__main__":
    main()
