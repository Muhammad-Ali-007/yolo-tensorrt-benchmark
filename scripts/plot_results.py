#!/usr/bin/env python3
"""Generates the charts referenced in the README from the CSVs the other
scripts produce. Run after benchmark.py / evaluate_map.py / power_profile.py.

Usage:
    python scripts/plot_results.py
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_latency_by_precision(bench_csv: str, out_dir: Path) -> None:
    df = pd.read_csv(bench_csv)
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, group in df.groupby("model"):
        group = group.set_index("precision").reindex(["fp32", "fp16", "int8"])
        ax.plot(group.index, group["gpu_latency_ms_p50"], marker="o", label=model)
    ax.set_xlabel("Precision")
    ax.set_ylabel("GPU latency, p50 (ms)")
    ax.set_title("Latency by precision")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "latency_by_precision.png", dpi=150)
    plt.close(fig)


def plot_accuracy_vs_speed(bench_csv: str, acc_csv: str, out_dir: Path) -> None:
    bench = pd.read_csv(bench_csv)
    acc = pd.read_csv(acc_csv)
    merged = bench.merge(acc, on=["model", "precision"])

    fig, ax = plt.subplots(figsize=(8, 5))
    for (model, precision), row in merged.set_index(["model", "precision"]).iterrows():
        ax.scatter(row["gpu_latency_ms_p50"], row["mAP@0.5:0.95"])
        ax.annotate(f"{model}-{precision}",
                    (row["gpu_latency_ms_p50"], row["mAP@0.5:0.95"]),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("GPU latency, p50 (ms)")
    ax.set_ylabel("mAP@0.5:0.95")
    ax.set_title("Accuracy vs. speed tradeoff")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_vs_speed.png", dpi=150)
    plt.close(fig)


def plot_power(power_csv: str, out_dir: Path) -> None:
    df = pd.read_csv(power_csv)
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.35
    models = df["model"].unique()
    x = range(len(models))
    for i, precision in enumerate(sorted(df["precision"].unique())):
        sub = df[df["precision"] == precision].set_index("model").reindex(models)
        ax.bar([xi + i * width for xi in x], sub["avg_power_mw"], width, label=precision)
    ax.set_xticks([xi + width / 2 for xi in x])
    ax.set_xticklabels(models, rotation=20)
    ax.set_ylabel("Avg power draw (mW)")
    ax.set_title("Power draw by model / precision")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "power_by_model.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = results_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    bench_csv = results_dir / "benchmark_results.csv"
    acc_csv = results_dir / "accuracy_results.csv"
    power_csv = results_dir / "power_results.csv"

    if bench_csv.exists():
        plot_latency_by_precision(str(bench_csv), out_dir)
        print(f"wrote {out_dir / 'latency_by_precision.png'}")
    else:
        print(f"skip: {bench_csv} not found (run benchmark.py first)")

    if bench_csv.exists() and acc_csv.exists():
        plot_accuracy_vs_speed(str(bench_csv), str(acc_csv), out_dir)
        print(f"wrote {out_dir / 'accuracy_vs_speed.png'}")
    else:
        print(f"skip: need both {bench_csv} and {acc_csv} (run benchmark.py and evaluate_map.py)")

    if power_csv.exists():
        plot_power(str(power_csv), out_dir)
        print(f"wrote {out_dir / 'power_by_model.png'}")
    else:
        print(f"skip: {power_csv} not found (run power_profile.py first, optional)")


if __name__ == "__main__":
    main()
