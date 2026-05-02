#!/usr/bin/env python3
"""Plot the whole-subspace complement ablation for Experiment 3."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mpl-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ORDER = ["base", "keep_Q", "remove_Q"]
LABELS = {
    "base": "base",
    "keep_Q": r"keep $Q$",
    "remove_Q": r"remove $Q$",
}
COLORS = {
    "base": "#666666",
    "keep_Q": "#4C78A8",
    "remove_Q": "#D55E00",
}


def read_summary(path: Path) -> Dict[str, Dict[str, float]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    out: Dict[str, Dict[str, float]] = {}
    for row in rows:
        name = row["name"]
        out[name] = {k: float(v) for k, v in row.items() if k != "name"}
    return out


def panel(ax: plt.Axes, summary: Dict[str, Dict[str, float]], key: str, title: str) -> None:
    xs = list(range(len(ORDER)))
    raw_vals: List[float] = [summary[name][key] for name in ORDER]
    vals: List[float] = [max(val, 1e-5) for val in raw_vals]

    ax.bar(
        xs,
        vals,
        color=[COLORS[name] for name in ORDER],
        edgecolor="white",
        linewidth=0.8,
        width=0.68,
    )
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([LABELS[name] for name in ORDER])
    ax.set_title(title, loc="left")
    ax.yaxis.grid(True, which="major", color="#dddddd", linewidth=0.7)
    ax.yaxis.grid(True, which="minor", color="#eeeeee", linewidth=0.45)
    ax.set_axisbelow(True)
    for x, raw_val, val in zip(xs, raw_vals, vals):
        label = "0" if raw_val == 0.0 else f"{raw_val:.3g}"
        ax.text(x, val * 1.18, label, ha="center", va="bottom", fontsize=8)


def make_plot(summary_csv: Path, out_path: Path) -> None:
    summary = read_summary(summary_csv)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.55), constrained_layout=True)
    panel(axes[0], summary, "E_F_T_mean", r"$E_{\mathrm{task}}(F,T)$")
    panel(axes[1], summary, "point_err_mean", "pointwise error vs. KRR")
    axes[0].set_ylabel("relative error")
    fig.suptitle(r"Whole-subspace intervention at residual state $s=1$", x=0.01, ha="left", fontsize=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240)
    fig.savefig(out_path.with_suffix(".pdf"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path(__file__).resolve().parent / "results_complement_ablation" / "summary.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent
        / "results_complement_ablation"
        / "complement_ablation_bars.png",
    )
    args = parser.parse_args()
    make_plot(args.summary_csv, args.out)
    print(args.out)
    print(args.out.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
