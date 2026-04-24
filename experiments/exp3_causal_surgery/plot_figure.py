#!/usr/bin/env python3
"""Create a compact paper figure for Experiment 3.

The figure focuses on the main causal contrast at the early residual state
where surgery has the strongest interpretable effect.  It uses episode medians
with interquartile ranges so the visual summary is not dominated by a few
catastrophic outliers.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mpl-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


CONTROL_ORDER = [
    "low_task",
    "high_iso_low_task",
    "high_variance_low_task",
    "random_Q",
    "high_task",
]

CONTROL_LABELS = {
    "low_task": "low task\nleverage",
    "high_iso_low_task": "iso-only\nleverage",
    "high_variance_low_task": "variance-only\nlow task",
    "random_Q": "random\nQ subset",
    "high_task": "high task\nleverage",
}


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def quantile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def summarize(vals: Iterable[float]) -> Tuple[float, float, float]:
    vv = sorted(max(float(v), 1e-8) for v in vals)
    return median(vv), quantile(vv, 0.25), quantile(vv, 0.75)


def make_plot(records_csv: Path, out_path: Path, state_idx: int) -> None:
    rows = read_rows(records_csv)
    colors = {1: "#4C78A8", 2: "#D55E00", 4: "#333333"}
    offsets = {1: -0.18, 2: 0.0, 4: 0.18}

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, ax = plt.subplots(figsize=(5.6, 2.9), constrained_layout=True)

    for k in (1, 2, 4):
        xs: List[float] = []
        ys: List[float] = []
        yerr_low: List[float] = []
        yerr_high: List[float] = []
        for i, control in enumerate(CONTROL_ORDER):
            vals = [
                float(r["surg_E_task_damage"])
                for r in rows
                if r["control"] == control
                and int(float(r["k_remove"])) == k
                and int(float(r["state_idx"])) == state_idx
            ]
            med, q1, q3 = summarize(vals)
            xs.append(i + offsets[k])
            ys.append(med)
            yerr_low.append(max(med - q1, 1e-8))
            yerr_high.append(max(q3 - med, 1e-8))

        ax.errorbar(
            xs,
            ys,
            yerr=[yerr_low, yerr_high],
            fmt="o",
            color=colors[k],
            markerfacecolor=colors[k],
            markeredgecolor="white",
            markersize=6,
            linewidth=1.1,
            capsize=2.5,
            label=rf"$k={k}$ removed",
            zorder=3,
        )

    ax.axvspan(3.58, 4.42, color="#D55E00", alpha=0.08, zorder=0)
    ax.set_yscale("log")
    ax.set_ylim(7e-4, 1.5e2)
    ax.set_xlim(-0.55, len(CONTROL_ORDER) - 0.45)
    ax.set_xticks(range(len(CONTROL_ORDER)))
    ax.set_xticklabels([CONTROL_LABELS[c] for c in CONTROL_ORDER])
    ax.set_ylabel("task response damage")
    ax.set_title(r"Early surgery at residual state $s=1$", loc="left")
    ax.yaxis.grid(True, which="major", color="#dddddd", linewidth=0.7)
    ax.yaxis.grid(True, which="minor", color="#eeeeee", linewidth=0.45)
    ax.xaxis.grid(False)
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        4,
        9e1,
        "task-Galerkin\ndirections",
        ha="center",
        va="top",
        fontsize=8,
        color="#8A3B00",
    )

    fig.savefig(out_path, dpi=240)
    fig.savefig(out_path.with_suffix(".pdf"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--records-csv",
        type=Path,
        default=Path(__file__).resolve().parent / "results_L8_final" / "records.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent
        / "results_L8_final"
        / "experiment_3_key_surgery.png",
    )
    parser.add_argument("--state-idx", type=int, default=1)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    make_plot(args.records_csv, args.out, args.state_idx)
    print(args.out)
    print(args.out.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
