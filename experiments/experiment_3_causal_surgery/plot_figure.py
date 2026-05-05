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
    "high_iso_low_task": "isotropic\ncontrol",
    "high_variance_low_task": "activation\ncontrol",
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


def make_plot(
    records_csv: Path,
    out_path: Path,
    state_idx: int,
    metric: str = "surg_E_task_damage",
    ylabel: str = "task response damage",
    ylim: Tuple[float, float] = (7e-4, 1.5e2),
) -> None:
    rows = read_rows(records_csv)
    colors = {1: "#4477AA", 2: "#CC6677", 4: "#222222"}
    offsets = {1: -0.17, 2: 0.0, 4: 0.17}

    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "legend.handlelength": 1.0,
            "legend.handletextpad": 0.45,
        }
    )

    fig, ax = plt.subplots(figsize=(5.85, 2.8), constrained_layout=True)

    ax.axvspan(3.55, 4.45, color="#CC6677", alpha=0.055, zorder=0)
    ax.axvline(3.5, color="#bbbbbb", linewidth=0.8, linestyle=(0, (2.0, 2.0)), zorder=1)

    for k in (1, 2, 4):
        xs: List[float] = []
        ys: List[float] = []
        yerr_low: List[float] = []
        yerr_high: List[float] = []
        for i, control in enumerate(CONTROL_ORDER):
            vals = [
                float(r[metric])
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
            markeredgewidth=0.7,
            markersize=5.2,
            linewidth=1.0,
            elinewidth=1.15,
            capsize=2.2,
            capthick=1.0,
            label=rf"$k={k}$ removed",
            zorder=3,
        )

    ax.set_yscale("log")
    ax.set_ylim(*ylim)
    ax.set_xlim(-0.55, len(CONTROL_ORDER) - 0.45)
    ax.set_xticks(range(len(CONTROL_ORDER)))
    ax.set_xticklabels([CONTROL_LABELS[c] for c in CONTROL_ORDER])
    ax.set_ylabel(f"{ylabel}\nmedian and IQR")
    ax.set_title(r"Direction-removal response damage at residual state $s=1$", loc="left", pad=7)
    ax.yaxis.grid(True, which="major", color="#d8d8d8", linewidth=0.65)
    ax.yaxis.grid(True, which="minor", color="#eeeeee", linewidth=0.35)
    ax.xaxis.grid(False)
    ax.tick_params(axis="both", width=0.8, length=3.5)
    ax.tick_params(axis="y", which="minor", length=2.0)
    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        ncol=3,
        borderaxespad=0.0,
        columnspacing=1.0,
    )
    fig.savefig(out_path, dpi=240)
    fig.savefig(out_path.with_suffix(".pdf"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--records-csv",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "records.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent
        / "results"
        / "experiment_3_key_surgery.png",
    )
    parser.add_argument("--state-idx", type=int, default=1)
    parser.add_argument(
        "--metric",
        choices=("task_damage", "point_error"),
        default="task_damage",
        help="Plot task finite-difference response damage or relative KRR prediction error.",
    )
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.metric == "point_error":
        make_plot(
            args.records_csv,
            args.out,
            args.state_idx,
            metric="surg_pointwise_F_T",
            ylabel="relative KRR pred. error",
            ylim=(3e-3, 1e4),
        )
    else:
        make_plot(args.records_csv, args.out, args.state_idx)
    print(args.out)
    print(args.out.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
