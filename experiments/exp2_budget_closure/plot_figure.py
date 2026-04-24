#!/usr/bin/env python3
"""Create a compact paper figure for Experiment 2.

The figure deliberately avoids log-axis clipping and presentation-style
multi-panel decoration.  It shows the main threshold law and the main
qualification in two small panels.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mpl-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def f(row: Dict[str, str], key: str) -> float:
    return float(row[key])


def task_rows(rows: Iterable[Dict[str, str]], suite: str) -> List[Dict[str, str]]:
    return [
        r
        for r in rows
        if r.get("probe_kind") == "task" and r.get("suite_exp") == suite
    ]


def checkpoint_order(name: str) -> int:
    order = {
        "dx3": 3,
        "dx5": 5,
        "dx8": 8,
        "dx10": 10,
        "dx15": 15,
        "L2": 2,
        "L4": 4,
        "L6": 6,
        "L8": 8,
        "L12": 12,
        "H1": 101,
        "H2": 102,
        "H8": 108,
    }
    return order.get(name, 999)


def monotone_decreasing_interpolation(
    points: List[Tuple[float, float]]
) -> Tuple[List[float], List[float]]:
    """Return a nonincreasing isotonic guide through x/y points.

    This is a visual interpolation aid, not a parametric model.  Duplicate
    x-values are averaged, then a pool-adjacent-violators pass enforces the
    expected threshold monotonicity: larger reachable ratio should not require
    larger KRR-rule error.
    """
    grouped: Dict[float, List[float]] = {}
    for x, y in points:
        grouped.setdefault(x, []).append(y)

    xs = sorted(grouped)
    ys = [sum(grouped[x]) / len(grouped[x]) for x in xs]
    weights = [len(grouped[x]) for x in xs]

    blocks: List[Dict[str, float]] = []
    for x, y, w in zip(xs, ys, weights):
        blocks.append({"start": x, "end": x, "avg": y, "weight": float(w)})
        while len(blocks) >= 2 and blocks[-2]["avg"] < blocks[-1]["avg"]:
            b2 = blocks.pop()
            b1 = blocks.pop()
            weight = b1["weight"] + b2["weight"]
            avg = (b1["avg"] * b1["weight"] + b2["avg"] * b2["weight"]) / weight
            blocks.append(
                {
                    "start": b1["start"],
                    "end": b2["end"],
                    "avg": avg,
                    "weight": weight,
                }
            )

    fitted: Dict[float, float] = {}
    for block in blocks:
        for x in xs:
            if block["start"] <= x <= block["end"]:
                fitted[x] = block["avg"]
    return xs, [fitted[x] for x in xs]


def make_plot(summary_csv: Path, out_path: Path) -> None:
    rows = read_rows(summary_csv)
    rows_2b = sorted(
        task_rows(rows, "2b"),
        key=lambda r: (f(r, "suite_smax"), checkpoint_order(r["checkpoint"])),
    )
    rows_2a = sorted(
        task_rows(rows, "2a"),
        key=lambda r: (f(r, "suite_smax"), checkpoint_order(r["checkpoint"])),
    )

    colors = {
        1: "#777777",
        2: "#4C78A8",
        4: "#D55E00",
    }
    markers = {
        "dx3": "o",
        "dx5": "s",
        "dx8": "^",
        "dx10": "D",
        "dx15": "P",
    }

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

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65), constrained_layout=True)

    ax = axes[0]
    threshold_points: List[Tuple[float, float]] = []
    for row in rows_2b:
        smax = int(f(row, "suite_smax"))
        name = row["checkpoint"]
        ratio = f(row, "dim_R_nat_mean") / f(row, "r_eff_T_task_mean")
        err = f(row, "E_TQ_T_mean")
        threshold_points.append((ratio, err))
        ax.scatter(
            ratio,
            err,
            s=48,
            marker=markers.get(name, "o"),
            facecolor=colors[smax],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
    xs_fit, ys_fit = monotone_decreasing_interpolation(threshold_points)
    ax.plot(
        xs_fit,
        ys_fit,
        color="#222222",
        linewidth=1.2,
        alpha=0.45,
        zorder=2,
        label="monotone guide",
    )
    ax.axvline(1.0, color="#333333", linewidth=1.0, linestyle=(0, (4, 3)))
    ax.set_xlim(0.34, 1.03)
    ax.set_ylim(-0.035, 0.84)
    ax.set_xlabel(r"activation reachable ratio $d_{\mathrm{nat}}/r_T$")
    ax.set_ylabel(r"KRR-rule error $E(T_{Q_{\mathrm{nat}}},T)$")
    ax.set_title("A. Activation rank threshold for KRR rule", loc="left")
    ax.yaxis.grid(True, color="#dddddd", linewidth=0.7)
    ax.xaxis.grid(False)
    ax.text(
        0.985,
        0.78,
        "native rank\nmatches task rank",
        ha="right",
        va="top",
        fontsize=7,
        color="#444444",
    )

    ax = axes[1]
    rows_2a_s4 = [r for r in rows_2a if int(f(r, "suite_smax")) == 4]
    labels = ["L2", "L4", "L6", "L8", "L12", "H1", "H2", "H8"]
    x = list(range(len(labels)))
    e_model = {
        r["checkpoint"]: f(r, "E_F_T_mean")
        for r in rows_2a_s4
    }
    e_native = {
        r["checkpoint"]: f(r, "E_TQ_T_mean")
        for r in rows_2a_s4
    }
    ax.plot(
        x,
        [e_model.get(k, float("nan")) for k in labels],
        marker="o",
        color="#333333",
        linewidth=1.4,
        label=r"model vs KRR, $E(F,T)$",
    )
    ax.plot(
        x,
        [e_native.get(k, float("nan")) for k in labels],
        marker="s",
        color="#D55E00",
        linewidth=1.4,
        label=r"native span vs KRR, $E(T_Q,T)$",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(-0.012, 0.215)
    ax.set_ylabel("task error")
    ax.set_title("B. Native span is not model use", loc="left")
    ax.yaxis.grid(True, color="#dddddd", linewidth=0.7)
    ax.xaxis.grid(False)
    ax.legend(frameon=False, loc="upper right")

    handles = [
        Line2D(
            [0],
            [0],
            color="#222222",
            linewidth=1.2,
            alpha=0.45,
            label="monotone guide",
        )
    ]
    handles += [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=colors[s],
            markeredgecolor="white",
            markersize=6,
            label=rf"$s_{{\max}}={s}$",
        )
        for s in (1, 2, 4)
    ]
    axes[0].legend(handles=handles, frameon=False, loc="lower left")

    fig.savefig(out_path, dpi=240)
    fig.savefig(out_path.with_suffix(".pdf"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path(__file__).resolve().parent
        / "results_native_final_suite_both"
        / "aggregate_summary.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent
        / "results_native_final_suite_both"
        / "experiment_2_paper_threshold.png",
    )
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    make_plot(args.summary_csv, args.out)
    print(args.out)
    print(args.out.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
