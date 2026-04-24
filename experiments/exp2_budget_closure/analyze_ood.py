#!/usr/bin/env python3
"""Analyze whether task-Galerkinness predicts OOD/isotropic performance.

This script reads Experiment 2 native-budget outputs and compares task-probe
Galerkin diagnostics against isotropic model error.

Primary question:
    Do models/bases with stronger task-local Galerkin certificates have better
    OOD label-direction operator behavior?

Expected input:
    Results from run_native.py with --probe-kind both, either as:

    1. a final suite root containing aggregate_summary.csv, or
    2. a root containing subdirectories like 2a_smax1/, 2a_smax2/, ...

Example final data generation:

    python -m experiments.exp2_budget_closure.run \
      --suite \
      --only both \
      --episodes 16 \
      --n-build 8 \
      --n-eval 32 \
      --n-tgt 16 \
      --smax-list 1,2,4 \
      --probe-kind both \
      --out-root experiments/exp2_budget_closure/results

Then analyze:

    python -m experiments.exp2_budget_closure.analyze_ood \
      --in-root experiments/exp2_budget_closure/results

Outputs:
    ood_correlation_table.csv
    ood_correlation_report.txt
    galerkinness_vs_ood.png
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    keys: List[str] = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def f(row: Dict[str, object], key: str, default: float = float("nan")) -> float:
    try:
        v = row.get(key, default)
        if v in ("", None):
            return default
        return float(v)  # type: ignore[arg-type]
    except Exception:
        return default


def load_summary_rows(in_root: Path) -> List[Dict[str, object]]:
    """Load aggregate_summary.csv if present, otherwise collect */summary.csv."""
    aggregate = in_root / "aggregate_summary.csv"
    if aggregate.exists():
        return [dict(r) for r in read_csv(aggregate)]

    rows: List[Dict[str, object]] = []
    for sub in sorted(in_root.iterdir()):
        if not sub.is_dir():
            continue
        summary = sub / "summary.csv"
        if not summary.exists():
            continue

        # Infer suite_exp/smax from directory names like 2a_smax1.
        suite_exp = ""
        suite_smax = ""
        name = sub.name
        if "_smax" in name:
            left, right = name.split("_smax", 1)
            suite_exp = left
            suite_smax = right

        for r in read_csv(summary):
            rr: Dict[str, object] = dict(r)
            rr.setdefault("suite_exp", suite_exp)
            rr.setdefault("suite_smax", suite_smax)
            rows.append(rr)

    return rows


# ---------------------------------------------------------------------------
# Pair task and iso rows
# ---------------------------------------------------------------------------

def row_key(row: Dict[str, object]) -> Tuple[str, str, str, str]:
    """Key excluding probe kind."""
    return (
        str(row.get("suite_exp", row.get("exp", ""))),
        str(row.get("suite_smax", row.get("smax_mean", row.get("smax", "")))),
        str(row.get("checkpoint", "")),
        str(row.get("sweep", "")),
    )


def pair_task_iso(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    buckets: Dict[Tuple[str, str, str, str], Dict[str, Dict[str, object]]] = defaultdict(dict)

    for row in rows:
        pk = str(row.get("probe_kind", ""))
        if pk not in ("task", "iso"):
            continue
        buckets[row_key(row)][pk] = row

    paired: List[Dict[str, object]] = []
    for key, d in buckets.items():
        if "task" not in d or "iso" not in d:
            continue

        task = d["task"]
        iso = d["iso"]

        suite_exp, suite_smax, checkpoint, sweep = key
        rt = f(task, "r_eff_T_task_mean")
        dim = f(task, "dim_R_nat_mean")
        ratio = dim / rt if math.isfinite(dim) and math.isfinite(rt) and rt > 0 else float("nan")

        rec: Dict[str, object] = {
            "suite_exp": suite_exp,
            "suite_smax": suite_smax,
            "checkpoint": checkpoint,
            "sweep": sweep,
            "d_x": f(task, "d_x_mean"),
            "n_layers": f(task, "n_layers_mean"),
            "n_heads": f(task, "n_heads_mean"),
            "r_eff_T_task": rt,
            "dim_R_nat": dim,
            "dim_over_rT": ratio,

            # Task-probe Galerkin diagnostics.
            "task_E_F_T": f(task, "E_F_T_mean"),
            "task_E_TQ_T": f(task, "E_TQ_T_mean"),
            "task_E_F_TQ": f(task, "E_F_TQ_mean"),
            "task_X_use": f(task, "X_use_mean"),
            "task_X_sub": f(task, "X_sub_mean"),
            "task_max_closure": f(task, "max_closure_mean"),
            "task_mse_ratio": f(task, "mse_ratio_mean"),

            # OOD / isotropic diagnostics.
            "iso_E_F_T": f(iso, "E_F_T_mean"),
            "iso_E_TQ_T": f(iso, "E_TQ_T_mean"),
            "iso_E_F_TQ": f(iso, "E_F_TQ_mean"),
            "iso_X_use": f(iso, "X_use_mean"),
            "iso_X_sub": f(iso, "X_sub_mean"),
            "iso_mse_ratio": f(iso, "mse_ratio_mean"),

            # Useful ratios.
            "ood_gap_E_F_T": f(iso, "E_F_T_mean") / max(f(task, "E_F_T_mean"), 1e-12),
            "ood_gap_E_TQ_T": f(iso, "E_TQ_T_mean") / max(f(task, "E_TQ_T_mean"), 1e-12),
        }
        paired.append(rec)

    return paired


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def finite_xy(rows: List[Dict[str, object]], xkey: str, ykey: str) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for r in rows:
        x = f(r, xkey)
        y = f(r, ykey)
        if math.isfinite(x) and math.isfinite(y) and x > 0 and y > 0:
            xs.append(x)
            ys.append(y)
    return np.array(xs, dtype=float), np.array(ys, dtype=float)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    if np.std(x) <= 0 or np.std(y) <= 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rankdata(a: np.ndarray) -> np.ndarray:
    """Simple average-rank implementation without scipy."""
    order = np.argsort(a)
    ranks = np.empty_like(order, dtype=float)
    sorted_a = a[order]
    n = len(a)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_a[j] == sorted_a[i]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    return pearson(rankdata(x), rankdata(y))


def correlation_table(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    comparisons = [
        ("task_E_TQ_T", "iso_E_F_T", "subspace adequacy vs OOD model error"),
        ("task_E_F_TQ", "iso_E_F_T", "model-use residual vs OOD model error"),
        ("task_E_F_T", "iso_E_F_T", "task model error vs OOD model error"),
        ("dim_over_rT", "iso_E_F_T", "rank ratio vs OOD model error"),
        ("task_max_closure", "iso_E_F_T", "closure defect vs OOD model error"),
        ("task_E_TQ_T", "iso_E_TQ_T", "task Galerkin adequacy vs iso Galerkin adequacy"),
        ("task_E_F_TQ", "iso_E_F_TQ", "task model-use vs iso model-use"),
    ]

    out: List[Dict[str, object]] = []
    groups = {
        "all": rows,
        "2a": [r for r in rows if str(r.get("suite_exp")) == "2a"],
        "2b": [r for r in rows if str(r.get("suite_exp")) == "2b"],
    }

    for group_name, group_rows in groups.items():
        for xkey, ykey, desc in comparisons:
            x, y = finite_xy(group_rows, xkey, ykey)

            # Log-log correlations are more meaningful for error-like quantities.
            log_x = np.log10(np.maximum(x, 1e-12))
            log_y = np.log10(np.maximum(y, 1e-12))

            out.append({
                "group": group_name,
                "x": xkey,
                "y": ykey,
                "description": desc,
                "n": len(x),
                "pearson_loglog": pearson(log_x, log_y),
                "spearman": spearman(x, y),
            })

    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def color_for_smax(s: str) -> str:
    try:
        i = int(float(s))
    except Exception:
        i = -1
    return {
        1: "#1f77b4",
        2: "#ff7f0e",
        4: "#2ca02c",
        8: "#d62728",
    }.get(i, "#7f7f7f")


def marker_for_exp(exp: str) -> str:
    return {"2a": "o", "2b": "s"}.get(exp, "o")


def scatter_panel(ax, rows: List[Dict[str, object]], xkey: str, ykey: str, title: str) -> None:
    for r in rows:
        x = f(r, xkey)
        y = f(r, ykey)
        if not (math.isfinite(x) and math.isfinite(y) and x > 0 and y > 0):
            continue

        exp = str(r.get("suite_exp", ""))
        smax = str(r.get("suite_smax", ""))
        ckpt = str(r.get("checkpoint", ""))

        ax.scatter(
            x,
            y,
            s=60,
            marker=marker_for_exp(exp),
            color=color_for_smax(smax),
            alpha=0.85,
        )
        ax.text(x * 1.03, y * 1.03, ckpt, fontsize=7)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xkey)
    ax.set_ylabel(ykey)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)


def plot_ood(rows: List[Dict[str, object]], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))

    scatter_panel(
        axes[0, 0],
        rows,
        "task_E_TQ_T",
        "iso_E_F_T",
        "Does Galerkin subspace adequacy predict OOD model error?",
    )
    scatter_panel(
        axes[0, 1],
        rows,
        "task_E_F_TQ",
        "iso_E_F_T",
        "Does task model-use residual predict OOD model error?",
    )
    scatter_panel(
        axes[0, 2],
        rows,
        "task_E_F_T",
        "iso_E_F_T",
        "Does task operator error predict OOD model error?",
    )
    scatter_panel(
        axes[1, 0],
        rows,
        "dim_over_rT",
        "iso_E_F_T",
        "Does reachable-rank ratio predict OOD model error?",
    )
    scatter_panel(
        axes[1, 1],
        rows,
        "task_E_TQ_T",
        "iso_E_TQ_T",
        "Task vs isotropic Galerkin adequacy",
    )
    scatter_panel(
        axes[1, 2],
        rows,
        "task_E_F_TQ",
        "iso_E_F_TQ",
        "Task vs isotropic model-use residual",
    )

    # Add a small visual legend manually.
    handles = []
    labels = []
    for s in ["1", "2", "4"]:
        h = axes[0, 0].scatter([], [], color=color_for_smax(s), marker="o", s=60)
        handles.append(h)
        labels.append(f"smax={s}")
    for exp, marker in [("2a", "o"), ("2b", "s")]:
        h = axes[0, 0].scatter([], [], color="black", marker=marker, s=60)
        handles.append(h)
        labels.append(exp)

    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=9)
    fig.suptitle("Does task-local Galerkinness predict OOD/isotropic operator behavior?", fontsize=14)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_report(path: Path, corr_rows: List[Dict[str, object]], paired: List[Dict[str, object]]) -> None:
    lines: List[str] = []
    lines.append("Galerkinness vs OOD/isotropic performance analysis")
    lines.append("")
    lines.append(f"paired task/iso rows: {len(paired)}")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- x variables are task-probe diagnostics.")
    lines.append("- y variables are isotropic/OOD label-direction diagnostics.")
    lines.append("- Strong positive correlation means worse task Galerkinness predicts worse OOD behavior.")
    lines.append("- Weak correlation means the Galerkin certificate and OOD robustness can come apart.")
    lines.append("")
    lines.append("=== Correlations ===")
    lines.append(f"{'group':6s} {'n':>4s} {'pearson_loglog':>15s} {'spearman':>10s}  comparison")
    for r in corr_rows:
        lines.append(
            f"{str(r['group']):6s} {int(r['n']):4d} "
            f"{float(r['pearson_loglog']):15.4f} "
            f"{float(r['spearman']):10.4f}  "
            f"{r['x']} -> {r['y']}  ({r['description']})"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze task-Galerkinness vs OOD/isotropic performance.")
    p.add_argument(
        "--in-root",
        required=True,
        help="Root containing aggregate_summary.csv or subdirectories with summary.csv files.",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to <in-root>/ood_analysis.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_root = Path(args.in_root)
    out_dir = Path(args.out_dir) if args.out_dir else in_root / "ood_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_summary_rows(in_root)
    if not rows:
        raise FileNotFoundError(f"No summary rows found under {in_root}")

    paired = pair_task_iso(rows)
    if not paired:
        raise RuntimeError(
            "No paired task/iso rows found. Re-run run_native.py with --probe-kind both."
        )

    corr_rows = correlation_table(paired)

    write_csv(out_dir / "ood_paired_table.csv", paired)
    write_csv(out_dir / "ood_correlation_table.csv", corr_rows)
    plot_ood(paired, out_dir / "galerkinness_vs_ood.png")
    write_report(out_dir / "ood_correlation_report.txt", corr_rows, paired)

    print("Wrote:")
    print(" ", out_dir / "ood_paired_table.csv")
    print(" ", out_dir / "ood_correlation_table.csv")
    print(" ", out_dir / "galerkinness_vs_ood.png")
    print(" ", out_dir / "ood_correlation_report.txt")


if __name__ == "__main__":
    main()