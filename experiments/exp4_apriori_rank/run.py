#!/usr/bin/env python3
"""Experiment 4: a priori rank and capacity closure.

This experiment supplies the task-side anchor for the capture-and-refine
account. For the linear-teacher evaluation task used in Experiments 1--3, the
label covariance is C_G = A_G = K_G + sigma^2 I. In this case the best
rank-r Galerkin residual has the closed form

    delta_r(G)^2 = sum_{j>r} s_j(G)^2 / sum_j s_j(G)^2,

where s_j(G) are the singular values of K_t,G A_G^{-1/2}.

The script:
  1. samples evaluation geometries before looking at model activations,
  2. computes r_epsilon(G; C_G) and r_{epsilon, alpha},
  3. joins those task ranks with the existing Experiment 2 causal-native
     summaries, and
  4. writes figures and CSVs for the paper.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SUPPORT_DIR = REPO_ROOT / "experiments" / "shared"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SUPPORT_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mpl-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import sample_batch_eigenvalues  # noqa: E402
from support import flat_eigenvalues  # noqa: E402

FLOOR = 1e-12


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260425)
    parser.add_argument("--episodes", type=int, default=1024)
    parser.add_argument("--dx-list", default="3,5,8,10,15")
    parser.add_argument("--n-ctx", type=int, default=47)
    parser.add_argument("--n-tgt", type=int, default=16)
    parser.add_argument("--sigma2", type=float, default=0.1)
    parser.add_argument("--eps-list", default="0.2,0.1,0.05,0.02,0.01,0.005,0.001")
    parser.add_argument(
        "--target-eps",
        type=float,
        default=0.01,
        help="Accuracy used for the headline capacity comparison.",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--sv-tau", type=float, default=1e-3)
    parser.add_argument(
        "--target-summary",
        default=str(
            REPO_ROOT
            / "experiments"
            / "exp2_validated_rank"
            / "results_2b_causal_tau005_r16_nbuild16"
            / "summary.csv"
        ),
        help="Existing Experiment 2 target-rank summary to join against.",
    )
    parser.add_argument(
        "--arch-summary",
        default=str(
            REPO_ROOT
            / "experiments"
            / "exp2_validated_rank"
            / "results_2a_causal_tau005_r16_nbuild16"
            / "summary.csv"
        ),
        help="Existing Experiment 2 architecture summary to join against.",
    )
    parser.add_argument(
        "--results-dir",
        default=str(SCRIPT_DIR / "results"),
        help="Output directory for CSV, JSON, PNG, PDF, and report files.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def psd_invsqrt(A: torch.Tensor) -> torch.Tensor:
    eigvals, eigvecs = torch.linalg.eigh(0.5 * (A + A.T))
    eigvals = eigvals.clamp_min(1e-12)
    return (eigvecs * eigvals.rsqrt().unsqueeze(0)) @ eigvecs.T


def required_rank_from_energy(energy: torch.Tensor, eps: float) -> int:
    total = float(energy.sum())
    if total <= FLOOR:
        return 0
    for r in range(energy.numel() + 1):
        tail = float(energy[r:].sum()) if r < energy.numel() else 0.0
        if math.sqrt(tail / (total + FLOOR)) <= eps:
            return int(r)
    return int(energy.numel())


def numerical_rank(svals: torch.Tensor, tau: float) -> int:
    if svals.numel() == 0:
        return 0
    top = float(svals[0])
    if top <= 0:
        return 0
    return int((svals >= tau * top).sum().item())


def quantile_nearest(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(float(v) for v in values)
    idx = min(max(math.ceil(q * len(ordered)) - 1, 0), len(ordered) - 1)
    return ordered[idx]


def compute_apriori_records(args: argparse.Namespace) -> List[Dict[str, object]]:
    dx_list = parse_int_list(args.dx_list)
    eps_list = parse_float_list(args.eps_list)
    torch.manual_seed(args.seed)

    records: List[Dict[str, object]] = []
    for d_x in dx_list:
        x_ctx, _, x_tgt, _, _ = sample_batch_eigenvalues(
            args.episodes,
            d_x,
            args.n_ctx,
            args.n_tgt,
            args.sigma2,
            flat_eigenvalues,
            device="cpu",
        )

        eye = torch.eye(args.n_ctx, dtype=torch.float64)
        for episode in range(args.episodes):
            X = x_ctx[episode].double()
            Z = x_tgt[episode].double()
            K = X @ X.T
            Kt = Z @ X.T
            A = K + args.sigma2 * eye
            M = Kt @ psd_invsqrt(A)
            svals = torch.linalg.svdvals(M)
            energy = (svals.double() ** 2).clamp_min(0.0)
            total_energy = float(energy.sum())

            rec: Dict[str, object] = {
                "d_x": d_x,
                "episode": episode,
                "n_ctx": args.n_ctx,
                "n_tgt": args.n_tgt,
                "sigma2": args.sigma2,
                "total_energy": total_energy,
                "numerical_rank": numerical_rank(svals, args.sv_tau),
                "singular_values": " ".join(f"{float(s):.10g}" for s in svals),
            }
            for eps in eps_list:
                rec[f"r_eps_{eps:g}"] = required_rank_from_energy(energy, eps)
            records.append(rec)
    return records


def summarize_apriori(
    records: Sequence[Dict[str, object]],
    eps_list: Sequence[float],
    alpha: float,
) -> List[Dict[str, object]]:
    by_dx: Dict[int, List[Dict[str, object]]] = {}
    for rec in records:
        by_dx.setdefault(int(rec["d_x"]), []).append(rec)

    rows: List[Dict[str, object]] = []
    for d_x in sorted(by_dx):
        group = by_dx[d_x]
        numerical_ranks = [float(rec["numerical_rank"]) for rec in group]
        for eps in eps_list:
            key = f"r_eps_{eps:g}"
            vals = [float(rec[key]) for rec in group]
            rows.append(
                {
                    "d_x": d_x,
                    "epsilon": eps,
                    "alpha": alpha,
                    "n": len(vals),
                    "r_mean": float(np.mean(vals)),
                    "r_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "r_min": int(min(vals)),
                    "r_median": float(np.median(vals)),
                    "r_q90": quantile_nearest(vals, 0.90),
                    "r_q95": quantile_nearest(vals, 0.95),
                    "r_q99": quantile_nearest(vals, 0.99),
                    "r_max": int(max(vals)),
                    "r_epsilon_alpha": quantile_nearest(vals, 1.0 - alpha),
                    "numerical_rank_mean": float(np.mean(numerical_ranks)),
                    "numerical_rank_min": int(min(numerical_ranks)),
                    "numerical_rank_max": int(max(numerical_ranks)),
                }
            )
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def f(row: Dict[str, str], key: str, default: float = float("nan")) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def required_by_dx(summary_rows: Sequence[Dict[str, object]], target_eps: float) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for row in summary_rows:
        if abs(float(row["epsilon"]) - target_eps) < 1e-12:
            out[int(row["d_x"])] = float(row["r_epsilon_alpha"])
    return out


def architecture_bound(n_ctx: int, d_model: float, n_layers: float) -> float:
    return min(n_ctx, d_model) * (n_layers + 1.0) * (n_layers + 2.0) / 2.0


def join_target_sweep(
    summary_rows: Sequence[Dict[str, object]],
    args: argparse.Namespace,
) -> List[Dict[str, object]]:
    required = required_by_dx(summary_rows, args.target_eps)
    exp2_rows = read_csv_rows(Path(args.target_summary))
    joined: List[Dict[str, object]] = []
    for row in exp2_rows:
        d_x = int(round(f(row, "d_x_mean", f(row, "sweep_val_mean"))))
        r_req = required.get(d_x, float("nan"))
        d_nat = f(row, "dim_R_nat_mean")
        n_layers = f(row, "n_layers_mean", 8.0)
        d_model = f(row, "d_model_mean", 128.0)
        joined.append(
            {
                "checkpoint": row.get("checkpoint", ""),
                "d_x": d_x,
                "epsilon": args.target_eps,
                "alpha": args.alpha,
                "r_epsilon_alpha": r_req,
                "d_nat_mean": d_nat,
                "d_nat_std": f(row, "dim_R_nat_std"),
                "d_nat_over_r_req": d_nat / r_req if r_req else float("nan"),
                "r_eff_T_task_mean": f(row, "r_eff_T_task_mean"),
                "E_TQ_T_mean": f(row, "E_TQ_T_mean"),
                "E_TQ_T_std": f(row, "E_TQ_T_std"),
                "E_F_T_mean": f(row, "E_F_T_mean"),
                "E_F_TQ_mean": f(row, "E_F_TQ_mean"),
                "B_nat_mean": f(row, "B_nat_mean"),
                "B_arch": architecture_bound(args.n_ctx, d_model, n_layers),
            }
        )
    return sorted(joined, key=lambda r: int(r["d_x"]))


def join_arch_sweep(
    summary_rows: Sequence[Dict[str, object]],
    args: argparse.Namespace,
) -> List[Dict[str, object]]:
    required = required_by_dx(summary_rows, args.target_eps)
    exp2_rows = read_csv_rows(Path(args.arch_summary))
    joined: List[Dict[str, object]] = []
    for row in exp2_rows:
        d_x = int(round(f(row, "d_x_mean", 5.0)))
        r_req = required.get(d_x, float("nan"))
        d_nat = f(row, "dim_R_nat_mean")
        n_layers = f(row, "n_layers_mean")
        d_model = f(row, "d_model_mean")
        joined.append(
            {
                "checkpoint": row.get("checkpoint", ""),
                "d_x": d_x,
                "n_layers": n_layers,
                "n_heads": f(row, "n_heads_mean"),
                "epsilon": args.target_eps,
                "alpha": args.alpha,
                "r_epsilon_alpha": r_req,
                "d_nat_mean": d_nat,
                "d_nat_std": f(row, "dim_R_nat_std"),
                "d_nat_over_r_req": d_nat / r_req if r_req else float("nan"),
                "E_TQ_T_mean": f(row, "E_TQ_T_mean"),
                "E_F_T_mean": f(row, "E_F_T_mean"),
                "E_F_TQ_mean": f(row, "E_F_TQ_mean"),
                "B_nat_mean": f(row, "B_nat_mean"),
                "B_arch": architecture_bound(args.n_ctx, d_model, n_layers),
            }
        )
    order = {"L2": 0, "L4": 1, "L6": 2, "L8": 3, "L12": 4, "H1": 5, "H2": 6, "H8": 7}
    return sorted(joined, key=lambda r: order.get(str(r["checkpoint"]), 99))


def plot_rank_distribution(
    summary_rows: Sequence[Dict[str, object]],
    target_rows: Sequence[Dict[str, object]],
    arch_rows: Sequence[Dict[str, object]],
    args: argparse.Namespace,
    out_dir: Path,
) -> None:
    eps_to_show = [0.1, 0.05, args.target_eps]
    dxs = sorted({int(r["d_x"]) for r in summary_rows})

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0))

    ax = axes[0]
    for eps in eps_to_show:
        rows = [r for r in summary_rows if abs(float(r["epsilon"]) - eps) < 1e-12]
        rows = sorted(rows, key=lambda r: int(r["d_x"]))
        if not rows:
            continue
        ax.plot(
            [int(r["d_x"]) for r in rows],
            [float(r["r_epsilon_alpha"]) for r in rows],
            marker="o",
            label=rf"$\varepsilon={eps:g}$",
        )
    ax.plot(dxs, dxs, color="0.2", linestyle="--", linewidth=1.0, label=r"$d_x$")
    ax.set_xlabel(r"$d_x$")
    ax.set_ylabel(r"$r_{\varepsilon,\alpha}$")
    ax.set_title("A priori rank")
    ax.legend(frameon=False)

    ax = axes[1]
    target_rows = sorted(target_rows, key=lambda r: int(r["d_x"]))
    x = np.arange(len(target_rows))
    ax.bar(x - 0.18, [float(r["r_epsilon_alpha"]) for r in target_rows], width=0.36, label="a priori")
    ax.bar(x + 0.18, [float(r["d_nat_mean"]) for r in target_rows], width=0.36, label=r"$d_{\rm nat}$")
    ax.set_xticks(x, [str(int(r["d_x"])) for r in target_rows])
    ax.set_xlabel(r"$d_x$")
    ax.set_ylabel("rank")
    ax.set_title("Target-rank sweep")
    ax.legend(frameon=False)

    ax2 = ax.twinx()
    ax2.plot(x, [float(r["E_TQ_T_mean"]) for r in target_rows], color="crimson", marker="s")
    ax2.set_ylabel(r"$\mathcal{E}(T_{Q_{\mathrm{nat}}},T)$", color="crimson")
    ax2.tick_params(axis="y", colors="crimson")

    ax = axes[2]
    labels = [str(r["checkpoint"]) for r in arch_rows]
    x = np.arange(len(arch_rows))
    ax.bar(x, [float(r["d_nat_over_r_req"]) for r in arch_rows], color="0.55")
    ax.axhline(1.0, color="0.1", linestyle="--", linewidth=1.0)
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel(r"$d_{\rm nat}/r_{\varepsilon,\alpha}$")
    ax.set_title("Architecture sweep")

    ax2 = ax.twinx()
    ax2.plot(x, [float(r["E_F_T_mean"]) for r in arch_rows], color="crimson", marker="s")
    ax2.set_ylabel(r"$\mathcal{E}(F,T)$", color="crimson")
    ax2.tick_params(axis="y", colors="crimson")

    fig.suptitle(rf"Experiment 4: $\varepsilon={args.target_eps:g}$, $\alpha={args.alpha:g}$")
    fig.tight_layout()
    fig.savefig(out_dir / "experiment_4_apriori_rank_closure.png", dpi=200)
    fig.savefig(out_dir / "experiment_4_apriori_rank_closure.pdf")
    plt.close(fig)


def write_report(
    out_dir: Path,
    summary_rows: Sequence[Dict[str, object]],
    target_rows: Sequence[Dict[str, object]],
    arch_rows: Sequence[Dict[str, object]],
    args: argparse.Namespace,
) -> None:
    headline = [
        r for r in summary_rows if abs(float(r["epsilon"]) - args.target_eps) < 1e-12
    ]
    lines = []
    lines.append("Experiment 4: A Priori Rank and Capacity Closure")
    lines.append("=" * 56)
    lines.append("")
    lines.append(f"episodes per d_x: {args.episodes}")
    lines.append(f"n_ctx={args.n_ctx}, n_tgt={args.n_tgt}, sigma2={args.sigma2}")
    lines.append(f"headline epsilon={args.target_eps}, alpha={args.alpha}")
    lines.append("")
    lines.append("A priori distributional ranks:")
    for row in sorted(headline, key=lambda r: int(r["d_x"])):
        lines.append(
            "  d_x={d_x:>2}: r_mean={r_mean:.3f}, r_(1-alpha)={r_epsilon_alpha:.0f}, "
            "range=[{r_min},{r_max}], numerical_rank={numerical_rank_mean:.0f}".format(**row)
        )
    lines.append("")
    lines.append("Target-rank sweep joined to Experiment 2:")
    for row in target_rows:
        lines.append(
            "  {checkpoint}: r_req={r_epsilon_alpha:.0f}, d_nat={d_nat_mean:.2f}, "
            "d_nat/r={d_nat_over_r_req:.3f}, E(T_Qnat,T)={E_TQ_T_mean:.5g}, "
            "E(F,T)={E_F_T_mean:.5g}".format(**row)
        )
    lines.append("")
    lines.append("Architecture sweep joined to Experiment 2:")
    for row in arch_rows:
        lines.append(
            "  {checkpoint}: r_req={r_epsilon_alpha:.0f}, d_nat/r={d_nat_over_r_req:.3f}, "
            "E(T_Qnat,T)={E_TQ_T_mean:.5g}, E(F,T)={E_F_T_mean:.5g}".format(**row)
        )
    lines.append("")
    lines.append("Interpretation:")
    lines.append(
        "  At epsilon=0.01, the flat-spectrum linear-teacher task requires all "
        "task-visible linear directions: r_{epsilon,alpha}=d_x for the target sweep."
    )
    lines.append(
        "  Existing causal-native results match this threshold for d_x=3,5,8,10: "
        "d_nat reaches r_req and the reduced KRR residual is numerical zero."
    )
    lines.append(
        "  The d_x=15 checkpoint is the informative miss: mean d_nat is below the "
        "rank target and the reduced KRR residual remains nonzero."
    )
    lines.append(
        "  The architecture sweep shows the complementary point: once d_nat reaches "
        "the rank target, T_Qnat can be exact even when the shallow L2 model still "
        "has a large model-to-KRR residual. Capacity is necessary for this mechanism, "
        "but learning the exact map is an additional condition."
    )
    (out_dir / "analysis.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.results_dir)
    ensure_dir(out_dir)

    eps_list = parse_float_list(args.eps_list)
    if args.target_eps not in eps_list:
        eps_list = sorted(set(eps_list + [args.target_eps]), reverse=True)

    config = vars(args).copy()
    config["eps_list"] = eps_list
    config["dx_list"] = parse_int_list(args.dx_list)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    records = compute_apriori_records(args)
    summary_rows = summarize_apriori(records, eps_list, args.alpha)
    target_rows = join_target_sweep(summary_rows, args)
    arch_rows = join_arch_sweep(summary_rows, args)

    write_csv(out_dir / "records.csv", records)
    write_csv(out_dir / "rank_summary.csv", summary_rows)
    write_csv(out_dir / "target_sweep_closure.csv", target_rows)
    write_csv(out_dir / "architecture_sweep_closure.csv", arch_rows)
    plot_rank_distribution(summary_rows, target_rows, arch_rows, args, out_dir)
    write_report(out_dir, summary_rows, target_rows, arch_rows, args)

    print((out_dir / "analysis.txt").read_text())


if __name__ == "__main__":
    main()
