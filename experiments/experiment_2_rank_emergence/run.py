#!/usr/bin/env python3
"""Layerwise KRR subspace sufficiency diagnostic.

This is a new Experiment 2 variant. It keeps the layerwise goal but replaces
greedy single-direction causal selection with a risk-normalized sufficiency
curve at every layer.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
MPL_CACHE_DIR = Path(os.environ.get("TMPDIR", "/tmp")) / "mech_icl_krr_mpl_cache"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.experiment_1_operator_certificate.run import (  # noqa: E402
    symmetric_eig_factors,
    weighted_svd_basis,
)
from experiment_utils.layerwise_core import (  # noqa: E402
    CkptCfg,
    FLOOR,
    build_eval_kernels,
    build_eval_target_kernel,
    effective_rank_T_excess_risk,
    effective_rank_T_task,
    eval_operator_error,
    forward_with_ctx_hidden,
    hidden_response_matrices,
    load_model,
    prediction_fd_bundle,
    psd_sqrt_and_invsqrt,
    sample_eval_episode,
    sample_task_probes,
)
from experiment_utils.support import get_device, set_seed  # noqa: E402


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: Sequence[Dict[str, object]], group_keys: Sequence[str]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[object, ...], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k) for k in group_keys)].append(row)
    numeric = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float, np.integer, np.floating)) and key != "episode"
        }
    )
    out: List[Dict[str, object]] = []
    for key, vals in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        rec: Dict[str, object] = {name: value for name, value in zip(group_keys, key)}
        rec["n"] = len(vals)
        for field in numeric:
            arr = np.array([float(v[field]) for v in vals if field in v], dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                rec[f"{field}_mean"] = float(arr.mean())
                rec[f"{field}_std"] = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        out.append(rec)
    return out


def exact_operator_excess(T_q: torch.Tensor, T: torch.Tensor, A: torch.Tensor) -> float:
    delta = T_q - T
    return max(float(((delta @ A) * delta).sum()), 0.0)


def empirical_model_excess(
    eval_fd: torch.Tensor,
    T_q: torch.Tensor,
    probes: torch.Tensor,
) -> float:
    pred = T_q @ probes.T
    if probes.shape[0] == 0:
        return 0.0
    return float(((eval_fd - pred) ** 2).sum()) / probes.shape[0]


def a_orth_candidate(
    M: torch.Tensor,
    A_factors: Dict[str, torch.Tensor],
    tau_sv: float,
    candidate_rank: int,
) -> torch.Tensor:
    pack = weighted_svd_basis(
        M,
        A_factors,
        tau_sv=tau_sv,
        r_max=candidate_rank,
        force_rank=None,
        curve_r_max=candidate_rank,
    )
    return pack["Q"]  # type: ignore[return-value]


def krr_targeted_order(
    C: torch.Tensor,
    Kt: torch.Tensor,
    A_sqrt: torch.Tensor,
    A_invsqrt: torch.Tensor,
) -> torch.Tensor:
    """Order a candidate A-orthonormal span by KRR operator energy.

    If U=A^{1/2}C, prefixes of U V maximize ||K_t A^{-1/2} U V||_F
    within span(U).
    """
    if C.numel() == 0 or C.shape[1] == 0:
        return C
    U_c = A_sqrt @ C
    Z = Kt @ A_invsqrt
    gram = U_c.T @ (Z.T @ Z) @ U_c
    gram = 0.5 * (gram + gram.T)
    vals, vecs = torch.linalg.eigh(gram)
    order = torch.argsort(vals, descending=True)
    return C @ vecs[:, order]


def rank_at_or_below(rows: Sequence[Dict[str, object]], threshold: float) -> float:
    for row in rows:
        if float(row["op_excess_over_krr_risk"]) <= threshold:
            return float(row["rank"])
    return float("nan")


def make_layer_summary(
    curve_rows: Sequence[Dict[str, object]],
    args: argparse.Namespace,
) -> List[Dict[str, object]]:
    groups: Dict[Tuple[int, str, int], List[Dict[str, object]]] = defaultdict(list)
    for row in curve_rows:
        groups[(int(row["episode"]), str(row["basis"]), int(row["layer"]))].append(row)

    selected_rows: List[Dict[str, object]] = []
    for (episode, basis, layer), rows in groups.items():
        rows_sorted = sorted(rows, key=lambda r: int(r["rank"]))
        oracle_rank = int(round(float(rows_sorted[0]["r_eff_T_task"])))
        strict_rank = int(round(float(rows_sorted[0]["r_eff_T_task_strict"])))
        at_oracle = min(rows_sorted, key=lambda r: abs(int(r["rank"]) - oracle_rank))
        at_strict = min(rows_sorted, key=lambda r: abs(int(r["rank"]) - strict_rank))
        best = min(rows_sorted, key=lambda r: float(r["op_excess_over_krr_risk"]))
        selected_rows.append(
            {
                "episode": episode,
                "basis": basis,
                "layer": layer,
                "r_eff_T_task": float(rows_sorted[0]["r_eff_T_task"]),
                "r_eff_T_task_strict": float(rows_sorted[0]["r_eff_T_task_strict"]),
                "rank_to_alpha": rank_at_or_below(rows_sorted, args.excess_risk_frac),
                "reach_alpha": 1.0
                if math.isfinite(rank_at_or_below(rows_sorted, args.excess_risk_frac))
                else 0.0,
                "rank_at_oracle": int(at_oracle["rank"]),
                "op_excess_at_oracle": float(at_oracle["op_excess_over_krr_risk"]),
                "op_signal_error_at_oracle": float(at_oracle["op_signal_error"]),
                "model_excess_at_oracle": float(at_oracle["model_excess_over_krr_risk"]),
                "rank_at_strict": int(at_strict["rank"]),
                "op_excess_at_strict": float(at_strict["op_excess_over_krr_risk"]),
                "op_signal_error_at_strict": float(at_strict["op_signal_error"]),
                "model_excess_at_strict": float(at_strict["model_excess_over_krr_risk"]),
                "best_rank": int(best["rank"]),
                "best_op_excess": float(best["op_excess_over_krr_risk"]),
                "best_op_signal_error": float(best["op_signal_error"]),
                "best_model_excess": float(best["model_excess_over_krr_risk"]),
                "candidate_dim": int(rows_sorted[0]["candidate_dim"]),
            }
        )
    return summarize(selected_rows, ["basis", "layer"])


def _basis_label(basis: str) -> str:
    return {
        "response": "Response span",
        "raw": "Raw residual span",
        "combined": "Combined span",
    }.get(basis, basis.replace("_", " ").title())


def _as_float(value: object, default: float = float("nan")) -> float:
    if value is None:
        return default
    if isinstance(value, str) and value.strip() == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _summary_series(rows: Sequence[Dict[str, object]], field: str) -> List[float]:
    return [_as_float(r.get(field)) for r in rows]


def _mean_prediction_risk_rank(rows: Sequence[Dict[str, object]]) -> float:
    vals = [_as_float(r.get("r_eff_T_task_mean")) for r in rows]
    vals = [v for v in vals if math.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _save_plot(fig: plt.Figure, path_without_suffix: Path) -> None:
    fig.savefig(path_without_suffix.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(path_without_suffix.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)


def plot_emergence(
    layer_summary: Sequence[Dict[str, object]],
    out_dir: Path,
    threshold: float = 0.05,
) -> None:
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "semibold",
            "font.size": 10,
        }
    )
    basis_order = ["response", "raw", "combined"]
    bases = [b for b in basis_order if any(str(r["basis"]) == b for r in layer_summary)]
    bases.extend(sorted({str(r["basis"]) for r in layer_summary} - set(bases)))

    for basis in bases:
        rows = sorted([r for r in layer_summary if r["basis"] == basis], key=lambda r: int(r["layer"]))
        if not rows:
            continue
        layers = [int(r["layer"]) for r in rows]
        r_t = _mean_prediction_risk_rank(rows)

        fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), constrained_layout=True)
        axes[0].plot(
            layers,
            _summary_series(rows, "op_excess_at_oracle_mean"),
            marker="o",
            linewidth=2.0,
            label=r"prefix size $r_T$",
        )
        axes[0].plot(
            layers,
            _summary_series(rows, "best_op_excess_mean"),
            marker="s",
            linewidth=1.8,
            label=r"best prefix",
        )
        axes[0].axhline(
            threshold,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label=f"{100 * threshold:.0f}% target",
        )
        axes[0].set_xlabel("layer")
        axes[0].set_ylabel(r"excess-risk ratio $\rho_G$")
        axes[0].set_title(r"Excess at task rank: $\rho_G(T_{Q_{\ell,r_T}})$", fontsize=11)
        axes[0].grid(True, axis="y", alpha=0.22)
        axes[0].legend(frameon=False, fontsize=8)

        axes[1].plot(
            layers,
            _summary_series(rows, "rank_to_alpha_mean"),
            marker="o",
            linewidth=2.0,
            label=rf"$k_\ell^{{{100 * threshold:.0f}\%}}$",
        )
        if math.isfinite(r_t):
            axes[1].axhline(
                r_t,
                color="black",
                linestyle="--",
                linewidth=1.2,
                label=rf"$r_T={r_t:.1f}$",
            )
        axes[1].set_xlabel("layer")
        axes[1].set_ylabel("prefix rank")
        axes[1].set_title(r"Rank to 5% risk: $k_\ell^{5\%}$", fontsize=11)
        axes[1].set_ylim(bottom=0)
        axes[1].grid(True, axis="y", alpha=0.22)
        axes[1].legend(frameon=False, fontsize=8)

        _save_plot(fig, out_dir / f"emergence_{basis}")
        plt.close(fig)

    if len(bases) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), constrained_layout=True)
        for basis in bases:
            rows = sorted([r for r in layer_summary if r["basis"] == basis], key=lambda r: int(r["layer"]))
            layers = [int(r["layer"]) for r in rows]
            axes[0].plot(
                layers,
                _summary_series(rows, "op_excess_at_oracle_mean"),
                marker="o",
                linewidth=1.9,
                label=_basis_label(basis),
            )
            axes[1].plot(
                layers,
                _summary_series(rows, "best_op_excess_mean"),
                marker="o",
                linewidth=1.9,
                label=_basis_label(basis),
            )
        for ax in axes:
            ax.axhline(
                threshold,
                color="black",
                linestyle="--",
                linewidth=1.2,
                label=f"{100 * threshold:.0f}% target",
            )
            ax.set_xlabel("layer")
            ax.set_ylabel(r"excess-risk ratio $\rho_G$")
            ax.grid(True, axis="y", alpha=0.22)
        axes[0].set_title(r"At task rank: $\rho_G(T_{Q_{\ell,r_T}})$", fontsize=11)
        axes[1].set_title(r"Best prefix inside each candidate span", fontsize=11)
        handles, labels = axes[1].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axes[1].legend(by_label.values(), by_label.keys(), loc="upper right", frameon=False, fontsize=8)
        _save_plot(fig, out_dir / "emergence_basis_comparison")
        plt.close(fig)

    response_rows = sorted([r for r in layer_summary if str(r["basis"]) == "response"], key=lambda r: int(r["layer"]))
    if response_rows:
        layers = [int(r["layer"]) for r in response_rows]
        fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), constrained_layout=True)
        axes[0].plot(
            layers,
            _summary_series(response_rows, "op_excess_at_oracle_mean"),
            marker="o",
            linewidth=2.0,
            label=r"prefix size $r_T$",
        )
        axes[0].plot(
            layers,
            _summary_series(response_rows, "best_op_excess_mean"),
            marker="s",
            linewidth=1.8,
            label=r"best prefix",
        )
        axes[0].axhline(
            threshold,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label=f"{100 * threshold:.0f}% target",
        )
        axes[0].set_xlabel("layer")
        axes[0].set_ylabel(r"excess-risk ratio $\rho_G$")
        axes[0].set_title(r"Response-span KRR excess", fontsize=11)
        axes[0].grid(True, axis="y", alpha=0.22)
        axes[0].legend(frameon=False, fontsize=8)

        axes[1].plot(
            layers,
            _summary_series(response_rows, "reach_alpha_mean"),
            marker="o",
            linewidth=2.0,
        )
        axes[1].set_xlabel("layer")
        axes[1].set_ylabel("episode fraction")
        axes[1].set_ylim(-0.03, 1.03)
        axes[1].set_title(r"Episodes reaching $\rho_G \leq 0.05$", fontsize=11)
        axes[1].grid(True, axis="y", alpha=0.22)
        _save_plot(fig, out_dir / "emergence_response_reach")
        plt.close(fig)


def write_report(path: Path, layer_summary: Sequence[Dict[str, object]], args: argparse.Namespace) -> None:
    curve_desc = "all" if args.curve_r_max is None else str(args.curve_r_max)
    lines = [
        "Experiment 2 Layerwise Sufficiency",
        "",
        f"checkpoint={args.checkpoint}",
        f"episodes={args.episodes}, n_ctx={args.n_ctx}, n_tgt={args.n_tgt}, "
        f"alpha={args.excess_risk_frac}, curve_r_max={curve_desc}",
        "",
    ]
    for basis in ("response", "raw", "combined"):
        rows = sorted([r for r in layer_summary if r["basis"] == basis], key=lambda r: int(r["layer"]))
        if not rows:
            continue
        lines.append(f"{basis} basis")
        lines.append("layer cand  rT  strict  rank_alpha reach  excess@rT  best_excess  model@rT")
        for row in rows:
            lines.append(
                f"{int(row['layer']):5d} "
                f"{float(row.get('candidate_dim_mean', float('nan'))):4.0f} "
                f"{float(row.get('r_eff_T_task_mean', float('nan'))):4.1f} "
                f"{float(row.get('r_eff_T_task_strict_mean', float('nan'))):6.1f} "
                f"{float(row.get('rank_to_alpha_mean', float('nan'))):10.1f} "
                f"{float(row.get('reach_alpha_mean', float('nan'))):5.2f} "
                f"{float(row.get('op_excess_at_oracle_mean', float('nan'))):9.3f} "
                f"{float(row.get('best_op_excess_mean', float('nan'))):11.3f} "
                f"{float(row.get('model_excess_at_oracle_mean', float('nan'))):8.3f}"
            )
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        default=str(SCRIPT_DIR / "results_rbf128_sufficiency"),
    )
    parser.add_argument(
        "--checkpoint",
        default="final/rbf_fixed_l3_nctx128_ntgt128_seed42.pt",
    )
    parser.add_argument("--name", default="rbf_elbow_scratch_nctx128_ntgt128")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--d-x", type=int, default=5)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-ctx", type=int, default=128)
    parser.add_argument("--n-tgt", type=int, default=128)
    parser.add_argument("--sigma2", type=float, default=0.1)
    parser.add_argument("--kernel-family", choices=["linear", "rbf"], default="rbf")
    parser.add_argument("--kernel-lengthscale", type=float, default=3.0)
    parser.add_argument("--kernel-signal-var", type=float, default=1.0)
    parser.add_argument("--kernel-jitter", type=float, default=1e-5)
    parser.add_argument("--n-build", type=int, default=16)
    parser.add_argument("--n-eval", type=int, default=16)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--tau-sv", type=float, default=1e-3)
    parser.add_argument(
        "--candidate-rank",
        type=int,
        default=None,
        help="Maximum extracted candidate dimension. Omit to keep every numerically retained direction.",
    )
    parser.add_argument(
        "--curve-r-max",
        type=int,
        default=None,
        help="Maximum prefix rank to evaluate. Omit to evaluate every candidate prefix.",
    )
    parser.add_argument("--rank-tau", type=float, default=1e-2)
    parser.add_argument("--excess-risk-frac", type=float, default=0.05)
    parser.add_argument("--basis-kinds", default="response,raw,combined")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps({"args": vars(args)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    device = get_device() if args.device == "auto" else torch.device(args.device)
    set_seed(args.seed)
    gen = torch.Generator(device="cpu").manual_seed(args.seed + 7007)

    cfg = CkptCfg(
        args.name,
        args.checkpoint,
        args.d_x,
        args.d_model,
        args.n_layers,
        args.n_heads,
        args.kernel_family,
        args.kernel_lengthscale if args.kernel_family == "rbf" else float(args.d_x),
    )
    model = load_model(cfg, device)
    basis_kinds = [x.strip() for x in args.basis_kinds.split(",") if x.strip()]

    curve_rows: List[Dict[str, object]] = []
    episode_rows: List[Dict[str, object]] = []

    for episode in range(args.episodes):
        x_ctx, y_ctx, x_tgt, y_tgt = sample_eval_episode(cfg, args, device)
        y = y_ctx[0].detach().cpu().double()
        _K, Kt, A, T = build_eval_kernels(x_ctx, x_tgt, args)
        Ktt = build_eval_target_kernel(x_tgt, args)
        A_factors = symmetric_eig_factors(A)
        A_sqrt, A_invsqrt = psd_sqrt_and_invsqrt(A)
        risk_pack = effective_rank_T_excess_risk(T, A, Kt, Ktt, args.excess_risk_frac)
        r_t_strict = effective_rank_T_task(T, A, args.rank_tau)
        r_t = int(risk_pack["r_eff_T_task"])
        risk_total = float(risk_pack["krr_risk_total"])
        signal_total = float(risk_pack["krr_signal_total"])

        F_y, raw_layers = forward_with_ctx_hidden(model, x_ctx, y, x_tgt)
        response_layers = hidden_response_matrices(
            model,
            x_ctx,
            x_tgt,
            y,
            sample_task_probes(A, args.n_build, gen),
            args.eps,
        )
        eval_probes = sample_task_probes(A, args.n_eval, gen)
        eval_fd = prediction_fd_bundle(model, x_ctx, x_tgt, y, eval_probes, args.eps)
        e_f_t = eval_operator_error(T, eval_probes, eval_fd)
        f_t_excess = empirical_model_excess(eval_fd, T, eval_probes)

        episode_rows.append(
            {
                "episode": episode,
                "r_eff_T_task": r_t,
                "r_eff_T_task_strict": r_t_strict,
                "rank_tau_task": risk_pack["rank_tau_task"],
                "krr_risk_total": risk_total,
                "krr_signal_total": signal_total,
                "E_F_T": e_f_t,
                "F_T_excess_over_krr_risk": f_t_excess / (risk_total + FLOOR),
            }
        )

        for layer, (M_resp, M_raw) in enumerate(zip(response_layers, raw_layers)):
            matrices = {
                "response": M_resp,
                "raw": M_raw,
                "combined": torch.cat([M_resp, M_raw], dim=1),
            }
            for basis in basis_kinds:
                M = matrices[basis]
                candidate_limit = (
                    args.candidate_rank if args.candidate_rank is not None else max(M.shape)
                )
                cand_rank = min(candidate_limit, M.shape[0], M.shape[1])
                C = a_orth_candidate(M, A_factors, args.tau_sv, cand_rank)
                Q_all = krr_targeted_order(C, Kt, A_sqrt, A_invsqrt)
                max_rank = (
                    Q_all.shape[1]
                    if args.curve_r_max is None
                    else min(args.curve_r_max, Q_all.shape[1])
                )
                for rank in range(max_rank + 1):
                    Q = Q_all[:, :rank]
                    T_q = Kt @ Q @ Q.T if rank else torch.zeros_like(T)
                    op_excess = exact_operator_excess(T_q, T, A)
                    model_excess = empirical_model_excess(eval_fd, T_q, eval_probes)
                    curve_rows.append(
                        {
                            "episode": episode,
                            "basis": basis,
                            "layer": layer,
                            "rank": rank,
                            "candidate_dim": Q_all.shape[1],
                            "r_eff_T_task": r_t,
                            "r_eff_T_task_strict": r_t_strict,
                            "op_excess_over_krr_risk": op_excess / (risk_total + FLOOR),
                            "op_signal_error": math.sqrt(op_excess / (signal_total + FLOOR)),
                            "model_excess_over_krr_risk": model_excess / (risk_total + FLOOR),
                            "model_signal_error": math.sqrt(model_excess / (signal_total + FLOOR)),
                            "E_F_T": e_f_t,
                            "F_T_excess_over_krr_risk": f_t_excess / (risk_total + FLOOR),
                        }
                    )
        print(
            f"episode {episode + 1}/{args.episodes}: rT={r_t} strict={r_t_strict} "
            f"E(F,T)={e_f_t:.5f}",
            flush=True,
        )

    write_csv(out_dir / "episode_summary.csv", episode_rows)
    write_csv(out_dir / "rank_curves.csv", curve_rows)
    rank_summary = summarize(curve_rows, ["basis", "layer", "rank"])
    write_csv(out_dir / "rank_summary.csv", rank_summary)
    layer_summary = make_layer_summary(curve_rows, args)
    write_csv(out_dir / "layer_summary.csv", layer_summary)
    write_report(out_dir / "summary.txt", layer_summary, args)
    plot_emergence(layer_summary, out_dir, threshold=args.excess_risk_frac)
    print(out_dir / "summary.txt", flush=True)


if __name__ == "__main__":
    main()
