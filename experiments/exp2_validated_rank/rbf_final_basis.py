#!/usr/bin/env python3
"""Final-state RBF basis diagnostic.

This tests whether a fixed-RBF checkpoint exposes a final context-side subspace
whose induced reduced KRR operator approaches the RBF operator as rank increases.
It is intentionally smaller than Experiment 1 and focused on raw/response final
bases for the RBF checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SUPPORT_DIR = REPO_ROOT / "experiments" / "shared"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SUPPORT_DIR))

from experiments.exp1_operator_certificate.run import (  # noqa: E402
    finite_difference_bundle,
    forward_model,
    galerkin_operator,
    operator_metrics,
    sample_probes,
    symmetric_eig_factors,
    weighted_svd_basis,
)
from experiments.exp2_budget_closure.run import (  # noqa: E402
    CkptCfg,
    effective_rank_T_excess_risk,
    effective_rank_T_task,
    load_model,
)
from experiments.exp2_validated_rank.run import (  # noqa: E402
    build_eval_kernels,
    build_eval_target_kernel,
    sample_eval_episode,
)
from support import get_device, set_seed  # noqa: E402


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: Sequence[Dict[str, object]], group_keys: Sequence[str]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[object, ...], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k) for k in group_keys)].append(row)

    numeric = sorted(
        {
            k
            for row in rows
            for k, v in row.items()
            if isinstance(v, (int, float, np.integer, np.floating)) and k != "episode"
        }
    )
    out: List[Dict[str, object]] = []
    for key, vals in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        rec: Dict[str, object] = {g: v for g, v in zip(group_keys, key)}
        rec["n"] = len(vals)
        for field in numeric:
            arr = np.array([float(v[field]) for v in vals if field in v], dtype=float)
            if arr.size:
                rec[f"{field}_mean"] = float(arr.mean())
                rec[f"{field}_std"] = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        out.append(rec)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RBF final-state basis diagnostic")
    parser.add_argument("--results-dir", default=str(SCRIPT_DIR / "results_rbf_l3_final_basis"))
    parser.add_argument("--checkpoint", default="model_rbf_fixed_l3.pt")
    parser.add_argument("--name", default="rbf_l3")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--d-x", type=int, default=5)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-ctx", type=int, default=47)
    parser.add_argument("--n-tgt", type=int, default=16)
    parser.add_argument("--sigma2", type=float, default=0.1)
    parser.add_argument("--kernel-family", choices=["rbf"], default="rbf")
    parser.add_argument("--kernel-lengthscale", type=float, default=3.0)
    parser.add_argument("--kernel-signal-var", type=float, default=1.0)
    parser.add_argument("--kernel-jitter", type=float, default=1e-5)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--rank-tau", type=float, default=1e-2)
    parser.add_argument("--excess-risk-frac", type=float, default=0.05)
    parser.add_argument("--tau-sv", type=float, default=1e-3)
    parser.add_argument("--curve-r-max", type=int, default=16)
    parser.add_argument("--n-build", type=int, default=32)
    parser.add_argument("--n-eval", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device() if args.device == "auto" else torch.device(args.device)
    set_seed(args.seed)
    gen = torch.Generator(device="cpu").manual_seed(args.seed + 2024)
    cfg = CkptCfg(args.name, args.checkpoint, args.d_x, args.d_model, args.n_layers, args.n_heads, "rbf", args.kernel_lengthscale)
    model = load_model(cfg, device)

    rows: List[Dict[str, object]] = []
    config = {"args": vars(args), "device": str(device)}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

    for episode in range(args.episodes):
        x_ctx, y_ctx, x_tgt, y_tgt = sample_eval_episode(cfg, args, device)
        y = y_ctx[0].detach().cpu().double()
        _y_tgt = y_tgt[0].detach().cpu().double()
        _K, Kt, A, T = build_eval_kernels(x_ctx, x_tgt, args)
        Ktt = build_eval_target_kernel(x_tgt, args)
        A_factors = symmetric_eig_factors(A)

        F_y, H_raw = forward_model(model, x_ctx, x_tgt, y, return_hidden=True)
        assert H_raw is not None
        r_t_strict = effective_rank_T_task(T, A, args.rank_tau)
        risk_rank = effective_rank_T_excess_risk(T, A, Kt, Ktt, args.excess_risk_frac)
        r_t = int(risk_rank["r_eff_T_task"])

        build_probes = sample_probes("label", args.n_build, A, gen)
        _build_fd, M_resp = finite_difference_bundle(
            model, x_ctx, x_tgt, y, build_probes, args.eps, return_hidden=True
        )
        assert M_resp is not None
        eval_probes = sample_probes("label", args.n_eval, A, gen)
        eval_fd, _ = finite_difference_bundle(
            model, x_ctx, x_tgt, y, eval_probes, args.eps, return_hidden=False
        )

        bases = {
            "raw": H_raw,
            "response": M_resp,
        }
        for basis_name, M in bases.items():
            pack = weighted_svd_basis(
                M,
                A_factors,
                args.tau_sv,
                r_max=args.curve_r_max,
                curve_r_max=args.curve_r_max,
            )
            Q_all = pack["Q_all"]
            selected_rank = int(pack["rank"])
            max_rank = min(args.curve_r_max, Q_all.shape[1])
            for rank in range(max_rank + 1):
                Q = Q_all[:, :rank]
                S = galerkin_operator(Kt, Q)
                metrics = operator_metrics(S, T, y, F_y, eval_probes, eval_fd)
                rows.append(
                    {
                        "checkpoint": args.name,
                        "episode": episode,
                        "basis": basis_name,
                        "rank": rank,
                        "selected_rank": selected_rank,
                        "r_eff_T_task": r_t,
                        "r_eff_T_task_strict": r_t_strict,
                        "excess_risk_frac": args.excess_risk_frac,
                        "kernel_lengthscale": args.kernel_lengthscale,
                        "kernel_signal_var": args.kernel_signal_var,
                        **metrics,
                    }
                )
                rows[-1].update(risk_rank)
        print(f"episode {episode + 1}/{args.episodes} rT={r_t} strict={r_t_strict}", flush=True)

    write_csv(out_dir / "rank_curves.csv", rows)
    summary = summarize(rows, ["basis", "rank"])
    write_csv(out_dir / "rank_summary.csv", summary)

    selected = [
        r for r in summary
        if int(r["rank"]) in (0, 5, 10, 15, 16)
    ]
    lines = ["RBF final-state basis diagnostic", ""]
    lines.append(
        f"checkpoint={args.checkpoint}, episodes={args.episodes}, ell={args.kernel_lengthscale}, "
        f"sigma2={args.sigma2}, rank_tau={args.rank_tau}, excess_risk_frac={args.excess_risk_frac}"
    )
    lines.append("")
    lines.append(
        f"{'basis':10s} {'rank':>4s} {'rT':>6s} {'rStrict':>8s} "
        f"{'E(TQ,T)':>10s} {'E(F,TQ)':>10s} {'E(F,T)':>10s}"
    )
    for row in selected:
        lines.append(
            f"{str(row['basis']):10s} {int(row['rank']):4d} "
            f"{float(row.get('r_eff_T_task_mean', float('nan'))):6.2f} "
            f"{float(row.get('r_eff_T_task_strict_mean', float('nan'))):8.2f} "
            f"{float(row.get('E_operator_to_T_mean', float('nan'))):10.5f} "
            f"{float(row.get('E_model_to_operator_mean', float('nan'))):10.5f} "
            f"{float(row.get('E_model_to_T_mean', float('nan'))):10.5f}"
        )
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote results to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
