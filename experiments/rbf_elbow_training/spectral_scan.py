#!/usr/bin/env python3
"""Scan task-visible RBF rank over context/target sample sizes.

This script matches the data geometry used by ``train_rbf_elbow.py``:
inputs are sampled from N(0, I_d), labels come from a fixed RBF GP, and the
operator rank is measured for

    T A^{1/2} = K_t (K + sigma2 I)^{-1} (K + sigma2 I)^{1/2}.

The reported r_T is the smallest rank whose discarded operator energy would add
at most ``excess_risk_frac`` times the KRR posterior risk. The old relative-tail
rank is kept as r_T_strict.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch


FLOOR = 1e-12


def parse_int_list(text: str) -> List[int]:
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return vals


def squared_distances(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    sq1 = (x1 * x1).sum(-1, keepdim=True)
    sq2 = (x2 * x2).sum(-1, keepdim=True)
    return (sq1 + sq2.transpose(-2, -1) - 2.0 * (x1 @ x2.T)).clamp_min(0.0)


def rbf_kernel(x1: torch.Tensor, x2: torch.Tensor, lengthscale: float, signal_var: float) -> torch.Tensor:
    return signal_var * torch.exp(-squared_distances(x1, x2) / (2.0 * lengthscale * lengthscale))


def effective_rank_from_svals(svals: torch.Tensor, eps: float) -> int:
    energy = (svals.double() ** 2).clamp_min(0.0)
    total = float(energy.sum())
    if total <= FLOOR:
        return 0
    for r in range(energy.numel() + 1):
        tail = float(energy[r:].sum()) if r < energy.numel() else 0.0
        if math.sqrt(tail / (total + FLOOR)) <= eps:
            return int(r)
    return int(energy.numel())


def effective_rank_from_tail_budget(svals: torch.Tensor, tail_budget: float) -> int:
    energy = (svals.double() ** 2).clamp_min(0.0)
    budget = max(float(tail_budget), 0.0)
    for r in range(energy.numel() + 1):
        tail = float(energy[r:].sum()) if r < energy.numel() else 0.0
        if tail <= budget + FLOOR:
            return int(r)
    return int(energy.numel())


def soft_ranks(svals: torch.Tensor) -> Dict[str, float]:
    energy = (svals.double() ** 2).clamp_min(0.0)
    total = float(energy.sum())
    if total <= FLOOR:
        return {"participation_rank": 0.0, "entropy_rank": 0.0}
    participation = float((energy.sum() ** 2) / (torch.sum(energy * energy) + FLOOR))
    p = energy / energy.sum()
    entropy = float(torch.exp(-(p * torch.log(p.clamp_min(FLOOR))).sum()))
    return {"participation_rank": participation, "entropy_rank": entropy}


def episode_metrics(
    n_ctx: int,
    n_tgt: int,
    args: argparse.Namespace,
    gen: torch.Generator,
) -> Dict[str, float]:
    x_ctx = torch.randn(n_ctx, args.d_x, dtype=torch.float64, generator=gen)
    x_tgt = torch.randn(n_tgt, args.d_x, dtype=torch.float64, generator=gen)
    k = rbf_kernel(x_ctx, x_ctx, args.kernel_lengthscale, args.kernel_signal_var)
    kt = rbf_kernel(x_tgt, x_ctx, args.kernel_lengthscale, args.kernel_signal_var)
    ktt = rbf_kernel(x_tgt, x_tgt, args.kernel_lengthscale, args.kernel_signal_var)
    eye = torch.eye(n_ctx, dtype=torch.float64)
    a = k + args.sigma2 * eye
    vals, vecs = torch.linalg.eigh(0.5 * (a + a.T))
    vals = vals.clamp_min(1e-12)
    a_sqrt = (vecs * vals.sqrt().unsqueeze(0)) @ vecs.T
    t = kt @ torch.linalg.solve(a, eye)
    svals = torch.linalg.svdvals(t @ a_sqrt)
    energy_total = float(((svals.double() ** 2).clamp_min(0.0)).sum())
    posterior = ktt - kt @ torch.linalg.solve(a, kt.T)
    risk_total = max(float(torch.trace(posterior)), 0.0)
    risk_budget = max(float(args.excess_risk_frac), 0.0) * risk_total
    r_t_strict = effective_rank_from_svals(svals, args.rank_tau)
    r_t = effective_rank_from_tail_budget(svals, risk_budget)
    s = soft_ranks(svals)
    cap = min(n_ctx, n_tgt)
    return {
        "n_ctx": float(n_ctx),
        "n_tgt": float(n_tgt),
        "cap": float(cap),
        "r_T": float(r_t),
        "r_T_strict": float(r_t_strict),
        "r_T_over_cap": float(r_t / cap) if cap else float("nan"),
        "r_T_strict_over_cap": float(r_t_strict / cap) if cap else float("nan"),
        "rank_tau_task": float(math.sqrt(risk_budget / (energy_total + FLOOR))) if energy_total > FLOOR else 0.0,
        "krr_risk_total": float(risk_total),
        "krr_signal_total": float(energy_total),
        "krr_risk_per_tgt": float(risk_total / max(n_tgt, 1)),
        "krr_signal_per_tgt": float(energy_total / max(n_tgt, 1)),
        "krr_risk_over_signal": float(risk_total / (energy_total + FLOOR)),
        "top_sval": float(svals[0]) if svals.numel() else 0.0,
        "tail_at_32": tail_error(svals, 32),
        "tail_at_48": tail_error(svals, 48),
        "tail_at_64": tail_error(svals, 64),
        **s,
    }


def tail_error(svals: torch.Tensor, rank: int) -> float:
    energy = (svals.double() ** 2).clamp_min(0.0)
    total = float(energy.sum())
    if total <= FLOOR:
        return 0.0
    r = min(max(rank, 0), energy.numel())
    tail = float(energy[r:].sum()) if r < energy.numel() else 0.0
    return math.sqrt(tail / (total + FLOOR))


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


def summarize(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[tuple[int, int], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(int(row["n_ctx"]), int(row["n_tgt"]))].append(row)
    fields = [
        "r_T",
        "r_T_strict",
        "r_T_over_cap",
        "r_T_strict_over_cap",
        "rank_tau_task",
        "krr_risk_per_tgt",
        "krr_signal_per_tgt",
        "krr_risk_over_signal",
        "participation_rank",
        "entropy_rank",
        "tail_at_32",
        "tail_at_48",
        "tail_at_64",
    ]
    out: List[Dict[str, object]] = []
    for (n_ctx, n_tgt), vals in sorted(groups.items()):
        rec: Dict[str, object] = {
            "n_ctx": n_ctx,
            "n_tgt": n_tgt,
            "episodes": len(vals),
            "cap": min(n_ctx, n_tgt),
        }
        for field in fields:
            arr = np.array([float(v[field]) for v in vals], dtype=float)
            rec[f"{field}_mean"] = float(arr.mean())
            rec[f"{field}_std"] = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
            rec[f"{field}_p10"] = float(np.quantile(arr, 0.10))
            rec[f"{field}_p90"] = float(np.quantile(arr, 0.90))
        out.append(rec)
    return out


def write_analysis(path: Path, summary: Sequence[Dict[str, object]], args: argparse.Namespace) -> None:
    lines = [
        "RBF task-visible rank spectral scan",
        "",
        f"d_x={args.d_x}, ell={args.kernel_lengthscale}, sigma2={args.sigma2}, "
        f"rank_tau={args.rank_tau}, excess_risk_frac={args.excess_risk_frac}",
        f"episodes={args.episodes}, x distribution=N(0,I)",
        "",
        "Grid summary:",
        "n_ctx n_tgt cap   r_T_mean strict  r_T/cap  tau_eq  risk/tgt sig/tgt  p_rank  ent_rank  tail32  tail48  tail64",
    ]
    for row in summary:
        lines.append(
            f"{int(row['n_ctx']):5d} {int(row['n_tgt']):5d} {int(row['cap']):3d} "
            f"{float(row['r_T_mean']):8.2f} "
            f"{float(row['r_T_strict_mean']):6.2f} "
            f"{float(row['r_T_over_cap_mean']):8.3f} "
            f"{float(row['rank_tau_task_mean']):7.3f} "
            f"{float(row['krr_risk_per_tgt_mean']):8.4f} "
            f"{float(row['krr_signal_per_tgt_mean']):7.4f} "
            f"{float(row['participation_rank_mean']):7.2f} "
            f"{float(row['entropy_rank_mean']):8.2f} "
            f"{float(row['tail_at_32_mean']):7.3f} "
            f"{float(row['tail_at_48_mean']):7.3f} "
            f"{float(row['tail_at_64_mean']):7.3f}"
        )
    lines.extend(
        [
            "",
            "Selection rule used here:",
            "Pick the smallest context/target setting where increasing n_tgt no longer changes r_T much,",
            "then increase n_ctx only if we explicitly want a harder source-side operator.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="experiments/rbf_elbow_training/results/spectral_scan")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--d-x", type=int, default=5)
    parser.add_argument("--n-ctx-list", type=parse_int_list, default=parse_int_list("47,96,128,192,256"))
    parser.add_argument("--n-tgt-list", type=parse_int_list, default=parse_int_list("64,128,256"))
    parser.add_argument("--sigma2", type=float, default=0.1)
    parser.add_argument("--kernel-lengthscale", type=float, default=3.0)
    parser.add_argument("--kernel-signal-var", type=float, default=1.0)
    parser.add_argument("--rank-tau", type=float, default=0.01)
    parser.add_argument("--excess-risk-frac", type=float, default=0.05)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    rows: List[Dict[str, object]] = []
    for n_ctx in args.n_ctx_list:
        for n_tgt in args.n_tgt_list:
            for episode in range(args.episodes):
                row = episode_metrics(n_ctx, n_tgt, args, gen)
                row["episode"] = episode
                rows.append(row)
            print(f"done n_ctx={n_ctx} n_tgt={n_tgt}", flush=True)
    summary = summarize(rows)
    write_csv(out_dir / "records.csv", rows)
    write_csv(out_dir / "summary.csv", summary)
    write_analysis(out_dir / "analysis.txt", summary, args)
    print(out_dir / "analysis.txt", flush=True)


if __name__ == "__main__":
    main()
