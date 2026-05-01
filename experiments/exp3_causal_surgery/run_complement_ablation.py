#!/usr/bin/env python3
"""Complement ablation test for the final-state Q subspace.

This script adds a mediation-style check to Experiment 3. At a chosen residual
state, it compares:

  base:
      unmodified model;
  keep_Q:
      keep only the A-projection of context-token residual states onto Q;
  remove_Q:
      remove the A-projection of context-token residual states onto Q.

The interventions are

    keep_Q:   H_ctx <- Q Q^T A H_ctx,
    remove_Q: H_ctx <- H_ctx - Q Q^T A H_ctx,

where Q is the raw final activation basis and Q^T A Q = I.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "shared"))
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mpl-cache"))

from experiments.exp3_causal_surgery.run import (
    FLOOR,
    ModelCfg,
    build_kernels,
    extract_raw_final_basis,
    fd_operator_error,
    get_device,
    load_model,
    sample_episode,
    sample_task_probes,
    set_seed,
)


@torch.no_grad()
def states_for_labels(model, x_ctx, y_cpu: torch.Tensor, x_tgt):
    from experiments.exp3_causal_surgery.run import forward_base_with_states

    return forward_base_with_states(model, x_ctx, y_cpu, x_tgt)


@torch.no_grad()
def roll_from_state(model, h: torch.Tensor, n_ctx: int, state_idx: int) -> torch.Tensor:
    """Roll the transformer from residual state index `state_idx` to predictions."""
    for layer_idx in range(state_idx, len(model.layers)):
        h = model.layers[layer_idx](h)
    return model.head(h[:, n_ctx:, :]).squeeze(-1)[0].detach().cpu().double()


@torch.no_grad()
def project_context_component(
    h: torch.Tensor,
    n_ctx: int,
    A_cpu: torch.Tensor,
    Q_cpu: torch.Tensor,
) -> torch.Tensor:
    """Return the A-projection Q Q^T A H_ctx as a full residual state."""
    h_proj = h.clone()
    if Q_cpu.numel() == 0 or Q_cpu.shape[1] == 0:
        h_proj[0, :n_ctx, :] = 0
        return h_proj

    dtype = h.dtype
    device = h.device
    Q = Q_cpu.to(device=device, dtype=dtype)
    A = A_cpu.to(device=device, dtype=dtype)
    Hctx = h[0, :n_ctx, :]
    h_proj[0, :n_ctx, :] = Q @ (Q.T @ (A @ Hctx))
    return h_proj


@torch.no_grad()
def keep_q_state(h: torch.Tensor, n_ctx: int, A_cpu: torch.Tensor, Q_cpu: torch.Tensor) -> torch.Tensor:
    h_new = h.clone()
    h_new[0, :n_ctx, :] = project_context_component(h, n_ctx, A_cpu, Q_cpu)[0, :n_ctx, :]
    return h_new


@torch.no_grad()
def remove_q_state(h: torch.Tensor, n_ctx: int, A_cpu: torch.Tensor, Q_cpu: torch.Tensor) -> torch.Tensor:
    h_new = h.clone()
    proj = project_context_component(h, n_ctx, A_cpu, Q_cpu)
    h_new[0, :n_ctx, :] = h[0, :n_ctx, :] - proj[0, :n_ctx, :]
    return h_new


def fd_from_columns(cols: Sequence[torch.Tensor]) -> torch.Tensor:
    if not cols:
        return torch.zeros(0, 0, dtype=torch.float64)
    return torch.cat([c.reshape(-1, 1) for c in cols], dim=1)


def rel_point_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float((pred - target).norm() / (target.norm() + FLOOR))


def rel_drift(pred: torch.Tensor, base: torch.Tensor) -> float:
    return float((pred - base).norm() / (base.norm() + FLOOR))


def fd_damage_error(fd_surg: torch.Tensor, fd_base: torch.Tensor, T: torch.Tensor, probes: torch.Tensor) -> float:
    U = probes.T
    TU = T @ U
    denom = float((TU * TU).sum()) + FLOOR
    return math.sqrt(float(((fd_surg - fd_base) ** 2).sum()) / denom)


def mean_std(vals: Sequence[float]) -> Tuple[float, float]:
    x = torch.tensor(list(vals), dtype=torch.float64)
    if x.numel() == 0:
        return float("nan"), float("nan")
    if x.numel() == 1:
        return float(x.mean()), 0.0
    return float(x.mean()), float(x.std(unbiased=True))


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["name"])].append(row)

    metrics = ["point_err", "point_drift", "E_F_T", "E_F_TQ", "damage_vs_base"]
    out: List[Dict[str, object]] = []
    for name, group in sorted(grouped.items()):
        rec: Dict[str, object] = {"name": name, "n": len(group)}
        for metric in metrics:
            mu, sd = mean_std([float(r[metric]) for r in group])
            rec[f"{metric}_mean"] = mu
            rec[f"{metric}_std"] = sd
        out.append(rec)
    return out


def process_episode(model, cfg: ModelCfg, args, ep: int, device: torch.device, gen: torch.Generator):
    x_ctx, y_ctx, x_tgt, _y_tgt = sample_episode(cfg, args.sigma2, args.n_ctx, args.n_tgt, device)
    y = y_ctx[0].detach().cpu().double()

    _K, Kt, A, T = build_kernels(x_ctx, x_tgt, args.sigma2)
    Ty = T @ y

    base_pred, base_states = states_for_labels(model, x_ctx, y, x_tgt)
    Q, _svals = extract_raw_final_basis(base_states[-1], args.n_ctx, A, args.tau_sv, args.rmax)
    if Q.shape[1] == 0:
        return []

    TQ = Kt @ Q @ Q.T
    probes = sample_task_probes(A, args.n_eval, gen)
    state_idx = args.state_idx

    if state_idx < 0 or state_idx > cfg.n_layers:
        raise ValueError(f"state-idx must be in [0,{cfg.n_layers}], got {state_idx}")

    keep_pred = roll_from_state(
        model,
        keep_q_state(base_states[state_idx], args.n_ctx, A, Q),
        args.n_ctx,
        state_idx,
    )
    remove_pred = roll_from_state(
        model,
        remove_q_state(base_states[state_idx], args.n_ctx, A, Q),
        args.n_ctx,
        state_idx,
    )

    fd_cols: Dict[str, List[torch.Tensor]] = {"base": [], "keep_Q": [], "remove_Q": []}
    for u in probes:
        plus = y + args.eps * u
        minus = y - args.eps * u
        pred_p, states_p = states_for_labels(model, x_ctx, plus, x_tgt)
        pred_m, states_m = states_for_labels(model, x_ctx, minus, x_tgt)
        fd_cols["base"].append((pred_p - pred_m) / (2.0 * args.eps))

        keep_p = roll_from_state(model, keep_q_state(states_p[state_idx], args.n_ctx, A, Q), args.n_ctx, state_idx)
        keep_m = roll_from_state(model, keep_q_state(states_m[state_idx], args.n_ctx, A, Q), args.n_ctx, state_idx)
        fd_cols["keep_Q"].append((keep_p - keep_m) / (2.0 * args.eps))

        rem_p = roll_from_state(model, remove_q_state(states_p[state_idx], args.n_ctx, A, Q), args.n_ctx, state_idx)
        rem_m = roll_from_state(model, remove_q_state(states_m[state_idx], args.n_ctx, A, Q), args.n_ctx, state_idx)
        fd_cols["remove_Q"].append((rem_p - rem_m) / (2.0 * args.eps))

    fd = {name: fd_from_columns(cols) for name, cols in fd_cols.items()}
    preds = {"base": base_pred, "keep_Q": keep_pred, "remove_Q": remove_pred}

    rows: List[Dict[str, object]] = []
    for name in ["base", "keep_Q", "remove_Q"]:
        rows.append(
            {
                "episode": ep,
                "state_idx": state_idx,
                "rank_Q": Q.shape[1],
                "name": name,
                "point_err": rel_point_error(preds[name], Ty),
                "point_drift": rel_drift(preds[name], base_pred),
                "E_F_T": fd_operator_error(fd[name], T, T, probes),
                "E_F_TQ": fd_operator_error(fd[name], TQ, T, probes),
                "damage_vs_base": fd_damage_error(fd[name], fd["base"], T, probes),
            }
        )
    return rows


def write_summary_txt(path: Path, rows: Sequence[Dict[str, object]], summary: Sequence[Dict[str, object]]) -> None:
    lines = ["Complement ablation of Q", ""]
    lines.append(f"records={len(rows)}")
    lines.append("")
    lines.append(f"{'name':12s} {'n':>4s} {'pt_err':>10s} {'drift':>10s} {'E(F,T)':>10s} {'E(F,TQ)':>10s} {'damage':>10s}")
    for row in summary:
        lines.append(
            f"{str(row['name']):12s} {int(row['n']):4d} "
            f"{float(row['point_err_mean']):10.5f} "
            f"{float(row['point_drift_mean']):10.5f} "
            f"{float(row['E_F_T_mean']):10.5f} "
            f"{float(row['E_F_TQ_mean']):10.5f} "
            f"{float(row['damage_vs_base_mean']):10.5f}"
        )
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="model_L8.pt")
    p.add_argument("--d-x", type=int, default=5)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--n-ctx", type=int, default=47)
    p.add_argument("--n-tgt", type=int, default=16)
    p.add_argument("--n-eval", type=int, default=16)
    p.add_argument("--state-idx", type=int, default=1)
    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--sigma2", type=float, default=0.1)
    p.add_argument("--tau-sv", type=float, default=1e-3)
    p.add_argument("--rmax", type=int, default=12)
    p.add_argument("--seed", type=int, default=31415)
    p.add_argument("--results-dir", default="experiments/exp3_causal_surgery/results_complement_ablation")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    cfg = ModelCfg(args.checkpoint, args.d_x, args.d_model, args.n_layers, args.n_heads)
    model = load_model(cfg, device)
    gen = torch.Generator(device="cpu").manual_seed(args.seed + 123)

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, object]] = []
    for ep in range(args.episodes):
        print(f"episode {ep + 1}/{args.episodes}", flush=True)
        all_rows.extend(process_episode(model, cfg, args, ep, device, gen))

    summary = summarize(all_rows)
    write_csv(out_dir / "records.csv", all_rows)
    write_csv(out_dir / "summary.csv", summary)
    write_summary_txt(out_dir / "summary.txt", all_rows, summary)
    print((out_dir / "summary.txt").read_text())


if __name__ == "__main__":
    main()
