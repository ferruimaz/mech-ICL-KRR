#!/usr/bin/env python3
"""Experiment 2 variant: data-selected native innovation rank.

This keeps the original Experiment 2 intact and replaces only the per-layer
hand cap with data-driven criteria.

For each layer ell:

    N_build = (I - P_pre) M_build[ell]
    N_val   = (I - P_pre) M_val[ell]

The candidate directions are the A-weighted SVD directions of N_build.

In hidden-validation mode, s_ell is the smallest prefix whose held-out capture
reaches gamma:

    capture_s =
        (||N_val||_A^2 - ||(I-P_s)N_val||_A^2)
        / (||N_val||_A^2 - ||(I-P_full)N_val||_A^2).

In causal mode, each candidate direction is ablated from the context residual
stream after layer ell, the remaining model is rolled forward, and the direction
is kept only if it causes held-out finite-difference response damage above a
predeclared threshold.

After selecting all s_ell, the script freezes either the seed space
span(S_0,...,S_L) or the older A-reachable space, and evaluates the same
Galerkin operator metrics as the original Experiment 2.
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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SUPPORT_DIR = REPO_ROOT / "experiments" / "shared"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SUPPORT_DIR))
MPL_CACHE_DIR = Path(os.environ.get("TMPDIR", "/tmp")) / "mech_icl_krr_mpl_cache"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from support import get_device, set_seed
from experiments.shared.support import compute_kernel

# Reuse the original Experiment 2 mechanics without modifying it.
from experiments.exp2_budget_closure.run import (  # noqa: E402
    CHECKPOINTS_2A,
    CHECKPOINTS_2B,
    CkptCfg,
    FLOOR,
    a_norm_fro,
    a_orth_basis_from_cols,
    a_project,
    a_residual,
    build_kernels,
    effective_rank_T_excess_risk,
    effective_rank_T_task,
    eval_operator_error,
    forward_with_ctx_hidden,
    hidden_response_matrices,
    load_model,
    model_to_operator_error,
    mse_and_pointwise,
    operator_error,
    prediction_fd_bundle,
    psd_sqrt_and_invsqrt,
    reachable_basis,
    sample_episode,
    sample_probes,
    sample_task_probes,
)


CHECKPOINTS_RBF: List[CkptCfg] = [
    CkptCfg("rbf_l3", "model_rbf_fixed_l3.pt", 5, 128, 8, 4, "rbf", 3.0),
    CkptCfg("rbf_l3_smoke", "model_rbf_fixed_l3_smoke.pt", 5, 128, 8, 4, "rbf", 3.0),
]


def a_norm_fro_sq(M: torch.Tensor, A: torch.Tensor) -> float:
    if M.numel() == 0:
        return 0.0
    AM = A @ M
    return max(float((M * AM).sum()), 0.0)


def spectral_soft_ranks(svals: torch.Tensor) -> Dict[str, float]:
    if svals.numel() == 0:
        return {"participation_rank": 0.0, "entropy_rank": 0.0}
    e = (svals.double() ** 2).clamp_min(0.0)
    total = float(e.sum())
    if total <= 0:
        return {"participation_rank": 0.0, "entropy_rank": 0.0}
    participation = float((e.sum() ** 2) / (torch.sum(e * e) + FLOOR))
    p = e / e.sum()
    entropy = float(torch.exp(-(p * torch.log(p.clamp_min(FLOOR))).sum()))
    return {"participation_rank": participation, "entropy_rank": entropy}


def seed_basis(
    A: torch.Tensor,
    A_sqrt: torch.Tensor,
    A_invsqrt: torch.Tensor,
    S_list: Sequence[torch.Tensor],
    final_k: int,
    tau_subspace: float,
) -> torch.Tensor:
    n = A.shape[0]
    if final_k < 0:
        return torch.zeros(n, 0, dtype=torch.float64)

    cols = [S for S in S_list[: final_k + 1] if S.numel() > 0 and S.shape[1] > 0]
    if not cols:
        return torch.zeros(n, 0, dtype=torch.float64)

    M = torch.cat(cols, dim=1)
    Q, _ = a_orth_basis_from_cols(M, A_sqrt, A_invsqrt, tau_rel=tau_subspace, rmax=n)
    return Q.contiguous()


def layer_span_basis(
    A: torch.Tensor,
    A_sqrt: torch.Tensor,
    A_invsqrt: torch.Tensor,
    S_list: Sequence[torch.Tensor],
    final_k: int,
    tau_subspace: float,
    span_mode: str,
) -> torch.Tensor:
    if span_mode == "seed":
        return seed_basis(A, A_sqrt, A_invsqrt, S_list, final_k, tau_subspace)
    if span_mode == "reachable":
        return reachable_basis(A, A_sqrt, A_invsqrt, S_list, final_k, tau_subspace)
    raise ValueError(f"unknown span_mode: {span_mode}")


def select_rank_by_validation(
    Q_cand: torch.Tensor,
    N_val: torch.Tensor,
    A: torch.Tensor,
    gamma: float,
    min_explainable_frac: float,
) -> Dict[str, float]:
    r = Q_cand.shape[1]
    total_sq = a_norm_fro_sq(N_val, A)
    total_norm = math.sqrt(total_sq)

    empty = {
        "s": 0.0,
        "candidate_rank": float(r),
        "val_total_norm": total_norm,
        "val_explainable_frac": 0.0,
        "val_capture": 0.0,
        "val_resid_ratio": 1.0 if total_sq > FLOOR else 0.0,
        "val_full_resid_ratio": 1.0 if total_sq > FLOOR else 0.0,
    }

    if r == 0 or total_sq <= FLOOR:
        return empty

    full_resid_sq = a_norm_fro_sq(a_residual(Q_cand, N_val, A), A)
    explainable_sq = max(total_sq - full_resid_sq, 0.0)
    explainable_frac = explainable_sq / (total_sq + FLOOR)
    full_resid_ratio = math.sqrt(full_resid_sq / (total_sq + FLOOR))

    if explainable_frac < min_explainable_frac:
        empty.update(
            {
                "val_explainable_frac": explainable_frac,
                "val_full_resid_ratio": full_resid_ratio,
            }
        )
        return empty

    best: Optional[Dict[str, float]] = None
    for s in range(1, r + 1):
        Q_s = Q_cand[:, :s]
        resid_sq = a_norm_fro_sq(a_residual(Q_s, N_val, A), A)
        captured_sq = max(total_sq - resid_sq, 0.0)
        capture = captured_sq / (explainable_sq + FLOOR)
        resid_ratio = math.sqrt(resid_sq / (total_sq + FLOOR))
        best = {
            "s": float(s),
            "candidate_rank": float(r),
            "val_total_norm": total_norm,
            "val_explainable_frac": explainable_frac,
            "val_capture": max(0.0, min(capture, 1.0)),
            "val_resid_ratio": resid_ratio,
            "val_full_resid_ratio": full_resid_ratio,
        }
        if capture >= gamma:
            return best

    assert best is not None
    return best


@torch.no_grad()
def make_tokens(
    model: torch.nn.Module,
    x_ctx: torch.Tensor,
    y_cpu: torch.Tensor,
    x_tgt: torch.Tensor,
) -> Tuple[torch.Tensor, int]:
    dtype = next(model.parameters()).dtype
    device = x_ctx.device
    y = y_cpu.view(1, -1).to(device=device, dtype=dtype)
    B, n_ctx, _ = x_ctx.shape
    n_tgt = x_tgt.shape[1]

    ctx = torch.cat(
        [
            x_ctx,
            y.unsqueeze(-1),
            torch.zeros(B, n_ctx, 1, device=device, dtype=x_ctx.dtype),
        ],
        dim=-1,
    )
    tgt = torch.cat(
        [
            x_tgt,
            torch.zeros(B, n_tgt, 1, device=device, dtype=x_tgt.dtype),
            torch.ones(B, n_tgt, 1, device=device, dtype=x_tgt.dtype),
        ],
        dim=-1,
    )
    h = model.embed(torch.cat([ctx, tgt], dim=1))
    return h, n_ctx


@torch.no_grad()
def apply_context_projection_removal(
    h: torch.Tensor,
    n_ctx: int,
    A_cpu: torch.Tensor,
    D_cpu: torch.Tensor,
) -> torch.Tensor:
    if D_cpu.numel() == 0 or D_cpu.shape[1] == 0:
        return h

    dtype = h.dtype
    device = h.device
    D = D_cpu.to(device=device, dtype=dtype)
    A = A_cpu.to(device=device, dtype=dtype)

    h_new = h.clone()
    Hctx = h_new[0, :n_ctx, :]
    coeff = D.T @ (A @ Hctx)
    h_new[0, :n_ctx, :] = Hctx - D @ coeff
    return h_new


@torch.no_grad()
def forward_with_surgery(
    model: torch.nn.Module,
    x_ctx: torch.Tensor,
    y_cpu: torch.Tensor,
    x_tgt: torch.Tensor,
    A_cpu: torch.Tensor,
    D_cpu: torch.Tensor,
    state_idx: int,
) -> torch.Tensor:
    h, n_ctx = make_tokens(model, x_ctx, y_cpu, x_tgt)
    L = len(model.layers)

    if state_idx < 0 or state_idx > L:
        raise ValueError(f"state_idx must be in [0,{L}], got {state_idx}")

    if state_idx == 0:
        h = apply_context_projection_removal(h, n_ctx, A_cpu, D_cpu)

    for i, layer in enumerate(model.layers):
        h = layer(h)
        if state_idx == i + 1:
            h = apply_context_projection_removal(h, n_ctx, A_cpu, D_cpu)

    pred = model.head(h[:, n_ctx:, :]).squeeze(-1)[0].detach().cpu().double()
    return pred


@torch.no_grad()
def fd_bundle_surgery(
    model: torch.nn.Module,
    x_ctx: torch.Tensor,
    x_tgt: torch.Tensor,
    y: torch.Tensor,
    probes: torch.Tensor,
    eps: float,
    A: torch.Tensor,
    D: torch.Tensor,
    state_idx: int,
) -> torch.Tensor:
    cols: List[torch.Tensor] = []
    for u in probes:
        pp = forward_with_surgery(model, x_ctx, y + eps * u, x_tgt, A, D, state_idx)
        pn = forward_with_surgery(model, x_ctx, y - eps * u, x_tgt, A, D, state_idx)
        cols.append(((pp - pn) / (2.0 * eps)).reshape(-1, 1))
    return torch.cat(cols, dim=1) if cols else torch.zeros(x_tgt.shape[1], 0, dtype=torch.float64)


def fd_response_damage(fd_surg: torch.Tensor, fd_base: torch.Tensor) -> float:
    denom = math.sqrt(float((fd_base * fd_base).sum())) + FLOOR
    return math.sqrt(float(((fd_surg - fd_base) ** 2).sum())) / denom


def select_directions_by_causal_damage(
    model: torch.nn.Module,
    x_ctx: torch.Tensor,
    x_tgt: torch.Tensor,
    y: torch.Tensor,
    A: torch.Tensor,
    Q_cand: torch.Tensor,
    state_idx: int,
    causal_probes: torch.Tensor,
    fd_base: torch.Tensor,
    eps: float,
    damage_tau: float,
    max_keep: int,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    r = Q_cand.shape[1]
    if r == 0:
        return Q_cand, {
            "causal_max_damage": 0.0,
            "causal_mean_damage": 0.0,
            "causal_min_kept_damage": 0.0,
            "causal_sum_kept_damage": 0.0,
            "causal_selected_indices_json": "[]",
            "causal_damages_json": "[]",
        }

    damages: List[float] = []
    for j in range(r):
        D = Q_cand[:, j : j + 1]
        fd_surg = fd_bundle_surgery(
            model,
            x_ctx,
            x_tgt,
            y,
            causal_probes,
            eps,
            A,
            D,
            state_idx,
        )
        damages.append(fd_response_damage(fd_surg, fd_base))

    selected = [j for j, damage in enumerate(damages) if damage >= damage_tau]
    if max_keep > 0 and len(selected) > max_keep:
        selected = sorted(selected, key=lambda idx: damages[idx], reverse=True)[:max_keep]
        selected = sorted(selected)

    if selected:
        S_ell = Q_cand[:, selected].contiguous()
        kept = [damages[j] for j in selected]
        min_kept = min(kept)
        sum_kept = sum(kept)
    else:
        S_ell = Q_cand[:, :0].contiguous()
        min_kept = 0.0
        sum_kept = 0.0

    return S_ell, {
        "causal_max_damage": max(damages),
        "causal_mean_damage": float(np.mean(damages)),
        "causal_min_kept_damage": min_kept,
        "causal_sum_kept_damage": sum_kept,
        "causal_selected_indices_json": json.dumps(selected),
        "causal_damages_json": json.dumps(damages),
    }


def closure_and_refinement(
    A: torch.Tensor,
    A_sqrt: torch.Tensor,
    A_invsqrt: torch.Tensor,
    S_list: Sequence[torch.Tensor],
    M_layers: Sequence[torch.Tensor],
    tau_subspace: float,
    span_mode: str,
) -> Tuple[List[float], List[float]]:
    L = len(M_layers) - 1
    n = A.shape[0]
    closure: List[float] = []
    refinement: List[float] = []

    for k in range(L + 1):
        M_k = M_layers[k]
        denom = a_norm_fro(M_k, A) + FLOOR

        Q_k = layer_span_basis(A, A_sqrt, A_invsqrt, S_list, k, tau_subspace, span_mode)
        residual = a_residual(Q_k, M_k, A)
        closure.append(a_norm_fro(residual, A) / denom)

        if k == 0:
            refinement.append(0.0)
            continue

        Q_prev = layer_span_basis(A, A_sqrt, A_invsqrt, S_list, k - 1, tau_subspace, span_mode)
        if Q_prev.shape[1] == 0:
            refinement.append(0.0)
            continue

        B_cols = torch.cat([Q_prev, A @ Q_prev], dim=1)
        B_k, _ = a_orth_basis_from_cols(
            B_cols,
            A_sqrt,
            A_invsqrt,
            tau_rel=tau_subspace,
            rmax=n,
        )
        proj = a_project(B_k, M_k, A)
        refinement.append(a_norm_fro(proj, A) / denom)

    return closure, refinement


def build_validated_native_basis(
    A: torch.Tensor,
    M_build_layers: List[torch.Tensor],
    M_val_layers: List[torch.Tensor],
    tau_sv: float,
    tau_subspace: float,
    gamma: float,
    min_explainable_frac: float,
    max_candidate_rank: Optional[int],
    selection_mode: str,
    model: Optional[torch.nn.Module] = None,
    x_ctx: Optional[torch.Tensor] = None,
    x_tgt: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    causal_probes: Optional[torch.Tensor] = None,
    causal_fd_base: Optional[torch.Tensor] = None,
    causal_eps: float = 1e-3,
    causal_damage_tau: float = 0.01,
    causal_max_keep: int = 0,
    span_mode: str = "seed",
) -> Dict[str, object]:
    A_sqrt, A_invsqrt = psd_sqrt_and_invsqrt(A)
    L = len(M_build_layers) - 1
    n = A.shape[0]

    S_list: List[torch.Tensor] = []
    s_list: List[int] = []
    candidate_ranks: List[int] = []
    layer_stats: List[Dict[str, object]] = []

    rmax = max_candidate_rank if max_candidate_rank and max_candidate_rank > 0 else n

    if selection_mode == "causal":
        if model is None or x_ctx is None or x_tgt is None or y is None:
            raise ValueError("causal mode requires model, x_ctx, x_tgt, and y")
        if causal_probes is None or causal_fd_base is None:
            raise ValueError("causal mode requires causal_probes and causal_fd_base")

    for ell in range(L + 1):
        Q_pre = layer_span_basis(
            A,
            A_sqrt,
            A_invsqrt,
            S_list,
            ell - 1,
            tau_subspace,
            span_mode,
        )

        N_build = a_residual(Q_pre, M_build_layers[ell], A)
        N_val = a_residual(Q_pre, M_val_layers[ell], A)

        Q_cand, svals = a_orth_basis_from_cols(
            N_build,
            A_sqrt,
            A_invsqrt,
            tau_rel=tau_sv,
            rmax=rmax,
        )
        hidden_choice = select_rank_by_validation(
            Q_cand,
            N_val,
            A,
            gamma=gamma,
            min_explainable_frac=min_explainable_frac,
        )

        causal_stats: Dict[str, object] = {
            "causal_max_damage": 0.0,
            "causal_mean_damage": 0.0,
            "causal_min_kept_damage": 0.0,
            "causal_sum_kept_damage": 0.0,
            "causal_selected_indices_json": "[]",
            "causal_damages_json": "[]",
        }
        if selection_mode == "hidden":
            s = int(hidden_choice["s"])
            S_ell = Q_cand[:, :s].contiguous()
        elif selection_mode == "causal":
            assert model is not None
            assert x_ctx is not None
            assert x_tgt is not None
            assert y is not None
            assert causal_probes is not None
            assert causal_fd_base is not None
            S_ell, causal_stats = select_directions_by_causal_damage(
                model=model,
                x_ctx=x_ctx,
                x_tgt=x_tgt,
                y=y,
                A=A,
                Q_cand=Q_cand,
                state_idx=ell,
                causal_probes=causal_probes,
                fd_base=causal_fd_base,
                eps=causal_eps,
                damage_tau=causal_damage_tau,
                max_keep=causal_max_keep,
            )
            s = S_ell.shape[1]
        else:
            raise ValueError(f"unknown selection_mode: {selection_mode}")

        S_list.append(S_ell)
        s_list.append(s)
        candidate_ranks.append(Q_cand.shape[1])

        soft = spectral_soft_ranks(svals)
        top = [float(x) for x in svals[: min(8, svals.numel())]]
        layer_stats.append(
            {
                "ell": float(ell),
                "selection_mode": selection_mode,
                "s": float(s),
                "hidden_s": hidden_choice["s"],
                "candidate_rank": float(Q_cand.shape[1]),
                "participation_rank": soft["participation_rank"],
                "entropy_rank": soft["entropy_rank"],
                "val_total_norm": hidden_choice["val_total_norm"],
                "val_explainable_frac": hidden_choice["val_explainable_frac"],
                "val_capture": hidden_choice["val_capture"],
                "val_resid_ratio": hidden_choice["val_resid_ratio"],
                "val_full_resid_ratio": hidden_choice["val_full_resid_ratio"],
                "top_singular_values_json": json.dumps(top),
                **causal_stats,
            }
        )

    Q_nat = layer_span_basis(A, A_sqrt, A_invsqrt, S_list, L, tau_subspace, span_mode)
    if span_mode == "seed":
        B_nat = int(sum(s_list))
    elif span_mode == "reachable":
        B_nat = int(sum(s_list[ell] * (L - ell + 1) for ell in range(L + 1)))
    else:
        raise ValueError(f"unknown span_mode: {span_mode}")

    build_closure, build_refinement = closure_and_refinement(
        A,
        A_sqrt,
        A_invsqrt,
        S_list,
        M_build_layers,
        tau_subspace,
        span_mode,
    )
    val_closure, val_refinement = closure_and_refinement(
        A,
        A_sqrt,
        A_invsqrt,
        S_list,
        M_val_layers,
        tau_subspace,
        span_mode,
    )

    return {
        "Q_nat": Q_nat,
        "S_list": S_list,
        "s_list": s_list,
        "candidate_ranks": candidate_ranks,
        "B_nat": B_nat,
        "dim_R_nat": Q_nat.shape[1],
        "build_closure": build_closure,
        "build_refinement": build_refinement,
        "val_closure": val_closure,
        "val_refinement": val_refinement,
        "layer_stats": layer_stats,
    }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: List[Dict[str, object]], group_keys: List[str]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[object, ...], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k) for k in group_keys)].append(row)

    num_keys = {
        key
        for row in rows
        for key, value in row.items()
        if isinstance(value, (int, float)) and key != "episode"
    }

    out: List[Dict[str, object]] = []
    for group_tuple, group_rows in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        out_row: Dict[str, object] = {key: value for key, value in zip(group_keys, group_tuple)}
        out_row["n"] = len(group_rows)
        for key in sorted(num_keys):
            vals = [float(r[key]) for r in group_rows if key in r and r[key] not in ("", None)]
            if not vals:
                continue
            out_row[f"{key}_mean"] = float(np.mean(vals))
            out_row[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        out.append(out_row)
    return out


def write_summary(path: Path, summary: List[Dict[str, object]], args: argparse.Namespace) -> None:
    lines = ["Experiment 2 Validated Native Rank", ""]
    lines.append(f"selection_mode={args.selection_mode}")
    lines.append(f"span_mode={args.span_mode}")
    lines.append(
        f"episodes={args.episodes}, n_build={args.n_build}, n_val={args.n_val}, "
        f"n_eval={args.n_eval}, gamma={args.gamma}, min_explainable_frac={args.min_explainable_frac}"
    )
    if args.selection_mode == "causal":
        lines.append(
            f"n_causal={args.n_causal}, causal_damage_tau={args.causal_damage_tau}, "
            f"causal_max_keep={args.causal_max_keep}, max_candidate_rank={args.max_candidate_rank}"
        )
    lines.append(
        f"n_ctx={args.n_ctx}, n_tgt={args.n_tgt}, sigma2={args.sigma2}, "
        f"tau_sv={args.tau_sv}, tau_subspace={args.tau_subspace}, "
        f"excess_risk_frac={args.excess_risk_frac}"
    )
    lines.append("")
    lines.append(
        f"{'name':6s} {'probe':5s} {'L':3s} {'H':3s} {'B_nat':8s} {'dim':6s} "
        f"{'rT':5s} {'rStrict':7s} {'dim/rT':7s} {'mean_s':7s} {'E(TQ,T)':10s} "
        f"{'E(F,TQ)':10s} {'E(F,T)':10s} {'MSE/KRR':9s} {'valClos':8s} {'maxDmg':8s}"
    )

    for row in sorted(
        summary,
        key=lambda r: (
            str(r.get("probe_kind", "")),
            str(r.get("sweep", "")),
            float(r.get("sweep_val_mean", 0.0)),
        ),
    ):
        dim = float(row.get("dim_R_nat_mean", float("nan")))
        r_t = float(row.get("r_eff_T_task_mean", float("nan")))
        r_strict = float(row.get("r_eff_T_task_strict_mean", float("nan")))
        ratio = dim / r_t if math.isfinite(dim) and math.isfinite(r_t) and r_t > 0 else float("nan")
        lines.append(
            f"{str(row['checkpoint']):6s} {str(row['probe_kind']):5s} "
            f"{row.get('n_layers_mean', float('nan')):3.0f} "
            f"{row.get('n_heads_mean', float('nan')):3.0f} "
            f"{row.get('B_nat_mean', float('nan')):8.1f} "
            f"{dim:6.1f} "
            f"{r_t:5.1f} "
            f"{r_strict:7.1f} "
            f"{ratio:7.3f} "
            f"{row.get('mean_s_mean', float('nan')):7.2f} "
            f"{row.get('E_TQ_T_mean', float('nan')):10.5f} "
            f"{row.get('E_F_TQ_mean', float('nan')):10.5f} "
            f"{row.get('E_F_T_mean', float('nan')):10.5f} "
            f"{row.get('mse_ratio_mean', float('nan')):9.3f} "
            f"{row.get('max_val_closure_mean', float('nan')):8.5f} "
            f"{row.get('max_causal_damage_mean', float('nan')):8.5f}"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_summary(summary: List[Dict[str, object]], out_path: Path, probe_kind: str) -> None:
    rows = [r for r in summary if r.get("probe_kind") == probe_kind]
    if not rows:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))

    ax = axes[0]
    label_counts: Dict[Tuple[float, float], int] = defaultdict(int)
    for row in rows:
        r_t = float(row.get("r_eff_T_task_mean", float("nan")))
        dim = float(row.get("dim_R_nat_mean", float("nan")))
        if not math.isfinite(r_t) or r_t <= 0:
            continue
        x = dim / r_t
        y = float(row.get("E_TQ_T_mean", float("nan")))
        ax.scatter(x, y, s=55)
        key = (round(x, 4), round(y, 4))
        label_idx = label_counts[key]
        label_counts[key] += 1
        ax.annotate(
            str(row["checkpoint"]),
            (x, y),
            xytext=(4, 4 + 8 * label_idx),
            textcoords="offset points",
            fontsize=7,
        )
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("d_lay / r_T")
    ax.set_ylabel("E(T_Q,T)")
    ax.set_title(f"Reduced-operator error ({probe_kind})")
    ax.margins(x=0.06, y=0.18)
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    label_counts = defaultdict(int)
    for row in rows:
        x = float(row.get("mean_s_mean", float("nan")))
        y = float(row.get("max_val_closure_mean", float("nan")))
        ax.scatter(x, y, s=55)
        key = (round(x, 4), round(y, 4))
        label_idx = label_counts[key]
        label_counts[key] += 1
        ax.annotate(
            str(row["checkpoint"]),
            (x, y),
            xytext=(4, 4 + 8 * label_idx),
            textcoords="offset points",
            fontsize=7,
        )
    ax.set_xlabel("mean selected layer")
    ax.set_ylabel("max held-out closure defect")
    ax.set_title("Hidden-response validation")
    ax.margins(x=0.06, y=0.18)
    ax.grid(True, alpha=0.25)

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.88, wspace=0.32)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def sample_rbf_episode(
    cfg: CkptCfg,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    total = args.n_ctx + args.n_tgt
    X = torch.randn(1, total, cfg.d_x)
    K_full = compute_kernel(
        X,
        X,
        "rbf",
        args.kernel_lengthscale,
        args.kernel_signal_var,
    ).double()
    jitter = args.kernel_jitter * torch.eye(total, dtype=torch.float64).unsqueeze(0)
    L = torch.linalg.cholesky(K_full + jitter)
    f = (L @ torch.randn(1, total, 1, dtype=torch.float64)).squeeze(-1)
    y_ctx = f[:, : args.n_ctx] + torch.randn(1, args.n_ctx, dtype=torch.float64) * (args.sigma2 ** 0.5)
    y_tgt = f[:, args.n_ctx :]
    return (
        X[:, : args.n_ctx, :].to(device=device, dtype=torch.float32),
        y_ctx.to(device=device, dtype=torch.float32),
        X[:, args.n_ctx :, :].to(device=device, dtype=torch.float32),
        y_tgt.to(device=device, dtype=torch.float32),
    )


def sample_eval_episode(
    cfg: CkptCfg,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if args.kernel_family == "linear":
        return sample_episode(cfg, args.sigma2, args.n_ctx, args.n_tgt, device)
    if args.kernel_family == "rbf":
        return sample_rbf_episode(cfg, args, device)
    raise ValueError(f"unknown kernel_family: {args.kernel_family}")


def build_eval_kernels(
    x_ctx: torch.Tensor,
    x_tgt: torch.Tensor,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if args.kernel_family == "linear":
        return build_kernels(x_ctx, x_tgt, args.sigma2)
    if args.kernel_family != "rbf":
        raise ValueError(f"unknown kernel_family: {args.kernel_family}")

    xc = x_ctx[0].detach().cpu().double()
    xt = x_tgt[0].detach().cpu().double()
    K = compute_kernel(
        xc.unsqueeze(0),
        xc.unsqueeze(0),
        "rbf",
        args.kernel_lengthscale,
        args.kernel_signal_var,
    ).squeeze(0).double()
    Kt = compute_kernel(
        xt.unsqueeze(0),
        xc.unsqueeze(0),
        "rbf",
        args.kernel_lengthscale,
        args.kernel_signal_var,
    ).squeeze(0).double()
    eye = torch.eye(K.shape[0], dtype=torch.float64)
    A = K + args.sigma2 * eye
    T = Kt @ torch.linalg.solve(A, eye)
    return K, Kt, A, T


def build_eval_target_kernel(x_tgt: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    xt = x_tgt[0].detach().cpu().double()
    if args.kernel_family == "linear":
        return xt @ xt.T
    if args.kernel_family != "rbf":
        raise ValueError(f"unknown kernel_family: {args.kernel_family}")
    return compute_kernel(
        xt.unsqueeze(0),
        xt.unsqueeze(0),
        "rbf",
        args.kernel_lengthscale,
        args.kernel_signal_var,
    ).squeeze(0).double()


def checkpoint_list(args: argparse.Namespace) -> List[CkptCfg]:
    cfgs: List[CkptCfg] = []
    if args.exp in ("2a", "both"):
        cfgs.extend(CHECKPOINTS_2A)
    if args.exp in ("2b", "both"):
        cfgs.extend(CHECKPOINTS_2B)
    if args.exp == "rbf":
        cfgs.extend(CHECKPOINTS_RBF)

    if args.checkpoints:
        wanted = {x.strip() for x in args.checkpoints.split(",") if x.strip()}
        cfgs = [cfg for cfg in cfgs if cfg.name in wanted or cfg.checkpoint in wanted]
        missing = wanted - {cfg.name for cfg in cfgs} - {cfg.checkpoint for cfg in cfgs}
        if missing:
            raise ValueError(f"unknown checkpoint filters: {sorted(missing)}")

    return cfgs


def process_checkpoint(
    cfg: CkptCfg,
    args: argparse.Namespace,
    device: torch.device,
    gen: torch.Generator,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    model = load_model(cfg, device)
    eval_kinds = ["task", "iso"] if args.probe_kind == "both" else [args.probe_kind]
    records: List[Dict[str, object]] = []
    layer_records: List[Dict[str, object]] = []

    for ep in range(args.episodes):
        x_ctx, y_ctx, x_tgt, y_tgt = sample_eval_episode(
            cfg,
            args,
            device,
        )
        y = y_ctx[0].detach().cpu().double()
        y_tgt_cpu = y_tgt[0].detach().cpu().double()

        _K, Kt, A, T = build_eval_kernels(x_ctx, x_tgt, args)
        Ktt = build_eval_target_kernel(x_tgt, args)
        Ty = T @ y

        F_y, _ = forward_with_ctx_hidden(model, x_ctx, y, x_tgt)
        base_metrics = mse_and_pointwise(F_y, Ty, y_tgt_cpu)

        build_probes = sample_task_probes(A, args.n_build, gen)
        val_probes = sample_task_probes(A, args.n_val, gen)
        causal_probes = None
        causal_fd_base = None
        if args.selection_mode == "causal":
            causal_probes = sample_task_probes(A, args.n_causal, gen)
            causal_fd_base = prediction_fd_bundle(
                model,
                x_ctx,
                x_tgt,
                y,
                causal_probes,
                args.eps,
            )

        M_build = hidden_response_matrices(model, x_ctx, x_tgt, y, build_probes, args.eps)
        M_val = hidden_response_matrices(model, x_ctx, x_tgt, y, val_probes, args.eps)

        native = build_validated_native_basis(
            A,
            M_build,
            M_val,
            tau_sv=args.tau_sv,
            tau_subspace=args.tau_subspace,
            gamma=args.gamma,
            min_explainable_frac=args.min_explainable_frac,
            max_candidate_rank=args.max_candidate_rank,
            selection_mode=args.selection_mode,
            model=model,
            x_ctx=x_ctx,
            x_tgt=x_tgt,
            y=y,
            causal_probes=causal_probes,
            causal_fd_base=causal_fd_base,
            causal_eps=args.eps,
            causal_damage_tau=args.causal_damage_tau,
            causal_max_keep=args.causal_max_keep,
            span_mode=args.span_mode,
        )
        Q_nat: torch.Tensor = native["Q_nat"]  # type: ignore[assignment]
        T_Q = Kt @ Q_nat @ Q_nat.T if Q_nat.shape[1] else torch.zeros_like(T)

        s_list = [int(x) for x in native["s_list"]]  # type: ignore[arg-type]
        candidate_ranks = [int(x) for x in native["candidate_ranks"]]  # type: ignore[arg-type]
        val_closure = [float(x) for x in native["val_closure"]]  # type: ignore[arg-type]
        build_closure = [float(x) for x in native["build_closure"]]  # type: ignore[arg-type]
        val_refinement = [float(x) for x in native["val_refinement"]]  # type: ignore[arg-type]
        build_refinement = [float(x) for x in native["build_refinement"]]  # type: ignore[arg-type]

        r_t_strict = effective_rank_T_task(T, A, args.rank_tau)
        risk_rank = effective_rank_T_excess_risk(T, A, Kt, Ktt, args.excess_risk_frac)
        r_t = int(risk_rank["r_eff_T_task"])
        B_nat = int(native["B_nat"])
        dim_R = int(native["dim_R_nat"])
        mean_s = float(np.mean(s_list)) if s_list else 0.0
        layer_stats = list(native["layer_stats"])  # type: ignore[arg-type]
        causal_max_damage = max(float(stat.get("causal_max_damage", 0.0)) for stat in layer_stats)
        causal_mean_damage = float(np.mean([float(stat.get("causal_mean_damage", 0.0)) for stat in layer_stats]))
        causal_sum_kept_damage = float(
            np.sum([float(stat.get("causal_sum_kept_damage", 0.0)) for stat in layer_stats])
        )
        mean_hidden_s = float(np.mean([float(stat.get("hidden_s", 0.0)) for stat in layer_stats]))

        for stat in layer_stats:
            layer_records.append(
                {
                    "exp": args.exp,
                    "checkpoint": cfg.name,
                    "checkpoint_file": cfg.checkpoint,
                    "episode": ep,
                    "d_x": cfg.d_x,
                    "n_layers": cfg.n_layers,
                    "n_heads": cfg.n_heads,
                    "selection_mode": args.selection_mode,
                    "span_mode": args.span_mode,
                    "kernel_family": args.kernel_family,
                    "kernel_lengthscale": args.kernel_lengthscale if args.kernel_family == "rbf" else float("nan"),
                    "kernel_signal_var": args.kernel_signal_var if args.kernel_family == "rbf" else float("nan"),
                    "excess_risk_frac": args.excess_risk_frac,
                    "gamma": args.gamma,
                    "min_explainable_frac": args.min_explainable_frac,
                    "n_causal": args.n_causal if args.selection_mode == "causal" else 0,
                    "causal_damage_tau": args.causal_damage_tau,
                    "causal_max_keep": args.causal_max_keep,
                    **stat,
                }
            )

        for kind in eval_kinds:
            eval_probes = sample_probes(A, args.n_eval, kind, gen)
            eval_fd = prediction_fd_bundle(model, x_ctx, x_tgt, y, eval_probes, args.eps)

            E_F_T = eval_operator_error(T, eval_probes, eval_fd)
            E_TQ_T = operator_error(T_Q, T, eval_probes)
            E_F_TQ = model_to_operator_error(eval_fd, T_Q, T, eval_probes)

            E_F_0 = model_to_operator_error(eval_fd, torch.zeros_like(T), T, eval_probes)
            E_0_T = operator_error(torch.zeros_like(T), T, eval_probes)
            X_use = 1.0 - (E_F_TQ / (E_F_0 + FLOOR)) ** 2
            X_sub = 1.0 - (E_TQ_T / (E_0_T + FLOOR)) ** 2

            rec: Dict[str, object] = {
                "exp": args.exp,
                "checkpoint": cfg.name,
                "checkpoint_file": cfg.checkpoint,
                "sweep": cfg.sweep,
                "sweep_val": cfg.sweep_val,
                "d_x": cfg.d_x,
                "d_model": cfg.d_model,
                "n_layers": cfg.n_layers,
                "n_heads": cfg.n_heads,
                "episode": ep,
                "probe_kind": kind,
                "selection_mode": args.selection_mode,
                "span_mode": args.span_mode,
                "kernel_family": args.kernel_family,
                "kernel_lengthscale": args.kernel_lengthscale if args.kernel_family == "rbf" else float("nan"),
                "kernel_signal_var": args.kernel_signal_var if args.kernel_family == "rbf" else float("nan"),
                "gamma": args.gamma,
                "min_explainable_frac": args.min_explainable_frac,
                "n_causal": args.n_causal if args.selection_mode == "causal" else 0,
                "causal_damage_tau": args.causal_damage_tau,
                "causal_max_keep": args.causal_max_keep,
                "tau_sv": args.tau_sv,
                "n_build": args.n_build,
                "n_val": args.n_val,
                "n_eval": args.n_eval,
                "B_nat": B_nat,
                "dim_R_nat": dim_R,
                "dim_over_nctx": dim_R / A.shape[0],
                "dim_over_rT": dim_R / r_t if r_t > 0 else float("nan"),
                "dim_over_rT_strict": dim_R / r_t_strict if r_t_strict > 0 else float("nan"),
                "r_eff_T_task": r_t,
                "r_eff_T_task_strict": r_t_strict,
                "excess_risk_frac": args.excess_risk_frac,
                "mean_s": mean_s,
                "mean_hidden_s": mean_hidden_s,
                "sum_candidate_rank": int(sum(candidate_ranks)),
                "mean_candidate_rank": float(np.mean(candidate_ranks)) if candidate_ranks else 0.0,
                "max_causal_damage": causal_max_damage,
                "mean_causal_damage": causal_mean_damage,
                "sum_kept_causal_damage": causal_sum_kept_damage,
                "E_F_T": E_F_T,
                "E_TQ_T": E_TQ_T,
                "E_F_TQ": E_F_TQ,
                "X_use": X_use,
                "X_sub": X_sub,
                "max_val_closure": max(val_closure),
                "mean_val_closure": float(np.mean(val_closure)),
                "max_build_closure": max(build_closure),
                "mean_build_closure": float(np.mean(build_closure)),
                "max_val_refinement": max(val_refinement),
                "mean_val_refinement": float(np.mean(val_refinement)),
                "max_build_refinement": max(build_refinement),
                "mean_build_refinement": float(np.mean(build_refinement)),
                "s_list": json.dumps(s_list),
                "candidate_ranks": json.dumps(candidate_ranks),
            }
            rec.update(risk_rank)
            rec.update(base_metrics)
            records.append(rec)

            print(
                f"  {cfg.name:5s} ep={ep+1:03d}/{args.episodes} probe={kind:4s} "
                f"mode={args.selection_mode:6s}/{args.span_mode:9s} B={B_nat:4d} dim={dim_R:2d}/{A.shape[0]} "
                f"rT={r_t:2d} strict={r_t_strict:2d} mean_s={mean_s:.2f} E(TQ,T)={E_TQ_T:.5f} "
                f"E(F,TQ)={E_F_TQ:.5f} E(F,T)={E_F_T:.5f} "
                f"maxValClos={max(val_closure):.4f} maxDmg={causal_max_damage:.4f}",
                flush=True,
            )

    return records, layer_records


def run(args: argparse.Namespace) -> None:
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.kernel_family == "auto":
        args.kernel_family = "rbf" if args.exp == "rbf" else "linear"
    if args.exp == "rbf" and args.kernel_family != "rbf":
        raise ValueError("--exp rbf requires --kernel-family rbf or auto")
    if args.kernel_lengthscale <= 0.0:
        raise ValueError("--kernel-lengthscale must be positive")
    if args.kernel_signal_var <= 0.0:
        raise ValueError("--kernel-signal-var must be positive")
    if args.kernel_jitter < 0.0:
        raise ValueError("--kernel-jitter must be nonnegative")

    device = get_device() if args.device == "auto" else torch.device(args.device)
    set_seed(args.seed)
    gen = torch.Generator(device="cpu").manual_seed(args.seed + 54321)

    cfgs = checkpoint_list(args)
    if not cfgs:
        raise ValueError("no checkpoints selected")

    (results_dir / "config.json").write_text(
        json.dumps({"args": vars(args), "device": str(device)}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    all_records: List[Dict[str, object]] = []
    all_layer_records: List[Dict[str, object]] = []

    print("=== Experiment 2: Validated Native Rank ===", flush=True)
    print(
        f"device={device} mode={args.selection_mode} span={args.span_mode} gamma={args.gamma} "
        f"causal_tau={args.causal_damage_tau} checkpoints={[cfg.name for cfg in cfgs]}",
        flush=True,
    )

    for cfg in cfgs:
        print(f"-- {cfg.name} (L={cfg.n_layers}, H={cfg.n_heads}, d_x={cfg.d_x}) --", flush=True)
        records, layer_records = process_checkpoint(cfg, args, device, gen)
        all_records.extend(records)
        all_layer_records.extend(layer_records)

    write_csv(results_dir / "records.csv", all_records)
    write_csv(results_dir / "layer_records.csv", all_layer_records)
    summary = summarize(all_records, ["checkpoint", "probe_kind"])
    layer_summary = summarize(all_layer_records, ["checkpoint", "ell"])
    write_csv(results_dir / "summary.csv", summary)
    write_csv(results_dir / "layer_summary.csv", layer_summary)
    write_summary(results_dir / "summary.txt", summary, args)

    for pk in (["task", "iso"] if args.probe_kind == "both" else [args.probe_kind]):
        plot_summary(summary, results_dir / f"{args.selection_mode}_rank_{pk}.png", pk)

    print(f"wrote results to {results_dir}", flush=True)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 2 validated native rank")
    parser.add_argument("--exp", choices=["2a", "2b", "rbf", "both"], default="2b")
    parser.add_argument("--checkpoints", default="", help="Comma-separated checkpoint names/files to include.")
    parser.add_argument("--results-dir", default=str(SCRIPT_DIR / "results"))
    parser.add_argument("--selection-mode", choices=["hidden", "causal"], default="hidden")
    parser.add_argument(
        "--span-mode",
        choices=["seed", "reachable"],
        default="seed",
        help=(
            "Final layerwise span. 'seed' uses span(S_0,...,S_L). "
            "'reachable' also includes powers A^t S_ell."
        ),
    )
    parser.add_argument("--probe-kind", choices=["task", "iso", "both"], default="task")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--n-ctx", type=int, default=47)
    parser.add_argument("--n-tgt", type=int, default=16)
    parser.add_argument("--sigma2", type=float, default=0.1)
    parser.add_argument("--kernel-family", choices=["auto", "linear", "rbf"], default="auto")
    parser.add_argument("--kernel-lengthscale", type=float, default=3.0)
    parser.add_argument("--kernel-signal-var", type=float, default=1.0)
    parser.add_argument("--kernel-jitter", type=float, default=1e-5)
    parser.add_argument("--n-build", type=int, default=8)
    parser.add_argument("--n-val", type=int, default=16)
    parser.add_argument("--n-eval", type=int, default=16)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--tau-sv", type=float, default=1e-3)
    parser.add_argument("--tau-subspace", type=float, default=1e-9)
    parser.add_argument(
        "--rank-tau",
        type=float,
        default=1e-2,
        help=(
            "Tail RMS tolerance for task-visible rank: smallest r with "
            "sqrt(sum_{i>r} s_i^2 / sum_i s_i^2) <= rank_tau."
        ),
    )
    parser.add_argument(
        "--excess-risk-frac",
        type=float,
        default=0.05,
        help=(
            "Prediction-risk rank tolerance: smallest r whose discarded operator "
            "energy is at most this fraction of the KRR posterior risk."
        ),
    )
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--min-explainable-frac", type=float, default=1e-3)
    parser.add_argument("--n-causal", type=int, default=8)
    parser.add_argument("--causal-damage-tau", type=float, default=0.01)
    parser.add_argument("--causal-max-keep", type=int, default=0)
    parser.add_argument(
        "--max-candidate-rank",
        type=int,
        default=0,
        help="Optional SVD candidate cap. 0 means use n_ctx.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
