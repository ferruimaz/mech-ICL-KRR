#!/usr/bin/env python3
"""Experiment 3: Galerkin-Leverage Surgery.

Causal test for the operator-Galerkin theory.

For an extracted A-orthonormal final-state basis Q=[q_1,...,q_r], compute
Galerkin leverage scores and intervene on the residual stream by removing
selected context-token directions:

    H_ctx <- H_ctx - sum_{j in J} q_j q_j^T A H_ctx.

Then roll the remaining transformer layers forward and measure:
  - prediction drift relative to base model;
  - KRR-error inflation;
  - post-surgery finite-difference operator error;
  - post-surgery model-to-Galerkin error;
  - finite-difference response damage relative to the base model.

Primary intervention:
  - high_task: top-k directions by task-distribution operator leverage.

Controls:
  - low_task: bottom-k directions by task leverage;
  - random_Q: random subset of Q columns;
  - random_A: random A-orthonormal directions;
  - high_iso_low_task: directions high under isotropic leverage but low under task leverage;
  - high_variance_low_task: early SVD directions among low-task-leverage columns.

Layer rules:
  - final: intervene after the final transformer layer, before output head;
  - sweep: intervene after every residual state, including embedding output and every layer.

Run smoke:

    python -m experiments.experiment_3_causal_surgery.run \
      --checkpoint final/linear_baseline_dx5_L8.pt \
      --d-x 5 --d-model 128 --n-layers 8 --n-heads 4 \
      --episodes 2 --n-ctx 47 --n-tgt 16 \
      --n-causal 16 --k-remove 1 --layer-rule final

Run final:

    python -m experiments.experiment_3_causal_surgery.run \
      --checkpoint final/linear_baseline_dx5_L8.pt \
      --d-x 5 --d-model 128 --n-layers 8 --n-heads 4 \
      --episodes 32 --n-ctx 47 --n-tgt 16 \
      --n-causal 32 --k-remove-list 1,2,4 \
      --layer-rule sweep \
      --probe-kind both \
      --results-dir experiments/experiment_3_causal_surgery/results
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mpl-cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import ICLTransformer
from data import sample_batch_eigenvalues
from experiment_utils.support import get_device, set_seed, flat_eigenvalues, CHECKPOINT_DIR

FLOOR = 1e-12


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelCfg:
    checkpoint: str
    d_x: int
    d_model: int
    n_layers: int
    n_heads: int


# ---------------------------------------------------------------------------
# Model/data utilities
# ---------------------------------------------------------------------------

def load_model(cfg: ModelCfg, device: torch.device) -> ICLTransformer:
    path = Path(CHECKPOINT_DIR) / cfg.checkpoint
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    model = ICLTransformer(cfg.d_x, cfg.d_model, cfg.n_layers, cfg.n_heads)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device).eval()
    return model


def sample_episode(
    cfg: ModelCfg,
    sigma2: float,
    n_ctx: int,
    n_tgt: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_ctx, y_ctx, x_tgt, y_tgt, _ = sample_batch_eigenvalues(
        1,
        cfg.d_x,
        n_ctx,
        n_tgt,
        sigma2,
        flat_eigenvalues,
        device="cpu",
    )
    return x_ctx.to(device), y_ctx.to(device), x_tgt.to(device), y_tgt.to(device)


def build_kernels(
    x_ctx: torch.Tensor,
    x_tgt: torch.Tensor,
    sigma2: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return K, Kt, A, T=Kt A^{-1}, all CPU float64."""
    xc = x_ctx[0].detach().cpu().double()
    xt = x_tgt[0].detach().cpu().double()

    K = xc @ xc.T
    Kt = xt @ xc.T
    eye = torch.eye(K.shape[0], dtype=torch.float64)
    A = K + sigma2 * eye
    T = Kt @ torch.linalg.solve(A, eye)
    return K, Kt, A, T


# ---------------------------------------------------------------------------
# Probe distributions
# ---------------------------------------------------------------------------

def sample_task_probes(A: torch.Tensor, n: int, gen: torch.Generator) -> torch.Tensor:
    z = torch.randn(n, A.shape[0], dtype=torch.float64, generator=gen)
    jitter = 1e-10 * torch.eye(A.shape[0], dtype=torch.float64)
    L = torch.linalg.cholesky(A + jitter)
    return z @ L.T


def sample_isotropic_probes(n_ctx: int, n: int, gen: torch.Generator) -> torch.Tensor:
    return torch.randn(n, n_ctx, dtype=torch.float64, generator=gen)


def sample_probes(A: torch.Tensor, n: int, kind: str, gen: torch.Generator) -> torch.Tensor:
    if kind == "task":
        return sample_task_probes(A, n, gen)
    if kind == "iso":
        return sample_isotropic_probes(A.shape[0], n, gen)
    raise ValueError(f"Unknown probe kind: {kind}")


# ---------------------------------------------------------------------------
# Linear algebra
# ---------------------------------------------------------------------------

def psd_sqrt_and_invsqrt(
    A: torch.Tensor,
    floor: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    eigvals, eigvecs = torch.linalg.eigh(A)
    eigvals = eigvals.clamp_min(floor)
    A_sqrt = eigvecs @ torch.diag(torch.sqrt(eigvals)) @ eigvecs.T
    A_invsqrt = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals)) @ eigvecs.T
    return A_sqrt, A_invsqrt


def a_orth_basis_from_cols(
    M: torch.Tensor,
    A_sqrt: torch.Tensor,
    A_invsqrt: torch.Tensor,
    tau_rel: float,
    rmax: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """A-orthonormal basis Q=A^{-1/2}U for columns of M."""
    n = M.shape[0]
    if M.numel() == 0 or M.shape[1] == 0:
        return torch.zeros(n, 0, dtype=torch.float64), torch.zeros(0, dtype=torch.float64)

    W = A_sqrt @ M
    U, S, _ = torch.linalg.svd(W, full_matrices=False)

    if S.numel() == 0 or float(S[0]) <= 0:
        return torch.zeros(n, 0, dtype=torch.float64), S

    rank = int((S >= tau_rel * S[0]).sum().item())
    if rmax is not None:
        rank = min(rank, rmax)

    if rank <= 0:
        return torch.zeros(n, 0, dtype=torch.float64), S

    Q = A_invsqrt @ U[:, :rank]
    return Q.contiguous(), S


def random_a_orthonormal(
    n_ctx: int,
    k: int,
    A: torch.Tensor,
    gen: torch.Generator,
) -> torch.Tensor:
    A_sqrt, A_invsqrt = psd_sqrt_and_invsqrt(A)
    M = torch.randn(n_ctx, k, dtype=torch.float64, generator=gen)
    Q, _ = a_orth_basis_from_cols(M, A_sqrt, A_invsqrt, tau_rel=1e-12, rmax=k)
    return Q[:, :k]


def operator_error(S: torch.Tensor, T: torch.Tensor, probes: torch.Tensor) -> float:
    U = probes.T
    TU = T @ U
    SU = S @ U
    denom = float((TU * TU).sum()) + FLOOR
    return math.sqrt(float(((SU - TU) ** 2).sum()) / denom)


def fd_operator_error(eval_fd: torch.Tensor, S: torch.Tensor, T: torch.Tensor, probes: torch.Tensor) -> float:
    U = probes.T
    TU = T @ U
    SU = S @ U
    denom = float((TU * TU).sum()) + FLOOR
    return math.sqrt(float(((eval_fd - SU) ** 2).sum()) / denom)


def fd_damage_error(fd_surg: torch.Tensor, fd_base: torch.Tensor, T: torch.Tensor, probes: torch.Tensor) -> float:
    U = probes.T
    TU = T @ U
    denom = float((TU * TU).sum()) + FLOOR
    return math.sqrt(float(((fd_surg - fd_base) ** 2).sum()) / denom)


# ---------------------------------------------------------------------------
# Base forward and intervention forward
# ---------------------------------------------------------------------------

@torch.no_grad()
def make_tokens(
    model: ICLTransformer,
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
def forward_base_with_states(
    model: ICLTransformer,
    x_ctx: torch.Tensor,
    y_cpu: torch.Tensor,
    x_tgt: torch.Tensor,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Return predictions and residual states.

    states[0] = embedding output, before layer 0.
    states[k] = after k layers.
    states[L] = final state before head.
    """
    h, n_ctx = make_tokens(model, x_ctx, y_cpu, x_tgt)
    states = [h.detach().clone()]

    for layer in model.layers:
        h = layer(h)
        states.append(h.detach().clone())

    pred = model.head(h[:, n_ctx:, :]).squeeze(-1)[0].detach().cpu().double()
    return pred, states


@torch.no_grad()
def apply_context_projection_removal(
    h: torch.Tensor,
    n_ctx: int,
    A_cpu: torch.Tensor,
    D_cpu: torch.Tensor,
) -> torch.Tensor:
    """Remove A-projection onto columns of D from context residual state.

    D_cpu is n_ctx x k and should be A-orthonormal:
        D^T A D = I.

    For H_ctx: n_ctx x d_model,

        H_ctx <- H_ctx - D D^T A H_ctx.
    """
    if D_cpu.numel() == 0 or D_cpu.shape[1] == 0:
        return h

    dtype = h.dtype
    device = h.device

    D = D_cpu.to(device=device, dtype=dtype)
    A = A_cpu.to(device=device, dtype=dtype)

    h_new = h.clone()
    Hctx = h_new[0, :n_ctx, :]  # n_ctx x d_model

    AH = A @ Hctx
    coeff = D.T @ AH
    delta = D @ coeff

    h_new[0, :n_ctx, :] = Hctx - delta
    return h_new


@torch.no_grad()
def forward_with_surgery(
    model: ICLTransformer,
    x_ctx: torch.Tensor,
    y_cpu: torch.Tensor,
    x_tgt: torch.Tensor,
    A_cpu: torch.Tensor,
    D_cpu: torch.Tensor,
    state_idx: int,
) -> torch.Tensor:
    """Run model with projection surgery after residual state `state_idx`.

    state_idx=0 means after embedding, before first transformer layer.
    state_idx=L means after final transformer layer, before output head.
    """
    h, n_ctx = make_tokens(model, x_ctx, y_cpu, x_tgt)
    L = len(model.layers)

    if state_idx == 0:
        h = apply_context_projection_removal(h, n_ctx, A_cpu, D_cpu)

    for i, layer in enumerate(model.layers):
        h = layer(h)
        if state_idx == i + 1:
            h = apply_context_projection_removal(h, n_ctx, A_cpu, D_cpu)

    if state_idx < 0 or state_idx > L:
        raise ValueError(f"state_idx must be in [0,{L}], got {state_idx}")

    pred = model.head(h[:, n_ctx:, :]).squeeze(-1)[0].detach().cpu().double()
    return pred


@torch.no_grad()
def fd_bundle_base(
    model: ICLTransformer,
    x_ctx: torch.Tensor,
    x_tgt: torch.Tensor,
    y: torch.Tensor,
    probes: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    cols: List[torch.Tensor] = []
    for u in probes:
        pp, _ = forward_base_with_states(model, x_ctx, y + eps * u, x_tgt)
        pn, _ = forward_base_with_states(model, x_ctx, y - eps * u, x_tgt)
        cols.append(((pp - pn) / (2.0 * eps)).reshape(-1, 1))
    return torch.cat(cols, dim=1) if cols else torch.zeros(x_tgt.shape[1], 0, dtype=torch.float64)


@torch.no_grad()
def fd_bundle_surgery(
    model: ICLTransformer,
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


# ---------------------------------------------------------------------------
# Extract final raw basis
# ---------------------------------------------------------------------------

def extract_raw_final_basis(
    final_state: torch.Tensor,
    n_ctx: int,
    A: torch.Tensor,
    tau_sv: float,
    rmax: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract Q_raw from final context hidden state H_ctx."""
    Hctx = final_state[0, :n_ctx, :].detach().cpu().double()
    A_sqrt, A_invsqrt = psd_sqrt_and_invsqrt(A)
    Q, S = a_orth_basis_from_cols(Hctx, A_sqrt, A_invsqrt, tau_rel=tau_sv, rmax=rmax)
    return Q, S


# ---------------------------------------------------------------------------
# Leverage and intervention sets
# ---------------------------------------------------------------------------

def leverage_scores(
    Q: torch.Tensor,
    Kt: torch.Tensor,
    T: torch.Tensor,
    y: torch.Tensor,
    probes_task: torch.Tensor,
    probes_iso: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Compute pointwise, task operator, and isotropic operator leverage.

    For direction q_j,
        (T_Q - T_Qminusj)u = Kt q_j * (q_j^T u).
    """
    r = Q.shape[1]
    if r == 0:
        z = torch.zeros(0, dtype=torch.float64)
        return {"point": z, "task": z, "iso": z, "ktq_norm": z}

    KtQ = Kt @ Q  # n_tgt x r
    ktq_norm_sq = (KtQ * KtQ).sum(dim=0)  # r

    denom_point = float((T @ y).norm()) + FLOOR

    point_vals = []
    for j in range(r):
        coeff_y = torch.dot(Q[:, j], y)
        val = float(KtQ[:, j].norm() * abs(float(coeff_y)) / denom_point)
        point_vals.append(val)
    point = torch.tensor(point_vals, dtype=torch.float64)

    def op_lev(probes: torch.Tensor) -> torch.Tensor:
        U = probes.T  # n_ctx x m
        TU = T @ U
        denom = float((TU * TU).sum()) + FLOOR
        coeff = Q.T @ U  # r x m
        num = ktq_norm_sq[:, None] * (coeff * coeff)
        return torch.sqrt(num.sum(dim=1) / denom)

    return {
        "point": point,
        "task": op_lev(probes_task),
        "iso": op_lev(probes_iso),
        "ktq_norm": torch.sqrt(ktq_norm_sq.clamp_min(0.0)),
    }


def choose_topk(vals: torch.Tensor, k: int) -> List[int]:
    if vals.numel() == 0:
        return []
    k = min(k, vals.numel())
    return torch.argsort(vals, descending=True)[:k].cpu().tolist()


def choose_bottomk(vals: torch.Tensor, k: int) -> List[int]:
    if vals.numel() == 0:
        return []
    k = min(k, vals.numel())
    return torch.argsort(vals, descending=False)[:k].cpu().tolist()


def choose_high_iso_low_task(task: torch.Tensor, iso: torch.Tensor, k: int) -> List[int]:
    if task.numel() == 0:
        return []
    k = min(k, task.numel())
    task_med = torch.median(task)
    candidates = torch.where(task <= task_med)[0]
    if candidates.numel() < k:
        score = iso / (task + 1e-12)
        return choose_topk(score, k)
    cand_scores = iso[candidates]
    chosen_local = torch.argsort(cand_scores, descending=True)[:k]
    return candidates[chosen_local].cpu().tolist()


def choose_high_variance_low_task(task: torch.Tensor, k: int) -> List[int]:
    """Q columns are in SVD order, so low index = high activation variance."""
    if task.numel() == 0:
        return []
    k = min(k, task.numel())
    task_med = torch.median(task)
    candidates = torch.where(task <= task_med)[0].cpu().tolist()
    if len(candidates) < k:
        candidates = list(range(task.numel()))
    return candidates[:k]


def make_intervention_sets(
    Q: torch.Tensor,
    A: torch.Tensor,
    lev: Dict[str, torch.Tensor],
    k: int,
    gen: torch.Generator,
) -> Dict[str, torch.Tensor]:
    r = Q.shape[1]
    n = Q.shape[0]
    sets: Dict[str, torch.Tensor] = {}

    if r == 0:
        return sets

    high_idx = choose_topk(lev["task"], k)
    low_idx = choose_bottomk(lev["task"], k)
    hi_iso_idx = choose_high_iso_low_task(lev["task"], lev["iso"], k)
    high_var_low_idx = choose_high_variance_low_task(lev["task"], k)

    perm = torch.randperm(r, generator=gen)[: min(k, r)].cpu().tolist()

    sets["high_task"] = Q[:, high_idx]
    sets["low_task"] = Q[:, low_idx]
    sets["random_Q"] = Q[:, perm]
    sets["high_iso_low_task"] = Q[:, hi_iso_idx]
    sets["high_variance_low_task"] = Q[:, high_var_low_idx]
    sets["random_A"] = random_a_orthonormal(n, min(k, n), A, gen)

    return sets


def leverage_for_direction_matrix(
    D: torch.Tensor,
    Kt: torch.Tensor,
    T: torch.Tensor,
    probes: torch.Tensor,
) -> float:
    """Aggregate operator leverage of a set of A-orthonormal directions D."""
    if D.numel() == 0 or D.shape[1] == 0:
        return 0.0
    U = probes.T
    TU = T @ U
    denom = float((TU * TU).sum()) + FLOOR
    SD = Kt @ D @ (D.T @ U)
    return math.sqrt(float((SD * SD).sum()) / denom)


# ---------------------------------------------------------------------------
# Per-episode processing
# ---------------------------------------------------------------------------

def process_episode(
    model: ICLTransformer,
    cfg: ModelCfg,
    args: argparse.Namespace,
    ep: int,
    device: torch.device,
    gen: torch.Generator,
) -> List[Dict[str, object]]:
    x_ctx, y_ctx, x_tgt, y_tgt = sample_episode(cfg, args.sigma2, args.n_ctx, args.n_tgt, device)

    y = y_ctx[0].detach().cpu().double()
    y_tgt_cpu = y_tgt[0].detach().cpu().double()

    _K, Kt, A, T = build_kernels(x_ctx, x_tgt, args.sigma2)
    Ty = T @ y

    base_pred, states = forward_base_with_states(model, x_ctx, y, x_tgt)
    base_err = float((base_pred - Ty).norm() / (Ty.norm() + FLOOR))
    mse_model = float(((base_pred - y_tgt_cpu) ** 2).mean())
    mse_krr = float(((Ty - y_tgt_cpu) ** 2).mean())
    mse_ratio = mse_model / max(mse_krr, FLOOR)

    Q, svals = extract_raw_final_basis(
        final_state=states[-1],
        n_ctx=args.n_ctx,
        A=A,
        tau_sv=args.tau_sv,
        rmax=args.rmax,
    )
    r = Q.shape[1]

    if r == 0:
        print(f"  ep={ep+1:03d}: extracted rank 0; skipping", flush=True)
        return []

    TQ = Kt @ Q @ Q.T

    probes_task = sample_task_probes(A, args.n_causal, gen)
    probes_iso = sample_isotropic_probes(A.shape[0], args.n_causal, gen)

    fd_base_task = fd_bundle_base(model, x_ctx, x_tgt, y, probes_task, args.eps)
    fd_base_iso = fd_bundle_base(model, x_ctx, x_tgt, y, probes_iso, args.eps)

    base_E_task_F_T = fd_operator_error(fd_base_task, T, T, probes_task)
    base_E_task_F_TQ = fd_operator_error(fd_base_task, TQ, T, probes_task)
    base_E_task_TQ_T = operator_error(TQ, T, probes_task)

    base_E_iso_F_T = fd_operator_error(fd_base_iso, T, T, probes_iso)
    base_E_iso_F_TQ = fd_operator_error(fd_base_iso, TQ, T, probes_iso)
    base_E_iso_TQ_T = operator_error(TQ, T, probes_iso)

    lev = leverage_scores(Q, Kt, T, y, probes_task, probes_iso)

    if args.layer_rule == "final":
        layer_indices = [cfg.n_layers]
    elif args.layer_rule == "sweep":
        layer_indices = list(range(cfg.n_layers + 1))
    else:
        raise ValueError(f"Unknown layer rule: {args.layer_rule}")

    records: List[Dict[str, object]] = []

    for k_remove in args.k_remove_list:
        sets = make_intervention_sets(Q, A, lev, k_remove, gen)

        for control_name, D in sets.items():
            agg_lev_task = leverage_for_direction_matrix(D, Kt, T, probes_task)
            agg_lev_iso = leverage_for_direction_matrix(D, Kt, T, probes_iso)

            for state_idx in layer_indices:
                surg_pred = forward_with_surgery(model, x_ctx, y, x_tgt, A, D, state_idx)

                pred_drift = float((surg_pred - base_pred).norm() / (base_pred.norm() + FLOOR))
                surg_krr_err = float((surg_pred - Ty).norm() / (Ty.norm() + FLOOR))
                krr_err_inflation = surg_krr_err / max(base_err, FLOOR)

                fd_surg_task = fd_bundle_surgery(
                    model, x_ctx, x_tgt, y, probes_task, args.eps, A, D, state_idx
                )
                fd_surg_iso = fd_bundle_surgery(
                    model, x_ctx, x_tgt, y, probes_iso, args.eps, A, D, state_idx
                )

                E_surg_task_F_T = fd_operator_error(fd_surg_task, T, T, probes_task)
                E_surg_task_F_TQ = fd_operator_error(fd_surg_task, TQ, T, probes_task)
                E_surg_task_damage = fd_damage_error(fd_surg_task, fd_base_task, T, probes_task)

                E_surg_iso_F_T = fd_operator_error(fd_surg_iso, T, T, probes_iso)
                E_surg_iso_F_TQ = fd_operator_error(fd_surg_iso, TQ, T, probes_iso)
                E_surg_iso_damage = fd_damage_error(fd_surg_iso, fd_base_iso, T, probes_iso)

                rec: Dict[str, object] = {
                    "checkpoint": cfg.checkpoint,
                    "episode": ep,
                    "d_x": cfg.d_x,
                    "d_model": cfg.d_model,
                    "n_layers": cfg.n_layers,
                    "n_heads": cfg.n_heads,
                    "n_ctx": args.n_ctx,
                    "n_tgt": args.n_tgt,
                    "sigma2": args.sigma2,
                    "eps": args.eps,
                    "rank_Q": r,
                    "k_remove": k_remove,
                    "control": control_name,
                    "state_idx": state_idx,
                    "state_label": "embedding" if state_idx == 0 else ("final" if state_idx == cfg.n_layers else f"after_layer_{state_idx}"),
                    "agg_leverage_task": agg_lev_task,
                    "agg_leverage_iso": agg_lev_iso,
                    "pred_drift": pred_drift,
                    "base_pointwise_F_T": base_err,
                    "surg_pointwise_F_T": surg_krr_err,
                    "krr_err_inflation": krr_err_inflation,
                    "mse_model": mse_model,
                    "mse_krr": mse_krr,
                    "mse_ratio": mse_ratio,

                    "base_E_task_F_T": base_E_task_F_T,
                    "base_E_task_F_TQ": base_E_task_F_TQ,
                    "base_E_task_TQ_T": base_E_task_TQ_T,
                    "surg_E_task_F_T": E_surg_task_F_T,
                    "surg_E_task_F_TQ": E_surg_task_F_TQ,
                    "surg_E_task_damage": E_surg_task_damage,
                    "delta_E_task_F_T": E_surg_task_F_T - base_E_task_F_T,

                    "base_E_iso_F_T": base_E_iso_F_T,
                    "base_E_iso_F_TQ": base_E_iso_F_TQ,
                    "base_E_iso_TQ_T": base_E_iso_TQ_T,
                    "surg_E_iso_F_T": E_surg_iso_F_T,
                    "surg_E_iso_F_TQ": E_surg_iso_F_TQ,
                    "surg_E_iso_damage": E_surg_iso_damage,
                    "delta_E_iso_F_T": E_surg_iso_F_T - base_E_iso_F_T,
                }
                records.append(rec)

                print(
                    f"  ep={ep+1:03d} k={k_remove} {control_name:22s} "
                    f"state={state_idx:02d} drift={pred_drift:.4f} "
                    f"rho={krr_err_inflation:.2f} "
                    f"task_damage={E_surg_task_damage:.4f}",
                    flush=True,
                )

    return records


# ---------------------------------------------------------------------------
# CSV / summary
# ---------------------------------------------------------------------------

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
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def summarize(rows: List[Dict[str, object]], group_keys: List[str]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[object, ...], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k) for k in group_keys)].append(row)

    num_keys = {
        k
        for row in rows
        for k, v in row.items()
        if isinstance(v, (int, float)) and k != "episode"
    }

    out: List[Dict[str, object]] = []
    for g, grp in sorted(groups.items(), key=lambda x: tuple(str(v) for v in x[0])):
        rec: Dict[str, object] = {k: v for k, v in zip(group_keys, g)}
        rec["n"] = len(grp)
        for k in sorted(num_keys):
            vals = [float(r[k]) for r in grp if k in r and r[k] not in ("", None)]
            if not vals:
                continue
            rec[f"{k}_mean"] = float(np.mean(vals))
            rec[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        out.append(rec)

    return out


def write_report(path: Path, summary: List[Dict[str, object]], args: argparse.Namespace) -> None:
    lines: List[str] = []
    lines.append("Experiment 3: Galerkin-Leverage Surgery")
    lines.append("")
    lines.append(f"checkpoint={args.checkpoint}")
    lines.append(f"episodes={args.episodes}, n_causal={args.n_causal}, eps={args.eps}")
    lines.append(f"layer_rule={args.layer_rule}, k_remove_list={args.k_remove_list}")
    lines.append("")
    lines.append("Final-layer or layer-sweep summary by control:")
    lines.append(
        f"{'control':24s} {'k':>3s} {'state':>8s} {'drift':>10s} "
        f"{'rho':>10s} {'task_damage':>12s} {'delta_task':>12s}"
    )

    for row in summary:
        lines.append(
            f"{str(row.get('control','')):24s} "
            f"{float(row.get('k_remove_mean', float('nan'))):3.0f} "
            f"{float(row.get('state_idx_mean', float('nan'))):8.1f} "
            f"{float(row.get('pred_drift_mean', float('nan'))):10.5f} "
            f"{float(row.get('krr_err_inflation_mean', float('nan'))):10.3f} "
            f"{float(row.get('surg_E_task_damage_mean', float('nan'))):12.5f} "
            f"{float(row.get('delta_E_task_F_T_mean', float('nan'))):12.5f}"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def f(row: Dict[str, object], key: str, default: float = float("nan")) -> float:
    try:
        val = row.get(key, default)
        if val in ("", None):
            return default
        return float(val)  # type: ignore[arg-type]
    except Exception:
        return default


def plot_final_controls(summary: List[Dict[str, object]], out_path: Path, final_state_idx: int) -> None:
    rows = [r for r in summary if int(round(f(r, "state_idx_mean"))) == final_state_idx]
    if not rows:
        return

    rows = sorted(rows, key=lambda r: (f(r, "k_remove_mean"), str(r.get("control", ""))))

    labels = [f"{r['control']}\nk={f(r,'k_remove_mean'):.0f}" for r in rows]
    y = [f(r, "surg_E_task_damage_mean") for r in rows]
    yerr = [f(r, "surg_E_task_damage_std", 0.0) for r in rows]

    fig, ax = plt.subplots(figsize=(max(10, 0.6 * len(rows)), 4.5))
    ax.bar(np.arange(len(rows)), y, yerr=yerr, capsize=3)
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("task finite-difference response damage")
    ax.set_title("Experiment 3: final-state Galerkin-leverage surgery")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_layer_sweep(summary: List[Dict[str, object]], out_path: Path) -> None:
    rows = sorted(summary, key=lambda r: (str(r.get("control", "")), f(r, "k_remove_mean"), f(r, "state_idx_mean")))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))

    for metric, ax, title in [
        ("surg_E_task_damage_mean", axes[0], "Task response damage"),
        ("krr_err_inflation_mean", axes[1], "KRR-error inflation"),
    ]:
        grouped: Dict[Tuple[str, int], List[Dict[str, object]]] = defaultdict(list)
        for r in rows:
            grouped[(str(r.get("control", "")), int(round(f(r, "k_remove_mean"))))].append(r)

        for (control, k), grp in grouped.items():
            xs = [f(r, "state_idx_mean") for r in grp]
            ys = [f(r, metric) for r in grp]
            ax.plot(xs, ys, marker="o", label=f"{control}, k={k}")

        ax.set_xlabel("residual state index")
        ax.set_ylabel(metric.replace("_mean", ""))
        ax.set_title(title)
        ax.grid(True, alpha=0.25)

    axes[1].legend(fontsize=7, ncol=2)
    fig.suptitle("Experiment 3: layer sweep")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_leverage_damage(records: List[Dict[str, object]], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    for ax, ykey, title in [
        (axes[0], "surg_E_task_damage", "Task leverage vs task response damage"),
        (axes[1], "pred_drift", "Task leverage vs prediction drift"),
    ]:
        for r in records:
            x = f(r, "agg_leverage_task")
            y = f(r, ykey)
            if not (math.isfinite(x) and math.isfinite(y) and x > 0 and y > 0):
                continue
            control = str(r.get("control", ""))
            marker = "o" if control == "high_task" else "x"
            alpha = 0.9 if control == "high_task" else 0.45
            ax.scatter(x, y, marker=marker, alpha=alpha, s=35)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("aggregate task Galerkin leverage")
        ax.set_ylabel(ykey)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Args / main
# ---------------------------------------------------------------------------

def parse_k_list(s: str) -> List[int]:
    vals: List[int] = []
    for p in s.split(","):
        p = p.strip()
        if p:
            vals.append(int(p))
    if not vals:
        raise ValueError("Empty k-remove list")
    return vals


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 3: Galerkin-Leverage Surgery")

    p.add_argument("--checkpoint", default="final/linear_baseline_dx5_L8.pt")
    p.add_argument("--d-x", type=int, default=5)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--n-heads", type=int, default=4)

    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--n-ctx", type=int, default=47)
    p.add_argument("--n-tgt", type=int, default=16)
    p.add_argument("--sigma2", type=float, default=0.1)

    p.add_argument("--n-causal", type=int, default=32)
    p.add_argument("--eps", type=float, default=1e-3)

    p.add_argument("--tau-sv", type=float, default=1e-3)
    p.add_argument("--rmax", type=int, default=12)

    p.add_argument("--k-remove", type=int, default=None)
    p.add_argument("--k-remove-list", default="1,2,4")

    p.add_argument("--layer-rule", choices=["final", "sweep"], default="final")
    p.add_argument("--probe-kind", choices=["task", "iso", "both"], default="both")

    p.add_argument("--results-dir", default=str(SCRIPT_DIR / "results"))

    args = p.parse_args()

    if args.k_remove is not None:
        args.k_remove_list = [args.k_remove]
    else:
        args.k_remove_list = parse_k_list(args.k_remove_list)

    return args


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = get_device() if args.device == "auto" else torch.device(args.device)
    set_seed(args.seed)
    gen = torch.Generator(device="cpu").manual_seed(args.seed + 777)

    cfg = ModelCfg(
        checkpoint=args.checkpoint,
        d_x=args.d_x,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
    )

    model = load_model(cfg, device)

    (results_dir / "config.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "device": str(device),
                "checkpoint_dir": str(CHECKPOINT_DIR),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    all_records: List[Dict[str, object]] = []

    print("=== Experiment 3: Galerkin-Leverage Surgery ===", flush=True)
    print(f"checkpoint={args.checkpoint}, layer_rule={args.layer_rule}", flush=True)

    for ep in range(args.episodes):
        print(f"-- episode {ep+1}/{args.episodes} --", flush=True)
        recs = process_episode(model, cfg, args, ep, device, gen)
        all_records.extend(recs)

    write_csv(results_dir / "records.csv", all_records)

    summary = summarize(all_records, ["control", "k_remove", "state_idx"])
    write_csv(results_dir / "summary.csv", summary)
    write_report(results_dir / "summary.txt", summary, args)

    plot_final_controls(summary, results_dir / "final_state_controls.png", final_state_idx=args.n_layers)
    plot_layer_sweep(summary, results_dir / "layer_sweep.png")
    plot_leverage_damage(all_records, results_dir / "leverage_vs_damage.png")

    print("Wrote:")
    print(" ", results_dir / "records.csv")
    print(" ", results_dir / "summary.csv")
    print(" ", results_dir / "summary.txt")
    print(" ", results_dir / "final_state_controls.png")
    print(" ", results_dir / "layer_sweep.png")
    print(" ", results_dir / "leverage_vs_damage.png")


if __name__ == "__main__":
    main()
