#!/usr/bin/env python3
"""Experiment 2 native budget and closure.

Single-file version.

Worker mode:
    Runs one native-budget experiment for one exp and one smax.

    python -m experiments.exp2_budget_closure.run \
      --exp 2b \
      --episodes 4 \
      --n-build 8 \
      --n-eval 16 \
      --n-tgt 16 \
      --smax 4 \
      --probe-kind task \
      --results-dir experiments/exp2_budget_closure/results/2b_smax4

Suite mode:
    Runs 2a/2b over smax values, aggregates all summaries, and creates the
    main Experiment 2 resource-threshold figure.

    python -m experiments.exp2_budget_closure.run \
      --suite \
      --only both \
      --episodes 16 \
      --n-build 8 \
      --n-eval 32 \
      --n-tgt 16 \
      --smax-list 1,2,4 \
      --probe-kind task \
      --out-root experiments/exp2_budget_closure/results

This script computes activation-derived reachable subspaces from layerwise
finite-difference hidden responses:

    Z_l(u) = [H_l^ctx(y + eps u) - H_l^ctx(y - eps u)] / (2 eps),

then constructs A-novel innovations S_l, the reachable space

    R_nat = span{ A^t S_l : 0 <= l <= L, 0 <= t <= L-l },

and evaluates the frozen Galerkin operator

    T_Qnat = K_t Q_nat Q_nat^T.

Primary metrics:
  - E(F,T): model finite-difference operator error against exact KRR
  - E(T_Qnat,T): native Galerkin subspace adequacy
  - E(F,T_Qnat): model-to-native-Galerkin residual
  - B_nat = sum_l s_l (L-l+1)
  - dim R_nat and dim/n_ctx
  - max closure defect and max refinement fraction
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
SUPPORT_DIR = REPO_ROOT / "experiments" / "shared"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SUPPORT_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mpl-cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import ICLTransformer
from data import sample_batch_eigenvalues
from support import get_device, set_seed, flat_eigenvalues, CHECKPOINT_DIR

FLOOR = 1e-12


# ---------------------------------------------------------------------------
# Checkpoint registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CkptCfg:
    name: str
    checkpoint: str
    d_x: int
    d_model: int
    n_layers: int
    n_heads: int
    sweep: str
    sweep_val: float


CHECKPOINTS_2A: List[CkptCfg] = [
    CkptCfg("L2",  "model_L2.pt",  5, 128,  2, 4, "depth", 2),
    CkptCfg("L4",  "model_L4.pt",  5, 128,  4, 4, "depth", 4),
    CkptCfg("L6",  "model_L6.pt",  5, 128,  6, 4, "depth", 6),
    CkptCfg("L8",  "model_L8.pt",  5, 128,  8, 4, "anchor", 8),
    CkptCfg("L12", "model_L12.pt", 5, 128, 12, 4, "depth", 12),
    CkptCfg("H1",  "model_H1.pt",  5, 128,  8, 1, "head",  1),
    CkptCfg("H2",  "model_H2.pt",  5, 128,  8, 2, "head",  2),
    CkptCfg("H8",  "model_H8.pt",  5, 128,  8, 8, "head",  8),
]

CHECKPOINTS_2B: List[CkptCfg] = [
    CkptCfg("dx3",  "model_dx3.pt",   3, 128, 8, 4, "dx",  3),
    CkptCfg("dx5",  "model_dx5.pt",   5, 128, 8, 4, "dx",  5),
    CkptCfg("dx8",  "model_dx8.pt",   8, 128, 8, 4, "dx",  8),
    CkptCfg("dx10", "model_dx10.pt", 10, 128, 8, 4, "dx", 10),
    CkptCfg("dx15", "model_dx15.pt", 15, 128, 8, 4, "dx", 15),
    CkptCfg("dx30", "model_dx30.pt", 30, 128, 8, 4, "dx", 30),
]


# ---------------------------------------------------------------------------
# Model/data utilities
# ---------------------------------------------------------------------------

def load_model(cfg: CkptCfg, device: torch.device) -> torch.nn.Module:
    path = os.path.join(CHECKPOINT_DIR, cfg.checkpoint)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    model = ICLTransformer(cfg.d_x, cfg.d_model, cfg.n_layers, cfg.n_heads)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device).eval()
    return model


def sample_episode(
    cfg: CkptCfg,
    sigma2: float,
    n_ctx: int,
    n_tgt: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_ctx, y_ctx, x_tgt, y_tgt, _ = sample_batch_eigenvalues(
        1, cfg.d_x, n_ctx, n_tgt, sigma2, flat_eigenvalues, device="cpu"
    )
    return x_ctx.to(device), y_ctx.to(device), x_tgt.to(device), y_tgt.to(device)


def build_kernels(
    x_ctx: torch.Tensor,
    x_tgt: torch.Tensor,
    sigma2: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    xc = x_ctx[0].detach().cpu().double()
    xt = x_tgt[0].detach().cpu().double()

    K = xc @ xc.T
    Kt = xt @ xc.T
    eye = torch.eye(K.shape[0], dtype=torch.float64)
    A = K + sigma2 * eye

    A_inv = torch.linalg.solve(A, eye)
    T = Kt @ A_inv
    return K, Kt, A, T


def sample_task_probes(A: torch.Tensor, n: int, gen: torch.Generator) -> torch.Tensor:
    """Draw probes from the task label distribution N(0,A)."""
    z = torch.randn(n, A.shape[0], dtype=A.dtype, generator=gen)
    jitter = 1e-10 * torch.eye(A.shape[0], dtype=A.dtype)
    L = torch.linalg.cholesky(A + jitter)
    return z @ L.T


def sample_isotropic_probes(n_ctx: int, n: int, gen: torch.Generator) -> torch.Tensor:
    return torch.randn(n, n_ctx, dtype=torch.float64, generator=gen)


def sample_probes(
    A: torch.Tensor,
    n: int,
    kind: str,
    gen: torch.Generator,
) -> torch.Tensor:
    if kind == "task":
        return sample_task_probes(A, n, gen)
    if kind == "iso":
        return sample_isotropic_probes(A.shape[0], n, gen)
    raise ValueError(f"unknown probe kind: {kind}")


# ---------------------------------------------------------------------------
# Forward with context hidden states
# ---------------------------------------------------------------------------

@torch.no_grad()
def forward_with_ctx_hidden(
    model: ICLTransformer,
    x_ctx: torch.Tensor,
    y_cpu: torch.Tensor,
    x_tgt: torch.Tensor,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Forward pass returning final predictions and context residual states.

    Returns:
        preds: CPU float64, shape (n_tgt,)
        ctx_states: list of CPU float64 arrays, length L+1.
            ctx_states[0] is the embedding output on context tokens.
            ctx_states[l+1] is after transformer layer l.
            Each has shape (n_ctx, d_model).
    """
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
    ctx_states: List[torch.Tensor] = [h[0, :n_ctx, :].detach().cpu().double()]

    for layer in model.layers:
        h = layer(h)
        ctx_states.append(h[0, :n_ctx, :].detach().cpu().double())

    preds = model.head(h[:, n_ctx:, :]).squeeze(-1)[0].detach().cpu().double()
    return preds, ctx_states


@torch.no_grad()
def forward_pred_only(
    model: ICLTransformer,
    x_ctx: torch.Tensor,
    y_cpu: torch.Tensor,
    x_tgt: torch.Tensor,
) -> torch.Tensor:
    dtype = next(model.parameters()).dtype
    y = y_cpu.view(1, -1).to(device=x_ctx.device, dtype=dtype)
    return model(x_ctx, y, x_tgt)[0].detach().cpu().double()


@torch.no_grad()
def prediction_fd_bundle(
    model: ICLTransformer,
    x_ctx: torch.Tensor,
    x_tgt: torch.Tensor,
    y: torch.Tensor,
    probes: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    cols: List[torch.Tensor] = []
    for u in probes:
        pp = forward_pred_only(model, x_ctx, y + eps * u, x_tgt)
        pn = forward_pred_only(model, x_ctx, y - eps * u, x_tgt)
        cols.append(((pp - pn) / (2.0 * eps)).reshape(-1, 1))

    n_tgt = x_tgt.shape[1]
    return torch.cat(cols, dim=1) if cols else torch.zeros(n_tgt, 0, dtype=torch.float64)


def hidden_response_matrices(
    model: ICLTransformer,
    x_ctx: torch.Tensor,
    x_tgt: torch.Tensor,
    y: torch.Tensor,
    build_probes: torch.Tensor,
    eps: float,
) -> List[torch.Tensor]:
    """Return M_l=[Z_l(u_1),...,Z_l(u_m)] for all l=0..L.

    Each M_l has shape (n_ctx, n_build*d_model).
    """
    accum: Optional[List[List[torch.Tensor]]] = None

    for u in build_probes:
        _, hp = forward_with_ctx_hidden(model, x_ctx, y + eps * u, x_tgt)
        _, hn = forward_with_ctx_hidden(model, x_ctx, y - eps * u, x_tgt)
        z_list = [(a - b) / (2.0 * eps) for a, b in zip(hp, hn)]

        if accum is None:
            accum = [[] for _ in z_list]

        for l, z in enumerate(z_list):
            accum[l].append(z)

    assert accum is not None
    return [torch.cat(parts, dim=1) for parts in accum]


# ---------------------------------------------------------------------------
# Linear algebra for A-orthonormal bases
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


def a_project(Q: torch.Tensor, M: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    if Q.numel() == 0 or Q.shape[1] == 0:
        return torch.zeros_like(M)
    return Q @ (Q.T @ (A @ M))


def a_residual(Q: torch.Tensor, M: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    return M - a_project(Q, M, A)


def a_norm_fro(M: torch.Tensor, A: torch.Tensor) -> float:
    if M.numel() == 0:
        return 0.0
    AM = A @ M
    return math.sqrt(max(float((M * AM).sum()), 0.0))


def a_orth_basis_from_cols(
    M: torch.Tensor,
    A_sqrt: torch.Tensor,
    A_invsqrt: torch.Tensor,
    tau_rel: float,
    rmax: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """A-orthonormal basis for col(M) using weighted SVD of A^{1/2}M."""
    n = M.shape[0]
    if M.numel() == 0 or M.shape[1] == 0:
        return torch.zeros(n, 0, dtype=torch.float64), torch.zeros(0, dtype=torch.float64)

    W = A_sqrt @ M

    try:
        U, S, _Vh = torch.linalg.svd(W, full_matrices=False)
    except RuntimeError:
        U, S, _Vh = torch.linalg.svd(W + 1e-12 * torch.randn_like(W), full_matrices=False)

    if S.numel() == 0 or float(S[0]) <= 0:
        return torch.zeros(n, 0, dtype=torch.float64), S

    rank = int((S >= tau_rel * S[0]).sum().item())
    if rmax is not None:
        rank = min(rank, rmax)

    if rank <= 0:
        return torch.zeros(n, 0, dtype=torch.float64), S

    Q = A_invsqrt @ U[:, :rank]
    return Q.contiguous(), S


def powers_of_A(A: torch.Tensor, S: torch.Tensor, max_power: int) -> List[torch.Tensor]:
    if S.numel() == 0 or S.shape[1] == 0 or max_power < 0:
        return []

    out = [S]
    cur = S
    for _ in range(max_power):
        cur = A @ cur
        out.append(cur)
    return out


def reachable_basis(
    A: torch.Tensor,
    A_sqrt: torch.Tensor,
    A_invsqrt: torch.Tensor,
    S_list: Sequence[torch.Tensor],
    final_k: int,
    tau_subspace: float,
) -> torch.Tensor:
    cols: List[torch.Tensor] = []

    for ell, S in enumerate(S_list[: final_k + 1]):
        max_power = final_k - ell
        cols.extend(powers_of_A(A, S, max_power))

    n = A.shape[0]
    if not cols:
        return torch.zeros(n, 0, dtype=torch.float64)

    M = torch.cat(cols, dim=1)
    Q, _ = a_orth_basis_from_cols(M, A_sqrt, A_invsqrt, tau_rel=tau_subspace, rmax=n)
    return Q


# ---------------------------------------------------------------------------
# Native extractor
# ---------------------------------------------------------------------------

def build_native_basis(
    A: torch.Tensor,
    M_layers: List[torch.Tensor],
    tau_sv: float,
    tau_subspace: float,
    smax: int,
) -> Dict[str, object]:
    """Construct S_l, R_nat, B_nat, refinement fractions, and closure defects."""
    A_sqrt, A_invsqrt = psd_sqrt_and_invsqrt(A)
    L = len(M_layers) - 1
    n = A.shape[0]

    S_list: List[torch.Tensor] = []
    s_list: List[int] = []
    singvals: List[List[float]] = []

    for ell in range(L + 1):
        if ell > 0:
            Q_pre = reachable_basis(A, A_sqrt, A_invsqrt, S_list, ell - 1, tau_subspace)
        else:
            Q_pre = torch.zeros(n, 0, dtype=torch.float64)

        N_ell = a_residual(Q_pre, M_layers[ell], A)
        S_ell, Svals = a_orth_basis_from_cols(
            N_ell,
            A_sqrt,
            A_invsqrt,
            tau_rel=tau_sv,
            rmax=smax,
        )
        S_list.append(S_ell)
        s_list.append(S_ell.shape[1])
        singvals.append([float(x) for x in Svals[: min(10, Svals.numel())]])

    Q_nat = reachable_basis(A, A_sqrt, A_invsqrt, S_list, L, tau_subspace)
    B_nat = int(sum(s_list[ell] * (L - ell + 1) for ell in range(L + 1)))

    closure: List[float] = []
    refinement: List[float] = []

    for k in range(L + 1):
        M_k = M_layers[k]
        denom = a_norm_fro(M_k, A) + FLOOR

        Q_k = reachable_basis(A, A_sqrt, A_invsqrt, S_list, k, tau_subspace)
        residual = a_residual(Q_k, M_k, A)
        closure.append(a_norm_fro(residual, A) / denom)

        if k == 0:
            refinement.append(0.0)
        else:
            Q_prev = reachable_basis(A, A_sqrt, A_invsqrt, S_list, k - 1, tau_subspace)
            if Q_prev.shape[1] == 0:
                refinement.append(0.0)
            else:
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

    return {
        "Q_nat": Q_nat,
        "S_list": S_list,
        "s_list": s_list,
        "B_nat": B_nat,
        "dim_R_nat": Q_nat.shape[1],
        "closure": closure,
        "refinement": refinement,
        "singvals": singvals,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def operator_error(S: torch.Tensor, T: torch.Tensor, probes: torch.Tensor) -> float:
    U = probes.T
    TU = T @ U
    SU = S @ U
    denom = float((TU * TU).sum()) + FLOOR
    return math.sqrt(float(((SU - TU) ** 2).sum()) / denom)


def model_to_operator_error(
    eval_fd: torch.Tensor,
    S: torch.Tensor,
    T: torch.Tensor,
    probes: torch.Tensor,
) -> float:
    U = probes.T
    TU = T @ U
    SU = S @ U
    denom = float((TU * TU).sum()) + FLOOR
    return math.sqrt(float(((eval_fd - SU) ** 2).sum()) / denom)


def eval_operator_error(T: torch.Tensor, probes: torch.Tensor, eval_fd: torch.Tensor) -> float:
    U = probes.T
    TU = T @ U
    denom = float((TU * TU).sum()) + FLOOR
    return math.sqrt(float(((eval_fd - TU) ** 2).sum()) / denom)


def mse_and_pointwise(
    F_y: torch.Tensor,
    Ty: torch.Tensor,
    y_tgt: torch.Tensor,
) -> Dict[str, float]:
    mse_model = float(((F_y - y_tgt) ** 2).mean())
    mse_krr = float(((Ty - y_tgt) ** 2).mean())
    return {
        "mse_model": mse_model,
        "mse_krr": mse_krr,
        "mse_ratio": mse_model / max(mse_krr, FLOOR),
        "E_pointwise_F_T": float((F_y - Ty).norm() / (Ty.norm() + FLOOR)),
    }


def effective_rank_from_svals(svals: torch.Tensor, eps: float) -> int:
    if svals.numel() == 0:
        return 0
    energy = (svals.double() ** 2).clamp_min(0.0)
    total = float(energy.sum())
    if total <= FLOOR:
        return 0
    tol = max(float(eps), 0.0)
    for r in range(energy.numel() + 1):
        tail = float(energy[r:].sum()) if r < energy.numel() else 0.0
        if math.sqrt(tail / (total + FLOOR)) <= tol:
            return int(r)
    return int(energy.numel())


def effective_rank_from_tail_budget(svals: torch.Tensor, tail_budget: float) -> int:
    if svals.numel() == 0:
        return 0
    energy = (svals.double() ** 2).clamp_min(0.0)
    budget = max(float(tail_budget), 0.0)
    for r in range(energy.numel() + 1):
        tail = float(energy[r:].sum()) if r < energy.numel() else 0.0
        if tail <= budget + FLOOR:
            return int(r)
    return int(energy.numel())


def task_operator_svals(T: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    A_sqrt, _ = psd_sqrt_and_invsqrt(A)
    return torch.linalg.svdvals(T @ A_sqrt)


def effective_rank_T_task(T: torch.Tensor, A: torch.Tensor, eps: float) -> int:
    return effective_rank_from_svals(task_operator_svals(T, A), eps)


def krr_posterior_risk_total(Ktt: torch.Tensor, Kt: torch.Tensor, A: torch.Tensor) -> float:
    sol = torch.linalg.solve(A, Kt.T)
    posterior = Ktt - Kt @ sol
    return max(float(torch.trace(posterior)), 0.0)


def effective_rank_T_excess_risk(
    T: torch.Tensor,
    A: torch.Tensor,
    Kt: torch.Tensor,
    Ktt: torch.Tensor,
    excess_risk_frac: float,
) -> Dict[str, float]:
    svals = task_operator_svals(T, A)
    signal_total = float(((svals.double() ** 2).clamp_min(0.0)).sum())
    risk_total = krr_posterior_risk_total(Ktt, Kt, A)
    tail_budget = max(float(excess_risk_frac), 0.0) * risk_total
    rank = effective_rank_from_tail_budget(svals, tail_budget)
    tau_equiv = math.sqrt(tail_budget / (signal_total + FLOOR)) if signal_total > FLOOR else 0.0
    n_tgt = max(int(Ktt.shape[0]), 1)
    return {
        "r_eff_T_task": float(rank),
        "rank_tau_task": float(tau_equiv),
        "krr_risk_total": float(risk_total),
        "krr_signal_total": float(signal_total),
        "krr_risk_per_tgt": float(risk_total / n_tgt),
        "krr_signal_per_tgt": float(signal_total / n_tgt),
        "krr_risk_over_signal": float(risk_total / (signal_total + FLOOR)),
    }


# ---------------------------------------------------------------------------
# Per-checkpoint processing
# ---------------------------------------------------------------------------

def process_checkpoint(
    cfg: CkptCfg,
    args: argparse.Namespace,
    device: torch.device,
    gen: torch.Generator,
) -> List[Dict]:
    model = load_model(cfg, device)
    records: List[Dict] = []
    eval_kinds = ["task", "iso"] if args.probe_kind == "both" else [args.probe_kind]

    for ep in range(args.episodes):
        x_ctx, y_ctx, x_tgt, y_tgt = sample_episode(
            cfg,
            args.sigma2,
            args.n_ctx,
            args.n_tgt,
            device,
        )
        y = y_ctx[0].detach().cpu().double()
        y_tgt_cpu = y_tgt[0].detach().cpu().double()

        _K, Kt, A, T = build_kernels(x_ctx, x_tgt, args.sigma2)
        xt = x_tgt[0].detach().cpu().double()
        Ktt = xt @ xt.T
        Ty = T @ y

        F_y, _ = forward_with_ctx_hidden(model, x_ctx, y, x_tgt)
        base_metrics = mse_and_pointwise(F_y, Ty, y_tgt_cpu)

        build_probes = sample_task_probes(A, args.n_build, gen)
        M_layers = hidden_response_matrices(model, x_ctx, x_tgt, y, build_probes, args.eps)

        native = build_native_basis(
            A,
            M_layers,
            tau_sv=args.tau_sv,
            tau_subspace=args.tau_subspace,
            smax=args.smax,
        )
        Q_nat: torch.Tensor = native["Q_nat"]  # type: ignore[assignment]
        T_Q = Kt @ Q_nat @ Q_nat.T if Q_nat.shape[1] else torch.zeros_like(T)

        rT_strict = effective_rank_T_task(T, A, args.rank_tau)
        risk_rank = effective_rank_T_excess_risk(T, A, Kt, Ktt, args.excess_risk_frac)
        rT = int(risk_rank["r_eff_T_task"])
        B_nat = int(native["B_nat"])
        dim_R = int(native["dim_R_nat"])
        s_list = native["s_list"]
        closure = native["closure"]
        refinement = native["refinement"]

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
                "smax": args.smax,
                "tau_sv": args.tau_sv,
                "B_nat": B_nat,
                "dim_R_nat": dim_R,
                "dim_over_nctx": dim_R / A.shape[0],
                "r_eff_T_task": rT,
                "r_eff_T_task_strict": rT_strict,
                "dim_over_rT": dim_R / rT if rT > 0 else float("nan"),
                "dim_over_rT_strict": dim_R / rT_strict if rT_strict > 0 else float("nan"),
                "excess_risk_frac": args.excess_risk_frac,
                "E_F_T": E_F_T,
                "E_TQ_T": E_TQ_T,
                "E_F_TQ": E_F_TQ,
                "X_use": X_use,
                "X_sub": X_sub,
                "max_closure": max(float(x) for x in closure),
                "mean_closure": float(np.mean(closure)),
                "max_refinement": max(float(x) for x in refinement),
                "mean_refinement": float(np.mean(refinement)),
                "s_list": json.dumps(s_list),
            }
            rec.update(risk_rank)
            rec.update(base_metrics)
            records.append(rec)

            print(
                f"  {cfg.name:5s} ep={ep+1:03d}/{args.episodes} probe={kind:4s} "
                f"smax={args.smax} B_nat={B_nat:4d} dim={dim_R:2d}/{A.shape[0]} "
                f"E(TQ,T)={E_TQ_T:.5f} E(F,TQ)={E_F_TQ:.5f} E(F,T)={E_F_T:.5f} "
                f"mse_ratio={base_metrics['mse_ratio']:.3f} maxClos={max(closure):.4f}",
                flush=True,
            )

    return records


# ---------------------------------------------------------------------------
# IO and summaries
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: List[Dict]) -> None:
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


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarize(rows: List[Dict], group_keys: List[str]) -> List[Dict]:
    groups: Dict[tuple, List[Dict]] = defaultdict(list)

    for row in rows:
        groups[tuple(row.get(k) for k in group_keys)].append(row)

    out: List[Dict] = []
    num_keys = {
        k
        for row in rows
        for k, v in row.items()
        if isinstance(v, (int, float)) and k != "episode"
    }

    for group_tuple, group_rows in sorted(groups.items(), key=lambda x: tuple(str(v) for v in x[0])):
        out_row: Dict[str, object] = {key: value for key, value in zip(group_keys, group_tuple)}
        out_row["n"] = len(group_rows)

        for key in sorted(num_keys):
            vals = [float(r[key]) for r in group_rows if key in r and r[key] not in (None, "")]
            if not vals:
                continue
            out_row[f"{key}_mean"] = float(np.mean(vals))
            out_row[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

        out.append(out_row)

    return out


def write_summary(path: Path, summary: List[Dict], args: argparse.Namespace) -> None:
    lines = ["Experiment 2: Native Budget and Closure", ""]
    lines.append(f"episodes={args.episodes}, n_build={args.n_build}, n_eval={args.n_eval}, eps={args.eps}")
    lines.append(
        f"n_ctx={args.n_ctx}, n_tgt={args.n_tgt}, sigma2={args.sigma2}, "
        f"smax={args.smax}, tau_sv={args.tau_sv}, excess_risk_frac={args.excess_risk_frac}"
    )
    lines.append("")
    lines.append(
        f"{'name':6s} {'probe':5s} {'L':3s} {'H':3s} {'B_nat':8s} {'dim':6s} "
        f"{'rT':5s} {'rStrict':7s} {'E(TQ,T)':10s} {'E(F,TQ)':10s} "
        f"{'E(F,T)':10s} {'MSE/KRR':9s} {'maxClos':8s}"
    )

    for row in sorted(
        summary,
        key=lambda r: (
            str(r.get("probe_kind", "")),
            str(r.get("sweep", "")),
            float(r.get("sweep_val_mean", 0.0)),
        ),
    ):
        lines.append(
            f"{str(row['checkpoint']):6s} {str(row['probe_kind']):5s} "
            f"{row.get('n_layers_mean', float('nan')):3.0f} "
            f"{row.get('n_heads_mean', float('nan')):3.0f} "
            f"{row.get('B_nat_mean', float('nan')):8.1f} "
            f"{row.get('dim_R_nat_mean', float('nan')):6.1f} "
            f"{row.get('r_eff_T_task_mean', float('nan')):5.1f} "
            f"{row.get('r_eff_T_task_strict_mean', float('nan')):7.1f} "
            f"{row.get('E_TQ_T_mean', float('nan')):10.5f} "
            f"{row.get('E_F_TQ_mean', float('nan')):10.5f} "
            f"{row.get('E_F_T_mean', float('nan')):10.5f} "
            f"{row.get('mse_ratio_mean', float('nan')):9.3f} "
            f"{row.get('max_closure_mean', float('nan')):8.5f}"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_summary(summary: List[Dict], out_path: Path, probe_kind: str) -> None:
    rows = [r for r in summary if r.get("probe_kind") == probe_kind]
    if not rows:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    for ax, ykey, ylabel, title in [
        (axes[0], "E_TQ_T", r"$E(T_{Q_{nat}},T)$", "Subspace adequacy"),
        (axes[1], "E_F_TQ", r"$E(F,T_{Q_{nat}})$", "Model-Galerkin residual"),
        (axes[2], "E_F_T", r"$E(F,T)$", "Model-KRR residual"),
    ]:
        for row in rows:
            name = str(row["checkpoint"])
            x = float(row.get("B_nat_mean", 0.0))
            y = float(row.get(f"{ykey}_mean", float("nan")))
            ax.scatter(x, y, s=60, label=name)
            ax.text(x, y, name, fontsize=7)

        ax.set_xlabel(r"$B_{nat}$")
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.set_title(f"{title} ({probe_kind})")
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Worker mode
# ---------------------------------------------------------------------------

def run_worker(args: argparse.Namespace) -> None:
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = get_device() if args.device == "auto" else torch.device(args.device)
    set_seed(args.seed)
    gen = torch.Generator(device="cpu").manual_seed(args.seed + 12345)

    (results_dir / "config.json").write_text(
        json.dumps({"args": vars(args), "device": str(device)}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    cfgs: List[CkptCfg] = []
    if args.exp in ("2a", "both"):
        cfgs.extend(CHECKPOINTS_2A)
    if args.exp in ("2b", "both"):
        cfgs.extend(CHECKPOINTS_2B)

    all_records: List[Dict] = []

    print("=== Experiment 2: Native Budget and Closure ===", flush=True)

    for cfg in cfgs:
        print(f"-- {cfg.name} (L={cfg.n_layers}, H={cfg.n_heads}, d_x={cfg.d_x}) --", flush=True)
        all_records.extend(process_checkpoint(cfg, args, device, gen))

    write_csv(results_dir / "records.csv", all_records)
    summary = summarize(all_records, ["checkpoint", "probe_kind"])
    write_csv(results_dir / "summary.csv", summary)
    write_summary(results_dir / "summary.txt", summary, args)

    for pk in (["task", "iso"] if args.probe_kind == "both" else [args.probe_kind]):
        plot_summary(summary, results_dir / f"native_budget_{pk}.png", pk)

    print(f"wrote results to {results_dir}", flush=True)


# ---------------------------------------------------------------------------
# Suite mode
# ---------------------------------------------------------------------------

def parse_smax_list(s: str) -> List[int]:
    vals: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if part:
            vals.append(int(part))
    if not vals:
        raise ValueError("empty --smax-list")
    return vals


def run_suite_one(args: argparse.Namespace, exp: str, smax: int, out_dir: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "operator_galerkin_experiments.experiment_2_native_budget_closure.run_native",
        "--exp", exp,
        "--episodes", str(args.episodes),
        "--n-build", str(args.n_build),
        "--n-eval", str(args.n_eval),
        "--n-ctx", str(args.n_ctx),
        "--n-tgt", str(args.n_tgt),
        "--sigma2", str(args.sigma2),
        "--eps", str(args.eps),
        "--tau-sv", str(args.tau_sv),
        "--tau-subspace", str(args.tau_subspace),
        "--rank-tau", str(args.rank_tau),
        "--excess-risk-frac", str(args.excess_risk_frac),
        "--smax", str(smax),
        "--probe-kind", args.probe_kind,
        "--device", args.device,
        "--seed", str(args.seed),
        "--results-dir", str(out_dir),
    ]

    print("\n=== RUN", exp, "smax", smax, "===", flush=True)
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def float_get(row: Dict[str, object], key: str, default: float = float("nan")) -> float:
    try:
        v = row.get(key, default)
        if v in (None, ""):
            return default
        return float(v)  # type: ignore[arg-type]
    except Exception:
        return default


def aggregate_suite(
    out_root: Path,
    exps: List[str],
    smax_list: List[int],
) -> Tuple[List[Dict], List[Dict]]:
    summaries: List[Dict] = []
    records: List[Dict] = []

    for exp in exps:
        for smax in smax_list:
            d = out_root / f"{exp}_smax{smax}"

            for row in read_csv_rows(d / "summary.csv"):
                r = dict(row)
                r["suite_exp"] = exp
                r["suite_smax"] = smax
                summaries.append(r)

            for row in read_csv_rows(d / "records.csv"):
                r = dict(row)
                r["suite_exp"] = exp
                r["suite_smax"] = smax
                records.append(r)

    return summaries, records


def suite_color(smax: int) -> str:
    return {
        1: "#1f77b4",
        2: "#ff7f0e",
        4: "#2ca02c",
        8: "#d62728",
    }.get(smax, "#7f7f7f")


def suite_marker(name: str) -> str:
    if name.startswith("dx"):
        return {
            "dx3": "o",
            "dx5": "s",
            "dx8": "^",
            "dx10": "D",
            "dx15": "P",
        }.get(name, "o")

    return {
        "L2": "s",
        "L4": "^",
        "L6": "D",
        "L8": "o",
        "L12": "P",
        "H1": "X",
        "H2": "v",
        "H8": "h",
    }.get(name, "o")


def make_experiment_2_strong_figure(summaries: List[Dict], out_path: Path) -> None:
    rows_task = [r for r in summaries if str(r.get("probe_kind", "")) == "task"]
    rows_2b = [r for r in rows_task if str(r.get("suite_exp", "")) == "2b"]
    rows_2a = [r for r in rows_task if str(r.get("suite_exp", "")) == "2a"]

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    # Panel A: threshold law E(TQ,T) vs dim/rT
    ax = axes[0]
    for r in rows_2b:
        smax = int(float_get(r, "suite_smax"))
        name = str(r.get("checkpoint", ""))
        dim = float_get(r, "dim_R_nat_mean")
        rt = float_get(r, "r_eff_T_task_mean")
        y = float_get(r, "E_TQ_T_mean")
        yerr = float_get(r, "E_TQ_T_std", 0.0)

        if not math.isfinite(dim) or not math.isfinite(rt) or rt <= 0:
            continue

        x = dim / rt
        ax.errorbar(
            x,
            max(y, 1e-6),
            yerr=yerr if yerr > 0 else None,
            fmt=suite_marker(name),
            color=suite_color(smax),
            markersize=8,
            capsize=2,
            label=f"smax={smax}" if name == "dx3" else None,
        )
        ax.text(x * 1.01, max(y, 1e-6) * 1.08, name, fontsize=7)

    ax.axvline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_yscale("log")
    ax.set_xlabel(r"reachable ratio $\dim R_{nat}/r_T$")
    ax.set_ylabel(r"$E(T_{Q_{nat}},T)$")
    ax.set_title("A. Native rank threshold")
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8)

    # Panel B: model remains good while compressed Q can fail
    ax = axes[1]
    smax_values = sorted({int(float_get(r, "suite_smax")) for r in rows_2b})
    for smax in smax_values:
        rr = sorted(
            [r for r in rows_2b if int(float_get(r, "suite_smax")) == smax],
            key=lambda z: float_get(z, "r_eff_T_task_mean"),
        )
        xs = [float_get(r, "r_eff_T_task_mean") for r in rr]
        y_tq = [max(float_get(r, "E_TQ_T_mean"), 1e-6) for r in rr]
        y_f = [max(float_get(r, "E_F_T_mean"), 1e-6) for r in rr]

        ax.plot(xs, y_tq, "-", color=suite_color(smax), marker="o", label=f"TQ vs T, smax={smax}")
        ax.plot(xs, y_f, "--", color=suite_color(smax), marker="x", alpha=0.75, label=f"F vs T, smax={smax}")

    ax.set_yscale("log")
    ax.set_xlabel(r"task-visible rank $r_T$")
    ax.set_ylabel("operator error")
    ax.set_title("B. Compressed subspace vs model response")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)

    # Panel C: 2A model-use residual
    ax = axes[2]
    for r in rows_2a:
        smax = int(float_get(r, "suite_smax"))
        name = str(r.get("checkpoint", ""))
        x = max(float_get(r, "E_F_T_mean"), 1e-6)
        y = max(float_get(r, "E_F_TQ_mean"), 1e-6)
        dim = float_get(r, "dim_R_nat_mean")

        ax.scatter(
            x,
            y,
            s=45 + 6 * dim,
            marker=suite_marker(name),
            color=suite_color(smax),
            alpha=0.85,
        )
        if smax == 2:
            ax.text(x * 1.05, y * 1.05, name, fontsize=7)

    max_val = max([float_get(r, "E_F_TQ_mean") for r in rows_2a] + [1e-2]) * 1.4
    lims = [1e-3, max_val]
    ax.plot(lims, lims, "k--", linewidth=1, alpha=0.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(r"$E(F,T)$")
    ax.set_ylabel(r"$E(F,T_{Q_{nat}})$")
    ax.set_title("C. Low-rank 2A: model-use residual")
    ax.grid(True, alpha=0.25)

    fig.suptitle("Experiment 2: native Galerkin resource threshold and residual decomposition", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_suite_report(summaries: List[Dict], out_path: Path) -> None:
    rows_task = [r for r in summaries if str(r.get("probe_kind", "")) == "task"]
    rows_2b = [r for r in rows_task if str(r.get("suite_exp", "")) == "2b"]
    rows_2a = [r for r in rows_task if str(r.get("suite_exp", "")) == "2a"]

    lines: List[str] = []
    lines.append("Experiment 2 final suite summary")
    lines.append("")
    lines.append("Key derived quantity: ratio = dim_R_nat / r_eff_T_task.")
    lines.append("")

    if rows_2b:
        lines.append("=== 2B target-rank sweep: threshold table ===")
        lines.append(
            f"{'smax':>4s} {'name':>6s} {'rT':>6s} {'dim':>7s} {'ratio':>7s} "
            f"{'E(TQ,T)':>10s} {'E(F,T)':>10s} {'E(F,TQ)':>10s}"
        )
        for r in sorted(
            rows_2b,
            key=lambda z: (int(float_get(z, "suite_smax")), float_get(z, "r_eff_T_task_mean")),
        ):
            rt = float_get(r, "r_eff_T_task_mean")
            dim = float_get(r, "dim_R_nat_mean")
            ratio = dim / rt if rt > 0 else float("nan")
            lines.append(
                f"{int(float_get(r, 'suite_smax')):4d} {str(r.get('checkpoint','')):>6s} "
                f"{rt:6.2f} {dim:7.2f} {ratio:7.3f} "
                f"{float_get(r,'E_TQ_T_mean'):10.5f} "
                f"{float_get(r,'E_F_T_mean'):10.5f} "
                f"{float_get(r,'E_F_TQ_mean'):10.5f}"
            )
        lines.append("")

    if rows_2a:
        lines.append("=== 2A depth/head sweep: task-local use table ===")
        lines.append(
            f"{'smax':>4s} {'name':>6s} {'L':>3s} {'H':>3s} {'dim':>7s} "
            f"{'E(TQ,T)':>10s} {'E(F,T)':>10s} {'E(F,TQ)':>10s} {'maxClos':>9s}"
        )
        for r in sorted(
            rows_2a,
            key=lambda z: (int(float_get(z, "suite_smax")), str(z.get("checkpoint", ""))),
        ):
            lines.append(
                f"{int(float_get(r, 'suite_smax')):4d} {str(r.get('checkpoint','')):>6s} "
                f"{float_get(r,'n_layers_mean'):3.0f} "
                f"{float_get(r,'n_heads_mean'):3.0f} "
                f"{float_get(r,'dim_R_nat_mean'):7.2f} "
                f"{float_get(r,'E_TQ_T_mean'):10.5f} "
                f"{float_get(r,'E_F_T_mean'):10.5f} "
                f"{float_get(r,'E_F_TQ_mean'):10.5f} "
                f"{float_get(r,'max_closure_mean'):9.5f}"
            )
        lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_suite(args: argparse.Namespace) -> None:
    smax_list = parse_smax_list(args.smax_list)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    exps = ["2a", "2b"] if args.only == "both" else [args.only]

    (out_root / "suite_config.json").write_text(
        json.dumps({"args": vars(args), "smax_list": smax_list, "exps": exps}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if not args.skip_runs:
        for exp in exps:
            for smax in smax_list:
                run_suite_one(args, exp, smax, out_root / f"{exp}_smax{smax}")

    summaries, records = aggregate_suite(out_root, exps, smax_list)
    write_csv(out_root / "aggregate_summary.csv", summaries)
    write_csv(out_root / "aggregate_records.csv", records)
    make_experiment_2_strong_figure(summaries, out_root / "experiment_2_strong_resource_figure.png")
    write_suite_report(summaries, out_root / "final_report.txt")

    print("\nWrote:")
    print(" ", out_root / "aggregate_summary.csv")
    print(" ", out_root / "aggregate_records.csv")
    print(" ", out_root / "experiment_2_strong_resource_figure.png")
    print(" ", out_root / "final_report.txt")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 2 native budget and closure")

    # Suite mode.
    parser.add_argument("--suite", action="store_true", help="Run final suite over exps and smax values.")
    parser.add_argument("--only", choices=["2a", "2b", "both"], default="both")
    parser.add_argument("--smax-list", default="1,2,4")
    parser.add_argument("--out-root", default=str(SCRIPT_DIR / "results"))
    parser.add_argument("--skip-runs", action="store_true")

    # Worker mode.
    parser.add_argument("--exp", choices=["2a", "2b", "both"], default="2a")
    parser.add_argument("--results-dir", default=str(SCRIPT_DIR / "results"))

    # Shared args.
    parser.add_argument("--probe-kind", choices=["task", "iso", "both"], default="task")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--n-ctx", type=int, default=47)
    parser.add_argument("--n-tgt", type=int, default=16)
    parser.add_argument("--sigma2", type=float, default=0.1)
    parser.add_argument("--n-build", type=int, default=8)
    parser.add_argument("--n-eval", type=int, default=16)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--tau-sv", type=float, default=1e-3)
    parser.add_argument("--tau-subspace", type=float, default=1e-9)
    parser.add_argument("--smax", type=int, default=1)
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

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.suite:
        run_suite(args)
    else:
        run_worker(args)
