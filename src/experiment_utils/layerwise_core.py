"""Core model, kernel, and rank utilities for the final layerwise experiment."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from data import sample_batch_eigenvalues
from model import ICLTransformer

from .support import CHECKPOINT_DIR, compute_kernel, flat_eigenvalues


FLOOR = 1e-12


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


def checkpoint_file(checkpoint: str) -> str:
    return os.path.join(CHECKPOINT_DIR, checkpoint)


def load_model(cfg: CkptCfg, device: torch.device) -> torch.nn.Module:
    path = checkpoint_file(cfg.checkpoint)
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
    T = Kt @ torch.linalg.solve(A, eye)
    return K, Kt, A, T


def sample_rbf_episode(
    cfg: CkptCfg,
    args,
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
    args,
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
    args,
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


def build_eval_target_kernel(x_tgt: torch.Tensor, args) -> torch.Tensor:
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


def sample_task_probes(A: torch.Tensor, n: int, gen: torch.Generator) -> torch.Tensor:
    z = torch.randn(n, A.shape[0], dtype=A.dtype, generator=gen)
    jitter = 1e-10 * torch.eye(A.shape[0], dtype=A.dtype)
    L = torch.linalg.cholesky(A + jitter)
    return z @ L.T


@torch.no_grad()
def forward_with_ctx_hidden(
    model: ICLTransformer,
    x_ctx: torch.Tensor,
    y_cpu: torch.Tensor,
    x_tgt: torch.Tensor,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    dtype = next(model.parameters()).dtype
    device = x_ctx.device
    y = y_cpu.view(1, -1).to(device=device, dtype=dtype)
    x_ctx = x_ctx.to(device=device, dtype=dtype)
    x_tgt = x_tgt.to(device=device, dtype=dtype)

    batch_size, n_ctx, _ = x_ctx.shape
    n_tgt = x_tgt.shape[1]
    ctx = torch.cat(
        [
            x_ctx,
            y.unsqueeze(-1),
            torch.zeros(batch_size, n_ctx, 1, device=device, dtype=dtype),
        ],
        dim=-1,
    )
    tgt = torch.cat(
        [
            x_tgt,
            torch.zeros(batch_size, n_tgt, 1, device=device, dtype=dtype),
            torch.ones(batch_size, n_tgt, 1, device=device, dtype=dtype),
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
    return model(x_ctx.to(dtype=dtype), y, x_tgt.to(dtype=dtype))[0].detach().cpu().double()


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
    accum: Optional[List[List[torch.Tensor]]] = None
    for u in build_probes:
        _, hp = forward_with_ctx_hidden(model, x_ctx, y + eps * u, x_tgt)
        _, hn = forward_with_ctx_hidden(model, x_ctx, y - eps * u, x_tgt)
        z_list = [(a - b) / (2.0 * eps) for a, b in zip(hp, hn)]
        if accum is None:
            accum = [[] for _ in z_list]
        for layer, z in enumerate(z_list):
            accum[layer].append(z)
    if accum is None:
        _, states = forward_with_ctx_hidden(model, x_ctx, y, x_tgt)
        return [torch.zeros(state.shape[0], 0, dtype=torch.float64) for state in states]
    return [torch.cat(parts, dim=1) for parts in accum]


def psd_sqrt_and_invsqrt(
    A: torch.Tensor,
    floor: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    eigvals, eigvecs = torch.linalg.eigh(A)
    eigvals = eigvals.clamp_min(floor)
    A_sqrt = eigvecs @ torch.diag(torch.sqrt(eigvals)) @ eigvecs.T
    A_invsqrt = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals)) @ eigvecs.T
    return A_sqrt, A_invsqrt


def eval_operator_error(T: torch.Tensor, probes: torch.Tensor, eval_fd: torch.Tensor) -> float:
    U = probes.T
    TU = T @ U
    denom = float((TU * TU).sum()) + FLOOR
    return math.sqrt(float(((eval_fd - TU) ** 2).sum()) / denom)


def effective_rank_from_tail_budget(svals: torch.Tensor, tail_budget: float) -> int:
    if svals.numel() == 0:
        return 0
    energy = (svals.double() ** 2).clamp_min(0.0)
    budget = max(float(tail_budget), 0.0)
    for rank in range(energy.numel() + 1):
        tail = float(energy[rank:].sum()) if rank < energy.numel() else 0.0
        if tail <= budget + FLOOR:
            return int(rank)
    return int(energy.numel())


def effective_rank_from_svals(svals: torch.Tensor, eps: float) -> int:
    if svals.numel() == 0:
        return 0
    energy = (svals.double() ** 2).clamp_min(0.0)
    total = float(energy.sum())
    if total <= FLOOR:
        return 0
    tol = max(float(eps), 0.0)
    for rank in range(energy.numel() + 1):
        tail = float(energy[rank:].sum()) if rank < energy.numel() else 0.0
        if math.sqrt(tail / (total + FLOOR)) <= tol:
            return int(rank)
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


def final_linear_checkpoints() -> List[CkptCfg]:
    return [
        CkptCfg("dx3", "final/linear_sweep_dx3.pt", 3, 128, 8, 4, "dx", 3),
        CkptCfg("dx5", "final/linear_sweep_dx5.pt", 5, 128, 8, 4, "dx", 5),
        CkptCfg("dx8", "final/linear_sweep_dx8.pt", 8, 128, 8, 4, "dx", 8),
        CkptCfg("dx10", "final/linear_sweep_dx10.pt", 10, 128, 8, 4, "dx", 10),
        CkptCfg("dx15", "final/linear_sweep_dx15.pt", 15, 128, 8, 4, "dx", 15),
    ]


def cfg_by_name(candidates: Sequence[CkptCfg], name: str) -> CkptCfg:
    for cfg in candidates:
        if cfg.name == name or cfg.checkpoint == name:
            return cfg
    raise ValueError(f"unknown checkpoint: {name}")

