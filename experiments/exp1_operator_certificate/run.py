#!/usr/bin/env python3
"""Experiment 1: operator-Galerkin certificate.

This is a direct implementation of the first experiment specified in
`native_operator_galerkin_paper.tex`.

For each episode and probe distribution it:
  * freezes the geometry and exact KRR operator T = K_t (K + sigma^2 I)^-1,
  * extracts raw and response-only final-state A-weighted SVD bases,
  * forms frozen Galerkin operators T_Q = K_t Q Q^T,
  * fits activation-mediated low-rank operators on build probes,
  * evaluates all operators on held-out finite-difference probes,
  * runs controls with matched ranks,
  * reports pointwise, operator, explained-fraction, residual-alignment, rank
    curve, and local additivity diagnostics.
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
from typing import Dict, List, Optional, Sequence, Tuple

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

from support import (  # noqa: E402
    MODELS,
    Config,
    checkpoint_path,
    get_device,
    load_model,
    sample_episode_batch,
    set_seed,
)


FLOOR = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Experiment 1 from native_operator_galerkin_paper.tex"
    )
    parser.add_argument("--model-key", choices=sorted(MODELS), default="standard")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--n-ctx", type=int, default=47)
    parser.add_argument("--n-tgt", type=int, default=3)
    parser.add_argument("--sigma2", type=float, default=0.1)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--tau-sv", type=float, default=1e-3)
    parser.add_argument(
        "--r-max",
        type=int,
        default=None,
        help="Predeclared maximum extraction rank. Defaults to n_layers + 1.",
    )
    parser.add_argument("--curve-r-max", type=int, default=20)
    parser.add_argument("--n-build", type=int, default=16)
    parser.add_argument("--n-eval", type=int, default=32)
    parser.add_argument("--n-additivity", type=int, default=4)
    parser.add_argument("--n-interp", type=int, default=16)
    parser.add_argument(
        "--interp-lambdas",
        default="0,0.1,0.25,0.5,0.75,0.9,1",
        help="Comma-separated lambda values for u=sqrt(1-lambda)u_task+sqrt(lambda)u_iso.",
    )
    parser.add_argument(
        "--probe-kinds",
        default="isotropic,label",
        help="Comma-separated probe distributions: isotropic,label",
    )
    parser.add_argument(
        "--response-basis-probe-kind",
        choices=["label", "isotropic"],
        default="label",
        help=(
            "Probe distribution used to build the response-only final-state basis. "
            "The default keeps the response basis task-local and then tests it "
            "on each evaluation probe distribution."
        ),
    )
    parser.add_argument(
        "--additivity-eps-scales",
        default="1,10",
        help="Comma-separated scales multiplying --eps for additivity.",
    )
    parser.add_argument(
        "--skip-shuffled-label-control",
        action="store_true",
        help="Skip the shuffled-label extraction control to save forward passes.",
    )
    parser.add_argument(
        "--results-dir",
        default=str(SCRIPT_DIR / "results"),
        help="Directory for CSV/JSON/PNG outputs.",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return get_device()
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA, but torch.cuda.is_available() is false.")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Requested MPS, but torch.backends.mps.is_available() is false.")
    return torch.device(name)


def build_config(args: argparse.Namespace) -> Config:
    model_cfg = MODELS[args.model_key]
    return Config(
        d_x=model_cfg["d_x"],
        d_model=model_cfg["d_model"],
        n_layers=args.n_layers,
        n_heads=model_cfg["n_heads"],
        ffn_mult=2,
        n_ctx=args.n_ctx,
        n_tgt=args.n_tgt,
        sigma2=args.sigma2,
        batch_size=1,
        seed=args.seed,
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def symmetric_eig_factors(A: torch.Tensor) -> Dict[str, torch.Tensor]:
    vals, vecs = torch.linalg.eigh(0.5 * (A + A.T))
    vals = vals.clamp(min=1e-12)
    sqrt_vals = vals.sqrt()
    inv_sqrt_vals = vals.rsqrt()
    sqrtA = (vecs * sqrt_vals.unsqueeze(0)) @ vecs.T
    invsqrtA = (vecs * inv_sqrt_vals.unsqueeze(0)) @ vecs.T
    return {"sqrtA": sqrtA, "invsqrtA": invsqrtA, "eigvals": vals, "eigvecs": vecs}


def a_orthonormalize(
    Z: torch.Tensor,
    A: torch.Tensor,
    tol: float = 1e-10,
    target_rank: Optional[int] = None,
) -> torch.Tensor:
    n = A.shape[0]
    if Z.numel() == 0 or Z.shape[1] == 0 or target_rank == 0:
        return torch.zeros(n, 0, dtype=A.dtype)
    if target_rank is not None:
        Z = Z[:, : max(target_rank, 0)]
    G = 0.5 * (Z.T @ A @ Z + Z.T @ A.T @ Z)
    vals, vecs = torch.linalg.eigh(G)
    order = torch.argsort(vals, descending=True)
    vals = vals[order]
    vecs = vecs[:, order]
    if len(vals) == 0 or vals[0] <= 0:
        return torch.zeros(n, 0, dtype=A.dtype)
    keep = vals > tol * vals[0].clamp(min=1e-30)
    if target_rank is not None:
        idx = torch.nonzero(keep, as_tuple=False).flatten()[:target_rank]
    else:
        idx = torch.nonzero(keep, as_tuple=False).flatten()
    if len(idx) == 0:
        return torch.zeros(n, 0, dtype=A.dtype)
    vals_keep = vals[idx]
    vecs_keep = vecs[:, idx]
    return Z @ vecs_keep @ torch.diag(vals_keep.rsqrt())


def rank_from_singular_values(
    singular_values: torch.Tensor,
    tau_sv: float,
    r_max: int,
    force_rank: Optional[int] = None,
) -> int:
    if singular_values.numel() == 0:
        return 0
    if force_rank is not None:
        return int(max(0, min(force_rank, singular_values.numel())))
    if singular_values[0] <= 0:
        return 0
    count = int((singular_values >= tau_sv * singular_values[0]).sum().item())
    return max(0, min(r_max, count))


def weighted_svd_basis(
    M: torch.Tensor,
    A_factors: Dict[str, torch.Tensor],
    tau_sv: float,
    r_max: int,
    force_rank: Optional[int] = None,
    curve_r_max: Optional[int] = None,
) -> Dict[str, torch.Tensor | int | List[float]]:
    n = A_factors["sqrtA"].shape[0]
    if M.numel() == 0 or M.shape[1] == 0:
        q_empty = torch.zeros(n, 0, dtype=A_factors["sqrtA"].dtype)
        return {"Q": q_empty, "Q_all": q_empty, "rank": 0, "singular_values": []}

    W = A_factors["sqrtA"] @ M
    U, S, _ = torch.linalg.svd(W, full_matrices=False)
    rank = rank_from_singular_values(S, tau_sv, r_max, force_rank=force_rank)
    keep_for_curve = curve_r_max if curve_r_max is not None else r_max
    keep = int(max(rank, min(keep_for_curve, U.shape[1])))
    Q_all = A_factors["invsqrtA"] @ U[:, :keep]
    Q = Q_all[:, :rank]
    return {
        "Q": Q,
        "Q_all": Q_all,
        "rank": rank,
        "singular_values": [float(x) for x in S.cpu().tolist()],
    }


def euclidean_svd_basis(M: torch.Tensor, A: torch.Tensor, rank: int) -> torch.Tensor:
    if rank <= 0 or M.numel() == 0 or M.shape[1] == 0:
        return torch.zeros(A.shape[0], 0, dtype=A.dtype)
    U, _, _ = torch.linalg.svd(M, full_matrices=False)
    return a_orthonormalize(U[:, :rank], A, target_rank=rank)


def random_a_basis(
    n: int,
    rank: int,
    A: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    if rank <= 0:
        return torch.zeros(n, 0, dtype=A.dtype)
    Z = torch.randn(n, rank, dtype=A.dtype, generator=generator)
    return a_orthonormalize(Z, A, target_rank=rank)


def polynomial_y_basis(
    y: torch.Tensor,
    A: torch.Tensor,
    rank: int,
    max_degree: int,
) -> torch.Tensor:
    if rank <= 0:
        return torch.zeros(A.shape[0], 0, dtype=A.dtype)
    cols = []
    v = y.clone()
    for _ in range(max_degree + 1):
        cols.append(v)
        v = A @ v
    Z = torch.stack(cols, dim=1)
    return a_orthonormalize(Z, A, target_rank=rank)


def sample_probes(
    kind: str,
    count: int,
    A: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    n = A.shape[0]
    if count <= 0:
        return torch.zeros(0, n, dtype=A.dtype)
    z = torch.randn(count, n, dtype=A.dtype, generator=generator)
    if kind == "isotropic":
        return z
    if kind == "label":
        jitter = 1e-10 * torch.eye(n, dtype=A.dtype)
        L = torch.linalg.cholesky(A + jitter)
        return z @ L.T
    raise ValueError(f"Unknown probe kind: {kind}")


@torch.no_grad()
def forward_model(
    model: torch.nn.Module,
    x_ctx: torch.Tensor,
    x_tgt: torch.Tensor,
    y_cpu: torch.Tensor,
    return_hidden: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    y_dev = y_cpu.view(1, -1).to(device=x_ctx.device, dtype=x_ctx.dtype)
    if return_hidden:
        preds, internals = model(x_ctx, y_dev, x_tgt, return_internals=True)
        H = internals["ctx_hidden_states"][-1][0].detach().cpu().double()
        return preds[0].detach().cpu().double(), H
    preds = model(x_ctx, y_dev, x_tgt)
    return preds[0].detach().cpu().double(), None


@torch.no_grad()
def finite_difference_bundle(
    model: torch.nn.Module,
    x_ctx: torch.Tensor,
    x_tgt: torch.Tensor,
    y: torch.Tensor,
    probes: torch.Tensor,
    eps: float,
    return_hidden: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    pred_cols: List[torch.Tensor] = []
    hidden_blocks: List[torch.Tensor] = []
    for u in probes:
        p_plus, h_plus = forward_model(model, x_ctx, x_tgt, y + eps * u, return_hidden)
        p_minus, h_minus = forward_model(model, x_ctx, x_tgt, y - eps * u, return_hidden)
        pred_cols.append(((p_plus - p_minus) / (2.0 * eps)).reshape(-1, 1))
        if return_hidden:
            assert h_plus is not None and h_minus is not None
            hidden_blocks.append((h_plus - h_minus) / (2.0 * eps))

    n_tgt = x_tgt.shape[1]
    pred_fd = (
        torch.cat(pred_cols, dim=1)
        if pred_cols
        else torch.zeros(n_tgt, 0, dtype=torch.float64)
    )
    hidden_fd = torch.cat(hidden_blocks, dim=1) if return_hidden and hidden_blocks else None
    return pred_fd, hidden_fd


@torch.no_grad()
def additivity_defect(
    model: torch.nn.Module,
    x_ctx: torch.Tensor,
    x_tgt: torch.Tensor,
    y: torch.Tensor,
    U: torch.Tensor,
    V: torch.Tensor,
    T: torch.Tensor,
    eps: float,
) -> float:
    numer = 0.0
    denom = 0.0
    for u, v in zip(U, V):
        p_pp, _ = forward_model(model, x_ctx, x_tgt, y + eps * u + eps * v, False)
        p_pm, _ = forward_model(model, x_ctx, x_tgt, y + eps * u - eps * v, False)
        p_mp, _ = forward_model(model, x_ctx, x_tgt, y - eps * u + eps * v, False)
        p_mm, _ = forward_model(model, x_ctx, x_tgt, y - eps * u - eps * v, False)
        C = p_pp - p_pm - p_mp + p_mm
        exact = T @ (eps * u + eps * v)
        numer += float((C * C).sum().item())
        denom += float((exact * exact).sum().item())
    return math.sqrt(numer / (denom + FLOOR))


def galerkin_operator(Kt: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    if Q.shape[1] == 0:
        return torch.zeros(Kt.shape[0], Kt.shape[1], dtype=Kt.dtype)
    return Kt @ Q @ Q.T


def actlr_operator(Q: torch.Tensor, build_probes: torch.Tensor, build_fd: torch.Tensor) -> torch.Tensor:
    n_tgt = build_fd.shape[0]
    n_ctx = build_probes.shape[1]
    if Q.shape[1] == 0:
        return torch.zeros(n_tgt, n_ctx, dtype=build_fd.dtype)
    coords = Q.T @ build_probes.T
    C = build_fd @ torch.linalg.pinv(coords)
    return C @ Q.T


def best_rank_operator(
    T: torch.Tensor,
    rank: int,
    probe_kind: str,
    build_probes: torch.Tensor,
) -> torch.Tensor:
    if rank <= 0:
        return torch.zeros_like(T)
    rank = min(rank, min(T.shape))
    if probe_kind == "isotropic":
        U, S, Vh = torch.linalg.svd(T, full_matrices=False)
        return (U[:, :rank] * S[:rank].unsqueeze(0)) @ Vh[:rank, :]

    # Empirical best rank-r operator under the build-probe covariance:
    # min_rank(S)<=r ||(S - T) C_xx^{1/2}||_F.
    X = build_probes.T
    Cxx = (X @ X.T) / max(1, X.shape[1])
    vals, vecs = torch.linalg.eigh(0.5 * (Cxx + Cxx.T))
    vals = vals.clamp(min=0.0)
    sqrt_vals = vals.sqrt()
    tol = 1e-12 * max(1.0, float(vals.max().item()))
    inv_sqrt_vals = torch.where(vals > tol, vals.rsqrt(), torch.zeros_like(vals))
    sqrtC = (vecs * sqrt_vals.unsqueeze(0)) @ vecs.T
    pinv_sqrtC = (vecs * inv_sqrt_vals.unsqueeze(0)) @ vecs.T
    M = T @ sqrtC
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    rank = min(rank, len(S))
    M_r = (U[:, :rank] * S[:rank].unsqueeze(0)) @ Vh[:rank, :]
    return M_r @ pinv_sqrtC


def operator_metrics(
    S: torch.Tensor,
    T: torch.Tensor,
    y: torch.Tensor,
    F_y: torch.Tensor,
    eval_probes: torch.Tensor,
    eval_fd: torch.Tensor,
) -> Dict[str, float]:
    U = eval_probes.T
    exact_eval = T @ U
    S_eval = S @ U
    denom = float((exact_eval * exact_eval).sum().item()) + FLOOR

    model_to_T = math.sqrt(float(((eval_fd - exact_eval) ** 2).sum().item()) / denom)
    model_to_S = math.sqrt(float(((eval_fd - S_eval) ** 2).sum().item()) / denom)
    S_to_T = math.sqrt(float(((S_eval - exact_eval) ** 2).sum().item()) / denom)
    model_to_zero = math.sqrt(float((eval_fd * eval_fd).sum().item()) / denom)
    zero_to_T = math.sqrt(float((exact_eval * exact_eval).sum().item()) / denom)

    use_ratio = model_to_S / (model_to_zero + FLOOR)
    sub_ratio = S_to_T / (zero_to_T + FLOOR)
    X_use = 1.0 - use_ratio * use_ratio
    X_sub = 1.0 - sub_ratio * sub_ratio

    r_model = eval_fd - S_eval
    r_sub = S_eval - exact_eval
    r_model_norm = math.sqrt(float((r_model * r_model).sum().item()))
    r_sub_norm = math.sqrt(float((r_sub * r_sub).sum().item()))
    cos_res = float((r_model * r_sub).sum().item()) / (r_model_norm * r_sub_norm + FLOOR)

    Ty = T @ y
    Sy = S @ y
    point_denom = float(Ty.norm().item()) + FLOOR
    return {
        "point_model_to_T": float((F_y - Ty).norm().item() / point_denom),
        "point_operator_to_T": float((Sy - Ty).norm().item() / point_denom),
        "point_model_to_operator": float((F_y - Sy).norm().item() / point_denom),
        "E_model_to_T": model_to_T,
        "E_model_to_operator": model_to_S,
        "E_operator_to_T": S_to_T,
        "E_model_to_zero": model_to_zero,
        "E_zero_to_T": zero_to_T,
        "R_use": use_ratio,
        "R_sub": sub_ratio,
        "X_use": X_use,
        "X_sub": X_sub,
        "cos_res": cos_res,
    }


def append_operator_record(
    records: List[Dict[str, object]],
    episode: int,
    probe_kind: str,
    method: str,
    operator_kind: str,
    rank: int,
    S: torch.Tensor,
    T: torch.Tensor,
    y: torch.Tensor,
    F_y: torch.Tensor,
    eval_probes: torch.Tensor,
    eval_fd: torch.Tensor,
    extra: Optional[Dict[str, object]] = None,
) -> None:
    row: Dict[str, object] = {
        "episode": episode,
        "probe_kind": probe_kind,
        "method": method,
        "operator_kind": operator_kind,
        "rank": rank,
    }
    row.update(operator_metrics(S, T, y, F_y, eval_probes, eval_fd))
    if extra:
        row.update(extra)
    records.append(row)


def rank_curve_records(
    episode: int,
    probe_kind: str,
    basis_name: str,
    Q_all: torch.Tensor,
    Kt: torch.Tensor,
    T: torch.Tensor,
    y: torch.Tensor,
    F_y: torch.Tensor,
    eval_probes: torch.Tensor,
    eval_fd: torch.Tensor,
    max_rank: int,
    predeclared_rank: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    max_rank = min(max_rank, Q_all.shape[1])
    for rank in range(max_rank + 1):
        Q = Q_all[:, :rank]
        S = galerkin_operator(Kt, Q)
        metrics = operator_metrics(S, T, y, F_y, eval_probes, eval_fd)
        row: Dict[str, object] = {
            "episode": episode,
            "probe_kind": probe_kind,
            "basis": basis_name,
            "rank": rank,
            "is_predeclared_rank": int(rank == predeclared_rank),
        }
        row.update(metrics)
        rows.append(row)
    return rows


def numeric_fields(rows: Sequence[Dict[str, object]]) -> List[str]:
    fields = set()
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float, np.integer, np.floating)) and key != "episode":
                fields.add(key)
    return sorted(fields)


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


def summarize(
    rows: Sequence[Dict[str, object]],
    group_keys: Sequence[str],
) -> List[Dict[str, object]]:
    groups: Dict[Tuple[object, ...], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k) for k in group_keys)].append(row)

    summaries: List[Dict[str, object]] = []
    fields = numeric_fields(rows)
    for key, vals in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        out: Dict[str, object] = {g: v for g, v in zip(group_keys, key)}
        out["n"] = len(vals)
        for field in fields:
            arr = np.array(
                [float(v[field]) for v in vals if field in v and v[field] not in ("", None)],
                dtype=float,
            )
            if len(arr) == 0:
                continue
            out[f"{field}_mean"] = float(arr.mean())
            out[f"{field}_std"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        summaries.append(out)
    return summaries


def plot_summary_errors(rows: Sequence[Dict[str, object]], path: Path) -> None:
    selected = [
        "raw",
        "resp",
        "raw_actLR",
        "resp_actLR",
        "raw_random",
        "resp_random",
        "raw_poly_y",
        "resp_poly_y",
        "raw_best_rank",
        "resp_best_rank",
    ]
    probe_kinds = sorted({str(r["probe_kind"]) for r in rows})
    if not probe_kinds:
        return

    fig, axes = plt.subplots(len(probe_kinds), 1, figsize=(11, 4.5 * len(probe_kinds)))
    if len(probe_kinds) == 1:
        axes = [axes]
    for ax, probe_kind in zip(axes, probe_kinds):
        vals_model = []
        vals_sub = []
        labels = []
        for method in selected:
            subset = [r for r in rows if r["probe_kind"] == probe_kind and r["method"] == method]
            if not subset:
                continue
            labels.append(method)
            vals_model.append(np.mean([float(r["E_model_to_operator"]) for r in subset]))
            vals_sub.append(np.mean([float(r["E_operator_to_T"]) for r in subset]))
        x = np.arange(len(labels))
        width = 0.38
        ax.bar(x - width / 2, vals_model, width, label="model to operator")
        ax.bar(x + width / 2, vals_sub, width, label="operator to KRR")
        ax.axhline(1.0, color="0.4", linestyle=":", linewidth=1.0)
        ax.set_title(f"Held-out errors, {probe_kind} probes")
        ax.set_ylabel("relative error")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_rank_curves(rows: Sequence[Dict[str, object]], path: Path) -> None:
    if not rows:
        return
    probe_kinds = sorted({str(r["probe_kind"]) for r in rows})
    bases = sorted({str(r["basis"]) for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    metrics = [("E_operator_to_T", "operator to KRR"), ("E_model_to_operator", "model to Galerkin")]
    for ax, (metric, title) in zip(axes, metrics):
        for probe_kind in probe_kinds:
            for basis in bases:
                subset = [r for r in rows if r["probe_kind"] == probe_kind and r["basis"] == basis]
                by_rank: Dict[int, List[float]] = defaultdict(list)
                for row in subset:
                    by_rank[int(row["rank"])].append(float(row[metric]))
                ranks = sorted(by_rank)
                if not ranks:
                    continue
                vals = [np.mean(by_rank[r]) for r in ranks]
                ax.plot(ranks, vals, marker="o", label=f"{basis}, {probe_kind}")
        ax.set_title(title)
        ax.set_xlabel("rank")
        ax.set_ylabel("relative error")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
    axes[1].legend(fontsize=8, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_additivity(rows: Sequence[Dict[str, object]], path: Path) -> None:
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    probe_kinds = sorted({str(r["probe_kind"]) for r in rows})
    for probe_kind in probe_kinds:
        subset = [r for r in rows if r["probe_kind"] == probe_kind]
        by_eps: Dict[float, List[float]] = defaultdict(list)
        for row in subset:
            by_eps[float(row["eps"])].append(float(row["additivity_defect"]))
        eps_vals = sorted(by_eps)
        vals = [np.mean(by_eps[e]) for e in eps_vals]
        ax.plot(eps_vals, vals, marker="o", label=probe_kind)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("epsilon")
    ax.set_ylabel("additivity defect")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_actlr_comparison(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    by_key: Dict[Tuple[object, object, str, str], Dict[str, object]] = {}
    for row in rows:
        method = str(row["method"])
        if method not in {"raw", "resp", "raw_actLR", "resp_actLR"}:
            continue
        basis = "raw" if method.startswith("raw") else "response"
        operator = "actLR" if method.endswith("actLR") else "TQ"
        by_key[(row["episode"], row["probe_kind"], basis, operator)] = row

    out: List[Dict[str, object]] = []
    keys = sorted({key[:3] for key in by_key})
    for episode, probe_kind, basis in keys:
        tq = by_key.get((episode, probe_kind, basis, "TQ"))
        act = by_key.get((episode, probe_kind, basis, "actLR"))
        if tq is None or act is None:
            continue
        out.append(
            {
                "episode": episode,
                "probe_kind": probe_kind,
                "basis": basis,
                "rank": tq["rank"],
                "TQ_E_model_to_operator": tq["E_model_to_operator"],
                "actLR_E_model_to_operator": act["E_model_to_operator"],
                "actLR_model_fit_gain": float(tq["E_model_to_operator"])
                - float(act["E_model_to_operator"]),
                "TQ_E_operator_to_T": tq["E_operator_to_T"],
                "actLR_E_operator_to_T": act["E_operator_to_T"],
                "actLR_KRR_cost": float(act["E_operator_to_T"]) - float(tq["E_operator_to_T"]),
                "TQ_X_use": tq["X_use"],
                "actLR_X_use": act["X_use"],
                "TQ_X_sub": tq["X_sub"],
                "actLR_X_sub": act["X_sub"],
            }
        )
    return out


def plot_actlr_comparison(rows: Sequence[Dict[str, object]], path: Path) -> None:
    if not rows:
        return
    groups = []
    for probe_kind in sorted({str(r["probe_kind"]) for r in rows}):
        for basis in ["raw", "response"]:
            subset = [r for r in rows if r["probe_kind"] == probe_kind and r["basis"] == basis]
            if subset:
                groups.append((probe_kind, basis, subset))

    labels = [f"{probe}\n{basis}" for probe, basis, _ in groups]
    x = np.arange(len(labels))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, metric_tq, metric_act, title in [
        (
            axes[0],
            "TQ_E_model_to_operator",
            "actLR_E_model_to_operator",
            "Model response fit",
        ),
        (
            axes[1],
            "TQ_E_operator_to_T",
            "actLR_E_operator_to_T",
            "KRR operator agreement",
        ),
    ]:
        tq_vals = [np.mean([float(r[metric_tq]) for r in subset]) for _, _, subset in groups]
        act_vals = [np.mean([float(r[metric_act]) for r in subset]) for _, _, subset in groups]
        ax.bar(x - width / 2, tq_vals, width, label="Galerkin T_Q")
        ax.bar(x + width / 2, act_vals, width, label="actLR")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].set_ylabel("relative error")
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def append_interpolation_records(
    records: List[Dict[str, object]],
    episode: int,
    lam: float,
    basis: str,
    operator_kind: str,
    rank: int,
    S: torch.Tensor,
    T: torch.Tensor,
    y: torch.Tensor,
    F_y: torch.Tensor,
    probes: torch.Tensor,
    fd: torch.Tensor,
) -> None:
    row: Dict[str, object] = {
        "episode": episode,
        "lambda": lam,
        "basis": basis,
        "operator_kind": operator_kind,
        "rank": rank,
    }
    row.update(operator_metrics(S, T, y, F_y, probes, fd))
    records.append(row)


def plot_interpolation(rows: Sequence[Dict[str, object]], path: Path) -> None:
    if not rows:
        return
    bases = [b for b in ["raw", "response"] if any(r["basis"] == b for r in rows)]
    fig, axes = plt.subplots(1, len(bases), figsize=(7 * len(bases), 4.8), sharey=True)
    if len(bases) == 1:
        axes = [axes]

    for ax, basis in zip(axes, bases):
        subset = [r for r in rows if r["basis"] == basis]
        lambdas = sorted({float(r["lambda"]) for r in subset})

        def mean_for(operator_kind: str, metric: str) -> List[float]:
            vals = []
            for lam in lambdas:
                s = [
                    float(r[metric])
                    for r in subset
                    if float(r["lambda"]) == lam and r["operator_kind"] == operator_kind
                ]
                vals.append(float(np.mean(s)) if s else float("nan"))
            return vals

        model_to_T = []
        for lam in lambdas:
            s = [float(r["E_model_to_T"]) for r in subset if float(r["lambda"]) == lam]
            model_to_T.append(float(np.mean(s)) if s else float("nan"))

        ax.plot(lambdas, model_to_T, "k--", marker="o", label="model to KRR")
        ax.plot(
            lambdas,
            mean_for("galerkin", "E_model_to_operator"),
            marker="o",
            label="model to T_Q",
        )
        ax.plot(
            lambdas,
            mean_for("galerkin", "E_operator_to_T"),
            marker="o",
            label="T_Q to KRR",
        )
        ax.plot(
            lambdas,
            mean_for("actLR", "E_model_to_operator"),
            marker="s",
            label="model to actLR",
        )
        ax.plot(
            lambdas,
            mean_for("actLR", "E_operator_to_T"),
            marker="s",
            label="actLR to KRR",
        )
        ax.set_title(f"{basis} basis")
        ax.set_xlabel("isotropic mixing lambda")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("relative error")
    axes[-1].legend(fontsize=8, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_text_summary(
    path: Path,
    args: argparse.Namespace,
    task_summary: Sequence[Dict[str, object]],
    actlr_summary: Sequence[Dict[str, object]],
    interp_summary: Sequence[Dict[str, object]],
    metric_summary: Sequence[Dict[str, object]],
    add_summary: Sequence[Dict[str, object]],
) -> None:
    lines = []
    lines.append("Experiment 1: Operator-Galerkin Certificate")
    lines.append("")
    lines.append(
        f"model={args.model_key}, episodes={args.episodes}, eps={args.eps}, "
        f"tau_sv={args.tau_sv}, r_max={args.r_max or args.n_layers + 1}"
    )
    if task_summary:
        row = task_summary[0]
        lines.append("")
        lines.append(
            "Predictive task check: "
            + f"MSE(model,true)={row.get('mse_model_true_mean', float('nan')):.4g}, "
            + f"MSE(KRR,true)={row.get('mse_krr_true_mean', float('nan')):.4g}, "
            + f"MSE ratio={row.get('ratio_model_krr_mean', float('nan')):.4g}, "
            + f"rel(model,KRR)={row.get('rel_model_krr_norm_mean', float('nan')):.4g}"
        )
    if actlr_summary:
        lines.append("")
        lines.append("Galerkin vs actLR:")
        for row in actlr_summary:
            lines.append(
                "  "
                + f"{row.get('probe_kind')}/{row.get('basis')}: "
                + f"TQ model-fit={row.get('TQ_E_model_to_operator_mean', float('nan')):.4g}, "
                + f"actLR model-fit={row.get('actLR_E_model_to_operator_mean', float('nan')):.4g}, "
                + f"TQ-to-KRR={row.get('TQ_E_operator_to_T_mean', float('nan')):.4g}, "
                + f"actLR-to-KRR={row.get('actLR_E_operator_to_T_mean', float('nan')):.4g}"
            )
    if interp_summary:
        lines.append("")
        lines.append("Probe interpolation endpoints:")
        for basis in ["raw", "response"]:
            for lam in [0.0, 1.0]:
                rows = [
                    r
                    for r in interp_summary
                    if r.get("basis") == basis
                    and abs(float(r.get("lambda", -1.0)) - lam) < 1e-12
                    and r.get("operator_kind") == "galerkin"
                ]
                if not rows:
                    continue
                row = rows[0]
                lines.append(
                    "  "
                    + f"{basis}, lambda={lam:g}: "
                    + f"E_model_to_T={row.get('E_model_to_T_mean', float('nan')):.4g}, "
                    + f"E_model_to_TQ={row.get('E_model_to_operator_mean', float('nan')):.4g}, "
                    + f"E_TQ_to_T={row.get('E_operator_to_T_mean', float('nan')):.4g}"
                )
    lines.append("")
    lines.append("Key held-out means:")
    wanted = {"raw", "resp", "raw_actLR", "resp_actLR", "raw_random", "resp_random"}
    for row in metric_summary:
        method = row.get("method")
        if method not in wanted:
            continue
        lines.append(
            "  "
            + f"{row.get('probe_kind')}/{method}: "
            + f"rank={row.get('rank_mean', float('nan')):.2f}, "
            + f"E_model_to_op={row.get('E_model_to_operator_mean', float('nan')):.4g}, "
            + f"E_op_to_T={row.get('E_operator_to_T_mean', float('nan')):.4g}, "
            + f"X_use={row.get('X_use_mean', float('nan')):.4g}, "
            + f"X_sub={row.get('X_sub_mean', float('nan')):.4g}"
        )
    if add_summary:
        lines.append("")
        lines.append("Additivity defects:")
        for row in add_summary:
            lines.append(
                "  "
                + f"{row.get('probe_kind')}, eps={row.get('eps')}: "
                + f"{row.get('additivity_defect_mean', float('nan')):.4g}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> None:
    args = parse_args()
    r_max = args.r_max or (args.n_layers + 1)
    args.r_max = r_max
    results_dir = Path(args.results_dir)
    ensure_dir(results_dir)

    probe_kinds = [p.strip() for p in args.probe_kinds.split(",") if p.strip()]
    bad = sorted(set(probe_kinds) - {"isotropic", "label"})
    if bad:
        raise ValueError(f"Unsupported probe kind(s): {bad}")
    eps_scales = [float(x.strip()) for x in args.additivity_eps_scales.split(",") if x.strip()]
    interp_lambdas = [float(x.strip()) for x in args.interp_lambdas.split(",") if x.strip()]
    if any(lam < 0.0 or lam > 1.0 for lam in interp_lambdas):
        raise ValueError("--interp-lambdas must stay in [0, 1].")

    set_seed(args.seed)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1000)
    device = resolve_device(args.device)
    cfg = build_config(args)
    ckpt = args.checkpoint or checkpoint_path(MODELS[args.model_key]["checkpoint"])
    model = load_model(ckpt, cfg, device)
    model.eval()

    config_payload = {
        "args": vars(args),
        "checkpoint": ckpt,
        "device": str(device),
        "repo_root": str(REPO_ROOT),
    }
    (results_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    metric_records: List[Dict[str, object]] = []
    rank_records: List[Dict[str, object]] = []
    add_records: List[Dict[str, object]] = []
    task_records: List[Dict[str, object]] = []
    interp_records: List[Dict[str, object]] = []

    for episode in range(args.episodes):
        batch = sample_episode_batch(args.model_key, cfg, device, return_target_kernel=True)
        x_ctx, y_ctx_dev, x_tgt, y_tgt_dev, K_batch, Kt_batch = batch
        y = y_ctx_dev[0].detach().cpu().double()
        y_tgt = y_tgt_dev[0].detach().cpu().double()
        K = K_batch[0].detach().cpu().double()
        Kt = Kt_batch[0].detach().cpu().double()
        n_ctx = K.shape[0]
        I = torch.eye(n_ctx, dtype=torch.float64)
        A = K + args.sigma2 * I
        T = Kt @ torch.linalg.solve(A, I)
        A_factors = symmetric_eig_factors(A)

        F_y, H_raw = forward_model(model, x_ctx, x_tgt, y, return_hidden=True)
        assert H_raw is not None
        Ty = T @ y
        mse_model_true = float(((F_y - y_tgt) ** 2).mean().item())
        mse_krr_true = float(((Ty - y_tgt) ** 2).mean().item())
        task_records.append(
            {
                "episode": episode,
                "model_key": args.model_key,
                "checkpoint": ckpt,
                "mse_model_true": mse_model_true,
                "mse_krr_true": mse_krr_true,
                "ratio_model_krr": mse_model_true / max(mse_krr_true, FLOOR),
                "mse_model_krr": float(((F_y - Ty) ** 2).mean().item()),
                "rel_model_krr_norm": float(F_y.sub(Ty).norm().item() / (Ty.norm().item() + FLOOR)),
                "rel_model_true_norm": float(F_y.sub(y_tgt).norm().item() / (y_tgt.norm().item() + FLOOR)),
                "krr_pred_norm": float(Ty.norm().item()),
                "model_pred_norm": float(F_y.norm().item()),
            }
        )
        raw_pack = weighted_svd_basis(
            H_raw,
            A_factors,
            args.tau_sv,
            r_max,
            curve_r_max=args.curve_r_max,
        )
        raw_rank = int(raw_pack["rank"])
        raw_Q = raw_pack["Q"]  # type: ignore[assignment]
        raw_Q_all = raw_pack["Q_all"]  # type: ignore[assignment]

        perm_y = torch.randperm(n_ctx, generator=generator)
        y_shuf = y[perm_y]
        _, H_raw_shuf = forward_model(model, x_ctx, x_tgt, y_shuf, return_hidden=True)
        assert H_raw_shuf is not None
        raw_shuf_pack = weighted_svd_basis(
            H_raw_shuf,
            A_factors,
            args.tau_sv,
            r_max,
            force_rank=raw_rank,
            curve_r_max=args.curve_r_max,
        )

        response_build_probes = sample_probes(
            args.response_basis_probe_kind, args.n_build, A, generator
        )
        response_build_fd, M_resp = finite_difference_bundle(
            model, x_ctx, x_tgt, y, response_build_probes, args.eps, return_hidden=True
        )
        assert M_resp is not None
        resp_pack = weighted_svd_basis(
            M_resp,
            A_factors,
            args.tau_sv,
            r_max,
            curve_r_max=args.curve_r_max,
        )
        resp_rank = int(resp_pack["rank"])
        resp_Q = resp_pack["Q"]  # type: ignore[assignment]
        resp_Q_all = resp_pack["Q_all"]  # type: ignore[assignment]

        if args.skip_shuffled_label_control:
            resp_shuf_Q = torch.zeros(n_ctx, 0, dtype=torch.float64)
        else:
            _, M_resp_shuf = finite_difference_bundle(
                model,
                x_ctx,
                x_tgt,
                y_shuf,
                response_build_probes,
                args.eps,
                return_hidden=True,
            )
            assert M_resp_shuf is not None
            resp_shuf_pack = weighted_svd_basis(
                M_resp_shuf,
                A_factors,
                args.tau_sv,
                r_max,
                force_rank=resp_rank,
                curve_r_max=args.curve_r_max,
            )
            resp_shuf_Q = resp_shuf_pack["Q"]  # type: ignore[assignment]

        raw_actlr_fit = actlr_operator(raw_Q, response_build_probes, response_build_fd)
        resp_actlr_fit = actlr_operator(resp_Q, response_build_probes, response_build_fd)

        for probe_kind in probe_kinds:
            build_probes = sample_probes(probe_kind, args.n_build, A, generator)
            eval_probes = sample_probes(probe_kind, args.n_eval, A, generator)
            add_U = sample_probes(probe_kind, args.n_additivity, A, generator)
            add_V = sample_probes(probe_kind, args.n_additivity, A, generator)

            eval_fd, _ = finite_difference_bundle(
                model, x_ctx, x_tgt, y, eval_probes, args.eps, return_hidden=False
            )

            for scale in eps_scales:
                eps_add = args.eps * scale
                add_records.append(
                    {
                        "episode": episode,
                        "probe_kind": probe_kind,
                        "eps": eps_add,
                        "additivity_defect": additivity_defect(
                            model, x_ctx, x_tgt, y, add_U, add_V, T, eps_add
                        ),
                    }
                )

            Kt_perm = Kt[:, torch.randperm(n_ctx, generator=generator)]
            basis_controls: List[Tuple[str, str, torch.Tensor, torch.Tensor, int, Dict[str, object]]] = [
                ("raw", "galerkin_main", raw_Q, Kt, raw_rank, {"basis_source": "raw"}),
                (
                    "resp",
                    "galerkin_main",
                    resp_Q,
                    Kt,
                    resp_rank,
                    {
                        "basis_source": "response",
                        "response_basis_probe_kind": args.response_basis_probe_kind,
                    },
                ),
                (
                    "raw_euclidean",
                    "galerkin_control",
                    euclidean_svd_basis(H_raw, A, raw_rank),
                    Kt,
                    raw_rank,
                    {"basis_source": "raw", "control": "euclidean_svd"},
                ),
                (
                    "resp_euclidean",
                    "galerkin_control",
                    euclidean_svd_basis(M_resp, A, resp_rank),
                    Kt,
                    resp_rank,
                    {
                        "basis_source": "response",
                        "control": "euclidean_svd",
                        "response_basis_probe_kind": args.response_basis_probe_kind,
                    },
                ),
                (
                    "raw_random",
                    "galerkin_control",
                    random_a_basis(n_ctx, raw_rank, A, generator),
                    Kt,
                    raw_rank,
                    {"basis_source": "raw", "control": "random_A_orthonormal"},
                ),
                (
                    "resp_random",
                    "galerkin_control",
                    random_a_basis(n_ctx, resp_rank, A, generator),
                    Kt,
                    resp_rank,
                    {
                        "basis_source": "response",
                        "control": "random_A_orthonormal",
                        "response_basis_probe_kind": args.response_basis_probe_kind,
                    },
                ),
                (
                    "raw_poly_y",
                    "galerkin_control",
                    polynomial_y_basis(y, A, raw_rank, args.n_layers),
                    Kt,
                    raw_rank,
                    {"basis_source": "raw", "control": "single_seed_polynomial_y"},
                ),
                (
                    "resp_poly_y",
                    "galerkin_control",
                    polynomial_y_basis(y, A, resp_rank, args.n_layers),
                    Kt,
                    resp_rank,
                    {
                        "basis_source": "response",
                        "control": "single_seed_polynomial_y",
                        "response_basis_probe_kind": args.response_basis_probe_kind,
                    },
                ),
                (
                    "raw_shuffled_labels",
                    "galerkin_control",
                    raw_shuf_pack["Q"],  # type: ignore[list-item]
                    Kt,
                    raw_rank,
                    {"basis_source": "raw", "control": "shuffled_labels"},
                ),
                (
                    "resp_shuffled_labels",
                    "galerkin_control",
                    resp_shuf_Q,
                    Kt,
                    resp_rank,
                    {
                        "basis_source": "response",
                        "control": "shuffled_labels",
                        "response_basis_probe_kind": args.response_basis_probe_kind,
                    },
                ),
                (
                    "raw_shuffled_Kt",
                    "galerkin_control",
                    raw_Q,
                    Kt_perm,
                    raw_rank,
                    {"basis_source": "raw", "control": "shuffled_Kt"},
                ),
                (
                    "resp_shuffled_Kt",
                    "galerkin_control",
                    resp_Q,
                    Kt_perm,
                    resp_rank,
                    {
                        "basis_source": "response",
                        "control": "shuffled_Kt",
                        "response_basis_probe_kind": args.response_basis_probe_kind,
                    },
                ),
            ]

            for method, operator_kind, Q, Kt_used, rank, extra in basis_controls:
                S = galerkin_operator(Kt_used, Q)
                append_operator_record(
                    metric_records,
                    episode,
                    probe_kind,
                    method,
                    operator_kind,
                    int(Q.shape[1]),
                    S,
                    T,
                    y,
                    F_y,
                    eval_probes,
                    eval_fd,
                    extra={**extra, "nominal_rank": rank},
                )

            for method, Q, rank, source in [
                ("raw_actLR", raw_Q, raw_rank, "raw"),
                ("resp_actLR", resp_Q, resp_rank, "response"),
            ]:
                S = raw_actlr_fit if source == "raw" else resp_actlr_fit
                append_operator_record(
                    metric_records,
                    episode,
                    probe_kind,
                    method,
                    "activation_low_rank",
                    int(Q.shape[1]),
                    S,
                    T,
                    y,
                    F_y,
                    eval_probes,
                    eval_fd,
                    extra={
                        "basis_source": source,
                        "nominal_rank": rank,
                        "actlr_fit_probe_kind": args.response_basis_probe_kind,
                    },
                )

            if probe_kind == "label" and args.n_interp > 0 and interp_lambdas:
                interp_task = sample_probes("label", args.n_interp, A, generator)
                interp_iso = sample_probes("isotropic", args.n_interp, A, generator)
                for lam in interp_lambdas:
                    probes = (
                        math.sqrt(1.0 - lam) * interp_task
                        + math.sqrt(lam) * interp_iso
                    )
                    fd, _ = finite_difference_bundle(
                        model, x_ctx, x_tgt, y, probes, args.eps, return_hidden=False
                    )
                    for basis, Q, rank, S_gal, S_act in [
                        ("raw", raw_Q, raw_rank, galerkin_operator(Kt, raw_Q), raw_actlr_fit),
                        ("response", resp_Q, resp_rank, galerkin_operator(Kt, resp_Q), resp_actlr_fit),
                    ]:
                        append_interpolation_records(
                            interp_records,
                            episode,
                            lam,
                            basis,
                            "galerkin",
                            int(Q.shape[1]),
                            S_gal,
                            T,
                            y,
                            F_y,
                            probes,
                            fd,
                        )
                        append_interpolation_records(
                            interp_records,
                            episode,
                            lam,
                            basis,
                            "actLR",
                            int(Q.shape[1]),
                            S_act,
                            T,
                            y,
                            F_y,
                            probes,
                            fd,
                        )

            for method, rank, source in [
                ("raw_best_rank", raw_rank, "raw"),
                ("resp_best_rank", resp_rank, "response"),
            ]:
                S = best_rank_operator(T, rank, probe_kind, build_probes)
                append_operator_record(
                    metric_records,
                    episode,
                    probe_kind,
                    method,
                    "best_rank_control",
                    rank,
                    S,
                    T,
                    y,
                    F_y,
                    eval_probes,
                    eval_fd,
                    extra={"basis_source": source},
                )

            rank_records.extend(
                rank_curve_records(
                    episode,
                    probe_kind,
                    "raw",
                    raw_Q_all,
                    Kt,
                    T,
                    y,
                    F_y,
                    eval_probes,
                    eval_fd,
                    args.curve_r_max,
                    raw_rank,
                )
            )
            rank_records.extend(
                rank_curve_records(
                    episode,
                    probe_kind,
                    "response",
                    resp_Q_all,
                    Kt,
                    T,
                    y,
                    F_y,
                    eval_probes,
                    eval_fd,
                    args.curve_r_max,
                    resp_rank,
                )
            )

        print(f"episode {episode + 1}/{args.episodes} done", flush=True)

    metric_summary = summarize(metric_records, ["probe_kind", "method", "operator_kind"])
    rank_summary = summarize(rank_records, ["probe_kind", "basis", "rank"])
    add_summary = summarize(add_records, ["probe_kind", "eps"])
    task_summary = summarize(task_records, ["model_key"])
    actlr_records = build_actlr_comparison(metric_records)
    actlr_summary = summarize(actlr_records, ["probe_kind", "basis"])
    interp_summary = summarize(interp_records, ["basis", "operator_kind", "lambda"])

    write_csv(results_dir / "task_metrics.csv", task_records)
    write_csv(results_dir / "task_summary.csv", task_summary)
    write_csv(results_dir / "actlr_comparison.csv", actlr_records)
    write_csv(results_dir / "actlr_comparison_summary.csv", actlr_summary)
    write_csv(results_dir / "probe_interpolation.csv", interp_records)
    write_csv(results_dir / "probe_interpolation_summary.csv", interp_summary)
    write_csv(results_dir / "episode_metrics.csv", metric_records)
    write_csv(results_dir / "summary_metrics.csv", metric_summary)
    write_csv(results_dir / "rank_curves.csv", rank_records)
    write_csv(results_dir / "rank_curve_summary.csv", rank_summary)
    write_csv(results_dir / "additivity.csv", add_records)
    write_csv(results_dir / "additivity_summary.csv", add_summary)

    plot_summary_errors(metric_records, results_dir / "summary_errors.png")
    plot_rank_curves(rank_records, results_dir / "rank_curves.png")
    plot_additivity(add_records, results_dir / "additivity.png")
    plot_actlr_comparison(actlr_records, results_dir / "actlr_comparison.png")
    plot_interpolation(interp_records, results_dir / "probe_interpolation.png")
    write_text_summary(
        results_dir / "summary.txt",
        args,
        task_summary,
        actlr_summary,
        interp_summary,
        metric_summary,
        add_summary,
    )

    print(f"wrote results to {results_dir}")


if __name__ == "__main__":
    run()
