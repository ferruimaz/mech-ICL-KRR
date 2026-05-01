#!/usr/bin/env python3
"""Monte Carlo RBF-GP Galerkin rank experiment.

This script implements the task-side calculation in Section 8 of
``paper/final_paper.tex``.  It samples RBF episode geometries, computes the
GP-label task operator

    T_G C_G^{1/2} = K_t (K + sigma2 I)^{-1} C_G^{1/2},

and estimates the episode-level rank needed by reduced KRR trial spaces.  It
also reports the unconstrained SVD lower benchmark from the section.

For the noiseless GP setting, C_G=K.  The constrained Galerkin optimization has
no closed spectral form in general, so the script computes a nested greedy
Galerkin upper curve in whitened A-coordinates.  When C_G=A_G, it uses the
exact spectral formula from Section 7.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
MPL_CACHE_DIR = Path(os.environ.get("TMPDIR", "/tmp")) / "mech_icl_krr_mpl_cache"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FLOOR = 1e-12


def parse_float_list(text: str) -> List[float]:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError(f"expected at least one float in {text!r}")
    return vals


def parse_int_list(text: str) -> List[int]:
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError(f"expected at least one int in {text!r}")
    return vals


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    ensure_dir(path.parent)
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


def squared_distances(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    x1_sq = np.sum(X1 * X1, axis=1, keepdims=True)
    x2_sq = np.sum(X2 * X2, axis=1, keepdims=True).T
    return np.maximum(x1_sq + x2_sq - 2.0 * (X1 @ X2.T), 0.0)


def rbf_kernel(
    X1: np.ndarray,
    X2: np.ndarray,
    lengthscale: float,
    signal_var: float,
) -> np.ndarray:
    dist_sq = squared_distances(X1, X2)
    return signal_var * np.exp(-dist_sq / (2.0 * lengthscale * lengthscale))


def sym_eig_factors(
    M: np.ndarray,
    floor: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    vals, vecs = np.linalg.eigh(0.5 * (M + M.T))
    vals = np.maximum(vals, floor)
    sqrt = (vecs * np.sqrt(vals)[None, :]) @ vecs.T
    invsqrt = (vecs * (1.0 / np.sqrt(vals))[None, :]) @ vecs.T
    return sqrt, invsqrt, vals


def sample_geometry(
    d: int,
    n_ctx: int,
    n_tgt: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    X = rng.random((n_ctx, d), dtype=np.float64)
    Z = rng.random((n_tgt, d), dtype=np.float64)
    return X, Z


def label_covariance(
    K: np.ndarray,
    A: np.ndarray,
    mode: str,
    label_noise: float,
) -> np.ndarray:
    if mode == "gp":
        return K
    if mode == "noisy-gp":
        return K + (label_noise * label_noise) * np.eye(K.shape[0], dtype=np.float64)
    if mode == "matched-ridge":
        return A
    raise ValueError(f"unknown label covariance mode: {mode}")


def task_matrices(
    X: np.ndarray,
    Z: np.ndarray,
    lengthscale: float,
    sigma2: float,
    signal_var: float,
    label_cov_mode: str,
    label_noise: float,
) -> Dict[str, np.ndarray | float]:
    K = rbf_kernel(X, X, lengthscale, signal_var)
    Kt = rbf_kernel(Z, X, lengthscale, signal_var)
    eye = np.eye(K.shape[0], dtype=np.float64)
    A = K + sigma2 * eye
    C = label_covariance(K, A, label_cov_mode, label_noise)
    _A_sqrt, A_invsqrt, A_eigs = sym_eig_factors(A)
    C_sqrt, _C_invsqrt, C_eigs = sym_eig_factors(C)
    A_inv_Csqrt = np.linalg.solve(A, C_sqrt)

    B = Kt @ A_invsqrt
    D = A_invsqrt @ C_sqrt
    M = Kt @ A_inv_Csqrt
    denom = float(np.sum(M * M)) + FLOOR
    return {
        "K": K,
        "Kt": Kt,
        "A": A,
        "C": C,
        "A_invsqrt": A_invsqrt,
        "C_sqrt": C_sqrt,
        "B": B,
        "D": D,
        "M": M,
        "denom": denom,
        "A_cond": float(np.max(A_eigs) / np.min(A_eigs)),
        "C_cond_jittered": float(np.max(C_eigs) / np.min(C_eigs)),
    }


def tail_errors_from_svals(svals: np.ndarray, max_rank: int) -> List[float]:
    energy = np.asarray(svals, dtype=np.float64) ** 2
    total = float(np.sum(energy)) + FLOOR
    errors: List[float] = []
    for r in range(max_rank + 1):
        tail = float(np.sum(energy[r:])) if r < len(energy) else 0.0
        errors.append(math.sqrt(max(tail, 0.0) / total))
    return errors


def projection_error_sq(
    B: np.ndarray,
    D: np.ndarray,
    U: np.ndarray,
    denom: float,
) -> float:
    if U.size == 0 or U.shape[1] == 0:
        resid = B @ D
    else:
        resid = B @ (D - U @ (U.T @ D))
    return max(float(np.sum(resid * resid)) / denom, 0.0)


def orthonormalize(Z: np.ndarray) -> np.ndarray:
    if Z.size == 0 or Z.shape[1] == 0:
        return np.zeros((Z.shape[0], 0), dtype=np.float64)
    Q, R = np.linalg.qr(Z, mode="reduced")
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    return Q * signs[None, :]


def orthogonal_complement(U: np.ndarray) -> np.ndarray:
    n = U.shape[0]
    if U.shape[1] == 0:
        return np.eye(n, dtype=np.float64)
    full = np.concatenate([U, np.eye(n, dtype=np.float64)], axis=1)
    Q = orthonormalize(full)
    return Q[:, U.shape[1] :]


def top_eigvecs(M: np.ndarray, count: int) -> List[np.ndarray]:
    if M.size == 0:
        return []
    vals, vecs = np.linalg.eigh(0.5 * (M + M.T))
    order = np.argsort(vals)[::-1]
    return [vecs[:, idx].copy() for idx in order[:count]]


def normalized(v: np.ndarray) -> Optional[np.ndarray]:
    norm = float(np.linalg.norm(v))
    if norm <= FLOOR:
        return None
    return v / norm


def improve_direction(
    B: np.ndarray,
    D: np.ndarray,
    U: np.ndarray,
    N: np.ndarray,
    z_init: np.ndarray,
    denom: float,
    steps: int,
    restarts: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float]:
    z_best = normalized(z_init)
    if z_best is None:
        z_best = normalized(rng.standard_normal(N.shape[1]))
    assert z_best is not None

    u_best = N @ z_best
    U_try = np.concatenate([U, u_best.reshape(-1, 1)], axis=1)
    loss_best = projection_error_sq(B, D, U_try, denom)
    radius = 0.75

    for _ in range(max(0, steps)):
        candidates = [z_best]
        for _j in range(max(1, restarts)):
            proposal = normalized(z_best + radius * rng.standard_normal(N.shape[1]))
            if proposal is not None:
                candidates.append(proposal)

        improved = False
        for z in candidates:
            u = N @ z
            U_try = np.concatenate([U, u.reshape(-1, 1)], axis=1)
            loss = projection_error_sq(B, D, U_try, denom)
            if loss < loss_best:
                z_best = z
                u_best = u
                loss_best = loss
                improved = True
        radius *= 0.92 if improved else 0.80

    return u_best, loss_best


def polish_subspace(
    B: np.ndarray,
    D: np.ndarray,
    U_init: np.ndarray,
    denom: float,
    steps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if steps <= 0 or U_init.shape[1] == 0:
        return U_init

    U_best = U_init
    loss_best = projection_error_sq(B, D, U_best, denom)
    radius = 0.15
    for _ in range(steps):
        proposal = orthonormalize(U_best + radius * rng.standard_normal(U_best.shape))
        loss = projection_error_sq(B, D, proposal, denom)
        if loss < loss_best:
            U_best = proposal
            loss_best = loss
            radius *= 0.95
        else:
            radius *= 0.80
    return U_best


def greedy_galerkin_errors(
    B: np.ndarray,
    D: np.ndarray,
    max_rank: int,
    steps: int,
    restarts: int,
    eig_candidates: int,
    polish_steps: int,
    rng: np.random.Generator,
) -> List[float]:
    n = D.shape[0]
    max_rank = min(max_rank, n)
    denom = float(np.sum((B @ D) * (B @ D))) + FLOOR
    U = np.zeros((n, 0), dtype=np.float64)
    errors = [math.sqrt(projection_error_sq(B, D, U, denom))]

    S = B.T @ B
    R = D @ D.T
    H = 0.5 * (S @ R + R @ S)

    for _rank in range(1, max_rank + 1):
        N = orthogonal_complement(U)
        if N.shape[1] == 0:
            errors.append(0.0)
            continue

        local_mats = [N.T @ S @ N, N.T @ R @ N, N.T @ H @ N]
        seed_vectors: List[np.ndarray] = []
        for local in local_mats:
            seed_vectors.extend(top_eigvecs(local, eig_candidates))
        while len(seed_vectors) < max(1, restarts):
            seed_vectors.append(rng.standard_normal(N.shape[1]))

        best_u: Optional[np.ndarray] = None
        best_loss = float("inf")
        for seed in seed_vectors:
            u, loss = improve_direction(
                B,
                D,
                U,
                N,
                seed,
                denom,
                steps=steps,
                restarts=max(1, restarts),
                rng=rng,
            )
            if loss < best_loss:
                best_u = u
                best_loss = loss

        assert best_u is not None
        U = orthonormalize(np.concatenate([U, best_u.reshape(-1, 1)], axis=1))
        U = polish_subspace(B, D, U, denom, polish_steps, rng)
        err = math.sqrt(projection_error_sq(B, D, U, denom))
        errors.append(min(errors[-1], err))

    if max_rank == n:
        errors[-1] = 0.0
    return errors


def exact_matched_ridge_errors(Kt: np.ndarray, A_invsqrt: np.ndarray, max_rank: int) -> List[float]:
    B = Kt @ A_invsqrt
    svals = np.linalg.svd(B, compute_uv=False)
    return tail_errors_from_svals(svals, max_rank)


def first_rank_below(errors: Sequence[float], eps: float) -> int:
    for r, err in enumerate(errors):
        if err <= eps:
            return r
    return len(errors) - 1


def empirical_required_rank(ranks: Sequence[int], alpha: float) -> int:
    if not ranks:
        return 0
    sorted_ranks = sorted(int(x) for x in ranks)
    idx = max(0, math.ceil((1.0 - alpha) * len(sorted_ranks)) - 1)
    return sorted_ranks[min(idx, len(sorted_ranks) - 1)]


def eps_key(eps: float) -> str:
    return f"{eps:g}".replace(".", "p")


def summarize_records(
    records: List[Dict[str, object]],
    epsilons: Sequence[float],
    alpha: float,
) -> List[Dict[str, object]]:
    groups: Dict[Tuple[float, float, str, float], List[Dict[str, object]]] = defaultdict(list)
    for row in records:
        groups[
            (
                float(row["lengthscale"]),
                float(row["sigma2"]),
                str(row["label_covariance"]),
                float(row["label_noise"]),
            )
        ].append(row)

    summary: List[Dict[str, object]] = []
    for key, rows in sorted(groups.items()):
        lengthscale, sigma2, label_cov, label_noise = key
        for eps in epsilons:
            key_eps = eps_key(eps)
            gal = [int(row[f"r_galerkin_eps_{key_eps}"]) for row in rows]
            svd = [int(row[f"r_svd_lower_eps_{key_eps}"]) for row in rows]
            gal_req = empirical_required_rank(gal, alpha)
            svd_req = empirical_required_rank(svd, alpha)
            summary.append(
                {
                    "lengthscale": lengthscale,
                    "sigma2": sigma2,
                    "label_covariance": label_cov,
                    "label_noise": label_noise,
                    "epsilon": eps,
                    "alpha": alpha,
                    "episodes": len(rows),
                    "r_galerkin_mean": float(np.mean(gal)),
                    "r_galerkin_std": float(np.std(gal, ddof=1)) if len(gal) > 1 else 0.0,
                    "r_galerkin_median": float(np.median(gal)),
                    "r_galerkin_required": gal_req,
                    "r_svd_lower_mean": float(np.mean(svd)),
                    "r_svd_lower_std": float(np.std(svd, ddof=1)) if len(svd) > 1 else 0.0,
                    "r_svd_lower_median": float(np.median(svd)),
                    "r_svd_lower_required": svd_req,
                    "gap_required": gal_req - svd_req,
                }
            )
    return summary


def architecture_rows(
    summary: List[Dict[str, object]],
    n_ctx: int,
    depths: Sequence[int],
    widths: Sequence[int],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for item in summary:
        r_req = int(item["r_galerkin_required"])
        for L in depths:
            tri = (L + 1) * (L + 2) // 2
            required_width = math.ceil(r_req / tri) if tri > 0 else r_req
            for D_width in widths:
                capacity = min(n_ctx, D_width) * tri
                rows.append(
                    {
                        **item,
                        "L": L,
                        "D": D_width,
                        "triangular_depth_factor": tri,
                        "B_arch": capacity,
                        "passes_bound": int(capacity >= r_req),
                        "required_width_if_D_lt_n": required_width,
                    }
                )
    return rows


def median_curve(rows: List[Dict[str, object]], key: str, max_rank: int) -> List[float]:
    curves = [json.loads(str(row[key])) for row in rows]
    out: List[float] = []
    for r in range(max_rank + 1):
        vals = [float(curve[r]) for curve in curves if r < len(curve)]
        med = float(np.median(vals)) if vals else float("nan")
        out.append(max(med, 1e-12))
    return out


def plot_rank_curves(
    records: List[Dict[str, object]],
    out_path: Path,
    max_rank: int,
) -> None:
    groups: Dict[Tuple[float, float, str, float], List[Dict[str, object]]] = defaultdict(list)
    for row in records:
        groups[
            (
                float(row["lengthscale"]),
                float(row["sigma2"]),
                str(row["label_covariance"]),
                float(row["label_noise"]),
            )
        ].append(row)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ranks = list(range(max_rank + 1))
    for key, rows in sorted(groups.items()):
        lengthscale, sigma2, label_cov, label_noise = key
        gal = median_curve(rows, "galerkin_errors_json", max_rank)
        svd = median_curve(rows, "svd_lower_errors_json", max_rank)
        label = f"ell={lengthscale:g}, sigma2={sigma2:g}, {label_cov}"
        if label_cov == "noisy-gp":
            label += f", noise={label_noise:g}"
        ax.plot(ranks, gal, linewidth=1.8, label=label)
        ax.plot(ranks, svd, linewidth=1.2, linestyle="--", alpha=0.75)

    ax.set_xlabel("rank r")
    ax.set_ylabel("relative task-operator error")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_required_ranks(summary: List[Dict[str, object]], out_path: Path) -> None:
    if not summary:
        return
    epsilons = sorted({float(row["epsilon"]) for row in summary})
    groups = sorted(
        {
            (
                float(row["lengthscale"]),
                float(row["sigma2"]),
                str(row["label_covariance"]),
                float(row["label_noise"]),
            )
            for row in summary
        }
    )
    x = np.arange(len(groups))
    width = 0.8 / max(1, len(epsilons))

    fig, ax = plt.subplots(figsize=(max(7.5, 1.25 * len(groups)), 4.8))
    for i, eps in enumerate(epsilons):
        vals = []
        lower = []
        for group in groups:
            row = next(
                r
                for r in summary
                if (
                    float(r["lengthscale"]),
                    float(r["sigma2"]),
                    str(r["label_covariance"]),
                    float(r["label_noise"]),
                )
                == group
                and float(r["epsilon"]) == eps
            )
            vals.append(float(row["r_galerkin_required"]))
            lower.append(float(row["r_svd_lower_required"]))
        offset = (i - (len(epsilons) - 1) / 2.0) * width
        ax.bar(x + offset, vals, width=width, label=f"eps={eps:g}")
        ax.scatter(x + offset, lower, marker="_", color="black", s=90, zorder=3)

    labels = [f"ell={g[0]:g}\ns2={g[1]:g}" for g in groups]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("empirical required rank")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def write_summary_text(
    path: Path,
    summary: List[Dict[str, object]],
    args: argparse.Namespace,
) -> None:
    lines = [
        "RBF-GP Required Galerkin Rank",
        "",
        f"d={args.d}, n_ctx={args.n_ctx}, n_tgt={args.n_tgt}, episodes={args.episodes}",
        f"label_covariance={args.label_covariance}, alpha={args.alpha}",
        f"max_rank={args.max_rank or args.n_ctx}, greedy_steps={args.greedy_steps}, restarts={args.restarts}",
        "",
        "ell     sigma2  eps     r_GP    r_SVD   gap   mean_GP  std_GP",
    ]
    for row in summary:
        lines.append(
            f"{row['lengthscale']:<7g} {row['sigma2']:<7g} {row['epsilon']:<7g} "
            f"{row['r_galerkin_required']:<7.0f} {row['r_svd_lower_required']:<7.0f} "
            f"{row['gap_required']:<5.0f} {row['r_galerkin_mean']:<8.2f} "
            f"{row['r_galerkin_std']:<7.2f}"
        )
    lines.append("")
    lines.append("r_GP is the empirical (1-alpha)-quantile of the Galerkin curve.")
    lines.append("r_SVD is the unconstrained low-rank lower benchmark from Section 8.")
    if args.label_covariance != "matched-ridge":
        lines.append("For C_G != A_G, the Galerkin curve is a greedy upper curve.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def process_episode(
    args: argparse.Namespace,
    lengthscale: float,
    sigma2: float,
    episode: int,
    rng: np.random.Generator,
    epsilons: Sequence[float],
    max_rank: int,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    X, Z = sample_geometry(args.d, args.n_ctx, args.n_tgt, rng)
    mats = task_matrices(
        X,
        Z,
        lengthscale,
        sigma2,
        args.signal_var,
        args.label_covariance,
        args.label_noise,
    )
    M = np.asarray(mats["M"], dtype=np.float64)
    svd_errors = tail_errors_from_svals(np.linalg.svd(M, compute_uv=False), max_rank)

    if args.label_covariance == "matched-ridge":
        galerkin_errors = exact_matched_ridge_errors(
            np.asarray(mats["Kt"], dtype=np.float64),
            np.asarray(mats["A_invsqrt"], dtype=np.float64),
            max_rank,
        )
        galerkin_method = "exact-matched-ridge"
    else:
        galerkin_errors = greedy_galerkin_errors(
            np.asarray(mats["B"], dtype=np.float64),
            np.asarray(mats["D"], dtype=np.float64),
            max_rank=max_rank,
            steps=args.greedy_steps,
            restarts=args.restarts,
            eig_candidates=args.eig_candidates,
            polish_steps=args.polish_steps,
            rng=rng,
        )
        galerkin_method = "greedy-upper"

    row: Dict[str, object] = {
        "episode": episode,
        "d": args.d,
        "n_ctx": args.n_ctx,
        "n_tgt": args.n_tgt,
        "lengthscale": lengthscale,
        "sigma2": sigma2,
        "signal_var": args.signal_var,
        "label_covariance": args.label_covariance,
        "label_noise": args.label_noise,
        "galerkin_method": galerkin_method,
        "operator_energy": float(mats["denom"]),
        "A_cond": float(mats["A_cond"]),
        "C_cond_jittered": float(mats["C_cond_jittered"]),
        "galerkin_errors_json": json.dumps(galerkin_errors),
        "svd_lower_errors_json": json.dumps(svd_errors),
    }
    for eps in epsilons:
        key = eps_key(eps)
        row[f"r_galerkin_eps_{key}"] = first_rank_below(galerkin_errors, eps)
        row[f"r_svd_lower_eps_{key}"] = first_rank_below(svd_errors, eps)

    curve_rows: List[Dict[str, object]] = []
    for r, (gal, svd) in enumerate(zip(galerkin_errors, svd_errors)):
        curve_rows.append(
            {
                "episode": episode,
                "d": args.d,
                "n_ctx": args.n_ctx,
                "n_tgt": args.n_tgt,
                "lengthscale": lengthscale,
                "sigma2": sigma2,
                "label_covariance": args.label_covariance,
                "label_noise": args.label_noise,
                "rank": r,
                "galerkin_error": gal,
                "svd_lower_error": svd,
            }
        )
    return row, curve_rows


def run(args: argparse.Namespace) -> None:
    results_dir = Path(args.results_dir)
    ensure_dir(results_dir)
    epsilons = parse_float_list(args.epsilons)
    lengthscales = parse_float_list(args.lengthscales)
    sigma2_values = parse_float_list(args.sigma2_values)
    depths = parse_int_list(args.depths)
    widths = parse_int_list(args.widths)
    if any(x <= 0.0 for x in epsilons):
        raise ValueError("all --epsilons values must be positive")
    if any(x <= 0.0 for x in lengthscales):
        raise ValueError("all --lengthscales values must be positive")
    if any(x < 0.0 for x in sigma2_values):
        raise ValueError("all --sigma2-values values must be nonnegative")
    if any(x < 0 for x in depths) or any(x <= 0 for x in widths):
        raise ValueError("--depths must be nonnegative and --widths must be positive")

    (results_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    max_rank = args.max_rank if args.max_rank > 0 else args.n_ctx
    max_rank = min(max_rank, args.n_ctx)
    rng = np.random.default_rng(args.seed)

    records: List[Dict[str, object]] = []
    curve_rows: List[Dict[str, object]] = []

    print("=== RBF-GP required Galerkin rank ===", flush=True)
    print(
        f"d={args.d} n={args.n_ctx} m={args.n_tgt} episodes={args.episodes} "
        f"eps={epsilons} alpha={args.alpha}",
        flush=True,
    )

    for lengthscale in lengthscales:
        for sigma2 in sigma2_values:
            print(f"-- lengthscale={lengthscale:g} sigma2={sigma2:g} --", flush=True)
            for ep in range(args.episodes):
                row, rows = process_episode(
                    args,
                    lengthscale,
                    sigma2,
                    ep,
                    rng,
                    epsilons,
                    max_rank,
                )
                records.append(row)
                curve_rows.extend(rows)
                rank_bits = []
                for eps in epsilons:
                    key = eps_key(eps)
                    rank_bits.append(f"eps={eps:g}:r={row[f'r_galerkin_eps_{key}']}")
                print(f"  ep={ep + 1:03d}/{args.episodes} " + " ".join(rank_bits), flush=True)

    summary = summarize_records(records, epsilons, args.alpha)
    arch = architecture_rows(summary, args.n_ctx, depths, widths)

    write_csv(results_dir / "records.csv", records)
    write_csv(results_dir / "rank_curves.csv", curve_rows)
    write_csv(results_dir / "summary.csv", summary)
    write_csv(results_dir / "architecture_bounds.csv", arch)
    write_summary_text(results_dir / "summary.txt", summary, args)
    plot_rank_curves(records, results_dir / "rank_curves.png", max_rank)
    plot_required_ranks(summary, results_dir / "required_ranks.png")
    print(f"wrote results to {results_dir}", flush=True)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RBF-GP required Galerkin rank experiment")
    parser.add_argument("--results-dir", default=str(SCRIPT_DIR / "rbf_rank_results"))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--d", type=int, default=5)
    parser.add_argument("--n-ctx", type=int, default=47)
    parser.add_argument("--n-tgt", type=int, default=64)
    parser.add_argument("--lengthscales", default="0.15,0.3,0.6,1.2")
    parser.add_argument("--sigma2-values", default="0.1")
    parser.add_argument("--signal-var", type=float, default=1.0)
    parser.add_argument(
        "--label-covariance",
        choices=["gp", "noisy-gp", "matched-ridge"],
        default="gp",
        help="gp uses C=K; noisy-gp uses C=K+label_noise^2 I; matched-ridge uses C=A.",
    )
    parser.add_argument("--label-noise", type=float, default=0.0)
    parser.add_argument("--epsilons", default="0.1,0.05,0.02")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--max-rank", type=int, default=0, help="0 means n_ctx.")
    parser.add_argument("--greedy-steps", type=int, default=12)
    parser.add_argument("--restarts", type=int, default=8)
    parser.add_argument("--eig-candidates", type=int, default=3)
    parser.add_argument("--polish-steps", type=int, default=8)
    parser.add_argument("--depths", default="0,1,2,3,4,6,8,12")
    parser.add_argument("--widths", default="4,8,16,32,64,128")
    args = parser.parse_args(argv)
    if args.alpha <= 0.0 or args.alpha >= 1.0:
        raise ValueError("--alpha must be in (0,1)")
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.n_ctx <= 0 or args.n_tgt <= 0 or args.d <= 0:
        raise ValueError("--d, --n-ctx, and --n-tgt must be positive")
    if args.signal_var <= 0.0:
        raise ValueError("--signal-var must be positive")
    if args.max_rank < 0:
        raise ValueError("--max-rank must be nonnegative")
    if args.label_covariance == "noisy-gp" and args.label_noise < 0.0:
        raise ValueError("--label-noise must be nonnegative")
    if args.greedy_steps < 0 or args.restarts <= 0 or args.polish_steps < 0:
        raise ValueError(
            "--greedy-steps and --polish-steps must be nonnegative; "
            "--restarts must be positive"
        )
    if args.eig_candidates <= 0:
        raise ValueError("--eig-candidates must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
