#!/usr/bin/env python3
"""Population Mercer-spectrum estimate for the RBF task-visible rank.

For x ~ N(0, I_d) and an isotropic RBF kernel, the Mercer operator factorizes
across coordinates. We estimate the 1D Mercer eigenvalues by Gauss-Hermite
quadrature, enumerate the largest d-dimensional product eigenvalues, and plug
them into population approximations for the KRR recoverable energy.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from numpy.polynomial.hermite import hermgauss


FLOOR = 1e-14


def parse_int_list(text: str) -> List[int]:
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return vals


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


def one_dimensional_eigs(num_quad: int, lengthscale: float) -> np.ndarray:
    z, w = hermgauss(num_quad)
    x = np.sqrt(2.0) * z
    weights = w / np.sqrt(np.pi)
    sqdist = (x[:, None] - x[None, :]) ** 2
    kernel = np.exp(-sqdist / (2.0 * lengthscale * lengthscale))
    sym = np.sqrt(weights)[:, None] * kernel * np.sqrt(weights)[None, :]
    vals = np.linalg.eigvalsh(0.5 * (sym + sym.T))[::-1]
    return np.maximum(vals, 0.0)


def product_eigs(lam1: np.ndarray, d_x: int, max_modes: int, min_eig: float) -> Tuple[np.ndarray, List[Tuple[int, ...]]]:
    lam1 = lam1[lam1 > min_eig]
    start = (0,) * d_x
    heap: List[Tuple[float, Tuple[int, ...]]] = [(-(float(lam1[0]) ** d_x), start)]
    seen = {start}
    vals: List[float] = []
    indices: List[Tuple[int, ...]] = []
    while heap and len(vals) < max_modes:
        neg_val, idx = heapq.heappop(heap)
        vals.append(-neg_val)
        indices.append(idx)
        for axis in range(d_x):
            if idx[axis] + 1 >= len(lam1):
                continue
            nxt = list(idx)
            nxt[axis] += 1
            nxt_t = tuple(nxt)
            if nxt_t in seen:
                continue
            seen.add(nxt_t)
            prod = 1.0
            for i in nxt_t:
                prod *= float(lam1[i])
            heapq.heappush(heap, (-prod, nxt_t))
    return np.array(vals, dtype=np.float64), indices


def tail_rank(energy: np.ndarray, budget: float) -> int:
    tail = np.cumsum(energy[::-1])[::-1]
    for rank in range(len(energy) + 1):
        current = float(tail[rank]) if rank < len(tail) else 0.0
        if current <= budget + FLOOR:
            return rank
    return len(energy)


def self_consistent_risk(lam: np.ndarray, n_ctx: int, sigma2: float, tail_mass: float) -> Tuple[float, float]:
    """Solve eps = sum lambda*kappa/(n lambda+kappa), kappa=sigma2+eps.

    This is the standard random-design learning-curve correction. The truncated
    residual Mercer mass is treated as posterior risk, which is conservative for
    the very small omitted eigenvalues.
    """
    eps = min(max(tail_mass, 0.0) + sigma2, 1.0)
    for _ in range(10_000):
        kappa = sigma2 + eps
        new_eps = float(np.sum(lam * kappa / (n_ctx * lam + kappa))) + max(tail_mass, 0.0)
        if abs(new_eps - eps) < 1e-13:
            eps = new_eps
            break
        eps = 0.5 * eps + 0.5 * new_eps
    return eps, sigma2 + eps


def empirical_lookup(path: Path) -> Dict[Tuple[int, int], Dict[str, float]]:
    if not path.exists():
        return {}
    out: Dict[Tuple[int, int], Dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["n_ctx"]), int(row["n_tgt"]))
            out[key] = {
                "empirical_r_T": float(row["r_T_mean"]),
                "empirical_strict": float(row["r_T_strict_mean"]),
                "empirical_risk_per_tgt": float(row["krr_risk_per_tgt_mean"]),
                "empirical_signal_per_tgt": float(row["krr_signal_per_tgt_mean"]),
            }
    return out


def rows_for(
    lam: np.ndarray,
    total_mass: float,
    n_ctx_values: Iterable[int],
    n_tgt_values: Iterable[int],
    sigma2: float,
    alpha: float,
    empirical: Dict[Tuple[int, int], Dict[str, float]],
) -> List[Dict[str, object]]:
    tail_mass = max(0.0, 1.0 - total_mass)
    rows: List[Dict[str, object]] = []
    for n_ctx in n_ctx_values:
        naive_w = n_ctx * lam * lam / (n_ctx * lam + sigma2)
        naive_risk = float(np.sum(lam * sigma2 / (n_ctx * lam + sigma2))) + tail_mass
        naive_rank = tail_rank(naive_w, alpha * naive_risk)

        corrected_risk, kappa = self_consistent_risk(lam, n_ctx, sigma2, tail_mass)
        corrected_w = n_ctx * lam * lam / (n_ctx * lam + kappa)
        corrected_rank = tail_rank(corrected_w, alpha * corrected_risk)

        for n_tgt in n_tgt_values:
            rec: Dict[str, object] = {
                "n_ctx": n_ctx,
                "n_tgt": n_tgt,
                "cap": min(n_ctx, n_tgt),
                "naive_risk_per_tgt": naive_risk,
                "naive_signal_per_tgt": float(np.sum(naive_w)),
                "naive_rank": naive_rank,
                "naive_rank_capped": min(naive_rank, n_ctx, n_tgt),
                "corrected_kappa": kappa,
                "corrected_risk_per_tgt": corrected_risk,
                "corrected_signal_per_tgt": float(np.sum(corrected_w)),
                "corrected_rank": corrected_rank,
                "corrected_rank_capped": min(corrected_rank, n_ctx, n_tgt),
            }
            rec.update(empirical.get((n_ctx, n_tgt), {}))
            rows.append(rec)
    return rows


def write_analysis(path: Path, lam1: np.ndarray, lam: np.ndarray, rows: Sequence[Dict[str, object]], args: argparse.Namespace) -> None:
    lines = [
        "RBF population Mercer-spectrum rank estimate",
        "",
        f"d_x={args.d_x}, ell={args.kernel_lengthscale}, sigma2={args.sigma2}, alpha={args.excess_risk_frac}",
        f"Gauss-Hermite nodes={args.quad_nodes}, enumerated product modes={len(lam)}",
        "",
        "Top 1D Mercer eigenvalues:",
        " ".join(f"{x:.6g}" for x in lam1[:12]),
        "",
        "Top 5D product eigenvalues:",
        " ".join(f"{x:.6g}" for x in lam[:20]),
        "",
        "Population rank comparison:",
        "n_ctx n_tgt cap  naive_r risk_n  corr_r risk_c kappa  empirical_r empirical_strict empirical_risk",
    ]
    for row in rows:
        emp = row.get("empirical_r_T", float("nan"))
        strict = row.get("empirical_strict", float("nan"))
        emp_risk = row.get("empirical_risk_per_tgt", float("nan"))
        lines.append(
            f"{int(row['n_ctx']):5d} {int(row['n_tgt']):5d} {int(row['cap']):3d} "
            f"{int(row['naive_rank']):8d} {float(row['naive_risk_per_tgt']):6.4f} "
            f"{int(row['corrected_rank']):7d} {float(row['corrected_risk_per_tgt']):6.4f} "
            f"{float(row['corrected_kappa']):6.4f} "
            f"{float(emp):11.2f} {float(strict):15.2f} {float(emp_risk):14.4f}"
        )
    lines.extend(
        [
            "",
            "The naive column is the literal plug-in w_j=n*lambda_j^2/(n*lambda_j+sigma2).",
            "The corrected column replaces sigma2 by kappa=sigma2+posterior_risk, the standard random-design learning-curve correction.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="experiments/rbf_elbow_training/results/mercer_population_l3")
    parser.add_argument("--d-x", type=int, default=5)
    parser.add_argument("--kernel-lengthscale", type=float, default=3.0)
    parser.add_argument("--sigma2", type=float, default=0.1)
    parser.add_argument("--excess-risk-frac", type=float, default=0.05)
    parser.add_argument("--quad-nodes", type=int, default=200)
    parser.add_argument("--max-modes", type=int, default=200_000)
    parser.add_argument("--min-1d-eig", type=float, default=1e-16)
    parser.add_argument("--n-ctx-list", type=parse_int_list, default=parse_int_list("47,96,128,192,256"))
    parser.add_argument("--n-tgt-list", type=parse_int_list, default=parse_int_list("64,128,256"))
    parser.add_argument(
        "--empirical-summary",
        default="experiments/rbf_elbow_training/results/spectral_scan_risk_rank_seed42/summary.csv",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lam1 = one_dimensional_eigs(args.quad_nodes, args.kernel_lengthscale)
    lam, indices = product_eigs(lam1, args.d_x, args.max_modes, args.min_1d_eig)
    empirical = empirical_lookup(Path(args.empirical_summary))
    rows = rows_for(
        lam,
        float(lam.sum()),
        args.n_ctx_list,
        args.n_tgt_list,
        args.sigma2,
        args.excess_risk_frac,
        empirical,
    )

    write_csv(
        out_dir / "mercer_1d_eigenvalues.csv",
        [{"index": i, "lambda": float(v)} for i, v in enumerate(lam1)],
    )
    write_csv(
        out_dir / "mercer_5d_top_eigenvalues.csv",
        [
            {"rank": i + 1, "lambda": float(v), "multi_index": " ".join(map(str, idx))}
            for i, (v, idx) in enumerate(zip(lam[:1000], indices[:1000]))
        ],
    )
    write_csv(out_dir / "population_rank_summary.csv", rows)
    write_analysis(out_dir / "analysis.txt", lam1, lam, rows, args)
    print(out_dir / "analysis.txt", flush=True)


if __name__ == "__main__":
    main()
