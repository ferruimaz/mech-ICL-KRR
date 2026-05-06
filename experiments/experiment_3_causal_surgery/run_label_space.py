#!/usr/bin/env python3
"""Experiment 3: label-space Q projection diagnostic.

This replaces hidden-state complement surgery with an on-manifold label test.
For each episode, extract the raw final context basis Q with Q^T A Q = I, then
decompose the observed context labels as

    y_Q = A Q Q^T y,
    y_perp = y - y_Q.

Running the model on y_Q or y_perp recomputes all hidden states normally.  The
KRR prediction induced by y_Q is exactly T y_Q = K_t Q Q^T y = T_Q y, so this
is the input-space counterpart of the certified reduced operator.
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
from typing import Dict, List, Sequence

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mpl-cache"))

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.experiment_3_causal_surgery.run import (  # noqa: E402
    FLOOR,
    ModelCfg,
    build_kernels,
    extract_raw_final_basis,
    forward_base_with_states,
    get_device,
    load_model,
    sample_episode,
    set_seed,
)
from experiment_utils.layerwise_core import (  # noqa: E402
    effective_rank_T_excess_risk,
)


def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(((pred - target) ** 2).mean())


def linear_target_kernel(x_tgt: torch.Tensor) -> torch.Tensor:
    xt = x_tgt[0].detach().cpu().double()
    return xt @ xt.T


def rho_tq(T: torch.Tensor, TQ: torch.Tensor, A: torch.Tensor, risk_total: float) -> float:
    eigvals, eigvecs = torch.linalg.eigh(A)
    eigvals = eigvals.clamp_min(1e-12)
    A_sqrt = eigvecs @ torch.diag(torch.sqrt(eigvals)) @ eigvecs.T
    diff = (T - TQ) @ A_sqrt
    return float((diff * diff).sum() / (risk_total + FLOOR))


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
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["name"])].append(row)
    metrics = [
        "mse_ratio",
        "krr_input_ratio",
        "model_vs_input_krr",
        "pred_drift",
        "krr_drift",
        "rank_Q",
        "r_T",
        "rho_TQ",
        "q_label_energy_frac",
    ]
    out: List[Dict[str, object]] = []
    order = {"y": 0, "y_Q": 1, "y_perp": 2}
    for name, group in sorted(grouped.items(), key=lambda item: order.get(item[0], 99)):
        rec: Dict[str, object] = {"name": name, "n": len(group)}
        for metric in metrics:
            vals = [float(row[metric]) for row in group if row.get(metric) not in ("", None)]
            vals = [v for v in vals if math.isfinite(v)]
            if vals:
                arr = np.array(vals, dtype=float)
                rec[f"{metric}_mean"] = float(arr.mean())
                rec[f"{metric}_median"] = float(np.median(arr))
                rec[f"{metric}_std"] = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        out.append(rec)
    return out


def save_plot(out_dir: Path, summary: Sequence[Dict[str, object]]) -> None:
    names = [str(row["name"]) for row in summary]
    model = [float(row["mse_ratio_median"]) for row in summary]
    krr = [float(row["krr_input_ratio_median"]) for row in summary]
    x = np.arange(len(names))
    width = 0.36

    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.bar(x - width / 2, model, width, label="Transformer")
    ax.bar(x + width / 2, krr, width, label="KRR oracle")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([r"$y$", r"$y_Q$", r"$y_\perp$"])
    ax.set_ylabel("median MSE / MSE(KRR)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, axis="y", which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "label_space_mse_ratios.pdf")
    fig.savefig(out_dir / "label_space_mse_ratios.png", dpi=200)
    plt.close(fig)


def process_episode(model, cfg: ModelCfg, args, ep: int, device: torch.device) -> List[Dict[str, object]]:
    x_ctx, y_ctx, x_tgt, y_tgt = sample_episode(cfg, args.sigma2, args.n_ctx, args.n_tgt, device)
    y = y_ctx[0].detach().cpu().double()
    target = y_tgt[0].detach().cpu().double()

    _K, Kt, A, T = build_kernels(x_ctx, x_tgt, args.sigma2)
    Ktt = linear_target_kernel(x_tgt)
    risk = effective_rank_T_excess_risk(T, A, Kt, Ktt, args.excess_risk_frac)
    r_t = int(risk["r_eff_T_task"])

    base_pred, states = forward_base_with_states(model, x_ctx, y, x_tgt)
    Q, _svals = extract_raw_final_basis(states[-1], args.n_ctx, A, args.tau_sv, args.rmax)
    TQ = Kt @ Q @ Q.T if Q.shape[1] else torch.zeros_like(T)
    rho = rho_tq(T, TQ, A, float(risk["krr_risk_total"]))

    y_q = A @ Q @ (Q.T @ y) if Q.shape[1] else torch.zeros_like(y)
    y_perp = y - y_q
    labels = {"y": y, "y_Q": y_q, "y_perp": y_perp}
    preds = {"y": base_pred}
    for name, label in labels.items():
        if name == "y":
            continue
        pred, _ = forward_base_with_states(model, x_ctx, label, x_tgt)
        preds[name] = pred

    krr_outputs = {name: T @ label for name, label in labels.items()}
    base_krr_mse = mse(krr_outputs["y"], target)
    label_energy_denom = float(y @ torch.linalg.solve(A, y)) + FLOOR
    q_energy = float(y_q @ torch.linalg.solve(A, y_q)) / label_energy_denom

    rows: List[Dict[str, object]] = []
    for name in ["y", "y_Q", "y_perp"]:
        current_mse = mse(preds[name], target)
        input_krr_mse = mse(krr_outputs[name], target)
        rows.append(
            {
                "episode": ep,
                "name": name,
                "rank_Q": int(Q.shape[1]),
                "r_T": r_t,
                "rho_TQ": rho,
                "q_label_energy_frac": q_energy,
                "base_krr_mse": base_krr_mse,
                "mse": current_mse,
                "mse_ratio": current_mse / max(base_krr_mse, FLOOR),
                "krr_input_mse": input_krr_mse,
                "krr_input_ratio": input_krr_mse / max(base_krr_mse, FLOOR),
                "model_vs_input_krr": mse(preds[name], krr_outputs[name])
                / max(input_krr_mse, FLOOR),
                "pred_drift": float((preds[name] - preds["y"]).norm() / (preds["y"].norm() + FLOOR)),
                "krr_drift": float(
                    (krr_outputs[name] - krr_outputs["y"]).norm()
                    / (krr_outputs["y"].norm() + FLOOR)
                ),
            }
        )
    return rows


def write_summary_txt(path: Path, args, rows: Sequence[Dict[str, object]], summary: Sequence[Dict[str, object]]) -> None:
    base_rows = [row for row in rows if row["name"] == "y"]
    lines = [
        "Experiment 3 label-space Q projection",
        "",
        f"episodes={args.episodes}, seed={args.seed}, n_ctx={args.n_ctx}, n_tgt={args.n_tgt}",
        f"checkpoint={args.checkpoint}, eps not used, tau_sv={args.tau_sv}, rmax={args.rmax}",
        "",
        f"mean rank_Q={np.mean([float(row['rank_Q']) for row in base_rows]):.3g}",
        f"mean r_T={np.mean([float(row['r_T']) for row in base_rows]):.3g}",
        f"mean rho_TQ={np.mean([float(row['rho_TQ']) for row in base_rows]):.5g}",
        "",
        f"{'input':8s} {'n':>4s} {'MSE/KRR mean':>13s} {'MSE/KRR med':>12s} "
        f"{'KRR input med':>13s} {'drift med':>10s}",
    ]
    for row in summary:
        lines.append(
            f"{str(row['name']):8s} {int(row['n']):4d} "
            f"{float(row['mse_ratio_mean']):13.5g} {float(row['mse_ratio_median']):12.5g} "
            f"{float(row['krr_input_ratio_median']):13.5g} "
            f"{float(row['pred_drift_median']):10.5g}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(SCRIPT_DIR / "results_label_space"))
    parser.add_argument("--checkpoint", default="final/linear_baseline_dx5_L8.pt")
    parser.add_argument("--d-x", type=int, default=5)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--n-ctx", type=int, default=47)
    parser.add_argument("--n-tgt", type=int, default=16)
    parser.add_argument("--sigma2", type=float, default=0.1)
    parser.add_argument("--tau-sv", type=float, default=1e-3)
    parser.add_argument("--rmax", type=int, default=12)
    parser.add_argument("--excess-risk-frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=31415)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    cfg = ModelCfg(args.checkpoint, args.d_x, args.d_model, args.n_layers, args.n_heads)
    model = load_model(cfg, device)
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps({"args": vars(args), "device": str(device)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    rows: List[Dict[str, object]] = []
    for ep in range(args.episodes):
        print(f"episode {ep + 1}/{args.episodes}", flush=True)
        rows.extend(process_episode(model, cfg, args, ep, device))

    summary = summarize(rows)
    write_csv(out_dir / "records.csv", rows)
    write_csv(out_dir / "summary.csv", summary)
    write_summary_txt(out_dir / "summary.txt", args, rows, summary)
    save_plot(out_dir, summary)
    print((out_dir / "summary.txt").read_text(), flush=True)


if __name__ == "__main__":
    main()
