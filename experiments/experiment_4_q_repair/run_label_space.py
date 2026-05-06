#!/usr/bin/env python3
"""dx40 label-space Q projection and repair diagnostic.

For each dx40 episode, extract the final-layer response prefix Q and decompose
the context labels into

    y_Q = A Q Q^T y,
    y_perp = y - y_Q.

The model is then run normally on y, y_Q, and y_perp.  This is an on-manifold
alternative to residual-stream projection: no hidden state is edited.
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
MPL_CACHE_DIR = Path(os.environ.get("TMPDIR", "/tmp")) / "mech_icl_krr_mpl_cache"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.experiment_3_causal_surgery.run_dx40_complement import (  # noqa: E402
    select_response_prefix,
)
from experiments.experiment_4_q_repair.run import (  # noqa: E402
    parse_gamma_values,
    sample_dx40_batch_with_meta,
)
from experiment_utils.dx40 import (  # noqa: E402
    DX40_CHECKPOINT,
    Dx40SamplerCfg,
    dx40_forward_pred_only,
    dx40_hidden_response_matrices,
    load_dx40_model,
)
from experiment_utils.layerwise_core import (  # noqa: E402
    FLOOR,
    build_eval_kernels,
    build_eval_target_kernel,
    effective_rank_T_excess_risk,
    sample_task_probes,
)
from experiment_utils.support import set_seed  # noqa: E402


def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(((pred - target) ** 2).mean())


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


def scale_bin(log10_lambda1: float) -> str:
    x = float(log10_lambda1)
    if x < -1.0:
        return "[-2,-1)"
    if x < 0.0:
        return "[-1,0)"
    if x < 1.0:
        return "[0,1)"
    return "[1,2]"


def finite_vals(rows: Sequence[Dict[str, object]], metric: str) -> np.ndarray:
    vals = [float(row[metric]) for row in rows if row.get(metric) not in ("", None)]
    return np.array([v for v in vals if math.isfinite(v)], dtype=float)


def summarize(rows: Sequence[Dict[str, object]], keys: Sequence[str]) -> List[Dict[str, object]]:
    groups: Dict[tuple[str, ...], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row[key]) for key in keys)].append(row)

    metrics = [
        "mse_ratio",
        "krr_input_ratio",
        "model_vs_input_krr",
        "pred_drift",
        "krr_drift",
        "rank_Q",
        "r_T",
        "rho_TQ",
        "reach_alpha",
        "base_krr_mse",
        "q_label_energy_frac",
    ]
    name_order = {"y": 0, "y_Q": 1, "y_perp": 2}
    bin_order = {"[-2,-1)": 0, "[-1,0)": 1, "[0,1)": 2, "[1,2]": 3}

    def sort_key(item):
        key, _group = item
        parts = list(key)
        order = []
        for part in parts:
            if part in name_order:
                order.append(name_order[part])
            elif part in bin_order:
                order.append(bin_order[part])
            else:
                order.append(part)
        return order

    out: List[Dict[str, object]] = []
    for group_key, group in sorted(groups.items(), key=sort_key):
        rec: Dict[str, object] = {key: value for key, value in zip(keys, group_key)}
        rec["n"] = len(group)
        for metric in metrics:
            vals = finite_vals(group, metric)
            if vals.size:
                rec[f"{metric}_mean"] = float(vals.mean())
                rec[f"{metric}_median"] = float(np.median(vals))
                rec[f"{metric}_std"] = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
                rec[f"{metric}_q25"] = float(np.quantile(vals, 0.25))
                rec[f"{metric}_q75"] = float(np.quantile(vals, 0.75))
        return_key_vals = [row.get("reach_alpha") for row in group]
        if return_key_vals:
            reach_vals = np.array([float(v) for v in return_key_vals if v not in ("", None)], dtype=float)
            if reach_vals.size:
                rec["cert_fraction"] = float(np.mean(reach_vals >= 1.0))
        out.append(rec)
    return out


def scale_table(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    by_bin: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["name"] == "y":
            by_bin[str(row["scale_bin"])].append(row)
    out: List[Dict[str, object]] = []
    for bin_name in ["[-2,-1)", "[-1,0)", "[0,1)", "[1,2]"]:
        base_rows = by_bin.get(bin_name, [])
        if not base_rows:
            continue
        all_bin_rows = [row for row in rows if row["scale_bin"] == bin_name]
        by_name = defaultdict(list)
        for row in all_bin_rows:
            by_name[str(row["name"])].append(row)
        rec: Dict[str, object] = {
            "scale_bin": bin_name,
            "n": len(base_rows),
            "reach": float(np.mean([float(row["reach_alpha"]) >= 1.0 for row in base_rows])),
            "r_T_mean": float(np.mean([float(row["r_T"]) for row in base_rows])),
            "r_T_median": float(np.median([float(row["r_T"]) for row in base_rows])),
            "rank_Q_mean": float(np.mean([float(row["rank_Q"]) for row in base_rows])),
            "rank_Q_median": float(np.median([float(row["rank_Q"]) for row in base_rows])),
            "rho_TQ_mean": float(np.mean([float(row["rho_TQ"]) for row in base_rows])),
            "rho_TQ_median": float(np.median([float(row["rho_TQ"]) for row in base_rows])),
        }
        for name, prefix in [("y", "base"), ("y_Q", "label_keep"), ("y_perp", "label_perp")]:
            vals = finite_vals(by_name[name], "mse_ratio")
            if vals.size:
                rec[f"{prefix}_mean"] = float(vals.mean())
                rec[f"{prefix}_median"] = float(np.median(vals))
        out.append(rec)
    return out


def bank_distribution(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    base_rows = [row for row in rows if row["name"] == "y"]
    if not base_rows:
        return []
    rec: Dict[str, object] = {"n": len(base_rows)}
    for metric in ["mse_ratio", "r_T", "rank_Q", "rho_TQ", "base_krr_mse"]:
        vals = finite_vals(base_rows, metric)
        if vals.size:
            rec[f"{metric}_mean"] = float(vals.mean())
            rec[f"{metric}_median"] = float(np.median(vals))
            rec[f"{metric}_q25"] = float(np.quantile(vals, 0.25))
            rec[f"{metric}_q75"] = float(np.quantile(vals, 0.75))
            rec[f"{metric}_q90"] = float(np.quantile(vals, 0.90))
            rec[f"{metric}_q99"] = float(np.quantile(vals, 0.99))
            rec[f"{metric}_min"] = float(vals.min())
            rec[f"{metric}_max"] = float(vals.max())
    rec["certified"] = int(sum(float(row["reach_alpha"]) >= 1.0 for row in base_rows))
    rec["cert_fraction"] = float(rec["certified"] / len(base_rows))
    return [rec]


def outlier_table(rows: Sequence[Dict[str, object]], threshold: float) -> List[Dict[str, object]]:
    base_by_episode = {
        int(row["episode"]): row
        for row in rows
        if row["name"] == "y" and float(row["mse_ratio"]) >= float(threshold)
    }
    out: List[Dict[str, object]] = []
    for ep, base in sorted(base_by_episode.items(), key=lambda item: float(item[1]["mse_ratio"]), reverse=True):
        members = {str(row["name"]): row for row in rows if int(row["episode"]) == ep}
        rec = {
            "episode": ep,
            "bank_index": base["bank_index"],
            "scale_bin": base["scale_bin"],
            "log10_lambda1": base["log10_lambda1"],
            "tau_actual": base["tau_actual"],
            "n_ctx": base["n_ctx"],
            "n_tgt": base["n_tgt"],
            "r_T": base["r_T"],
            "rank_Q": base["rank_Q"],
            "rho_TQ": base["rho_TQ"],
            "reach_alpha": base["reach_alpha"],
            "krr_mse": base["base_krr_mse"],
            "base_mse_ratio": members["y"]["mse_ratio"],
            "label_keep_mse_ratio": members["y_Q"]["mse_ratio"],
            "label_perp_mse_ratio": members["y_perp"]["mse_ratio"],
            "label_keep_krr_ratio": members["y_Q"]["krr_input_ratio"],
            "label_perp_krr_ratio": members["y_perp"]["krr_input_ratio"],
        }
        out.append(rec)
    return out


def save_all_summary_plot(out_dir: Path, summary: Sequence[Dict[str, object]]) -> None:
    summary_by_name = {str(row["name"]): row for row in summary if "name" in row}
    names = ["y", "y_Q", "y_perp"]
    model = [float(summary_by_name[name]["mse_ratio_median"]) for name in names]
    krr = [float(summary_by_name[name]["krr_input_ratio_median"]) for name in names]
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
    fig.savefig(out_dir / "label_space_summary.pdf")
    fig.savefig(out_dir / "label_space_summary.png", dpi=200)
    plt.close(fig)


def save_scale_plot(out_dir: Path, scale_rows: Sequence[Dict[str, object]]) -> None:
    if not scale_rows:
        return
    bins = [str(row["scale_bin"]) for row in scale_rows]
    series = [
        ("base", [float(row["base_median"]) for row in scale_rows]),
        (r"$y_Q$", [float(row["label_keep_median"]) for row in scale_rows]),
        (r"$y_\perp$", [float(row["label_perp_median"]) for row in scale_rows]),
    ]
    x = np.arange(len(bins))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    for idx, (label, vals) in enumerate(series):
        ax.bar(x + (idx - 1) * width, vals, width, label=label)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(bins)
    ax.set_xlabel(r"$\log_{10}\lambda_1$")
    ax.set_ylabel("median MSE / MSE(KRR)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, axis="y", which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "label_space_by_scale.pdf")
    fig.savefig(out_dir / "label_space_by_scale.png", dpi=200)
    plt.close(fig)


def save_outlier_plot(out_dir: Path, outliers: Sequence[Dict[str, object]]) -> None:
    if not outliers:
        return
    labels = [str(row["bank_index"]) for row in outliers]
    x = np.arange(len(labels))
    width = 0.25
    series = [
        ("base", [float(row["base_mse_ratio"]) for row in outliers]),
        (r"$y_Q$", [float(row["label_keep_mse_ratio"]) for row in outliers]),
        (r"$y_\perp$", [float(row["label_perp_mse_ratio"]) for row in outliers]),
    ]
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    for idx, (label, vals) in enumerate(series):
        ax.bar(x + (idx - 1) * width, vals, width, label=label)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("bank id")
    ax.set_ylabel("MSE / MSE(KRR)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, axis="y", which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "label_space_outliers.pdf")
    fig.savefig(out_dir / "label_space_outliers.png", dpi=200)
    plt.close(fig)


def process_episode(
    model,
    args: argparse.Namespace,
    episode: int,
    bank_index: int,
    member: int,
    x_ctx: torch.Tensor,
    y_ctx: torch.Tensor,
    x_tgt: torch.Tensor,
    y_tgt: torch.Tensor,
    meta: Dict[str, object],
    gen: torch.Generator,
) -> List[Dict[str, object]]:
    y = y_ctx[0].detach().cpu().double()
    target = y_tgt[0].detach().cpu().double()
    _K, Kt, A, T = build_eval_kernels(x_ctx, x_tgt, args)
    Ktt = build_eval_target_kernel(x_tgt, args)
    risk = effective_rank_T_excess_risk(T, A, Kt, Ktt, args.excess_risk_frac)
    r_t = int(risk["r_eff_T_task"])

    build_probes = sample_task_probes(A, args.n_build, gen)
    response_layers = dx40_hidden_response_matrices(model, x_ctx, x_tgt, y, build_probes, args.eps)
    Q, q_meta = select_response_prefix(
        response_layers[args.basis_layer],
        A,
        Kt,
        T,
        float(risk["krr_risk_total"]),
        args.excess_risk_frac,
        args.tau_sv,
    )
    y_q = A @ Q @ (Q.T @ y) if Q.shape[1] else torch.zeros_like(y)
    y_perp = y - y_q
    labels = {"y": y, "y_Q": y_q, "y_perp": y_perp}
    preds = {name: dx40_forward_pred_only(model, x_ctx, label, x_tgt) for name, label in labels.items()}
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
                "episode": episode,
                "bank_index": bank_index,
                "bank_batch_index": meta.get("bank_batch_index", ""),
                "bank_batch_member": member,
                "name": name,
                "sampling_mode": args.sampling_mode,
                "profile": meta.get("profile", ""),
                "tau_actual": meta.get("tau_actual", ""),
                "log10_lambda1": meta.get("log10_lambda1", ""),
                "scale_bin": scale_bin(float(meta.get("log10_lambda1", 0.0))),
                "lambda1": meta.get("lambda1", ""),
                "gamma": meta.get("gamma", ""),
                "n_ctx": int(x_ctx.shape[1]),
                "n_tgt": int(x_tgt.shape[1]),
                "r_T": r_t,
                "rank_Q": q_meta["rank_Q"],
                "candidate_dim": q_meta["candidate_dim"],
                "rank_to_alpha": q_meta["rank_to_alpha"],
                "best_rank": q_meta["best_rank"],
                "rho_TQ": q_meta["rho_TQ"],
                "best_rho": q_meta["best_rho"],
                "reach_alpha": q_meta["reach_alpha"],
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


def run_bank(model, args: argparse.Namespace, device: torch.device) -> List[Dict[str, object]]:
    sampler_cfg = Dx40SamplerCfg(
        tau_min=args.tau_min,
        tau_max=args.tau_max,
        log10_lambda1_min=args.log10_lambda1_min,
        log10_lambda1_max=args.log10_lambda1_max,
    )
    gen = torch.Generator(device="cpu").manual_seed(args.seed + 777)
    rows: List[Dict[str, object]] = []
    bank_index = 0
    bank_batch_index = 0
    while bank_index < int(args.bank_size):
        current = min(int(args.bank_batch_size), int(args.bank_size) - bank_index)
        x_ctx_b, y_ctx_b, x_tgt_b, y_tgt_b, meta = sample_dx40_batch_with_meta(
            args, sampler_cfg, device, current
        )
        batch_meta = {**meta, "bank_batch_index": bank_batch_index}
        for member in range(current):
            print(f"episode {bank_index + 1}/{args.bank_size}", flush=True)
            rows.extend(
                process_episode(
                    model,
                    args,
                    bank_index,
                    bank_index,
                    member,
                    x_ctx_b[member : member + 1],
                    y_ctx_b[member : member + 1],
                    x_tgt_b[member : member + 1],
                    y_tgt_b[member : member + 1],
                    batch_meta,
                    gen,
                )
            )
            bank_index += 1
        bank_batch_index += 1
    return rows


def write_summary_txt(
    path: Path,
    args: argparse.Namespace,
    summary: Sequence[Dict[str, object]],
    bank_rows: Sequence[Dict[str, object]],
    scale_rows: Sequence[Dict[str, object]],
    outliers: Sequence[Dict[str, object]],
) -> None:
    dist = bank_rows[0] if bank_rows else {}
    lines = [
        "dx40 label-space Q projection",
        "",
        f"sampling={args.sampling_mode}, seed={args.seed}, bank_size={args.bank_size}, "
        f"bank_batch_size={args.bank_batch_size}",
        f"n_build={args.n_build}, eps={args.eps}, alpha={args.excess_risk_frac}, basis_layer={args.basis_layer}",
        f"log10(lambda1)=[{args.log10_lambda1_min}, {args.log10_lambda1_max}]",
        "",
    ]
    if dist:
        lines.extend(
            [
                "Bank distribution from base rows:",
                f"  n={int(dist['n'])}",
                f"  MSE/KRR mean={float(dist['mse_ratio_mean']):.5g}, "
                f"median={float(dist['mse_ratio_median']):.5g}, "
                f"q25-q75={float(dist['mse_ratio_q25']):.5g}-{float(dist['mse_ratio_q75']):.5g}, "
                f"q90={float(dist['mse_ratio_q90']):.5g}, "
                f"q99={float(dist['mse_ratio_q99']):.5g}, "
                f"max={float(dist['mse_ratio_max']):.5g}",
                f"  r_T mean={float(dist['r_T_mean']):.5g}, median={float(dist['r_T_median']):.5g}",
                f"  rank_Q mean={float(dist['rank_Q_mean']):.5g}, median={float(dist['rank_Q_median']):.5g}",
                f"  rho mean={float(dist['rho_TQ_mean']):.5g}, median={float(dist['rho_TQ_median']):.5g}, "
                f"certified={int(dist['certified'])}/{int(dist['n'])}",
                "",
            ]
        )
    lines.append(
        f"{'input':8s} {'n':>4s} {'MSE/KRR mean':>13s} {'MSE/KRR med':>12s} "
        f"{'KRR input med':>13s} {'drift med':>10s}"
    )
    for row in summary:
        lines.append(
            f"{str(row['name']):8s} {int(row['n']):4d} "
            f"{float(row['mse_ratio_mean']):13.5g} {float(row['mse_ratio_median']):12.5g} "
            f"{float(row['krr_input_ratio_median']):13.5g} "
            f"{float(row['pred_drift_median']):10.5g}"
        )
    if scale_rows:
        lines.append("")
        lines.append(
            f"{'scale':9s} {'n':>4s} {'reach':>6s} {'rT':>6s} {'k':>6s} "
            f"{'rho':>9s} {'base':>9s} {'y_Q':>9s} {'y_perp':>9s}"
        )
        for row in scale_rows:
            lines.append(
                f"{str(row['scale_bin']):9s} {int(row['n']):4d} "
                f"{float(row['reach']):6.3f} {float(row['r_T_mean']):6.2f} "
                f"{float(row['rank_Q_mean']):6.2f} {float(row['rho_TQ_mean']):9.5f} "
                f"{float(row['base_median']):9.3g} {float(row['label_keep_median']):9.3g} "
                f"{float(row['label_perp_median']):9.3g}"
            )
    if outliers:
        lines.append("")
        lines.append(f"Outliers with base MSE/KRR >= {args.min_base_ratio:g}: {len(outliers)}")
        lines.append(
            f"{'bank':>5s} {'rT':>4s} {'k':>4s} {'rho':>9s} {'base':>10s} {'y_Q':>10s} {'y_perp':>10s}"
        )
        for row in outliers:
            lines.append(
                f"{int(row['bank_index']):5d} {float(row['r_T']):4.0f} {float(row['rank_Q']):4.0f} "
                f"{float(row['rho_TQ']):9.5f} {float(row['base_mse_ratio']):10.4g} "
                f"{float(row['label_keep_mse_ratio']):10.4g} {float(row['label_perp_mse_ratio']):10.4g}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(SCRIPT_DIR / "results_label_space"))
    parser.add_argument("--checkpoint", default=str(DX40_CHECKPOINT))
    parser.add_argument("--seed", type=int, default=122)
    parser.add_argument("--bank-size", type=int, default=480)
    parser.add_argument("--bank-batch-size", type=int, default=16)
    parser.add_argument("--sampling-mode", choices=["train_distribution", "fixed_target_rich"], default="train_distribution")
    parser.add_argument("--gamma-values", default="4,8,12")
    parser.add_argument("--train-n-tgt", type=int, default=8)
    parser.add_argument("--n-ctx", type=int, default=47)
    parser.add_argument("--n-tgt", type=int, default=40)
    parser.add_argument("--device", choices=["cpu"], default="cpu")
    parser.add_argument("--sigma2", type=float, default=0.1)
    parser.add_argument("--kernel-family", choices=["linear"], default="linear")
    parser.add_argument("--n-build", type=int, default=16)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--tau-sv", type=float, default=1e-3)
    parser.add_argument("--excess-risk-frac", type=float, default=0.05)
    parser.add_argument("--basis-layer", type=int, default=8)
    parser.add_argument("--tau-min", type=float, default=1e-3)
    parser.add_argument("--tau-max", type=float, default=16.0)
    parser.add_argument("--log10-lambda1-min", type=float, default=-2.0)
    parser.add_argument("--log10-lambda1-max", type=float, default=2.0)
    parser.add_argument("--min-base-ratio", type=float, default=5.0)
    args = parser.parse_args()
    parse_gamma_values(args.gamma_values)
    return args


def main() -> None:
    args = parse_args()
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = load_dx40_model(args.checkpoint, device)

    # Match the existing dx40 scripts: sample bank after model construction.
    set_seed(args.seed)
    (out_dir / "config.json").write_text(
        json.dumps({"args": vars(args), "device": str(device)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    rows = run_bank(model, args, device)
    summary = summarize(rows, ["name"])
    summary_by_scale_name = summarize(rows, ["scale_bin", "name"])
    scale_rows = scale_table(rows)
    bank_rows = bank_distribution(rows)
    outliers = outlier_table(rows, args.min_base_ratio)
    outlier_rows = [
        row
        for row in rows
        if int(row["episode"]) in {int(outlier["episode"]) for outlier in outliers}
    ]
    outlier_summary = summarize(outlier_rows, ["name"]) if outlier_rows else []

    write_csv(out_dir / "records.csv", rows)
    write_csv(out_dir / "summary.csv", summary)
    write_csv(out_dir / "summary_by_scale_name.csv", summary_by_scale_name)
    write_csv(out_dir / "scale_summary.csv", scale_rows)
    write_csv(out_dir / "bank_distribution.csv", bank_rows)
    write_csv(out_dir / "outlier_table.csv", outliers)
    write_csv(out_dir / "outlier_summary.csv", outlier_summary)
    write_summary_txt(out_dir / "summary.txt", args, summary, bank_rows, scale_rows, outliers)
    save_all_summary_plot(out_dir, summary)
    save_scale_plot(out_dir, scale_rows)
    save_outlier_plot(out_dir, outliers)
    print((out_dir / "summary.txt").read_text(), flush=True)


if __name__ == "__main__":
    main()
