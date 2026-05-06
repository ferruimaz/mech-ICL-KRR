#!/usr/bin/env python3
"""Experiment 4: controlled response-prefix repair on bad dx40 episodes.

The experiment samples a fixed bank of dx40 episodes, selects the episodes with
the largest base-model MSE/KRR-MSE ratio, and asks whether the KRR-sufficient
response prefix Q repairs the model:

    base:     unmodified model;
    keep_Q:   keep only Q Q^T A H_ctx at a residual state;
    remove_Q: remove Q Q^T A H_ctx.

The prefix Q is extracted from the final-layer response span and chosen as the
first KRR-targeted prefix with rho_G(T_Q) <= alpha, falling back to the best
prefix if the threshold is not reached.

The default sampling mode now matches the dx40 training distribution:
n_tgt=8 and n_ctx=round(gamma tau(Sigma)) with gamma in {4,8,12}.  The older
fixed target-rich setup remains available as a stress test.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
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

from experiments.experiment_3_causal_surgery.run_dx40_complement import (  # noqa: E402
    fd_damage_error,
    fd_from_cols,
    fd_operator_error,
    forward_with_states_and_pack,
    keep_q_state,
    rel_drift,
    rel_point_error,
    remove_q_state,
    roll_from_state,
    select_response_prefix,
)
from experiments.experiment_1_operator_certificate.run import (  # noqa: E402
    random_a_basis,
    symmetric_eig_factors,
)
from experiments.experiment_2_rank_emergence.run import a_orth_candidate  # noqa: E402
from experiment_utils import dx40_data  # noqa: E402
from experiment_utils.dx40 import (  # noqa: E402
    DX40_CHECKPOINT,
    Dx40SamplerCfg,
    dx40_hidden_response_matrices,
    load_dx40_model,
)
from experiment_utils.layerwise_core import (  # noqa: E402
    CkptCfg,
    FLOOR,
    build_eval_kernels,
    build_eval_target_kernel,
    effective_rank_T_excess_risk,
    psd_sqrt_and_invsqrt,
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
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: Sequence[Dict[str, object]], key: str) -> List[Dict[str, object]]:
    groups: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)

    metrics = [
        "mse",
        "mse_ratio",
        "mse_repair_factor",
        "point_err",
        "point_drift",
        "E_F_T",
        "damage_vs_base",
        "rank_Q",
        "r_T",
        "rho_TQ",
        "base_mse",
        "krr_mse",
    ]
    out: List[Dict[str, object]] = []
    for name, group in sorted(groups.items()):
        rec: Dict[str, object] = {key: name, "n": len(group)}
        for metric in metrics:
            vals = [float(row[metric]) for row in group if row.get(metric) not in ("", None)]
            vals = [v for v in vals if math.isfinite(v)]
            if vals:
                rec[f"{metric}_mean"] = float(np.mean(vals))
                rec[f"{metric}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                rec[f"{metric}_median"] = float(np.median(vals))
                if metric == "mse_ratio":
                    rec["mse_ratio_frac_le_2"] = float(np.mean(np.array(vals) <= 2.0))
        out.append(rec)
    return out


def summarize_by_keys(rows: Sequence[Dict[str, object]], keys: Sequence[str]) -> List[Dict[str, object]]:
    groups: Dict[tuple[str, ...], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row[key]) for key in keys)].append(row)

    metrics = [
        "mse",
        "mse_ratio",
        "mse_repair_factor",
        "point_err",
        "point_drift",
        "E_F_T",
        "damage_vs_base",
        "rank_Q",
        "r_T",
        "rho_TQ",
        "base_mse",
        "krr_mse",
    ]
    out: List[Dict[str, object]] = []
    for group_key, group in sorted(groups.items()):
        rec: Dict[str, object] = {key: value for key, value in zip(keys, group_key)}
        rec["n"] = len(group)
        for metric in metrics:
            vals = [float(row[metric]) for row in group if row.get(metric) not in ("", None)]
            vals = [v for v in vals if math.isfinite(v)]
            if vals:
                rec[f"{metric}_mean"] = float(np.mean(vals))
                rec[f"{metric}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                rec[f"{metric}_median"] = float(np.median(vals))
                if metric == "mse_ratio":
                    rec["mse_ratio_frac_le_2"] = float(np.mean(np.array(vals) <= 2.0))
        out.append(rec)
    return out


def parse_gamma_values(text: str) -> List[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("--gamma-values must contain at least one integer")
    return values


def sample_training_spectrum(sampler_cfg: Dx40SamplerCfg, sigma2: float):
    """Sample one dx40 spectrum from the minimal scale-tau family."""
    return dx40_data._sample_minimal_batch_spectrum(
        d=40,
        sigma2=sigma2,
        profile=None,
        sampling_scheme="scale_tau_direct",
        tau_min=sampler_cfg.tau_min,
        tau_max=sampler_cfg.tau_max,
        log10_lambda1_min=sampler_cfg.log10_lambda1_min,
        log10_lambda1_max=sampler_cfg.log10_lambda1_max,
        rejection_attempts=sampler_cfg.rejection_attempts,
        smooth_span_min=sampler_cfg.smooth_span_min,
        smooth_span_max=sampler_cfg.smooth_span_max,
        step_rank_values=list(range(1, 17)),
        step_depth_min=sampler_cfg.step_depth_min,
        step_depth_max=sampler_cfg.step_depth_max,
        step_rank_distribution=sampler_cfg.step_rank_distribution,
        scale_distribution="uniform",
        scale_distribution_power=2.0,
        tau_target=None,
    )


def sample_dx40_batch_with_meta(
    args: argparse.Namespace,
    sampler_cfg: Dx40SamplerCfg,
    device: torch.device,
    batch_size: int,
):
    profile, eig, tau_actual, log10_lambda1 = sample_training_spectrum(
        sampler_cfg, args.sigma2
    )
    gamma = ""
    if args.sampling_mode == "train_distribution":
        gamma_values = parse_gamma_values(args.gamma_values)
        gamma = random.choice(gamma_values)
        n_ctx = max(1, int(round(float(gamma) * float(tau_actual))))
        n_tgt = int(args.train_n_tgt)
    elif args.sampling_mode == "fixed_target_rich":
        n_ctx = int(args.n_ctx)
        n_tgt = int(args.n_tgt)
    else:
        raise ValueError(f"unknown sampling_mode: {args.sampling_mode}")

    batch_size = max(1, int(batch_size))
    eigenvalues = eig.unsqueeze(0).repeat(batch_size, 1)
    x_ctx, y_ctx, x_tgt, y_tgt, _ = dx40_data._build_batch(
        eigenvalues,
        batch_size=batch_size,
        d=40,
        n_ctx=n_ctx,
        n_tgt=n_tgt,
        sigma2=args.sigma2,
        device=device,
    )
    meta = {
        "sampling_mode": args.sampling_mode,
        "profile": profile,
        "tau_actual": float(tau_actual),
        "log10_lambda1": float(log10_lambda1),
        "lambda1": float(10.0 ** float(log10_lambda1)),
        "gamma": gamma,
        "n_ctx_actual": n_ctx,
        "n_tgt_actual": n_tgt,
        "spectrum_batch_size": batch_size,
    }
    return x_ctx, y_ctx, x_tgt, y_tgt, meta


def sample_dx40_episode_with_meta(
    args: argparse.Namespace,
    sampler_cfg: Dx40SamplerCfg,
    device: torch.device,
):
    x_ctx, y_ctx, x_tgt, y_tgt, meta = sample_dx40_batch_with_meta(
        args, sampler_cfg, device, batch_size=1
    )
    return x_ctx, y_ctx, x_tgt, y_tgt, meta


def scan_summary_rows(scan_rows: Sequence[Dict[str, object]]) -> List[str]:
    if not scan_rows:
        return ["No scanned episodes."]
    ratios = np.array([float(row["base_mse_ratio"]) for row in scan_rows], dtype=float)
    lines = [
        "Bank scan:",
        f"  n={len(scan_rows)}",
        f"  median base/KRR={np.median(ratios):.3g}",
        f"  q90 base/KRR={np.quantile(ratios, 0.90):.3g}",
        f"  q99 base/KRR={np.quantile(ratios, 0.99):.3g}",
        f"  max base/KRR={np.max(ratios):.3g}",
    ]
    for threshold in (2.0, 5.0, 10.0):
        lines.append(f"  fraction >= {threshold:g}: {float(np.mean(ratios >= threshold)):.3f}")
    if any(row.get("bank_batch_index") not in ("", None) for row in scan_rows):
        by_batch: Dict[int, List[float]] = defaultdict(list)
        for row in scan_rows:
            if row.get("bank_batch_index") not in ("", None):
                by_batch[int(row["bank_batch_index"])].append(float(row["base_mse_ratio"]))
        batch_means = np.array([float(np.mean(vals)) for vals in by_batch.values()], dtype=float)
        if batch_means.size:
            lines.extend(
                [
                    "Batch-mean scan:",
                    f"  batches={len(batch_means)}",
                    f"  mean batch base/KRR={np.mean(batch_means):.5g}",
                    f"  median batch base/KRR={np.median(batch_means):.3g}",
                    f"  max batch base/KRR={np.max(batch_means):.3g}",
                ]
            )
            for threshold in (2.0, 5.0, 10.0):
                lines.append(
                    f"  batch fraction >= {threshold:g}: {float(np.mean(batch_means >= threshold)):.3f}"
                )
    return lines


def scan_episode_bank(model, cfg: CkptCfg, args: argparse.Namespace, sampler_cfg: Dx40SamplerCfg, device: torch.device):
    episodes = []
    scan_rows: List[Dict[str, object]] = []
    bank_batch_size = max(1, int(args.bank_batch_size))
    bank_batch_index = 0
    while len(episodes) < int(args.bank_size):
        current_batch_size = min(bank_batch_size, int(args.bank_size) - len(episodes))
        x_ctx_batch, y_ctx_batch, x_tgt_batch, y_tgt_batch, meta = sample_dx40_batch_with_meta(
            args, sampler_cfg, device, current_batch_size
        )
        for member in range(current_batch_size):
            idx = len(episodes)
            x_ctx = x_ctx_batch[member : member + 1]
            y_ctx = y_ctx_batch[member : member + 1]
            x_tgt = x_tgt_batch[member : member + 1]
            y_tgt = y_tgt_batch[member : member + 1]
            episode_meta = {
                **meta,
                "bank_batch_index": bank_batch_index,
                "bank_batch_member": member,
                "bank_batch_size": current_batch_size,
            }
            y = y_ctx[0].detach().cpu().double()
            target = y_tgt[0].detach().cpu().double()
            _K, _Kt, _A, T = build_eval_kernels(x_ctx, x_tgt, args)
            base_pred, _states, _pack = forward_with_states_and_pack(model, x_ctx, y, x_tgt)
            krr_pred = T @ y
            base_mse = mse(base_pred, target)
            krr_mse = mse(krr_pred, target)
            ratio = base_mse / max(krr_mse, FLOOR)
            episodes.append(
                {
                    "bank_index": idx,
                    "x_ctx": x_ctx,
                    "y_ctx": y_ctx,
                    "x_tgt": x_tgt,
                    "y_tgt": y_tgt,
                    "base_mse": base_mse,
                    "krr_mse": krr_mse,
                    "base_mse_ratio": ratio,
                    **episode_meta,
                }
            )
            scan_rows.append(
                {
                    "bank_index": idx,
                    "base_mse": base_mse,
                    "krr_mse": krr_mse,
                    "base_mse_ratio": ratio,
                    **episode_meta,
                }
            )
        bank_batch_index += 1
    if args.selection_mode == "all_bank":
        selected = episodes
    else:
        pathological = [
            row for row in episodes if float(row["base_mse_ratio"]) >= float(args.min_base_ratio)
        ]
        selected = sorted(pathological, key=lambda row: float(row["base_mse_ratio"]), reverse=True)[
            : args.select_top
        ]
    return selected, scan_rows


def zero_context_prediction(model, states, args: argparse.Namespace, pack):
    h_zero = states[args.state_idx].clone()
    h_zero[0, : pack.n_ctx, :] = 0
    return roll_from_state(model, h_zero, args.state_idx, pack)


def keep_prediction(model, states, args: argparse.Namespace, pack, A: torch.Tensor, Q: torch.Tensor):
    return roll_from_state(
        model,
        keep_q_state(states[args.state_idx], pack.n_ctx, A, Q),
        args.state_idx,
        pack,
    )


def response_svd_prefix(M_resp: torch.Tensor, A: torch.Tensor, tau_sv: float, rank: int) -> torch.Tensor:
    if rank <= 0:
        return torch.zeros(A.shape[0], 0, dtype=torch.float64)
    cand_rank = min(max(M_resp.shape), M_resp.shape[0], M_resp.shape[1])
    C = a_orth_candidate(M_resp, symmetric_eig_factors(A), tau_sv, cand_rank)
    return C[:, : min(rank, C.shape[1])]


def krr_oracle_prefix(Kt: torch.Tensor, A: torch.Tensor, rank: int) -> torch.Tensor:
    if rank <= 0:
        return torch.zeros(A.shape[0], 0, dtype=torch.float64)
    _A_sqrt, A_invsqrt = psd_sqrt_and_invsqrt(A)
    Z = Kt @ A_invsqrt
    _U, _S, Vh = torch.linalg.svd(Z, full_matrices=False)
    k = min(rank, Vh.shape[0])
    return A_invsqrt @ Vh[:k, :].T


def process_selected_episode(
    model,
    args: argparse.Namespace,
    episode: Dict[str, object],
    selection_rank: int,
    gen: torch.Generator,
) -> List[Dict[str, object]]:
    x_ctx = episode["x_ctx"]
    y_ctx = episode["y_ctx"]
    x_tgt = episode["x_tgt"]
    y_tgt = episode["y_tgt"]
    assert isinstance(x_ctx, torch.Tensor)
    assert isinstance(y_ctx, torch.Tensor)
    assert isinstance(x_tgt, torch.Tensor)
    assert isinstance(y_tgt, torch.Tensor)

    y = y_ctx[0].detach().cpu().double()
    target = y_tgt[0].detach().cpu().double()
    _K, Kt, A, T = build_eval_kernels(x_ctx, x_tgt, args)
    Ktt = build_eval_target_kernel(x_tgt, args)
    krr_pred = T @ y
    krr_mse = mse(krr_pred, target)

    risk_pack = effective_rank_T_excess_risk(T, A, Kt, Ktt, args.excess_risk_frac)
    r_t = int(risk_pack["r_eff_T_task"])
    risk_total = float(risk_pack["krr_risk_total"])

    build_probes = sample_task_probes(A, args.n_build, gen)
    response_layers = dx40_hidden_response_matrices(model, x_ctx, x_tgt, y, build_probes, args.eps)
    Q, q_meta = select_response_prefix(
        response_layers[args.basis_layer],
        A,
        Kt,
        T,
        risk_total,
        args.excess_risk_frac,
        args.tau_sv,
    )
    TQ = Kt @ Q @ Q.T if Q.shape[1] else torch.zeros_like(T)
    rank_q = int(q_meta["rank_Q"])
    Q_svd = response_svd_prefix(response_layers[args.basis_layer], A, args.tau_sv, rank_q)
    Q_oracle = krr_oracle_prefix(Kt, A, rank_q)

    base_pred, base_states, base_pack = forward_with_states_and_pack(model, x_ctx, y, x_tgt)
    n_ctx_actual = int(base_pack.n_ctx)
    n_tgt_actual = int(x_tgt.shape[1])
    keep_pred = keep_prediction(model, base_states, args, base_pack, A, Q)
    remove_pred = roll_from_state(
        model,
        remove_q_state(base_states[args.state_idx], n_ctx_actual, A, Q),
        args.state_idx,
        base_pack,
    )
    zero_ctx_pred = zero_context_prediction(model, base_states, args, base_pack)
    svd_pred = keep_prediction(model, base_states, args, base_pack, A, Q_svd)
    oracle_pred = keep_prediction(model, base_states, args, base_pack, A, Q_oracle)
    preds = {
        "base": base_pred,
        "keep_Q": keep_pred,
        "remove_Q": remove_pred,
        "zero_context": zero_ctx_pred,
        "keep_response_svd": svd_pred,
        "keep_krr_oracle": oracle_pred,
    }
    for random_idx in range(args.n_random_controls):
        Q_random = random_a_basis(A.shape[0], rank_q, A, gen)
        preds[f"keep_random_A_{random_idx:02d}"] = keep_prediction(
            model, base_states, args, base_pack, A, Q_random
        )

    probes = sample_task_probes(A, args.n_eval, gen)
    fd_cols: Dict[str, List[torch.Tensor]] = {"base": [], "keep_Q": [], "remove_Q": []}
    for u in probes:
        plus = y + args.eps * u
        minus = y - args.eps * u
        pred_p, states_p, pack_p = forward_with_states_and_pack(model, x_ctx, plus, x_tgt)
        pred_m, states_m, pack_m = forward_with_states_and_pack(model, x_ctx, minus, x_tgt)
        fd_cols["base"].append((pred_p - pred_m) / (2.0 * args.eps))

        keep_p = roll_from_state(
            model,
            keep_q_state(states_p[args.state_idx], n_ctx_actual, A, Q),
            args.state_idx,
            pack_p,
        )
        keep_m = roll_from_state(
            model,
            keep_q_state(states_m[args.state_idx], n_ctx_actual, A, Q),
            args.state_idx,
            pack_m,
        )
        fd_cols["keep_Q"].append((keep_p - keep_m) / (2.0 * args.eps))

        remove_p = roll_from_state(
            model,
            remove_q_state(states_p[args.state_idx], n_ctx_actual, A, Q),
            args.state_idx,
            pack_p,
        )
        remove_m = roll_from_state(
            model,
            remove_q_state(states_m[args.state_idx], n_ctx_actual, A, Q),
            args.state_idx,
            pack_m,
        )
        fd_cols["remove_Q"].append((remove_p - remove_m) / (2.0 * args.eps))

    fd = {name: fd_from_cols(cols, n_tgt_actual) for name, cols in fd_cols.items()}
    base_mse = mse(base_pred, target)

    rows: List[Dict[str, object]] = []
    names = list(preds.keys())
    for name in names:
        current_mse = mse(preds[name], target)
        fd_name = name if name in fd else ""
        rows.append(
            {
                "selection_rank": selection_rank,
                "bank_index": episode["bank_index"],
                "bank_batch_index": episode.get("bank_batch_index", ""),
                "bank_batch_member": episode.get("bank_batch_member", ""),
                "bank_batch_size": episode.get("bank_batch_size", ""),
                "name": "keep_random_A" if name.startswith("keep_random_A_") else name,
                "replicate": name,
                "state_idx": args.state_idx,
                "basis_layer": args.basis_layer,
                "certificate": "certified" if float(q_meta["reach_alpha"]) >= 1.0 else "not_certified",
                "sampling_mode": episode.get("sampling_mode", ""),
                "profile": episode.get("profile", ""),
                "tau_actual": episode.get("tau_actual", ""),
                "log10_lambda1": episode.get("log10_lambda1", ""),
                "gamma": episode.get("gamma", ""),
                "n_ctx": n_ctx_actual,
                "n_tgt": n_tgt_actual,
                "r_T": r_t,
                "rank_Q": q_meta["rank_Q"],
                "candidate_dim": q_meta["candidate_dim"],
                "rank_to_alpha": q_meta["rank_to_alpha"],
                "best_rank": q_meta["best_rank"],
                "rho_TQ": q_meta["rho_TQ"],
                "best_rho": q_meta["best_rho"],
                "reach_alpha": q_meta["reach_alpha"],
                "krr_mse": krr_mse,
                "base_mse": base_mse,
                "mse": current_mse,
                "mse_ratio": current_mse / max(krr_mse, FLOOR),
                "mse_repair_factor": base_mse / max(current_mse, FLOOR),
                "point_err": rel_point_error(preds[name], krr_pred),
                "point_drift": rel_drift(preds[name], base_pred),
                "E_F_T": fd_operator_error(fd[fd_name], T, T, probes) if fd_name else "",
                "E_F_TQ": fd_operator_error(fd[fd_name], TQ, T, probes) if fd_name else "",
                "damage_vs_base": fd_damage_error(fd[fd_name], fd["base"], T, probes)
                if fd_name
                else "",
            }
        )
    return rows


def sampling_description(args: argparse.Namespace) -> str:
    if args.sampling_mode == "train_distribution":
        return (
            f"sampling=train_distribution, gamma in {{{args.gamma_values}}}, "
            f"n_tgt={args.train_n_tgt}, n_ctx=round(gamma*tau)"
        )
    return f"sampling=fixed_target_rich, n_ctx={args.n_ctx}, n_tgt={args.n_tgt}"


def write_summary_txt(
    path: Path,
    args: argparse.Namespace,
    rows: Sequence[Dict[str, object]],
    summary: Sequence[Dict[str, object]],
    scan_rows: Sequence[Dict[str, object]],
    summary_by_certificate: Sequence[Dict[str, object]] | None = None,
) -> None:
    lines = [
        "Experiment 4: dx40 response-prefix repair",
        "",
        f"selection_mode={args.selection_mode}, bank_size={args.bank_size}, min_base_ratio={args.min_base_ratio}, "
        f"select_top={args.select_top}, seed={args.seed}, bank_batch_size={args.bank_batch_size}",
        sampling_description(args),
        f"state_idx={args.state_idx}, basis_layer={args.basis_layer}",
        f"alpha={args.excess_risk_frac}, n_build={args.n_build}, n_eval={args.n_eval}",
        f"log10(lambda1)=[{args.log10_lambda1_min}, {args.log10_lambda1_max}], "
        f"tau=[{args.tau_min}, {args.tau_max}]",
        "",
        *scan_summary_rows(scan_rows),
        "",
        f"{'name':10s} {'n':>3s} {'rankQ':>7s} {'rT':>6s} {'rhoTQ':>9s} {'MSE':>11s} {'MSE/KRR':>11s} {'repair':>9s} {'E(F,T)':>9s} {'damage':>9s}",
    ]
    for row in summary:
        lines.append(
            f"{str(row['name']):10s} {int(row['n']):3d} "
            f"{float(row.get('rank_Q_mean', float('nan'))):7.2f} "
            f"{float(row.get('r_T_mean', float('nan'))):6.2f} "
            f"{float(row.get('rho_TQ_mean', float('nan'))):9.5f} "
            f"{float(row.get('mse_mean', float('nan'))):11.5g} "
            f"{float(row.get('mse_ratio_mean', float('nan'))):11.3f} "
            f"{float(row.get('mse_repair_factor_mean', float('nan'))):9.1f} "
            f"{float(row.get('E_F_T_mean', float('nan'))):9.3f} "
            f"{float(row.get('damage_vs_base_mean', float('nan'))):9.3f}"
        )
    if summary_by_certificate:
        lines.append("")
        lines.append("By response-prefix certificate:")
        lines.append(
            f"{'cert':>13s} {'name':10s} {'n':>4s} {'rankQ':>7s} {'rhoTQ':>9s} "
            f"{'MSE/KRR':>11s} {'repair':>9s} {'E(F,T)':>9s} {'damage':>9s}"
        )
        for row in summary_by_certificate:
            lines.append(
                f"{str(row['certificate']):>13s} {str(row['name']):10s} {int(row['n']):4d} "
                f"{float(row.get('rank_Q_mean', float('nan'))):7.2f} "
                f"{float(row.get('rho_TQ_mean', float('nan'))):9.5f} "
                f"{float(row.get('mse_ratio_mean', float('nan'))):11.3f} "
                f"{float(row.get('mse_repair_factor_mean', float('nan'))):9.2f} "
                f"{float(row.get('E_F_T_mean', float('nan'))):9.3f} "
                f"{float(row.get('damage_vs_base_mean', float('nan'))):9.3f}"
            )
    lines.append("")
    by_sel: Dict[int, Dict[str, List[Dict[str, object]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_sel[int(row["selection_rank"])][str(row["name"])].append(row)
    if len(by_sel) > 25:
        lines.append(f"Episode details omitted from text summary ({len(by_sel)} processed episodes).")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    lines.append("Selected episodes:")
    lines.append(
        f"{'sel':>3s} {'bank':>5s} {'bb':>3s} {'bm':>3s} {'nctx':>4s} {'ntgt':>4s} {'tau':>6s} "
        f"{'loglam':>6s} {'g':>3s} {'rankQ':>5s} {'rT':>4s} {'base/KRR':>10s} "
        f"{'keep/KRR':>10s} {'zero/KRR':>10s} {'rand/KRR':>10s}"
    )
    for sel in sorted(by_sel):
        base = by_sel[sel]["base"][0]
        keep = by_sel[sel]["keep_Q"][0]
        zero = by_sel[sel]["zero_context"][0]
        random_rows = by_sel[sel].get("keep_random_A", [])
        rand_ratio = (
            float(np.mean([float(r["mse_ratio"]) for r in random_rows]))
            if random_rows
            else float("nan")
        )
        lines.append(
            f"{sel:3d} {int(base['bank_index']):5d} "
            f"{int(base.get('bank_batch_index') or 0):3d} {int(base.get('bank_batch_member') or 0):3d} "
            f"{int(base['n_ctx']):4d} {int(base['n_tgt']):4d} {float(base['tau_actual']):6.2f} "
            f"{float(base['log10_lambda1']):6.2f} {str(base.get('gamma', '')):>3s} "
            f"{float(base['rank_Q']):5.0f} {float(base['r_T']):4.0f} {float(base['mse_ratio']):10.1f} "
            f"{float(keep['mse_ratio']):10.3f} {float(zero['mse_ratio']):10.3f} "
            f"{rand_ratio:10.3f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default=str(SCRIPT_DIR / "results"))
    p.add_argument("--checkpoint", default=str(DX40_CHECKPOINT))
    p.add_argument("--name", default="dx40_tau16_s122_target_rich")
    p.add_argument("--seed", type=int, default=20260506)
    p.add_argument("--bank-size", type=int, default=200)
    p.add_argument(
        "--bank-batch-size",
        type=int,
        default=1,
        help="Number of independent episodes sharing each sampled spectrum/gamma; use 16 to mimic dx40 validation batches.",
    )
    p.add_argument("--min-base-ratio", type=float, default=5.0)
    p.add_argument("--select-top", type=int, default=8)
    p.add_argument(
        "--selection-mode",
        choices=["bad_tail", "all_bank"],
        default="bad_tail",
        help="Select high-MSE-ratio tail episodes, or process the whole sampled bank.",
    )
    p.add_argument(
        "--sampling-mode",
        choices=["train_distribution", "fixed_target_rich"],
        default="train_distribution",
    )
    p.add_argument("--gamma-values", default="4,8,12")
    p.add_argument("--train-n-tgt", type=int, default=8)
    p.add_argument("--scan-only", action="store_true")
    p.add_argument("--device", choices=["cpu"], default="cpu")
    p.add_argument("--n-ctx", type=int, default=47)
    p.add_argument("--n-tgt", type=int, default=40)
    p.add_argument("--sigma2", type=float, default=0.1)
    p.add_argument("--kernel-family", choices=["linear"], default="linear")
    p.add_argument("--n-build", type=int, default=16)
    p.add_argument("--n-eval", type=int, default=16)
    p.add_argument("--n-random-controls", type=int, default=20)
    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--tau-sv", type=float, default=1e-3)
    p.add_argument("--excess-risk-frac", type=float, default=0.05)
    p.add_argument("--basis-layer", type=int, default=8)
    p.add_argument("--state-idx", type=int, default=1)
    p.add_argument("--tau-min", type=float, default=1e-3)
    p.add_argument("--tau-max", type=float, default=16.0)
    p.add_argument("--log10-lambda1-min", type=float, default=-2.0)
    p.add_argument("--log10-lambda1-max", type=float, default=2.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model = load_dx40_model(args.checkpoint, device)

    # Seed after model loading so the episode bank is independent of constructor RNG.
    set_seed(args.seed)
    gen = torch.Generator(device="cpu").manual_seed(args.seed + 777)

    cfg = CkptCfg(args.name, args.checkpoint, 40, 128, 8, 4, args.kernel_family, 40.0)
    sampler_cfg = Dx40SamplerCfg(
        tau_min=args.tau_min,
        tau_max=args.tau_max,
        log10_lambda1_min=args.log10_lambda1_min,
        log10_lambda1_max=args.log10_lambda1_max,
    )

    (out_dir / "config.json").write_text(
        json.dumps({"args": vars(args), "device": str(device)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    selected, scan_rows = scan_episode_bank(model, cfg, args, sampler_cfg, device)
    write_csv(out_dir / "bank_scan.csv", scan_rows)
    if args.scan_only:
        write_summary_txt(out_dir / "summary.txt", args, [], [], scan_rows)
        print((out_dir / "summary.txt").read_text(), flush=True)
        return
    if not selected:
        write_summary_txt(out_dir / "summary.txt", args, [], [], scan_rows)
        raise RuntimeError(
            f"no episodes in the bank reached --min-base-ratio={args.min_base_ratio}"
        )

    rows: List[Dict[str, object]] = []
    for selection_rank, episode in enumerate(selected, start=1):
        print(
            f"selected {selection_rank}/{len(selected)}: bank={episode['bank_index']} "
            f"base/KRR={float(episode['base_mse_ratio']):.3f}",
            flush=True,
        )
        rows.extend(process_selected_episode(model, args, episode, selection_rank, gen))

    summary = summarize(rows, "name")
    summary_by_certificate = summarize_by_keys(rows, ["certificate", "name"])
    write_csv(out_dir / "records.csv", rows)
    write_csv(out_dir / "summary.csv", summary)
    write_csv(out_dir / "summary_by_certificate.csv", summary_by_certificate)
    write_summary_txt(
        out_dir / "summary.txt",
        args,
        rows,
        summary,
        scan_rows,
        summary_by_certificate,
    )
    print((out_dir / "summary.txt").read_text(), flush=True)


if __name__ == "__main__":
    main()
