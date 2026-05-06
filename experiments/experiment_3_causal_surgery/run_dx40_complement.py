#!/usr/bin/env python3
"""dx40 complement ablation for Experiment 3.

This is the causal-use counterpart of the dx40 Experiment 2 response-span
diagnostic.  For each episode, extract the final-layer response span, order it
by KRR-targeted energy, keep the first prefix reaching the five-percent
prediction-risk target, and then test whether that token subspace is used by
the model:

    keep_Q:   H_ctx <- Q Q^T A H_ctx
    remove_Q: H_ctx <- H_ctx - Q Q^T A H_ctx

The intervention is applied at a chosen residual state and the rest of the
dx40 transformer is rolled forward.
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
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
MPL_CACHE_DIR = Path(os.environ.get("TMPDIR", "/tmp")) / "mech_icl_krr_mpl_cache"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments.experiment_1_operator_certificate.run import symmetric_eig_factors  # noqa: E402
from experiments.experiment_2_rank_emergence.run import (  # noqa: E402
    a_orth_candidate,
    exact_operator_excess,
    krr_targeted_order,
)
from experiment_utils.dx40 import (  # noqa: E402
    DX40_CHECKPOINT,
    Dx40SamplerCfg,
    _dx40_head,
    _layer_kwargs,
    _prepare_dx40_tokens,
    dx40_hidden_response_matrices,
    load_dx40_model,
    sample_dx40_episode,
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
from experiment_utils.support import get_device, set_seed  # noqa: E402


@dataclass
class StatePack:
    n_ctx: int
    output_scale: torch.Tensor | float
    scale_info: object
    attn_gates: object
    ffn_gates: object
    scale_embedding: object


@torch.no_grad()
def forward_with_states_and_pack(model, x_ctx, y_cpu: torch.Tensor, x_tgt):
    h, n_ctx, output_scale, scale_info, attn_gates, ffn_gates, scale_embedding = _prepare_dx40_tokens(
        model, x_ctx, y_cpu, x_tgt
    )
    states = [h.detach().clone()]
    for layer_idx, layer in enumerate(model.layers):
        h = layer(h, **_layer_kwargs(model, n_ctx, attn_gates, ffn_gates, scale_embedding, layer_idx))
        states.append(h.detach().clone())
    pack = StatePack(n_ctx, output_scale, scale_info, attn_gates, ffn_gates, scale_embedding)
    return _dx40_head(model, h, n_ctx, output_scale, scale_info), states, pack


@torch.no_grad()
def roll_from_state(model, h: torch.Tensor, state_idx: int, pack: StatePack) -> torch.Tensor:
    for layer_idx in range(state_idx, len(model.layers)):
        h = model.layers[layer_idx](
            h,
            **_layer_kwargs(
                model,
                pack.n_ctx,
                pack.attn_gates,
                pack.ffn_gates,
                pack.scale_embedding,
                layer_idx,
            ),
        )
    return _dx40_head(model, h, pack.n_ctx, pack.output_scale, pack.scale_info)


@torch.no_grad()
def project_context_component(h: torch.Tensor, n_ctx: int, A_cpu: torch.Tensor, Q_cpu: torch.Tensor) -> torch.Tensor:
    h_proj = h.clone()
    if Q_cpu.numel() == 0 or Q_cpu.shape[1] == 0:
        h_proj[0, :n_ctx, :] = 0
        return h_proj
    dtype = h.dtype
    device = h.device
    A = A_cpu.to(device=device, dtype=dtype)
    Q = Q_cpu.to(device=device, dtype=dtype)
    Hctx = h[0, :n_ctx, :]
    h_proj[0, :n_ctx, :] = Q @ (Q.T @ (A @ Hctx))
    return h_proj


@torch.no_grad()
def keep_q_state(h: torch.Tensor, n_ctx: int, A_cpu: torch.Tensor, Q_cpu: torch.Tensor) -> torch.Tensor:
    h_new = h.clone()
    h_new[0, :n_ctx, :] = project_context_component(h, n_ctx, A_cpu, Q_cpu)[0, :n_ctx, :]
    return h_new


@torch.no_grad()
def remove_q_state(h: torch.Tensor, n_ctx: int, A_cpu: torch.Tensor, Q_cpu: torch.Tensor) -> torch.Tensor:
    h_new = h.clone()
    proj = project_context_component(h, n_ctx, A_cpu, Q_cpu)
    h_new[0, :n_ctx, :] = h[0, :n_ctx, :] - proj[0, :n_ctx, :]
    return h_new


def fd_from_cols(cols: Sequence[torch.Tensor], n_tgt: int) -> torch.Tensor:
    if not cols:
        return torch.zeros(n_tgt, 0, dtype=torch.float64)
    return torch.cat([c.reshape(-1, 1) for c in cols], dim=1)


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


def rel_point_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float((pred - target).norm() / (target.norm() + FLOOR))


def rel_drift(pred: torch.Tensor, base: torch.Tensor) -> float:
    return float((pred - base).norm() / (base.norm() + FLOOR))


def select_response_prefix(
    M_resp: torch.Tensor,
    A: torch.Tensor,
    Kt: torch.Tensor,
    T: torch.Tensor,
    risk_total: float,
    alpha: float,
    tau_sv: float,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    A_factors = symmetric_eig_factors(A)
    A_sqrt, A_invsqrt = psd_sqrt_and_invsqrt(A)
    cand_rank = min(max(M_resp.shape), M_resp.shape[0], M_resp.shape[1])
    C = a_orth_candidate(M_resp, A_factors, tau_sv, cand_rank)
    Q_all = krr_targeted_order(C, Kt, A_sqrt, A_invsqrt)

    best_rho = float("inf")
    best_rank = 0
    rank_to_alpha = None
    rho_by_rank: List[float] = []
    for rank in range(Q_all.shape[1] + 1):
        Q = Q_all[:, :rank]
        Tq = Kt @ Q @ Q.T if rank else torch.zeros_like(T)
        rho = exact_operator_excess(Tq, T, A) / (risk_total + FLOOR)
        rho_by_rank.append(float(rho))
        if rho < best_rho:
            best_rho = float(rho)
            best_rank = rank
        if rank_to_alpha is None and rho <= alpha:
            rank_to_alpha = rank

    chosen = rank_to_alpha if rank_to_alpha is not None else best_rank
    meta = {
        "candidate_dim": int(Q_all.shape[1]),
        "rank_Q": int(chosen),
        "rank_to_alpha": rank_to_alpha if rank_to_alpha is not None else "",
        "best_rank": int(best_rank),
        "rho_TQ": float(rho_by_rank[chosen]) if rho_by_rank else float("nan"),
        "best_rho": float(best_rho),
        "reach_alpha": 1.0 if rank_to_alpha is not None else 0.0,
    }
    return Q_all[:, :chosen], meta


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


def summarize(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["name"])].append(row)
    metrics = [
        "point_err",
        "point_drift",
        "E_F_T",
        "E_F_TQ",
        "damage_vs_base",
        "mse_ratio",
        "rank_Q",
        "candidate_dim",
        "r_T",
        "rho_TQ",
        "reach_alpha",
    ]
    out: List[Dict[str, object]] = []
    for name, group in sorted(groups.items()):
        rec: Dict[str, object] = {"name": name, "n": len(group)}
        for metric in metrics:
            vals = [float(r[metric]) for r in group if r.get(metric) not in ("", None)]
            vals = [v for v in vals if math.isfinite(v)]
            if vals:
                rec[f"{metric}_mean"] = float(np.mean(vals))
                rec[f"{metric}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        out.append(rec)
    return out


def write_summary_txt(path: Path, args: argparse.Namespace, rows: Sequence[Dict[str, object]], summary: Sequence[Dict[str, object]]) -> None:
    lines = [
        "dx40 Experiment 3 complement ablation",
        "",
        f"episodes={args.episodes}, n_ctx={args.n_ctx}, n_tgt={args.n_tgt}, state_idx={args.state_idx}",
        f"basis=response layer {args.basis_layer}, alpha={args.excess_risk_frac}, n_build={args.n_build}, n_eval={args.n_eval}",
        f"records={len(rows)}",
        "",
        f"{'name':10s} {'n':>3s} {'rankQ':>7s} {'rT':>6s} {'rhoTQ':>9s} {'mse/KRR':>9s} {'pt_err':>9s} {'drift':>9s} {'E(F,T)':>9s} {'damage':>9s}",
    ]
    for row in summary:
        lines.append(
            f"{str(row['name']):10s} {int(row['n']):3d} "
            f"{float(row.get('rank_Q_mean', float('nan'))):7.2f} "
            f"{float(row.get('r_T_mean', float('nan'))):6.2f} "
            f"{float(row.get('rho_TQ_mean', float('nan'))):9.5f} "
            f"{float(row.get('mse_ratio_mean', float('nan'))):9.3f} "
            f"{float(row.get('point_err_mean', float('nan'))):9.5f} "
            f"{float(row.get('point_drift_mean', float('nan'))):9.5f} "
            f"{float(row.get('E_F_T_mean', float('nan'))):9.5f} "
            f"{float(row.get('damage_vs_base_mean', float('nan'))):9.5f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def process_episode(model, cfg: CkptCfg, args: argparse.Namespace, sampler_cfg: Dx40SamplerCfg, ep: int, device: torch.device, gen: torch.Generator) -> List[Dict[str, object]]:
    x_ctx, y_ctx, x_tgt, y_tgt = sample_dx40_episode(cfg, args, device, sampler_cfg)
    y = y_ctx[0].detach().cpu().double()
    y_tgt_cpu = y_tgt[0].detach().cpu().double()

    _K, Kt, A, T = build_eval_kernels(x_ctx, x_tgt, args)
    Ktt = build_eval_target_kernel(x_tgt, args)
    Ty = T @ y

    risk_pack = effective_rank_T_excess_risk(T, A, Kt, Ktt, args.excess_risk_frac)
    r_t = int(risk_pack["r_eff_T_task"])
    risk_total = float(risk_pack["krr_risk_total"])

    build_probes = sample_task_probes(A, args.n_build, gen)
    response_layers = dx40_hidden_response_matrices(model, x_ctx, x_tgt, y, build_probes, args.eps)
    M_resp = response_layers[args.basis_layer]
    Q, q_meta = select_response_prefix(
        M_resp,
        A,
        Kt,
        T,
        risk_total,
        args.excess_risk_frac,
        args.tau_sv,
    )
    TQ = Kt @ Q @ Q.T if Q.shape[1] else torch.zeros_like(T)

    base_pred, base_states, base_pack = forward_with_states_and_pack(model, x_ctx, y, x_tgt)
    keep_pred = roll_from_state(
        model,
        keep_q_state(base_states[args.state_idx], args.n_ctx, A, Q),
        args.state_idx,
        base_pack,
    )
    remove_pred = roll_from_state(
        model,
        remove_q_state(base_states[args.state_idx], args.n_ctx, A, Q),
        args.state_idx,
        base_pack,
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
            keep_q_state(states_p[args.state_idx], args.n_ctx, A, Q),
            args.state_idx,
            pack_p,
        )
        keep_m = roll_from_state(
            model,
            keep_q_state(states_m[args.state_idx], args.n_ctx, A, Q),
            args.state_idx,
            pack_m,
        )
        fd_cols["keep_Q"].append((keep_p - keep_m) / (2.0 * args.eps))

        rem_p = roll_from_state(
            model,
            remove_q_state(states_p[args.state_idx], args.n_ctx, A, Q),
            args.state_idx,
            pack_p,
        )
        rem_m = roll_from_state(
            model,
            remove_q_state(states_m[args.state_idx], args.n_ctx, A, Q),
            args.state_idx,
            pack_m,
        )
        fd_cols["remove_Q"].append((rem_p - rem_m) / (2.0 * args.eps))

    fd = {name: fd_from_cols(cols, args.n_tgt) for name, cols in fd_cols.items()}
    preds = {"base": base_pred, "keep_Q": keep_pred, "remove_Q": remove_pred}
    mse_model = float(((base_pred - y_tgt_cpu) ** 2).mean())
    mse_krr = float(((Ty - y_tgt_cpu) ** 2).mean())
    mse_ratio = mse_model / max(mse_krr, FLOOR)

    rows: List[Dict[str, object]] = []
    for name in ["base", "keep_Q", "remove_Q"]:
        rows.append(
            {
                "episode": ep,
                "name": name,
                "state_idx": args.state_idx,
                "basis_layer": args.basis_layer,
                "n_ctx": args.n_ctx,
                "n_tgt": args.n_tgt,
                "r_T": r_t,
                "rank_Q": q_meta["rank_Q"],
                "candidate_dim": q_meta["candidate_dim"],
                "rank_to_alpha": q_meta["rank_to_alpha"],
                "best_rank": q_meta["best_rank"],
                "rho_TQ": q_meta["rho_TQ"],
                "best_rho": q_meta["best_rho"],
                "reach_alpha": q_meta["reach_alpha"],
                "mse_model": mse_model,
                "mse_krr": mse_krr,
                "mse_ratio": mse_ratio,
                "point_err": rel_point_error(preds[name], Ty),
                "point_drift": rel_drift(preds[name], base_pred),
                "E_F_T": fd_operator_error(fd[name], T, T, probes),
                "E_F_TQ": fd_operator_error(fd[name], TQ, T, probes),
                "damage_vs_base": fd_damage_error(fd[name], fd["base"], T, probes),
            }
        )
    print(
        f"episode {ep + 1}/{args.episodes}: rT={r_t}, rankQ={q_meta['rank_Q']}, "
        f"rho={float(q_meta['rho_TQ']):.4f}, mse_ratio={mse_ratio:.3f}",
        flush=True,
    )
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default=str(SCRIPT_DIR / "results_dx40_complement"))
    p.add_argument("--checkpoint", default=str(DX40_CHECKPOINT))
    p.add_argument("--name", default="dx40_tau16_s122_target_rich")
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    p.add_argument("--seed", type=int, default=122)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--n-ctx", type=int, default=47)
    p.add_argument("--n-tgt", type=int, default=40)
    p.add_argument("--sigma2", type=float, default=0.1)
    p.add_argument("--kernel-family", choices=["linear"], default="linear")
    p.add_argument("--n-build", type=int, default=16)
    p.add_argument("--n-eval", type=int, default=16)
    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--tau-sv", type=float, default=1e-3)
    p.add_argument("--excess-risk-frac", type=float, default=0.05)
    p.add_argument("--basis-layer", type=int, default=8)
    p.add_argument("--state-idx", type=int, default=1)
    p.add_argument("--tau-min", type=float, default=1e-3)
    p.add_argument("--tau-max", type=float, default=16.0)
    p.add_argument("--log10-lambda1-min", type=float, default=-2.0)
    p.add_argument("--log10-lambda1-max", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.basis_layer < 0 or args.basis_layer > 8:
        raise ValueError("--basis-layer must be in [0, 8]")
    if args.state_idx < 0 or args.state_idx > 8:
        raise ValueError("--state-idx must be in [0, 8]")

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device() if args.device == "auto" else torch.device(args.device)
    set_seed(args.seed)
    gen = torch.Generator(device="cpu").manual_seed(args.seed + 777)
    cfg = CkptCfg(args.name, args.checkpoint, 40, 128, 8, 4, args.kernel_family, 40.0)
    sampler_cfg = Dx40SamplerCfg(
        tau_min=args.tau_min,
        tau_max=args.tau_max,
        log10_lambda1_min=args.log10_lambda1_min,
        log10_lambda1_max=args.log10_lambda1_max,
    )
    model = load_dx40_model(args.checkpoint, device)

    (out_dir / "config.json").write_text(
        json.dumps({"args": vars(args), "device": str(device)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    rows: List[Dict[str, object]] = []
    for ep in range(args.episodes):
        rows.extend(process_episode(model, cfg, args, sampler_cfg, ep, device, gen))

    summary = summarize(rows)
    write_csv(out_dir / "records.csv", rows)
    write_csv(out_dir / "summary.csv", summary)
    write_summary_txt(out_dir / "summary.txt", args, rows, summary)
    print((out_dir / "summary.txt").read_text(), flush=True)


if __name__ == "__main__":
    main()
