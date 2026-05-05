#!/usr/bin/env python3
"""Run repo experiments 1-3 on the dx40 scale-aware checkpoint.

This file intentionally keeps the baseline experiment scripts unchanged. It
imports them, replaces only the model/data entry points in memory, and writes
outputs under this dx40 checkpoint directory.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import torch


DX40_DIR = Path(__file__).resolve().parent
REPO_ROOT = DX40_DIR.parent
SRC_DIR = REPO_ROOT / "src"
SUPPORT_DIR = REPO_ROOT / "experiments" / "shared"
CHECKPOINT = DX40_DIR / "minimal_scale_canonical_single_dx40_tau16_am2p0_2p0_inv_rank_bs16_ga1_150000_s122.best.pt"


def _setup_paths() -> None:
    """Make repo imports resolve to the baseline src modules, not this folder."""
    dx40_str = str(DX40_DIR)
    while dx40_str in sys.path:
        sys.path.remove(dx40_str)
    for path in (SUPPORT_DIR, SRC_DIR, REPO_ROOT):
        path_str = str(path)
        if path_str in sys.path:
            sys.path.remove(path_str)
        sys.path.insert(0, path_str)
    os.environ.setdefault("MPLCONFIGDIR", str(DX40_DIR / ".mpl-cache"))


def _load_file_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_setup_paths()
DX40_MODEL = _load_file_module("_dx40_checkpoint_model", DX40_DIR / "model.py")
DX40_DATA = _load_file_module("_dx40_checkpoint_data", DX40_DIR / "data.py")


@dataclass
class Dx40SamplerCfg:
    tau_min: float = 1e-3
    tau_max: float = 16.0
    log10_lambda1_min: float = -2.0
    log10_lambda1_max: float = 1.0
    smooth_span_min: float = 0.0
    smooth_span_max: float = 8.0
    step_depth_min: float = 1.0
    step_depth_max: float = 8.0
    step_rank_distribution: str = "inverse_rank"
    rejection_attempts: int = 512


def _torch_load_state(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_dx40_model(path: Optional[str], device: torch.device) -> torch.nn.Module:
    ckpt = Path(path) if path else CHECKPOINT
    model = DX40_MODEL.ICLTransformer(
        d_x=40,
        d_model=128,
        n_layers=8,
        n_heads=4,
        ffn_mult=2,
        mask_tgt_tgt=True,
        scale_canonical=True,
        scale_stat="mean_x2",
        scale_eps=1e-8,
        scale_y=True,
        scale_controller="layer_gates",
        scale_gate_hidden=16,
        scale_gate_bound=3.0,
        scale_log_clip=8.0,
        scale_conditioner="none",
    )
    model.load_state_dict(_torch_load_state(ckpt, device))
    model.to(device).eval()
    return model


def sample_dx40_batch(
    batch_size: int,
    d: int,
    n_ctx: int,
    n_tgt: int,
    sigma2: float,
    device: torch.device | str,
    sampler_cfg: Dx40SamplerCfg,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if int(d) != 40:
        raise ValueError(f"dx40 runner expected d=40, got d={d}")

    _, eig, _, _ = DX40_DATA._sample_minimal_batch_spectrum(
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
    eigenvalues = eig.unsqueeze(0).repeat(batch_size, 1)
    return DX40_DATA._build_batch(eigenvalues, batch_size, 40, n_ctx, n_tgt, sigma2, device)


def linear_kernel_batches(
    x_ctx: torch.Tensor,
    x_tgt: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xc = x_ctx.detach().cpu().double()
    xt = x_tgt.detach().cpu().double()
    return xc @ xc.transpose(-2, -1), xt @ xc.transpose(-2, -1)


@torch.no_grad()
def _prepare_dx40_tokens(
    model: torch.nn.Module,
    x_ctx: torch.Tensor,
    y_cpu: torch.Tensor,
    x_tgt: torch.Tensor,
):
    dtype = next(model.parameters()).dtype
    device = x_ctx.device
    y = y_cpu.view(1, -1).to(device=device, dtype=dtype)
    x_ctx = x_ctx.to(device=device, dtype=dtype)
    x_tgt = x_tgt.to(device=device, dtype=dtype)

    x_ctx_p, y_p, x_tgt_p, output_scale, scale_info = model._prepare_scale(x_ctx, y, x_tgt)
    attn_gates, ffn_gates = model._compute_gates(scale_info) if scale_info else (None, None)
    scale_embedding = model._compute_scale_embedding(scale_info) if scale_info else None

    batch_size, n_ctx, _ = x_ctx_p.shape
    n_tgt = x_tgt_p.shape[1]
    ctx = torch.cat(
        [
            x_ctx_p,
            y_p.unsqueeze(-1),
            torch.zeros(batch_size, n_ctx, 1, device=device, dtype=dtype),
        ],
        dim=-1,
    )
    tgt = torch.cat(
        [
            x_tgt_p,
            torch.zeros(batch_size, n_tgt, 1, device=device, dtype=dtype),
            torch.ones(batch_size, n_tgt, 1, device=device, dtype=dtype),
        ],
        dim=-1,
    )
    h = model.embed(torch.cat([ctx, tgt], dim=1))
    return h, n_ctx, output_scale, scale_info, attn_gates, ffn_gates, scale_embedding


def _layer_kwargs(model: torch.nn.Module, n_ctx: int, attn_gates, ffn_gates, scale_embedding, layer_idx: int):
    return {
        "n_ctx": n_ctx,
        "attn_gate": model._gate_at(attn_gates, layer_idx),
        "ffn_gate": model._gate_at(ffn_gates, layer_idx),
        "scale_embedding": scale_embedding,
    }


def _dx40_head(
    model: torch.nn.Module,
    h: torch.Tensor,
    n_ctx: int,
    output_scale,
    scale_info,
) -> torch.Tensor:
    h_tgt = h[:, n_ctx:, :]
    preds = model.head(h_tgt).squeeze(-1)
    preds = model._apply_final_scale_adapter(preds, h_tgt, scale_info)
    preds = model._rescale_preds(preds, output_scale)
    return preds[0].detach().cpu().double()


@torch.no_grad()
def dx40_forward_with_states(
    model: torch.nn.Module,
    x_ctx: torch.Tensor,
    y_cpu: torch.Tensor,
    x_tgt: torch.Tensor,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    h, n_ctx, output_scale, scale_info, attn_gates, ffn_gates, scale_embedding = _prepare_dx40_tokens(
        model, x_ctx, y_cpu, x_tgt
    )
    states = [h.detach().clone()]
    for i, layer in enumerate(model.layers):
        h = layer(h, **_layer_kwargs(model, n_ctx, attn_gates, ffn_gates, scale_embedding, i))
        states.append(h.detach().clone())
    return _dx40_head(model, h, n_ctx, output_scale, scale_info), states


@torch.no_grad()
def dx40_forward_with_ctx_hidden(
    model: torch.nn.Module,
    x_ctx: torch.Tensor,
    y_cpu: torch.Tensor,
    x_tgt: torch.Tensor,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    pred, states = dx40_forward_with_states(model, x_ctx, y_cpu, x_tgt)
    n_ctx = x_ctx.shape[1]
    return pred, [s[0, :n_ctx, :].detach().cpu().double() for s in states]


@torch.no_grad()
def dx40_forward_pred_only(
    model: torch.nn.Module,
    x_ctx: torch.Tensor,
    y_cpu: torch.Tensor,
    x_tgt: torch.Tensor,
) -> torch.Tensor:
    pred, _states = dx40_forward_with_states(model, x_ctx, y_cpu, x_tgt)
    return pred


@torch.no_grad()
def dx40_forward_with_surgery(
    exp3_module,
    model: torch.nn.Module,
    x_ctx: torch.Tensor,
    y_cpu: torch.Tensor,
    x_tgt: torch.Tensor,
    A_cpu: torch.Tensor,
    D_cpu: torch.Tensor,
    state_idx: int,
) -> torch.Tensor:
    h, n_ctx, output_scale, scale_info, attn_gates, ffn_gates, scale_embedding = _prepare_dx40_tokens(
        model, x_ctx, y_cpu, x_tgt
    )
    n_layers = len(model.layers)
    if state_idx < 0 or state_idx > n_layers:
        raise ValueError(f"state_idx must be in [0,{n_layers}], got {state_idx}")

    if state_idx == 0:
        h = exp3_module.apply_context_projection_removal(h, n_ctx, A_cpu, D_cpu)

    for i, layer in enumerate(model.layers):
        h = layer(h, **_layer_kwargs(model, n_ctx, attn_gates, ffn_gates, scale_embedding, i))
        if state_idx == i + 1:
            h = exp3_module.apply_context_projection_removal(h, n_ctx, A_cpu, D_cpu)

    return _dx40_head(model, h, n_ctx, output_scale, scale_info)


def _run_with_argv(argv: Sequence[str], fn) -> None:
    old_argv = sys.argv[:]
    try:
        sys.argv = list(argv)
        fn()
    finally:
        sys.argv = old_argv


def patch_exp1(module, sampler_cfg: Dx40SamplerCfg):
    module.MODELS["dx40"] = {
        "checkpoint": str(CHECKPOINT),
        "d_x": 40,
        "d_model": 128,
        "n_heads": 4,
        "label": r"$d_x=40$, scale-aware gated",
    }

    def load_model(path, cfg, device, n_layers=None):
        if int(cfg.d_x) != 40:
            raise ValueError("dx40 runner expected Experiment 1 to use --model-key dx40")
        if n_layers not in (None, 8):
            raise ValueError("dx40 checkpoint has exactly 8 layers")
        return load_dx40_model(path, device)

    def sample_episode_batch(model_key, cfg, device, return_target_kernel=False):
        if model_key != "dx40":
            raise ValueError("dx40 runner expected Experiment 1 to use --model-key dx40")
        x_ctx, y_ctx, x_tgt, y_tgt, _ = sample_dx40_batch(
            cfg.batch_size, cfg.d_x, cfg.n_ctx, cfg.n_tgt, cfg.sigma2, device, sampler_cfg
        )
        k_batch, kt_batch = linear_kernel_batches(x_ctx, x_tgt)
        if return_target_kernel:
            return x_ctx, y_ctx, x_tgt, y_tgt, k_batch, kt_batch
        return x_ctx, y_ctx, x_tgt, y_tgt, k_batch

    module.load_model = load_model
    module.sample_episode_batch = sample_episode_batch


def patch_exp2(module, sampler_cfg: Dx40SamplerCfg):
    budget_module = importlib.import_module("experiments.exp2_budget_closure.run")
    dx40_cfg = module.CkptCfg("dx40", str(CHECKPOINT), 40, 128, 8, 4, "dx", 40)

    for mod in (module, budget_module):
        mod.REPO_ROOT = REPO_ROOT
        mod.CHECKPOINTS_2A = []
        mod.CHECKPOINTS_2B = [dx40_cfg]

    def load_model(cfg, device):
        if cfg.name != "dx40":
            raise ValueError("dx40 runner patched Experiment 2 for dx40 only")
        return load_dx40_model(cfg.checkpoint, device)

    def sample_episode(cfg, sigma2, n_ctx, n_tgt, device):
        x_ctx, y_ctx, x_tgt, y_tgt, _ = sample_dx40_batch(
            1, cfg.d_x, n_ctx, n_tgt, sigma2, device, sampler_cfg
        )
        return x_ctx, y_ctx, x_tgt, y_tgt

    def forward_with_surgery(model, x_ctx, y_cpu, x_tgt, A_cpu, D_cpu, state_idx):
        return dx40_forward_with_surgery(module, model, x_ctx, y_cpu, x_tgt, A_cpu, D_cpu, state_idx)

    for mod in (module, budget_module):
        mod.load_model = load_model
        mod.sample_episode = sample_episode
        mod.forward_with_ctx_hidden = dx40_forward_with_ctx_hidden
        mod.forward_pred_only = dx40_forward_pred_only

    # The validated-rank script imports these helpers from the budget script.
    # Rebind them after patching the budget module's dx40-aware globals.
    module.prediction_fd_bundle = budget_module.prediction_fd_bundle
    module.hidden_response_matrices = budget_module.hidden_response_matrices
    module.forward_with_surgery = forward_with_surgery


def patch_exp3(module, sampler_cfg: Dx40SamplerCfg):
    def load_model(cfg, device):
        if int(cfg.d_x) != 40:
            raise ValueError("dx40 runner patched Experiment 3 for d_x=40 only")
        return load_dx40_model(cfg.checkpoint, device)

    def sample_episode(cfg, sigma2, n_ctx, n_tgt, device):
        x_ctx, y_ctx, x_tgt, y_tgt, _ = sample_dx40_batch(
            1, cfg.d_x, n_ctx, n_tgt, sigma2, device, sampler_cfg
        )
        return x_ctx, y_ctx, x_tgt, y_tgt

    def forward_base_with_states(model, x_ctx, y_cpu, x_tgt):
        return dx40_forward_with_states(model, x_ctx, y_cpu, x_tgt)

    def forward_with_surgery(model, x_ctx, y_cpu, x_tgt, A_cpu, D_cpu, state_idx):
        return dx40_forward_with_surgery(module, model, x_ctx, y_cpu, x_tgt, A_cpu, D_cpu, state_idx)

    module.load_model = load_model
    module.sample_episode = sample_episode
    module.forward_base_with_states = forward_base_with_states
    module.forward_with_surgery = forward_with_surgery


def parse_int_list(raw: str) -> List[int]:
    vals = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not vals:
        raise ValueError(f"empty integer list: {raw!r}")
    return vals


def selected_experiments(raw: str) -> List[int]:
    vals = parse_int_list(raw)
    bad = sorted(set(vals) - {1, 2, 3})
    if bad:
        raise ValueError(f"unknown experiment id(s): {bad}")
    return vals


def write_runner_config(args: argparse.Namespace, sampler_cfg: Dx40SamplerCfg, results_root: Path) -> None:
    payload = {
        "checkpoint": str(CHECKPOINT),
        "repo_root": str(REPO_ROOT),
        "sampler": sampler_cfg.__dict__,
        "runner_args": vars(args),
        "note": "Existing experiment modules were patched in memory only; baseline files were not modified.",
    }
    results_root.mkdir(parents=True, exist_ok=True)
    (results_root / "dx40_runner_config.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_exp1(args: argparse.Namespace, sampler_cfg: Dx40SamplerCfg, results_root: Path) -> None:
    exp1 = importlib.import_module("experiments.exp1_operator_certificate.run")
    patch_exp1(exp1, sampler_cfg)
    out_dir = results_root / "results_exp1_operator_certificate"
    argv = [
        "exp1_dx40",
        "--model-key",
        "dx40",
        "--checkpoint",
        str(CHECKPOINT),
        "--device",
        args.device,
        "--episodes",
        str(args.exp1_episodes),
        "--n-ctx",
        str(args.n_ctx),
        "--n-tgt",
        str(args.exp1_n_tgt),
        "--r-max",
        str(args.exp1_r_max),
        "--curve-r-max",
        str(args.exp1_curve_r_max),
        "--n-build",
        str(args.exp1_n_build),
        "--n-eval",
        str(args.exp1_n_eval),
        "--results-dir",
        str(out_dir),
    ]
    print("\n=== Running Experiment 1 on dx40 ===", flush=True)
    print(" ".join(argv), flush=True)
    _run_with_argv(argv, exp1.run)


def run_exp2(args: argparse.Namespace, sampler_cfg: Dx40SamplerCfg, results_root: Path) -> None:
    exp2 = importlib.import_module("experiments.exp2_validated_rank.run")
    patch_exp2(exp2, sampler_cfg)
    out_dir = results_root / "results_exp2_validated_rank"
    argv = [
        "exp2_validated_dx40",
        "--exp",
        "2b",
        "--checkpoints",
        "dx40",
        "--selection-mode",
        args.exp2_selection_mode,
        "--span-mode",
        args.exp2_span_mode,
        "--device",
        args.device,
        "--episodes",
        str(args.exp2_episodes),
        "--n-ctx",
        str(args.n_ctx),
        "--n-tgt",
        str(args.exp2_n_tgt),
        "--n-build",
        str(args.exp2_n_build),
        "--n-val",
        str(args.exp2_n_val),
        "--n-eval",
        str(args.exp2_n_eval),
        "--n-causal",
        str(args.exp2_n_causal),
        "--causal-damage-tau",
        str(args.exp2_causal_damage_tau),
        "--max-candidate-rank",
        str(args.exp2_max_candidate_rank),
        "--excess-risk-frac",
        str(args.exp2_excess_risk_frac),
        "--probe-kind",
        args.exp2_probe_kind,
        "--results-dir",
        str(out_dir),
    ]
    print("\n=== Running Experiment 2 validated native rank on dx40 ===", flush=True)
    print(" ".join(argv), flush=True)
    _run_with_argv(argv, lambda: exp2.run(exp2.parse_args()))


def run_exp3(args: argparse.Namespace, sampler_cfg: Dx40SamplerCfg, results_root: Path) -> None:
    exp3 = importlib.import_module("experiments.exp3_causal_surgery.run")
    patch_exp3(exp3, sampler_cfg)
    out_dir = results_root / "results_exp3_causal_surgery"
    argv = [
        "exp3_dx40",
        "--checkpoint",
        str(CHECKPOINT),
        "--d-x",
        "40",
        "--d-model",
        "128",
        "--n-layers",
        "8",
        "--n-heads",
        "4",
        "--device",
        args.device,
        "--episodes",
        str(args.exp3_episodes),
        "--n-ctx",
        str(args.n_ctx),
        "--n-tgt",
        str(args.exp3_n_tgt),
        "--n-causal",
        str(args.exp3_n_causal),
        "--rmax",
        str(args.exp3_rmax),
        "--k-remove-list",
        args.exp3_k_remove_list,
        "--layer-rule",
        args.exp3_layer_rule,
        "--results-dir",
        str(out_dir),
    ]
    print("\n=== Running Experiment 3 on dx40 ===", flush=True)
    print(" ".join(argv), flush=True)
    _run_with_argv(argv, exp3.main)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Experiments 1-3 on the dx40 checkpoint.")
    parser.add_argument("--only", default="1,2,3", help="Comma-separated experiment ids to run.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--results-root", default=str(DX40_DIR))

    parser.add_argument("--n-ctx", type=int, default=47)
    parser.add_argument("--exp1-n-tgt", type=int, default=40)
    parser.add_argument("--exp2-n-tgt", type=int, default=40)
    parser.add_argument("--exp3-n-tgt", type=int, default=40)

    parser.add_argument("--exp1-episodes", type=int, default=4)
    parser.add_argument("--exp1-r-max", type=int, default=40)
    parser.add_argument("--exp1-curve-r-max", type=int, default=40)
    parser.add_argument("--exp1-n-build", type=int, default=16)
    parser.add_argument("--exp1-n-eval", type=int, default=32)
    parser.add_argument("--exp2-episodes", type=int, default=4)
    parser.add_argument("--exp3-episodes", type=int, default=8)
    parser.add_argument("--exp2-selection-mode", choices=["hidden", "causal"], default="causal")
    parser.add_argument("--exp2-span-mode", choices=["seed", "reachable"], default="seed")
    parser.add_argument("--exp2-n-build", type=int, default=16)
    parser.add_argument("--exp2-n-val", type=int, default=16)
    parser.add_argument("--exp2-n-eval", type=int, default=16)
    parser.add_argument("--exp2-n-causal", type=int, default=8)
    parser.add_argument("--exp2-causal-damage-tau", type=float, default=0.05)
    parser.add_argument("--exp2-max-candidate-rank", type=int, default=40)
    parser.add_argument("--exp2-excess-risk-frac", type=float, default=0.05)
    parser.add_argument("--exp2-probe-kind", choices=["task", "iso", "both"], default="task")
    parser.add_argument("--exp3-n-causal", type=int, default=40)
    parser.add_argument("--exp3-rmax", type=int, default=40)
    parser.add_argument("--exp3-k-remove-list", default="1,2,4,8,16")
    parser.add_argument("--exp3-layer-rule", choices=["final", "sweep"], default="sweep")

    parser.add_argument("--tau-min", type=float, default=1e-3)
    parser.add_argument("--tau-max", type=float, default=16.0)
    parser.add_argument("--log10-lambda1-min", type=float, default=-2.0)
    parser.add_argument(
        "--log10-lambda1-max",
        type=float,
        default=1.0,
        help="Default 1.0 means lambda_1 <= 10, matching the collaborator's robustness note.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(0)
    sampler_cfg = Dx40SamplerCfg(
        tau_min=args.tau_min,
        tau_max=args.tau_max,
        log10_lambda1_min=args.log10_lambda1_min,
        log10_lambda1_max=args.log10_lambda1_max,
    )
    results_root = Path(args.results_root).resolve()
    write_runner_config(args, sampler_cfg, results_root)

    experiments = selected_experiments(args.only)
    if 1 in experiments:
        run_exp1(args, sampler_cfg, results_root)
    if 2 in experiments:
        run_exp2(args, sampler_cfg, results_root)
    if 3 in experiments:
        run_exp3(args, sampler_cfg, results_root)

    print("\nDone. Results root:", results_root, flush=True)


if __name__ == "__main__":
    main()
