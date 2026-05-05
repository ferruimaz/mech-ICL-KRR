"""Utilities for the final d_x=40 scale-aware gated checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from . import dx40_data, dx40_model


REPO_ROOT = Path(__file__).resolve().parents[2]
DX40_CHECKPOINT = REPO_ROOT / "checkpoints" / "final" / "dx40_tau16_s122.pt"


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


def load_dx40_model(path: Optional[str | Path], device: torch.device) -> torch.nn.Module:
    ckpt = Path(path) if path else DX40_CHECKPOINT
    model = dx40_model.ICLTransformer(
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
        raise ValueError(f"dx40 sampler expected d=40, got d={d}")

    _, eig, _, _ = dx40_data._sample_minimal_batch_spectrum(
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
    return dx40_data._build_batch(eigenvalues, batch_size, 40, n_ctx, n_tgt, sigma2, device)


def sample_dx40_episode(cfg, args, device: torch.device, sampler_cfg: Dx40SamplerCfg):
    x_ctx, y_ctx, x_tgt, y_tgt, _ = sample_dx40_batch(
        1, cfg.d_x, args.n_ctx, args.n_tgt, args.sigma2, device, sampler_cfg
    )
    return x_ctx, y_ctx, x_tgt, y_tgt


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
    for layer_idx, layer in enumerate(model.layers):
        h = layer(h, **_layer_kwargs(model, n_ctx, attn_gates, ffn_gates, scale_embedding, layer_idx))
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
def dx40_prediction_fd_bundle(
    model: torch.nn.Module,
    x_ctx: torch.Tensor,
    x_tgt: torch.Tensor,
    y: torch.Tensor,
    probes: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    cols: List[torch.Tensor] = []
    for u in probes:
        pp = dx40_forward_pred_only(model, x_ctx, y + eps * u, x_tgt)
        pn = dx40_forward_pred_only(model, x_ctx, y - eps * u, x_tgt)
        cols.append(((pp - pn) / (2.0 * eps)).reshape(-1, 1))
    n_tgt = x_tgt.shape[1]
    return torch.cat(cols, dim=1) if cols else torch.zeros(n_tgt, 0, dtype=torch.float64)


def dx40_hidden_response_matrices(
    model: torch.nn.Module,
    x_ctx: torch.Tensor,
    x_tgt: torch.Tensor,
    y: torch.Tensor,
    build_probes: torch.Tensor,
    eps: float,
) -> List[torch.Tensor]:
    accum: List[List[torch.Tensor]] | None = None
    for u in build_probes:
        _, hp = dx40_forward_with_ctx_hidden(model, x_ctx, y + eps * u, x_tgt)
        _, hn = dx40_forward_with_ctx_hidden(model, x_ctx, y - eps * u, x_tgt)
        z_list = [(a - b) / (2.0 * eps) for a, b in zip(hp, hn)]
        if accum is None:
            accum = [[] for _ in z_list]
        for layer, z in enumerate(z_list):
            accum[layer].append(z)
    if accum is None:
        _, states = dx40_forward_with_ctx_hidden(model, x_ctx, y, x_tgt)
        return [torch.zeros(state.shape[0], 0, dtype=torch.float64) for state in states]
    return [torch.cat(parts, dim=1) for parts in accum]
