#!/usr/bin/env python3
"""
Training loop, evaluation, and CLI for in-context linear regression transformer.

There are two training modes:
  - ``legacy``: original polynomial / exponential / step curriculum
  - ``minimal``: bounded-d_eff curriculum using smooth / step normalized shapes,
    target tau values, and gamma-controlled context sizes
"""

import argparse
import multiprocessing as mp
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from model import ICLTransformer
from data import (
    MINIMAL_PROFILE_NAMES,
    PROFILE_NAMES,
    effective_dimension,
    sample_batch,
    sample_batch_minimal,
    sample_batch_minimal_same_shape_multiscale,
    sample_batch_profile,
)


# ── Configuration ────────────────────────────────────────────────────────

@dataclass
class Config:
    # Problem
    d_x: int = 5
    sigma2: float = 0.1
    device: str = "auto"  # auto | cpu | mps | cuda
    cpu_threads: int = 0
    sample_device: str = "train"  # train | cpu

    # Legacy episode structure
    n_points: int = 50
    min_tgt: int = 1
    max_tgt: int = 10
    n_ctx: int = 47
    n_tgt: int = 3

    # Minimal bounded-d_eff curriculum
    spectrum_family: str = "legacy"  # legacy | minimal
    fixed_n_tgt: int = 8
    minimal_sampling_scheme: str = "tau_exact"  # tau_exact | scale_uniform_reject_tau | tau_uniform_reject_scale | scale_tau_direct
    tau_values: str = "1,2,3"
    minimal_tau_min: float = 1e-3
    minimal_tau_max: float = 3.0
    log10_lambda1_min: float = -1.0
    log10_lambda1_max: float = 1.0
    minimal_scale_distribution: str = "uniform"  # uniform | large_power
    minimal_scale_distribution_power: float = 2.0
    minimal_rejection_attempts: int = 256
    gamma_values: str = "8,16,32"
    minimal_smooth_span_min: float = 0.0
    minimal_smooth_span_max: float = 3.0
    minimal_step_rank_values: str = "1,2,3,4"
    minimal_step_rank_distribution: str = "uniform"
    minimal_step_depth_min: float = 1.0
    minimal_step_depth_max: float = 3.0
    minimal_multiscale_mode: str = "off"  # off | same_shape_iid
    minimal_multiscale_k: int = 1

    # Model
    d_model: int = 128
    n_layers: int = 8
    n_heads: int = 4
    ffn_mult: int = 2
    mask_tgt_tgt: bool = False
    finite_attn_mask: bool = False
    attn_logit_clip: float = 30.0
    scale_canonical: bool = False
    scale_stat: str = "mean_x2"
    scale_eps: float = 1e-8
    scale_y: bool = True
    scale_controller: str = "none"  # none | layer_gates
    scale_gate_hidden: int = 16
    scale_gate_bound: float = 3.0
    scale_log_clip: float = 8.0
    scale_conditioner: str = "none"  # none | final_adapter | ffn_bias
    scale_condition_dim: int = 8
    scale_condition_hidden: int = 16

    # Training
    batch_size: int = 64
    train_steps: int = 10000
    lr: float = 1e-3
    min_lr: float = 0.0
    warmup_steps: int = 0
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    abort_nonfinite_grad: bool = True
    prefetch_batches: int = 0
    prefetch_worker_threads: int = 1
    seed: int = 42

    # Evaluation
    eval_every: int = 500
    eval_batches: int = 10
    fixed_validation_bank: bool = False
    validation_bank_size: int = 0
    validation_bank_seed: int = 12345
    validation_bank_path: str = ""

    # Output
    save_path: str = ""
    init_checkpoint: str = ""
    save_best: bool = True
    save_every: int = 0  # <=0 disables periodic snapshots
    final_eval: bool = True
    sentinel_mode: str = "off"  # off | warn | stop
    sentinel_train_loss_max: float = 0.0  # <=0 disables threshold
    sentinel_eval_ratio_max: float = 0.0  # <=0 disables threshold


# ── Helpers ──────────────────────────────────────────────────────────────

def get_device(cfg):
    choice = str(getattr(cfg, "device", "auto")).lower()
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("requested device=mps but MPS is not available")
        return torch.device("mps")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("requested device=cuda but CUDA is not available")
        return torch.device("cuda")
    if choice != "auto":
        raise ValueError(f"unknown device option: {cfg.device}")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"cannot parse boolean value: {value}")


def _parse_int_list(text):
    if isinstance(text, (list, tuple)):
        return [int(x) for x in text]
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def _parse_float_list(text):
    if isinstance(text, (list, tuple)):
        return [float(x) for x in text]
    return [float(part.strip()) for part in str(text).split(",") if part.strip()]


def _derive_checkpoint_path(base_path, suffix):
    root, ext = os.path.splitext(base_path)
    if ext:
        return f"{root}{suffix}{ext}"
    return f"{base_path}{suffix}"


def _save_state_dict(model, path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(model.state_dict(), path)


def _set_lr(optimizer, lr_value):
    for group in optimizer.param_groups:
        group["lr"] = lr_value


def _restore_rng_states(py_state, np_state, torch_state):
    random.setstate(py_state)
    np.random.set_state(np_state)
    torch.random.set_rng_state(torch_state)


def _materialize_eval_batch(batch, device):
    return tuple(t.to(device) for t in batch)


def _resolve_sample_device(cfg, train_device):
    choice = str(getattr(cfg, "sample_device", "train")).strip().lower()
    if choice in {"train", "device"}:
        return train_device
    if choice == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unknown sample_device: {cfg.sample_device}")


def _build_validation_bank(cfg):
    """
    Build a fixed validation bank on CPU, restoring the caller RNG states
    afterwards so training randomness is unaffected.
    """
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    try:
        set_seed(int(cfg.validation_bank_seed))
        bank = []
        for _ in range(int(cfg.validation_bank_size)):
            batch = _sample_eval(cfg, torch.device("cpu"))
            bank.append(tuple(t.cpu() for t in batch))
        return bank
    finally:
        _restore_rng_states(py_state, np_state, torch_state)


def _load_or_create_validation_bank(cfg):
    if int(cfg.validation_bank_size) <= 0:
        raise ValueError("validation_bank_size must be positive when fixed_validation_bank=true")
    if cfg.validation_bank_path and os.path.exists(cfg.validation_bank_path):
        bank = torch.load(cfg.validation_bank_path, map_location="cpu")
        print(f"Loaded fixed validation bank from {cfg.validation_bank_path}")
        return bank
    bank = _build_validation_bank(cfg)
    if cfg.validation_bank_path:
        directory = os.path.dirname(cfg.validation_bank_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(bank, cfg.validation_bank_path)
        print(f"Saved fixed validation bank to {cfg.validation_bank_path}")
    return bank


def _sample_minimal(cfg, device, profile=None, tau_target=None, gamma=None, allow_multiscale=False):
    gamma_values = _parse_int_list(cfg.gamma_values)
    step_rank_values = _parse_int_list(cfg.minimal_step_rank_values)

    if tau_target is not None:
        tau_target = float(tau_target)
    elif cfg.minimal_sampling_scheme == "tau_exact":
        tau_target = float(random.choice(_parse_float_list(cfg.tau_values)))
    else:
        tau_target = None
    gamma = int(gamma if gamma is not None else random.choice(gamma_values))
    if (
        allow_multiscale
        and
        profile is None
        and tau_target is None
        and str(getattr(cfg, "minimal_multiscale_mode", "off")).lower() == "same_shape_iid"
        and int(getattr(cfg, "minimal_multiscale_k", 1)) > 1
    ):
        return sample_batch_minimal_same_shape_multiscale(
            batch_size=cfg.batch_size,
            d=cfg.d_x,
            n_tgt=cfg.fixed_n_tgt,
            sigma2=cfg.sigma2,
            gamma=gamma,
            k=cfg.minimal_multiscale_k,
            profile=None,
            device=device,
            tau_min=cfg.minimal_tau_min,
            tau_max=cfg.minimal_tau_max,
            log10_lambda1_min=cfg.log10_lambda1_min,
            log10_lambda1_max=cfg.log10_lambda1_max,
            scale_distribution=cfg.minimal_scale_distribution,
            scale_distribution_power=cfg.minimal_scale_distribution_power,
            rejection_attempts=cfg.minimal_rejection_attempts,
            smooth_span_min=cfg.minimal_smooth_span_min,
            smooth_span_max=cfg.minimal_smooth_span_max,
            step_rank_values=step_rank_values,
            step_rank_distribution=cfg.minimal_step_rank_distribution,
            step_depth_min=cfg.minimal_step_depth_min,
            step_depth_max=cfg.minimal_step_depth_max,
        )
    return sample_batch_minimal(
        batch_size=cfg.batch_size,
        d=cfg.d_x,
        n_tgt=cfg.fixed_n_tgt,
        sigma2=cfg.sigma2,
        tau_target=tau_target,
        gamma=gamma,
        profile=profile,
        device=device,
        sampling_scheme=cfg.minimal_sampling_scheme,
        tau_min=cfg.minimal_tau_min,
        tau_max=cfg.minimal_tau_max,
        log10_lambda1_min=cfg.log10_lambda1_min,
        log10_lambda1_max=cfg.log10_lambda1_max,
        scale_distribution=cfg.minimal_scale_distribution,
        scale_distribution_power=cfg.minimal_scale_distribution_power,
        rejection_attempts=cfg.minimal_rejection_attempts,
        smooth_span_min=cfg.minimal_smooth_span_min,
        smooth_span_max=cfg.minimal_smooth_span_max,
        step_rank_values=step_rank_values,
        step_rank_distribution=cfg.minimal_step_rank_distribution,
        step_depth_min=cfg.minimal_step_depth_min,
        step_depth_max=cfg.minimal_step_depth_max,
    )


def _sample_train(cfg, device):
    if cfg.spectrum_family == "legacy":
        n_tgt = torch.randint(cfg.min_tgt, cfg.max_tgt + 1, (1,)).item()
        n_ctx = cfg.n_points - n_tgt
        return sample_batch(cfg.batch_size, cfg.d_x, n_ctx, n_tgt, cfg.sigma2, device)
    if cfg.spectrum_family == "minimal":
        return _sample_minimal(cfg, device, allow_multiscale=True)
    raise ValueError(f"unknown spectrum_family: {cfg.spectrum_family}")


def _sample_eval(cfg, device):
    if cfg.spectrum_family == "legacy":
        return sample_batch(cfg.batch_size, cfg.d_x, cfg.n_ctx, cfg.n_tgt, cfg.sigma2, device)
    if cfg.spectrum_family == "minimal":
        return _sample_minimal(cfg, device, allow_multiscale=False)
    raise ValueError(f"unknown spectrum_family: {cfg.spectrum_family}")


def _materialize_train_batch(batch, device):
    if isinstance(batch, list):
        return [_materialize_eval_batch(item, device) for item in batch]
    return [_materialize_eval_batch(batch, device)]


def _prefetch_worker(cfg_dict, sample_device_name, seed, queue, stop_event):
    """Generate CPU batches in a side process so training can consume them."""
    cfg = Config(**cfg_dict)
    if int(getattr(cfg, "prefetch_worker_threads", 1)) > 0:
        torch.set_num_threads(int(cfg.prefetch_worker_threads))
    set_seed(int(seed))
    sample_device = torch.device(sample_device_name)
    try:
        while not stop_event.is_set():
            batch = _sample_train(cfg, sample_device)
            queue.put(("ok", batch))
    except Exception as exc:
        queue.put(("error", repr(exc)))


def _start_prefetcher(cfg, sample_device):
    if int(getattr(cfg, "prefetch_batches", 0)) <= 0:
        return None, None, None
    if sample_device.type != "cpu":
        raise ValueError("prefetch_batches currently requires sample_device=cpu")
    ctx = mp.get_context("spawn")
    queue = ctx.Queue(maxsize=max(1, int(cfg.prefetch_batches)))
    stop_event = ctx.Event()
    proc = ctx.Process(
        target=_prefetch_worker,
        args=(cfg.__dict__, str(sample_device), int(cfg.seed) + 10_000_000, queue, stop_event),
        daemon=True,
    )
    proc.start()
    return proc, queue, stop_event


def _next_train_batch(cfg, sample_device, prefetch_queue):
    if prefetch_queue is None:
        return _sample_train(cfg, sample_device)
    status, payload = prefetch_queue.get()
    if status != "ok":
        raise RuntimeError(f"prefetch worker failed: {payload}")
    return payload


def _ridge_predictions(x_ctx, y_ctx, x_tgt, sigma2):
    """Bayes-optimal ridge predictions (λ = σ²)."""
    d = x_ctx.shape[-1]
    XtX = x_ctx.transpose(-2, -1) @ x_ctx
    Xty = x_ctx.transpose(-2, -1) @ y_ctx.unsqueeze(-1)
    I = torch.eye(d, device=x_ctx.device).unsqueeze(0)
    beta_hat = torch.linalg.solve(XtX + sigma2 * I, Xty)
    return (x_tgt @ beta_hat).squeeze(-1)


# ── Evaluation ───────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, cfg, device, validation_bank=None, sample_device=None):
    """Compute average MSE and MSE/MSE_ridge ratio over eval_batches."""
    model.eval()
    total_mse = 0.0
    total_ratio = 0.0
    if validation_bank is not None:
        batches = [_materialize_eval_batch(batch, device) for batch in validation_bank]
    else:
        src_device = sample_device or device
        batches = [_materialize_eval_batch(_sample_eval(cfg, src_device), device) for _ in range(cfg.eval_batches)]
    for x_ctx, y_ctx, x_tgt, y_tgt, _ in batches:
        preds = model(x_ctx, y_ctx, x_tgt)
        ridge_preds = _ridge_predictions(x_ctx, y_ctx, x_tgt, cfg.sigma2)

        mse_transformer = ((preds - y_tgt) ** 2).mean(-1)
        mse_ridge = ((ridge_preds - y_tgt) ** 2).mean(-1)
        total_mse += mse_transformer.mean().item()
        total_ratio += (mse_transformer / mse_ridge.clamp(min=1e-10)).mean().item()

    model.train()
    n = len(batches)
    return total_mse / n, total_ratio / n


@torch.no_grad()
def evaluate_legacy_profiles(model, cfg, device, batches_per_profile=20, sample_device=None):
    """Compute MSE/MSE_ridge ratio per legacy spectral profile."""
    model.eval()
    results = {}
    for name in PROFILE_NAMES:
        total_ratio = 0.0
        for _ in range(batches_per_profile):
            batch = sample_batch_profile(
                cfg.batch_size, cfg.d_x, cfg.n_ctx, cfg.n_tgt, cfg.sigma2, name, sample_device or device,
            )
            x_ctx, y_ctx, x_tgt, y_tgt, _ = _materialize_eval_batch(batch, device)
            preds = model(x_ctx, y_ctx, x_tgt)
            ridge_preds = _ridge_predictions(x_ctx, y_ctx, x_tgt, cfg.sigma2)
            mse_t = ((preds - y_tgt) ** 2).mean(-1)
            mse_r = ((ridge_preds - y_tgt) ** 2).mean(-1)
            total_ratio += (mse_t / mse_r.clamp(min=1e-10)).mean().item()
        results[name] = total_ratio / batches_per_profile
    model.train()
    return results


@torch.no_grad()
def evaluate_minimal_profiles_matched(model, cfg, device, batches_per_profile=20, sample_device=None):
    """Evaluate each minimal profile under the current training sampler measure."""
    model.eval()
    results = {}
    for profile in MINIMAL_PROFILE_NAMES:
        total_ratio = 0.0
        tau_vals = []
        a_vals = []
        n_ctx_vals = []
        for _ in range(batches_per_profile):
            batch = _sample_minimal(cfg, sample_device or device, profile=profile)
            x_ctx, y_ctx, x_tgt, y_tgt, eig = _materialize_eval_batch(batch, device)
            preds = model(x_ctx, y_ctx, x_tgt)
            ridge_preds = _ridge_predictions(x_ctx, y_ctx, x_tgt, cfg.sigma2)
            mse_t = ((preds - y_tgt) ** 2).mean(-1)
            mse_r = ((ridge_preds - y_tgt) ** 2).mean(-1)
            total_ratio += (mse_t / mse_r.clamp(min=1e-10)).mean().item()

            batch_tau = effective_dimension(eig, cfg.sigma2).cpu()
            batch_a = torch.log10(eig.max(dim=-1).values.clamp(min=1e-12)).cpu()
            tau_vals.append(batch_tau)
            a_vals.append(batch_a)
            n_ctx_vals.append(torch.full((eig.shape[0],), float(x_ctx.shape[1])))

        tau_cat = torch.cat(tau_vals)
        a_cat = torch.cat(a_vals)
        n_ctx_cat = torch.cat(n_ctx_vals)
        results[profile] = {
            "ratio": total_ratio / batches_per_profile,
            "tau_mean": float(tau_cat.mean()),
            "tau_p05": float(torch.quantile(tau_cat, 0.05)),
            "tau_p50": float(torch.quantile(tau_cat, 0.50)),
            "tau_p95": float(torch.quantile(tau_cat, 0.95)),
            "a_mean": float(a_cat.mean()),
            "a_p05": float(torch.quantile(a_cat, 0.05)),
            "a_p50": float(torch.quantile(a_cat, 0.50)),
            "a_p95": float(torch.quantile(a_cat, 0.95)),
            "n_ctx_mean": float(n_ctx_cat.mean()),
        }

    model.train()
    return results


@torch.no_grad()
def evaluate_minimal_reference_grid(model, cfg, device, batches_per_cell=10, sample_device=None):
    """Compute MSE/MSE_ridge on the tau-exact minimal profile × tau × gamma grid."""
    model.eval()
    tau_values = _parse_float_list(cfg.tau_values)
    gamma_values = _parse_int_list(cfg.gamma_values)

    profile_means = {}
    cell_results = {}
    for profile in MINIMAL_PROFILE_NAMES:
        profile_total = 0.0
        n_cells = 0
        for tau in tau_values:
            for gamma in gamma_values:
                total_ratio = 0.0
                for _ in range(batches_per_cell):
                    batch = _sample_minimal(
                        cfg, sample_device or device, profile=profile, tau_target=tau, gamma=gamma,
                    )
                    x_ctx, y_ctx, x_tgt, y_tgt, _ = _materialize_eval_batch(batch, device)
                    preds = model(x_ctx, y_ctx, x_tgt)
                    ridge_preds = _ridge_predictions(x_ctx, y_ctx, x_tgt, cfg.sigma2)
                    mse_t = ((preds - y_tgt) ** 2).mean(-1)
                    mse_r = ((ridge_preds - y_tgt) ** 2).mean(-1)
                    total_ratio += (mse_t / mse_r.clamp(min=1e-10)).mean().item()
                cell_ratio = total_ratio / batches_per_cell
                cell_results[(profile, tau, gamma)] = cell_ratio
                profile_total += cell_ratio
                n_cells += 1
        profile_means[profile] = profile_total / max(1, n_cells)

    model.train()
    return profile_means, cell_results


# ── Training ─────────────────────────────────────────────────────────────

def train(cfg):
    device = get_device(cfg)
    sample_device = _resolve_sample_device(cfg, device)
    if int(getattr(cfg, "cpu_threads", 0)) > 0 and (device.type == "cpu" or sample_device.type == "cpu"):
        torch.set_num_threads(int(cfg.cpu_threads))
    set_seed(cfg.seed)
    print(f"Device: {device}")
    print(f"Sample device: {sample_device}")
    print(
        f"Config: family={cfg.spectrum_family} d_x={cfg.d_x} d_model={cfg.d_model} "
        f"L={cfg.n_layers} heads={cfg.n_heads} mask_tgt_tgt={cfg.mask_tgt_tgt}"
    )
    if (
        cfg.scale_canonical
        or str(cfg.scale_controller).lower() != "none"
        or str(getattr(cfg, "scale_conditioner", "none")).lower() != "none"
    ):
        print(
            "Scale mechanism: "
            f"canonical={cfg.scale_canonical} stat={cfg.scale_stat} scale_y={cfg.scale_y} "
            f"controller={cfg.scale_controller} gate_bound={cfg.scale_gate_bound} "
            f"log_clip={cfg.scale_log_clip} "
            f"conditioner={cfg.scale_conditioner} condition_dim={cfg.scale_condition_dim}"
        )
    if cfg.spectrum_family == "legacy":
        print(
            f"Training split: n_tgt ~ U{{{cfg.min_tgt}, ..., {cfg.max_tgt}}}, "
            f"n_ctx = {cfg.n_points} - n_tgt"
        )
        print(f"Eval split: n_ctx={cfg.n_ctx}, n_tgt={cfg.n_tgt}")
        print("Spectral profiles: polynomial, exponential, step (uniform random)")
    elif cfg.spectrum_family == "minimal":
        print(f"Minimal sampling scheme: {cfg.minimal_sampling_scheme}")
        if str(cfg.minimal_multiscale_mode).lower() != "off":
            print(
                f"Training multiscale mode: {cfg.minimal_multiscale_mode} "
                f"K={cfg.minimal_multiscale_k}"
            )
        print(f"Training targets: tau in {_parse_float_list(cfg.tau_values)}")
        print(f"Tau window: [{cfg.minimal_tau_min}, {cfg.minimal_tau_max}]")
        print(f"Scale window: [{cfg.log10_lambda1_min}, {cfg.log10_lambda1_max}]")
        print(
            f"Scale distribution: {cfg.minimal_scale_distribution} "
            f"(power={cfg.minimal_scale_distribution_power})"
        )
        print(f"Training context multipliers: gamma in {_parse_int_list(cfg.gamma_values)}")
        print(f"Targets per episode: n_tgt={cfg.fixed_n_tgt}")
        print(
            "Minimal profiles: "
            f"smooth span in [{cfg.minimal_smooth_span_min}, {cfg.minimal_smooth_span_max}], "
            f"step ranks in {_parse_int_list(cfg.minimal_step_rank_values)}, "
            f"step rank distribution={cfg.minimal_step_rank_distribution}, "
            f"step depth in [{cfg.minimal_step_depth_min}, {cfg.minimal_step_depth_max}]"
        )
    else:
        raise ValueError(f"unknown spectrum_family: {cfg.spectrum_family}")

    model = ICLTransformer(
        d_x=cfg.d_x,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        ffn_mult=cfg.ffn_mult,
        mask_tgt_tgt=cfg.mask_tgt_tgt,
        finite_attn_mask=cfg.finite_attn_mask,
        attn_logit_clip=cfg.attn_logit_clip,
        scale_canonical=cfg.scale_canonical,
        scale_stat=cfg.scale_stat,
        scale_eps=cfg.scale_eps,
        scale_y=cfg.scale_y,
        scale_controller=cfg.scale_controller,
        scale_gate_hidden=cfg.scale_gate_hidden,
        scale_gate_bound=cfg.scale_gate_bound,
        scale_log_clip=cfg.scale_log_clip,
        scale_conditioner=cfg.scale_conditioner,
        scale_condition_dim=cfg.scale_condition_dim,
        scale_condition_hidden=cfg.scale_condition_hidden,
    ).to(device)

    if cfg.init_checkpoint:
        state = torch.load(cfg.init_checkpoint, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
        print(f"Initialized model weights from {cfg.init_checkpoint}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=max(1, int(cfg.train_steps)),
        eta_min=float(cfg.min_lr),
    )
    save_path = cfg.save_path or f"model_L{cfg.n_layers}.pt"
    best_path = _derive_checkpoint_path(save_path, ".best")
    sentinel_path = _derive_checkpoint_path(save_path, ".sentinel")
    best_eval_ratio = None

    if cfg.save_best:
        print(f"Best-checkpoint path: {best_path}")
    if int(cfg.save_every) > 0:
        print(f"Periodic snapshots every {cfg.save_every} steps")
    if int(cfg.grad_accum_steps) > 1:
        print(f"Gradient accumulation: {cfg.grad_accum_steps} microbatches per optimizer step")
    if float(cfg.min_lr) > 0.0:
        print(f"LR floor: {cfg.min_lr}")
    if int(cfg.warmup_steps) > 0:
        print(f"LR warmup: {cfg.warmup_steps} steps")
        _set_lr(opt, 0.0)
    validation_bank = None
    if cfg.fixed_validation_bank:
        validation_bank = _load_or_create_validation_bank(cfg)
        print(f"Fixed validation bank: {len(validation_bank)} batches")
    if str(cfg.sentinel_mode).lower() != "off":
        print(
            f"Sentinel: mode={cfg.sentinel_mode} "
            f"train_loss_max={cfg.sentinel_train_loss_max} "
            f"eval_ratio_max={cfg.sentinel_eval_ratio_max}"
        )
    prefetch_proc, prefetch_queue, prefetch_stop = _start_prefetcher(cfg, sample_device)
    if prefetch_proc is not None:
        print(
            f"Prefetch: {cfg.prefetch_batches} queued batches, "
            f"worker_threads={cfg.prefetch_worker_threads}"
        )

    opt.zero_grad()
    train_start_time = time.monotonic()
    try:
        for step in range(1, cfg.train_steps + 1):
            accum_loss = 0.0
            for _ in range(max(1, int(cfg.grad_accum_steps))):
                batch = _next_train_batch(cfg, sample_device, prefetch_queue)
                train_batches = _materialize_train_batch(batch, device)
                batch_weight = 1.0 / float(max(1, len(train_batches)))
                for x_ctx, y_ctx, x_tgt, y_tgt, _ in train_batches:
                    preds = model(x_ctx, y_ctx, x_tgt)
                    loss = F.mse_loss(preds, y_tgt)

                    sentinel_reasons = []
                    if not torch.isfinite(loss):
                        sentinel_reasons.append("non-finite train loss")
                    if cfg.sentinel_train_loss_max > 0.0 and float(loss.item()) > cfg.sentinel_train_loss_max:
                        sentinel_reasons.append(
                            f"train loss {loss.item():.4e} exceeded sentinel threshold {cfg.sentinel_train_loss_max:.4e}"
                        )
                    if sentinel_reasons:
                        if str(cfg.sentinel_mode).lower() != "off":
                            _save_state_dict(model, sentinel_path)
                            print(f"  sentinel at step {step}: {'; '.join(sentinel_reasons)}")
                            print(f"  saved sentinel snapshot to {sentinel_path}")
                            if str(cfg.sentinel_mode).lower() == "stop":
                                return model
                        if not torch.isfinite(loss):
                            raise RuntimeError(f"non-finite train loss at step {step}")

                    accum_loss += float(loss.item()) * batch_weight
                    (loss * batch_weight / max(1, int(cfg.grad_accum_steps))).backward()

            if int(cfg.warmup_steps) > 0 and step <= int(cfg.warmup_steps):
                current_lr = float(cfg.lr) * (float(step) / float(max(1, int(cfg.warmup_steps))))
                _set_lr(opt, current_lr)
            else:
                current_lr = float(sched.get_last_lr()[0])
                _set_lr(opt, current_lr)

            pre_clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
            if cfg.abort_nonfinite_grad and not torch.isfinite(pre_clip_norm):
                if str(cfg.sentinel_mode).lower() != "off":
                    _save_state_dict(model, sentinel_path)
                    print(f"  sentinel at step {step}: non-finite gradient norm")
                    print(f"  saved sentinel snapshot to {sentinel_path}")
                raise RuntimeError(f"non-finite gradient norm at step {step}")
            if cfg.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            opt.zero_grad()

            if step % cfg.eval_every == 0 or step == 1:
                elapsed_sec = time.monotonic() - train_start_time
                sec_per_step = elapsed_sec / max(1, step)
                eval_mse, eval_ratio = evaluate(
                    model, cfg, device, validation_bank=validation_bank, sample_device=sample_device
                )
                print(
                    f"  step {step:5d}/{cfg.train_steps}  train_loss={accum_loss / max(1, int(cfg.grad_accum_steps)):.4e}  "
                    f"eval_mse={eval_mse:.4e}  MSE/ridge={eval_ratio:.3f}  "
                    f"lr={current_lr:.2e}  elapsed={elapsed_sec/60:.1f}m  sec/step={sec_per_step:.3f}"
                )
                if cfg.save_best and torch.isfinite(torch.tensor(eval_ratio)):
                    if best_eval_ratio is None or eval_ratio < best_eval_ratio:
                        best_eval_ratio = float(eval_ratio)
                        _save_state_dict(model, best_path)
                        print(f"    new best checkpoint: MSE/ridge={best_eval_ratio:.3f} -> {best_path}")

                eval_sentinel_reasons = []
                if not torch.isfinite(torch.tensor(eval_ratio)):
                    eval_sentinel_reasons.append("non-finite eval ratio")
                if cfg.sentinel_eval_ratio_max > 0.0 and eval_ratio > cfg.sentinel_eval_ratio_max:
                    eval_sentinel_reasons.append(
                        f"eval ratio {eval_ratio:.4e} exceeded sentinel threshold {cfg.sentinel_eval_ratio_max:.4e}"
                    )
                if eval_sentinel_reasons and str(cfg.sentinel_mode).lower() != "off":
                    _save_state_dict(model, sentinel_path)
                    print(f"    sentinel at step {step}: {'; '.join(eval_sentinel_reasons)}")
                    print(f"    saved sentinel snapshot to {sentinel_path}")
                    if str(cfg.sentinel_mode).lower() == "stop":
                        break

            if int(cfg.save_every) > 0 and (step % int(cfg.save_every) == 0):
                snapshot_path = _derive_checkpoint_path(save_path, f".step{step}")
                _save_state_dict(model, snapshot_path)
                print(f"    saved snapshot: {snapshot_path}")
    finally:
        if prefetch_stop is not None:
            prefetch_stop.set()
        if prefetch_proc is not None:
            prefetch_proc.join(timeout=2.0)
            if prefetch_proc.is_alive():
                prefetch_proc.terminate()

    if cfg.final_eval:
        print(f"\n{'─'*50}")
        if cfg.spectrum_family == "legacy":
            print("Per-profile MSE / MSE_ridge (1.0 = matches Bayes-optimal):")
            profile_results = evaluate_legacy_profiles(model, cfg, device, sample_device=sample_device)
            for name, ratio in profile_results.items():
                print(f"  {name:15s}  {ratio:.3f}")
            avg_ratio = sum(profile_results.values()) / len(profile_results)
            print(f"  {'average':15s}  {avg_ratio:.3f}")
        else:
            print("Sampler-matched minimal profile MSE / MSE_ridge (in-distribution under current training measure):")
            matched_results = evaluate_minimal_profiles_matched(model, cfg, device, sample_device=sample_device)
            for name, stats in matched_results.items():
                print(
                    f"  {name:15s}  ratio={stats['ratio']:.3f}  "
                    f"tau[p05,p50,p95]=[{stats['tau_p05']:.2f},{stats['tau_p50']:.2f},{stats['tau_p95']:.2f}]  "
                    f"log10(lambda1)[p05,p50,p95]=[{stats['a_p05']:.2f},{stats['a_p50']:.2f},{stats['a_p95']:.2f}]  "
                    f"n_ctx_mean={stats['n_ctx_mean']:.1f}"
                )

            print("\nTau-exact reference grid MSE / MSE_ridge (transfer check; not necessarily scale-matched):")
            original_scheme = cfg.minimal_sampling_scheme
            cfg.minimal_sampling_scheme = "tau_exact"
            profile_means, cell_results = evaluate_minimal_reference_grid(
                model, cfg, device, sample_device=sample_device
            )
            cfg.minimal_sampling_scheme = original_scheme
            for name, ratio in profile_means.items():
                print(f"  {name:15s}  mean={ratio:.3f}")
            for profile in MINIMAL_PROFILE_NAMES:
                print(f"  {profile}:")
                for tau in _parse_float_list(cfg.tau_values):
                    row = []
                    for gamma in _parse_int_list(cfg.gamma_values):
                        ratio = cell_results[(profile, tau, gamma)]
                        row.append(f"tau={tau:g},g={gamma}:{ratio:.3f}")
                    print(f"    {' | '.join(row)}")
    else:
        print("\nSkipped final profile/reference-grid evaluation (final_eval=false).")

    _save_state_dict(model, save_path)
    print(f"\nSaved {save_path}")
    if cfg.save_best and best_eval_ratio is not None:
        print(f"Best checkpoint: {best_path}  (MSE/ridge={best_eval_ratio:.3f})")

    return model


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train ICL linear regression transformer")
    for field_name, field in Config.__dataclass_fields__.items():
        arg_type = _parse_bool if isinstance(field.default, bool) else type(field.default)
        option_strings = [f"--{field_name}"]
        hyphen_name = field_name.replace("_", "-")
        if hyphen_name != field_name:
            option_strings.append(f"--{hyphen_name}")
        parser.add_argument(*option_strings, dest=field_name, type=arg_type, default=field.default)
    args = parser.parse_args()
    cfg = Config(**vars(args))
    train(cfg)


if __name__ == "__main__":
    main()
