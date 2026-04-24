#!/usr/bin/env python3
"""
Training loop, evaluation, and CLI for in-context linear regression transformer.

Usage:
    python train.py
    python train.py --d_x 10 --n_layers 8 --train_steps 5000
"""

import argparse
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from model import ICLTransformer
from data import sample_batch, PROFILE_NAMES


# ── Configuration ────────────────────────────────────────────────────────

@dataclass
class Config:
    # Problem
    d_x: int = 5               # input dimension
    sigma2: float = 0.1        # noise variance

    # Episode structure
    n_points: int = 50          # total points per episode (context + target)
    min_tgt: int = 1            # min target points per batch
    max_tgt: int = 10           # max target points per batch
    n_ctx: int = 47             # fixed split for evaluation only
    n_tgt: int = 3              # fixed split for evaluation only

    # Model
    d_model: int = 128
    n_layers: int = 8
    n_heads: int = 4
    ffn_mult: int = 2

    # Training
    batch_size: int = 64
    train_steps: int = 10000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    seed: int = 42

    # Evaluation
    eval_every: int = 500
    eval_batches: int = 10

    # Output
    save_path: str = ""


# ── Helpers ──────────────────────────────────────────────────────────────

def get_device():
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


def _sample(cfg, device):
    """Sample one batch using current config."""
    return sample_batch(
        cfg.batch_size, cfg.d_x, cfg.n_ctx, cfg.n_tgt, cfg.sigma2, device,
    )


def _ridge_predictions(x_ctx, y_ctx, x_tgt, sigma2):
    """Bayes-optimal ridge predictions (λ = σ²)."""
    d = x_ctx.shape[-1]
    XtX = x_ctx.transpose(-2, -1) @ x_ctx  # (B, d, d)
    Xty = x_ctx.transpose(-2, -1) @ y_ctx.unsqueeze(-1)  # (B, d, 1)
    I = torch.eye(d, device=x_ctx.device).unsqueeze(0)
    beta_hat = torch.linalg.solve(XtX + sigma2 * I, Xty)  # (B, d, 1)
    return (x_tgt @ beta_hat).squeeze(-1)  # (B, n_tgt)


# ── Evaluation ───────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, cfg, device):
    """Compute average MSE and MSE/MSE_ridge ratio over eval_batches."""
    model.eval()
    total_mse = 0.0
    total_ratio = 0.0
    for _ in range(cfg.eval_batches):
        x_ctx, y_ctx, x_tgt, y_tgt, _ = _sample(cfg, device)
        preds = model(x_ctx, y_ctx, x_tgt)
        ridge_preds = _ridge_predictions(x_ctx, y_ctx, x_tgt, cfg.sigma2)

        mse_transformer = ((preds - y_tgt) ** 2).mean(-1)  # (B,)
        mse_ridge = ((ridge_preds - y_tgt) ** 2).mean(-1)  # (B,)
        total_mse += mse_transformer.mean().item()
        total_ratio += (mse_transformer / mse_ridge.clamp(min=1e-10)).mean().item()

    model.train()
    n = cfg.eval_batches
    return total_mse / n, total_ratio / n


@torch.no_grad()
def evaluate_per_profile(model, cfg, device, batches_per_profile=20):
    """Compute MSE/MSE_ridge ratio per spectral profile."""
    from data import sample_batch_profile
    model.eval()
    results = {}
    for name in PROFILE_NAMES:
        total_ratio = 0.0
        for _ in range(batches_per_profile):
            x_ctx, y_ctx, x_tgt, y_tgt, _ = sample_batch_profile(
                cfg.batch_size, cfg.d_x, cfg.n_ctx, cfg.n_tgt, cfg.sigma2,
                name, device,
            )
            preds = model(x_ctx, y_ctx, x_tgt)
            ridge_preds = _ridge_predictions(x_ctx, y_ctx, x_tgt, cfg.sigma2)

            mse_t = ((preds - y_tgt) ** 2).mean(-1)
            mse_r = ((ridge_preds - y_tgt) ** 2).mean(-1)
            total_ratio += (mse_t / mse_r.clamp(min=1e-10)).mean().item()

        results[name] = total_ratio / batches_per_profile
    model.train()
    return results


# ── Training ─────────────────────────────────────────────────────────────

def train(cfg):
    device = get_device()
    set_seed(cfg.seed)
    print(f"Device: {device}")
    print(f"Config: d_x={cfg.d_x} d_model={cfg.d_model} L={cfg.n_layers} "
          f"heads={cfg.n_heads} n_points={cfg.n_points}")
    print(f"Training split: n_tgt ~ U{{{cfg.min_tgt}, ..., {cfg.max_tgt}}}, "
          f"n_ctx = {cfg.n_points} - n_tgt")
    print(f"Eval split: n_ctx={cfg.n_ctx}, n_tgt={cfg.n_tgt}")
    print(f"Spectral profiles: polynomial, exponential, step (uniform random)")

    model = ICLTransformer(
        d_x=cfg.d_x,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        ffn_mult=cfg.ffn_mult,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.train_steps)

    for step in range(1, cfg.train_steps + 1):
        # Random ctx/tgt split each batch (total points stays fixed)
        n_tgt = torch.randint(cfg.min_tgt, cfg.max_tgt + 1, (1,)).item()
        n_ctx = cfg.n_points - n_tgt

        x_ctx, y_ctx, x_tgt, y_tgt, _ = sample_batch(
            cfg.batch_size, cfg.d_x, n_ctx, n_tgt, cfg.sigma2, device,
        )

        preds = model(x_ctx, y_ctx, x_tgt)
        loss = F.mse_loss(preds, y_tgt)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()

        if step % cfg.eval_every == 0 or step == 1:
            eval_mse, eval_ratio = evaluate(model, cfg, device)
            print(f"  step {step:5d}/{cfg.train_steps}  "
                  f"train_loss={loss.item():.4e}  eval_mse={eval_mse:.4e}  "
                  f"MSE/ridge={eval_ratio:.3f}  lr={sched.get_last_lr()[0]:.2e}")

    # Final per-profile evaluation
    print(f"\n{'─'*50}")
    print("Per-profile MSE / MSE_ridge (1.0 = matches Bayes-optimal):")
    profile_results = evaluate_per_profile(model, cfg, device)
    for name, ratio in profile_results.items():
        print(f"  {name:15s}  {ratio:.3f}")
    avg_ratio = sum(profile_results.values()) / len(profile_results)
    print(f"  {'average':15s}  {avg_ratio:.3f}")

    save_path = cfg.save_path or f"model_L{cfg.n_layers}.pt"
    torch.save(model.state_dict(), save_path)
    print(f"\nSaved {save_path}")

    return model


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train ICL linear regression transformer")
    for field_name, field in Config.__dataclass_fields__.items():
        parser.add_argument(f"--{field_name}", type=type(field.default), default=field.default)
    args = parser.parse_args()
    cfg = Config(**vars(args))
    train(cfg)


if __name__ == "__main__":
    main()
