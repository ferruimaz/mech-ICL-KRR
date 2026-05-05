#!/usr/bin/env python3
"""Train/fine-tune an ICL transformer on a fixed RBF GP task at the rank elbow."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SRC_DIR = REPO_ROOT / "src"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_DIR))

from model import ICLTransformer  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_int_list(text: str) -> List[int]:
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return vals


def squared_distances(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    sq1 = (x1 * x1).sum(-1, keepdim=True)
    sq2 = (x2 * x2).sum(-1, keepdim=True)
    cross = x1 @ x2.transpose(-2, -1)
    return (sq1 + sq2.transpose(-2, -1) - 2.0 * cross).clamp_min(0.0)


def rbf_kernel(
    x1: torch.Tensor,
    x2: torch.Tensor,
    lengthscale: float,
    signal_var: float,
) -> torch.Tensor:
    return signal_var * torch.exp(-squared_distances(x1, x2) / (2.0 * lengthscale * lengthscale))


def sample_rbf_batch(
    batch_size: int,
    d_x: int,
    n_ctx: int,
    n_tgt: int,
    sigma2: float,
    lengthscale: float,
    signal_var: float,
    kernel_jitter: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    total = n_ctx + n_tgt
    x = torch.randn(batch_size, total, d_x, device=device)
    k_full = rbf_kernel(x, x, lengthscale, signal_var)
    eye = torch.eye(total, device=device, dtype=x.dtype).unsqueeze(0)
    chol = torch.linalg.cholesky(k_full + kernel_jitter * eye)
    f = (chol @ torch.randn(batch_size, total, 1, device=device)).squeeze(-1)
    y_ctx = f[:, :n_ctx] + math.sqrt(sigma2) * torch.randn(batch_size, n_ctx, device=device)
    y_tgt = f[:, n_ctx:]
    return x[:, :n_ctx], y_ctx, x[:, n_ctx:], y_tgt


@torch.no_grad()
def krr_predict(
    x_ctx: torch.Tensor,
    y_ctx: torch.Tensor,
    x_tgt: torch.Tensor,
    sigma2: float,
    lengthscale: float,
    signal_var: float,
) -> torch.Tensor:
    k = rbf_kernel(x_ctx, x_ctx, lengthscale, signal_var)
    kt = rbf_kernel(x_tgt, x_ctx, lengthscale, signal_var)
    eye = torch.eye(k.shape[-1], device=k.device, dtype=k.dtype).unsqueeze(0)
    alpha = torch.linalg.solve(k + sigma2 * eye, y_ctx.unsqueeze(-1))
    return (kt @ alpha).squeeze(-1)


@torch.no_grad()
def evaluate(
    model: ICLTransformer,
    args: argparse.Namespace,
    device: torch.device,
    n_tgt_values: Sequence[int],
) -> List[Dict[str, float]]:
    model.eval()
    rows: List[Dict[str, float]] = []
    for n_tgt in n_tgt_values:
        mse_total = 0.0
        krr_mse_total = 0.0
        ratio_total = 0.0
        pred_to_krr_total = 0.0
        for _ in range(args.eval_batches):
            x_ctx, y_ctx, x_tgt, y_tgt = sample_rbf_batch(
                args.eval_batch_size,
                args.d_x,
                args.n_ctx,
                n_tgt,
                args.sigma2,
                args.kernel_lengthscale,
                args.kernel_signal_var,
                args.kernel_jitter,
                device,
            )
            pred = model(x_ctx, y_ctx, x_tgt)
            krr = krr_predict(
                x_ctx,
                y_ctx,
                x_tgt,
                args.sigma2,
                args.kernel_lengthscale,
                args.kernel_signal_var,
            )
            mse = ((pred - y_tgt) ** 2).mean(-1)
            krr_mse = ((krr - y_tgt) ** 2).mean(-1)
            mse_total += float(mse.mean())
            krr_mse_total += float(krr_mse.mean())
            ratio_total += float((mse / krr_mse.clamp_min(1e-10)).mean())
            pred_to_krr_total += float(((pred - krr) ** 2).mean())
        denom = float(args.eval_batches)
        rows.append(
            {
                "n_tgt": float(n_tgt),
                "eval_mse": mse_total / denom,
                "krr_mse": krr_mse_total / denom,
                "mse_over_krr": ratio_total / denom,
                "pred_to_krr_mse": pred_to_krr_total / denom,
            }
        )
    model.train()
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
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


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def linear_lr(args: argparse.Namespace, step: int) -> float:
    if args.train_steps <= 1:
        return args.lr_final
    t = (step - 1) / (args.train_steps - 1)
    return args.lr + t * (args.lr_final - args.lr)


def train(args: argparse.Namespace) -> None:
    out_dir = Path(args.results_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    device = get_device(args.device)
    set_seed(args.seed)
    train_n_tgt_choices = args.train_n_tgt
    eval_n_tgt_values = args.eval_n_tgt

    model = ICLTransformer(
        d_x=args.d_x,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ffn_mult=args.ffn_mult,
    ).to(device)

    init_checkpoint = Path(args.init_checkpoint).expanduser() if args.init_checkpoint else None
    if init_checkpoint:
        init_checkpoint = init_checkpoint.resolve()
        state = torch.load(init_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state)

    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    config = vars(args).copy()
    config.update(
        {
            "resolved_results_dir": str(out_dir),
            "resolved_init_checkpoint": str(init_checkpoint) if init_checkpoint else "",
            "device": str(device),
            "n_params": n_params,
        }
    )
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

    log_rows: List[Dict[str, object]] = []
    start = time.time()
    print("=== RBF elbow training ===", flush=True)
    print(
        f"device={device} params={n_params:,} n_ctx={args.n_ctx} "
        f"train_n_tgt={train_n_tgt_choices} eval_n_tgt={eval_n_tgt_values}",
        flush=True,
    )
    if init_checkpoint:
        print(f"initialized from {init_checkpoint}", flush=True)

    model.train()
    for step in range(1, args.train_steps + 1):
        lr = linear_lr(args, step)
        set_optimizer_lr(opt, lr)
        n_tgt = int(train_n_tgt_choices[torch.randint(len(train_n_tgt_choices), (1,)).item()])
        x_ctx, y_ctx, x_tgt, y_tgt = sample_rbf_batch(
            args.batch_size,
            args.d_x,
            args.n_ctx,
            n_tgt,
            args.sigma2,
            args.kernel_lengthscale,
            args.kernel_signal_var,
            args.kernel_jitter,
            device,
        )
        pred = model(x_ctx, y_ctx, x_tgt)
        loss = F.mse_loss(pred, y_tgt)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        should_eval = step == 1 or step % args.eval_every == 0 or step == args.train_steps
        if should_eval:
            eval_rows = evaluate(model, args, device, eval_n_tgt_values)
            elapsed = time.time() - start
            for row in eval_rows:
                rec: Dict[str, object] = {
                    "step": step,
                    "elapsed_sec": elapsed,
                    "train_loss": float(loss.detach().cpu()),
                    "lr": lr,
                    **row,
                }
                log_rows.append(rec)
            write_csv(out_dir / "training_log.csv", log_rows)
            eval_text = "  ".join(
                f"ntgt={int(row['n_tgt'])}: mse={row['eval_mse']:.4e}, "
                f"krr={row['krr_mse']:.4e}, ratio={row['mse_over_krr']:.3f}"
                for row in eval_rows
            )
            print(
                f"step {step:6d}/{args.train_steps} loss={float(loss.detach()):.4e} "
                f"lr={lr:.2e} elapsed={elapsed/60:.1f}m  {eval_text}",
                flush=True,
            )
            if args.save_every > 0 and (step % args.save_every == 0 or step == args.train_steps):
                ckpt_step = out_dir / f"checkpoint_step{step}.pt"
                torch.save(model.state_dict(), ckpt_step)
                print(f"saved {ckpt_step}", flush=True)

    ckpt_path = out_dir / "checkpoint_final.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"saved {ckpt_path}", flush=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(SCRIPT_DIR / "results" / "nctx47_ntgt64_seed42"))
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-threads", type=int, default=0)

    parser.add_argument("--d-x", type=int, default=5)
    parser.add_argument("--sigma2", type=float, default=0.1)
    parser.add_argument("--n-ctx", type=int, default=47)
    parser.add_argument("--train-n-tgt", type=parse_int_list, default=parse_int_list("64"))
    parser.add_argument("--eval-n-tgt", type=parse_int_list, default=parse_int_list("64,128"))
    parser.add_argument("--kernel-lengthscale", type=float, default=3.0)
    parser.add_argument("--kernel-signal-var", type=float, default=1.0)
    parser.add_argument("--kernel-jitter", type=float, default=1e-5)

    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--ffn-mult", type=int, default=2)

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-final", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--save-every", type=int, default=0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
