#!/usr/bin/env python3
"""Run final Experiment 2 on the d_x=40 scale-aware checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import torch

from experiments.experiment_2_rank_emergence import run as layerwise
from experiment_utils.dx40 import (
    DX40_CHECKPOINT,
    Dx40SamplerCfg,
    dx40_forward_with_ctx_hidden,
    dx40_hidden_response_matrices,
    dx40_prediction_fd_bundle,
    load_dx40_model,
    sample_dx40_episode,
)
from experiment_utils.layerwise_core import CkptCfg


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__, add_help=True)
    parser.add_argument("--results-dir", default=str(SCRIPT_DIR / "results" / "dx40_target_rich"))
    parser.add_argument("--checkpoint", default=str(DX40_CHECKPOINT))
    parser.add_argument("--name", default="dx40_tau16_s122_target_rich")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--seed", type=int, default=122)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--n-ctx", type=int, default=47)
    parser.add_argument("--n-tgt", type=int, default=40)
    parser.add_argument("--tau-min", type=float, default=1e-3)
    parser.add_argument("--tau-max", type=float, default=16.0)
    parser.add_argument("--log10-lambda1-min", type=float, default=-2.0)
    parser.add_argument(
        "--log10-lambda1-max",
        type=float,
        default=1.0,
        help="Default caps lambda_1 at 10 for the robust diagnostic regime.",
    )
    known, rest = parser.parse_known_args(argv)
    return known, rest


def main(argv: Sequence[str] | None = None) -> None:
    known, rest = parse_args(argv)
    sampler_cfg = Dx40SamplerCfg(
        tau_min=known.tau_min,
        tau_max=known.tau_max,
        log10_lambda1_min=known.log10_lambda1_min,
        log10_lambda1_max=known.log10_lambda1_max,
    )

    def patched_load_model(cfg: CkptCfg, device: torch.device):
        return load_dx40_model(cfg.checkpoint, device)

    def patched_sample_eval_episode(cfg: CkptCfg, args, device: torch.device):
        return sample_dx40_episode(cfg, args, device, sampler_cfg)

    layerwise.load_model = patched_load_model
    layerwise.sample_eval_episode = patched_sample_eval_episode
    layerwise.forward_with_ctx_hidden = dx40_forward_with_ctx_hidden
    layerwise.hidden_response_matrices = dx40_hidden_response_matrices
    layerwise.prediction_fd_bundle = dx40_prediction_fd_bundle

    forwarded = [
        "--results-dir",
        known.results_dir,
        "--checkpoint",
        known.checkpoint,
        "--name",
        known.name,
        "--device",
        known.device,
        "--seed",
        str(known.seed),
        "--episodes",
        str(known.episodes),
        "--d-x",
        "40",
        "--d-model",
        "128",
        "--n-layers",
        "8",
        "--n-heads",
        "4",
        "--n-ctx",
        str(known.n_ctx),
        "--n-tgt",
        str(known.n_tgt),
        "--sigma2",
        "0.1",
        "--kernel-family",
        "linear",
        "--candidate-rank",
        "40",
        "--curve-r-max",
        "40",
        "--rank-tau",
        "0.01",
        "--excess-risk-frac",
        "0.05",
    ] + rest
    layerwise.main(forwarded)


if __name__ == "__main__":
    main()

