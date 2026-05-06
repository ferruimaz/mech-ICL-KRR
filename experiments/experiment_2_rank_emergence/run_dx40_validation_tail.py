#!/usr/bin/env python3
"""Run Experiment 2 on dx40 validation-tail episodes selected by Experiment 4."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments.experiment_2_rank_emergence import run as layerwise  # noqa: E402
from experiments.experiment_4_q_repair.run import (  # noqa: E402
    parse_gamma_values,
    sample_dx40_batch_with_meta,
)
from experiment_utils.dx40 import (  # noqa: E402
    DX40_CHECKPOINT,
    Dx40SamplerCfg,
    dx40_forward_with_ctx_hidden,
    dx40_hidden_response_matrices,
    dx40_prediction_fd_bundle,
    load_dx40_model,
)
from experiment_utils.layerwise_core import CkptCfg  # noqa: E402
from experiment_utils.support import set_seed  # noqa: E402


DEFAULT_SELECTION_RECORDS = (
    REPO_ROOT
    / "experiments"
    / "experiment_4_q_repair"
    / "results_train_validation_like"
    / "records.csv"
)


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


def parse_index_list(text: str) -> List[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def selected_indices_from_records(path: Path) -> List[int]:
    if not path.exists():
        raise FileNotFoundError(
            f"selection records not found: {path}. Pass --selected-bank-indices or rerun Experiment 4."
        )
    rows: List[Dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("name") == "base":
                rows.append(row)
    if not rows:
        raise ValueError(f"no base rows found in {path}")
    rows.sort(key=lambda row: int(row["selection_rank"]))
    return [int(row["bank_index"]) for row in rows]


def reconstruct_validation_tail(
    args: argparse.Namespace,
    selected_indices: Sequence[int],
    device: torch.device,
) -> List[Dict[str, object]]:
    if not selected_indices:
        raise ValueError("no selected bank indices were provided")

    set_seed(args.bank_seed)
    sampler_cfg = Dx40SamplerCfg(
        tau_min=args.tau_min,
        tau_max=args.tau_max,
        log10_lambda1_min=args.log10_lambda1_min,
        log10_lambda1_max=args.log10_lambda1_max,
    )
    sample_args = SimpleNamespace(
        sampling_mode="train_distribution",
        gamma_values=args.gamma_values,
        train_n_tgt=args.train_n_tgt,
        n_ctx=args.n_ctx,
        n_tgt=args.n_tgt,
        sigma2=args.sigma2,
    )

    wanted = set(int(idx) for idx in selected_indices)
    max_index = max(wanted)
    bank_batch_size = max(1, int(args.bank_batch_size))
    found: Dict[int, Dict[str, object]] = {}
    bank_index = 0
    bank_batch_index = 0

    while bank_index <= max_index:
        current_batch_size = min(bank_batch_size, max_index + 1 - bank_index)
        x_ctx_b, y_ctx_b, x_tgt_b, y_tgt_b, meta = sample_dx40_batch_with_meta(
            sample_args,
            sampler_cfg,
            device,
            current_batch_size,
        )
        for member in range(current_batch_size):
            idx = bank_index + member
            if idx in wanted:
                found[idx] = {
                    "bank_index": idx,
                    "bank_batch_index": bank_batch_index,
                    "bank_batch_member": member,
                    "bank_batch_size": current_batch_size,
                    "x_ctx": x_ctx_b[member : member + 1].detach().cpu(),
                    "y_ctx": y_ctx_b[member : member + 1].detach().cpu(),
                    "x_tgt": x_tgt_b[member : member + 1].detach().cpu(),
                    "y_tgt": y_tgt_b[member : member + 1].detach().cpu(),
                    **meta,
                }
        bank_index += current_batch_size
        bank_batch_index += 1

    missing = [idx for idx in selected_indices if idx not in found]
    if missing:
        raise RuntimeError(f"failed to reconstruct selected bank indices: {missing}")
    return [found[int(idx)] for idx in selected_indices]


def metadata_rows(selected: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for episode_idx, row in enumerate(selected):
        rows.append(
            {
                "episode": episode_idx,
                "bank_index": row["bank_index"],
                "bank_batch_index": row["bank_batch_index"],
                "bank_batch_member": row["bank_batch_member"],
                "bank_batch_size": row["bank_batch_size"],
                "sampling_mode": row["sampling_mode"],
                "profile": row["profile"],
                "tau_actual": row["tau_actual"],
                "log10_lambda1": row["log10_lambda1"],
                "lambda1": row["lambda1"],
                "gamma": row["gamma"],
                "n_ctx": row["n_ctx_actual"],
                "n_tgt": row["n_tgt_actual"],
            }
        )
    return rows


def parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(description=__doc__, add_help=True)
    parser.add_argument(
        "--results-dir",
        default=str(SCRIPT_DIR / "results" / "dx40_validation_tail"),
    )
    parser.add_argument("--checkpoint", default=str(DX40_CHECKPOINT))
    parser.add_argument("--name", default="dx40_tau16_s122_validation_tail")
    parser.add_argument("--device", choices=["cpu"], default="cpu")
    parser.add_argument("--seed", type=int, default=122)
    parser.add_argument("--bank-seed", type=int, default=122)
    parser.add_argument("--bank-batch-size", type=int, default=16)
    parser.add_argument(
        "--selection-mode",
        choices=["records", "all_bank"],
        default="records",
        help="Use Experiment 4 selected records, or replay every episode in the native validation bank.",
    )
    parser.add_argument("--bank-size", type=int, default=480)
    parser.add_argument("--selection-records", default=str(DEFAULT_SELECTION_RECORDS))
    parser.add_argument(
        "--selected-bank-indices",
        default="",
        help="Comma-separated bank indices. If omitted, read base rows from --selection-records.",
    )
    parser.add_argument("--gamma-values", default="4,8,12")
    parser.add_argument("--train-n-tgt", type=int, default=8)
    parser.add_argument("--n-ctx", type=int, default=47)
    parser.add_argument("--n-tgt", type=int, default=40)
    parser.add_argument("--sigma2", type=float, default=0.1)
    parser.add_argument("--tau-min", type=float, default=1e-3)
    parser.add_argument("--tau-max", type=float, default=16.0)
    parser.add_argument("--log10-lambda1-min", type=float, default=-2.0)
    parser.add_argument("--log10-lambda1-max", type=float, default=2.0)
    known, rest = parser.parse_known_args(argv)
    parse_gamma_values(known.gamma_values)
    return known, rest


def main(argv: Sequence[str] | None = None) -> None:
    known, rest = parse_args(argv)
    device = torch.device(known.device)
    if known.selected_bank_indices.strip():
        selected_indices = parse_index_list(known.selected_bank_indices)
    elif known.selection_mode == "all_bank":
        selected_indices = list(range(int(known.bank_size)))
    else:
        selected_indices = selected_indices_from_records(Path(known.selection_records))
    selected = reconstruct_validation_tail(known, selected_indices, device)

    out_dir = Path(known.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "selected_episode_metadata.csv", metadata_rows(selected))
    (out_dir / "selection_config.json").write_text(
        json.dumps(
            {
                "selected_bank_indices": selected_indices,
                "wrapper_args": vars(known),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    def patched_load_model(cfg: CkptCfg, device: torch.device):
        return load_dx40_model(cfg.checkpoint, device)

    cursor = {"episode": 0}

    def patched_sample_eval_episode(cfg: CkptCfg, args: argparse.Namespace, device: torch.device):
        index = cursor["episode"]
        if index >= len(selected):
            raise RuntimeError("Experiment 2 requested more episodes than were reconstructed")
        cursor["episode"] = index + 1
        episode = selected[index]
        return (
            episode["x_ctx"].to(device),
            episode["y_ctx"].to(device),
            episode["x_tgt"].to(device),
            episode["y_tgt"].to(device),
        )

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
        str(len(selected)),
        "--d-x",
        "40",
        "--d-model",
        "128",
        "--n-layers",
        "8",
        "--n-heads",
        "4",
        "--n-ctx",
        "0",
        "--n-tgt",
        str(known.train_n_tgt),
        "--sigma2",
        str(known.sigma2),
        "--kernel-family",
        "linear",
        "--rank-tau",
        "0.01",
        "--excess-risk-frac",
        "0.05",
    ] + rest
    layerwise.main(forwarded)
    summary_path = out_dir / "summary.txt"
    if summary_path.exists():
        summary_text = summary_path.read_text(encoding="utf-8")
        summary_text = summary_text.replace(
            f"episodes={len(selected)}, n_ctx=0, n_tgt={known.train_n_tgt},",
            f"episodes={len(selected)}, n_ctx=variable, n_tgt={known.train_n_tgt},",
        )
        summary_path.write_text(summary_text, encoding="utf-8")


if __name__ == "__main__":
    main()
