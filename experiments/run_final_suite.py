#!/usr/bin/env python3
"""Run the final three-experiment suite from a clean results tree."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run(cmd: List[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def remove_result_dirs(exp_dirs: Iterable[Path]) -> None:
    for exp_dir in exp_dirs:
        if not exp_dir.exists():
            continue
        for child in exp_dir.iterdir():
            if child.is_dir() and child.name.startswith("results"):
                shutil.rmtree(child)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--clean", action="store_true", help="Remove existing final result directories first.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    return parser.parse_args()


def exp2_cmd(
    results_dir: Path,
    checkpoint: str,
    name: str,
    episodes: int,
    n_ctx: int,
    n_tgt: int,
    d_x: int,
    kernel_family: str,
    candidate_rank: int,
    curve_r_max: int,
    device: str,
    tau_sv: str = "0.001",
) -> List[str]:
    return [
        PYTHON,
        "-m",
        "experiments.experiment_2_rank_emergence.run",
        "--results-dir",
        str(results_dir),
        "--checkpoint",
        checkpoint,
        "--name",
        name,
        "--device",
        device,
        "--seed",
        "42",
        "--episodes",
        str(episodes),
        "--d-x",
        str(d_x),
        "--d-model",
        "128",
        "--n-layers",
        "8",
        "--n-heads",
        "4",
        "--n-ctx",
        str(n_ctx),
        "--n-tgt",
        str(n_tgt),
        "--sigma2",
        "0.1",
        "--kernel-family",
        kernel_family,
        "--kernel-lengthscale",
        "3.0",
        "--kernel-signal-var",
        "1.0",
        "--candidate-rank",
        str(candidate_rank),
        "--curve-r-max",
        str(curve_r_max),
        "--tau-sv",
        tau_sv,
        "--rank-tau",
        "0.01",
        "--excess-risk-frac",
        "0.05",
    ]


def main() -> None:
    args = parse_args()
    exp1_dir = ROOT / "experiments" / "experiment_1_operator_certificate"
    exp2_dir = ROOT / "experiments" / "experiment_2_rank_emergence"
    exp3_dir = ROOT / "experiments" / "experiment_3_causal_surgery"

    if args.clean:
        remove_result_dirs([exp1_dir, exp2_dir, exp3_dir])

    smoke = args.mode == "smoke"
    exp1_eps = 1 if smoke else 4
    small_eps = 1 if smoke else 16
    dx40_eps = 1 if smoke else 8
    rbf_old_eps = 1 if smoke else 16
    rbf_clean_eps = 1 if smoke else 8
    exp3_eps = 1 if smoke else 32
    complement_eps = 1 if smoke else 8
    result_prefix = "results_smoke" if smoke else "results"

    run(
        [
            PYTHON,
            "-m",
            "experiments.experiment_1_operator_certificate.run",
            "--model-key",
            "standard",
            "--device",
            args.device,
            "--seed",
            "123",
            "--episodes",
            str(exp1_eps),
            "--n-ctx",
            "47",
            "--n-tgt",
            "3",
            "--curve-r-max",
            "20",
            "--selection-alpha",
            "0.05",
            "--n-build",
            "16",
            "--n-eval",
            "32",
            "--results-dir",
            str(exp1_dir / result_prefix),
        ]
    )

    for d_x in (3, 5, 8, 10, 15):
        run(
            exp2_cmd(
                exp2_dir / result_prefix / "small_linear" / f"dx{d_x}",
                f"final/linear_sweep_dx{d_x}.pt",
                f"dx{d_x}",
                small_eps,
                47,
                16,
                d_x,
                "linear",
                47,
                47,
                args.device,
            )
        )

    run(
        [
            PYTHON,
            "-m",
            "experiments.experiment_2_rank_emergence.run_dx40",
            "--results-dir",
            str(exp2_dir / result_prefix / "dx40_target_rich"),
            "--device",
            args.device,
            "--seed",
            "122",
            "--episodes",
            str(dx40_eps),
            "--n-ctx",
            "47",
            "--n-tgt",
            "40",
        ]
    )

    for n_tgt in (16, 64, 128):
        run(
            exp2_cmd(
                exp2_dir / result_prefix / "rbf_fixed_l3_original" / f"ntgt{n_tgt}",
                "final/rbf_fixed_l3_original.pt",
                f"rbf_fixed_l3_ntgt{n_tgt}",
                rbf_old_eps,
                47,
                n_tgt,
                5,
                "rbf",
                47,
                47,
                args.device,
            )
        )

    run(
        exp2_cmd(
            exp2_dir / result_prefix / "rbf_128x128_seed42",
            "final/rbf_fixed_l3_nctx128_ntgt128_seed42.pt",
            "rbf_128x128_seed42",
            rbf_clean_eps,
            128,
            128,
            5,
            "rbf",
            128,
            64,
            args.device,
            tau_sv="0.0001",
        )
    )

    run(
        [
            PYTHON,
            "-m",
            "experiments.experiment_3_causal_surgery.run",
            "--checkpoint",
            "final/linear_baseline_dx5_L8.pt",
            "--d-x",
            "5",
            "--d-model",
            "128",
            "--n-layers",
            "8",
            "--n-heads",
            "4",
            "--device",
            args.device,
            "--seed",
            "42",
            "--episodes",
            str(exp3_eps),
            "--n-ctx",
            "47",
            "--n-tgt",
            "16",
            "--n-causal",
            "32",
            "--k-remove-list",
            "1,2,4",
            "--layer-rule",
            "sweep",
            "--probe-kind",
            "both",
            "--results-dir",
            str(exp3_dir / result_prefix),
        ]
    )
    run(
        [
            PYTHON,
            "-m",
            "experiments.experiment_3_causal_surgery.run_complement_ablation",
            "--checkpoint",
            "final/linear_baseline_dx5_L8.pt",
            "--episodes",
            str(complement_eps),
            "--n-ctx",
            "47",
            "--n-tgt",
            "16",
            "--state-idx",
            "1",
            "--results-dir",
            str(exp3_dir / f"{result_prefix}_complement_ablation"),
        ]
    )

    if not smoke:
        run([PYTHON, "-m", "experiments.experiment_3_causal_surgery.plot_figure"])
        run(
            [
                PYTHON,
                "-m",
                "experiments.experiment_3_causal_surgery.plot_figure",
                "--metric",
                "point_error",
                "--out",
                str(exp3_dir / "results" / "experiment_3_key_surgery_point_error.png"),
            ]
        )
        run(
            [
                PYTHON,
                "-m",
                "experiments.experiment_3_causal_surgery.plot_complement_ablation",
                "--summary-csv",
                str(exp3_dir / "results_complement_ablation" / "summary.csv"),
                "--out",
                str(exp3_dir / "results_complement_ablation" / "complement_ablation_bars.png"),
            ]
        )


if __name__ == "__main__":
    main()
