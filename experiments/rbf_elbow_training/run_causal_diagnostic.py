#!/usr/bin/env python3
"""Run the validated-rank causal diagnostic on a local RBF checkpoint."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SRC_DIR = REPO_ROOT / "src"
SUPPORT_DIR = REPO_ROOT / "experiments" / "shared"
MPL_CACHE_DIR = Path(os.environ.get("TMPDIR", "/tmp")) / "mech_icl_krr_mpl_cache"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SUPPORT_DIR))

from experiments.exp2_budget_closure.run import CkptCfg  # noqa: E402
from experiments.exp2_validated_rank import run as validated_rank  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__, add_help=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--checkpoint-name", default="rbf_elbow_nctx47_ntgt64")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--d-x", type=int, default=5)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=4)
    known, rest = parser.parse_known_args(argv)
    return known, rest


def main(argv: Sequence[str] | None = None) -> None:
    known, rest = parse_args(argv)
    checkpoint = str(Path(known.checkpoint_path).expanduser().resolve())
    validated_rank.CHECKPOINTS_RBF = [
        CkptCfg(
            known.checkpoint_name,
            checkpoint,
            known.d_x,
            known.d_model,
            known.n_layers,
            known.n_heads,
            "rbf_elbow",
            47,
        )
    ]
    forwarded = [
        "--exp",
        "rbf",
        "--checkpoints",
        known.checkpoint_name,
        "--results-dir",
        known.results_dir,
    ] + rest
    args = validated_rank.parse_args(forwarded)
    validated_rank.run(args)


if __name__ == "__main__":
    main()
