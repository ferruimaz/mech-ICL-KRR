#!/usr/bin/env python3
"""Refresh Experiment 2 plots from existing layer_summary.csv files."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.experiment_2_rank_emergence.run import plot_emergence  # noqa: E402


def read_csv(path: Path) -> List[Dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "result_dirs",
        nargs="*",
        type=Path,
        help="Result directories containing layer_summary.csv. Defaults to all final Experiment 2 results.",
    )
    parser.add_argument("--threshold", type=float, default=0.05)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result_dirs = args.result_dirs
    if not result_dirs:
        result_dirs = sorted(path.parent for path in (SCRIPT_DIR / "results").glob("**/layer_summary.csv"))

    for result_dir in result_dirs:
        summary_path = result_dir / "layer_summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        rows = read_csv(summary_path)
        plot_emergence(rows, result_dir, threshold=args.threshold)
        print(f"refreshed {result_dir}", flush=True)


if __name__ == "__main__":
    main()
