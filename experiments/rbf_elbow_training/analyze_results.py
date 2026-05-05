#!/usr/bin/env python3
"""Summarize the RBF elbow retraining diagnostics."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(row: Dict[str, str], key: str, default: float = float("nan")) -> float:
    try:
        return float(row[key])
    except Exception:
        return default


def best_rank_rows(rows: Sequence[Dict[str, str]], basis: str, ranks: Iterable[int]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    by_rank = {int(round(f(r, "rank"))): r for r in rows if r.get("basis") == basis}
    for rank in ranks:
        if rank in by_rank:
            out.append(by_rank[rank])
    return out


def nearest_rank(rows: Sequence[Dict[str, str]], basis: str, target: float) -> int:
    ranks = [int(round(f(r, "rank"))) for r in rows if r.get("basis") == basis]
    if not ranks or math.isnan(target):
        return 0
    return min(ranks, key=lambda r: abs(r - target))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args(argv)
    root = Path(args.results_dir)

    lines: List[str] = ["RBF elbow retraining analysis", ""]

    train_rows = read_csv(root / "training_log.csv")
    if train_rows:
        lines.append("Training endpoint")
        final_step = max(int(float(r["step"])) for r in train_rows)
        for row in train_rows:
            if int(float(row["step"])) == final_step:
                lines.append(
                    f"  n_tgt={int(f(row, 'n_tgt'))}: MSE={f(row, 'eval_mse'):.5f}, "
                    f"KRR_MSE={f(row, 'krr_mse'):.5f}, MSE/KRR={f(row, 'mse_over_krr'):.3f}, "
                    f"pred_to_KRR={f(row, 'pred_to_krr_mse'):.5f}"
                )
        lines.append("")

    n_tgts = sorted(
        {
            int(path.name.replace("final_basis_ntgt", ""))
            for path in root.glob("final_basis_ntgt*")
            if path.is_dir() and path.name.replace("final_basis_ntgt", "").isdigit()
        }
        | {
            int(path.name.replace("causal_ntgt", ""))
            for path in root.glob("causal_ntgt*")
            if path.is_dir() and path.name.replace("causal_ntgt", "").isdigit()
        }
    )

    for n_tgt in n_tgts:
        fb = read_csv(root / f"final_basis_ntgt{n_tgt}" / "rank_summary.csv")
        if fb:
            r_t = f(fb[0], "r_eff_T_task_mean")
            r_t_strict = f(fb[0], "r_eff_T_task_strict_mean")
            r_near = nearest_rank(fb, "raw", r_t)
            lines.append(f"Final-state basis diagnostic, n_tgt={n_tgt}")
            lines.append(
                f"  mean prediction-risk rank r_T={r_t:.2f}; strict relative rank={r_t_strict:.2f}; "
                f"nearest reported rank={r_near}"
            )
            for basis in ("raw", "response"):
                wanted = sorted(set([16, r_near, 47]))
                for row in best_rank_rows(fb, basis, wanted):
                    lines.append(
                        f"  {basis:8s} rank={int(f(row, 'rank')):2d}: "
                        f"E(TQ,T)={f(row, 'E_operator_to_T_mean'):.5f}, "
                        f"E(F,TQ)={f(row, 'E_model_to_operator_mean'):.5f}, "
                        f"E(F,T)={f(row, 'E_model_to_T_mean'):.5f}"
                    )
            lines.append("")

        causal = read_csv(root / f"causal_ntgt{n_tgt}" / "summary.csv")
        if causal:
            row = causal[0]
            lines.append(f"Causal validated-rank diagnostic, n_tgt={n_tgt}")
            lines.append(
                f"  dim={f(row, 'dim_R_nat_mean'):.2f}, r_T={f(row, 'r_eff_T_task_mean'):.2f}, "
                f"strict={f(row, 'r_eff_T_task_strict_mean'):.2f}, dim/r_T={f(row, 'dim_over_rT_mean'):.3f}"
            )
            lines.append(
                f"  E(TQ,T)={f(row, 'E_TQ_T_mean'):.5f}, "
                f"E(F,TQ)={f(row, 'E_F_TQ_mean'):.5f}, "
                f"E(F,T)={f(row, 'E_F_T_mean'):.5f}, "
                f"MSE ratio={f(row, 'mse_ratio_mean'):.3f}"
            )
            lines.append("")

    (root / "analysis.txt").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(root / "analysis.txt")


if __name__ == "__main__":
    main()
