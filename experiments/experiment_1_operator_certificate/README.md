# Experiment 1: Operator-Galerkin Certificate

This folder implements Experiment 1 from `native_operator_galerkin_paper.tex`.
It tests whether final context-token activations define a frozen
activation-derived Galerkin operator that explains both exact KRR and the
model's local response to label perturbations.

Run from the repo root:

```bash
conda activate 2t-beta-np
python -m operator_galerkin_experiments.experiment_1_operator_galerkin_certificate.run
```

Main outputs are written to `results/`:

- `episode_metrics.csv`: per-episode pointwise and operator metrics for raw,
  response-only, activation-low-rank, and control operators.
- `summary_metrics.csv`: grouped means and standard deviations.
- `rank_curves.csv`: raw and response weighted-SVD rank curves.
- `additivity.csv`: finite-scale local additivity defects.
- `actlr_comparison.csv`: paired Galerkin-vs-activation-low-rank diagnostics.
- `probe_interpolation.csv`: sweep from task-label probes to isotropic probes.
- `summary_errors.png`, `rank_curves.png`, `additivity.png`,
  `actlr_comparison.png`, `probe_interpolation.png`: quick-look plots.
- `summary.txt`: compact textual summary of the run.
