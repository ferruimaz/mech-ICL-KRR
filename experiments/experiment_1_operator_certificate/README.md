# Experiment 1: Operator-Galerkin Certificate

This folder implements Experiment 1 from `paper/krr_mechanism.tex`.
It tests whether final context-token activations define a frozen
activation-derived Galerkin operator whose prediction-risk excess is small
and whose action matches the model's local response to label perturbations.

Run from the repo root:

```bash
python -m experiments.experiment_1_operator_certificate.run
```

Main outputs are written to `results/`:

- `episode_metrics.csv`: per-episode pointwise and operator metrics for raw,
  response-only, activation-low-rank, and control operators.
- `summary_metrics.csv`: grouped means and standard deviations.
- `rank_curves.csv`: raw and response weighted-SVD rank curves for
  \(\rho_G(T_Q)\) and \(E_{\mathrm{task}}(F,T_Q)\).
- `additivity.csv`: finite-scale local additivity defects.
- `actlr_comparison.csv`: paired Galerkin-vs-activation-low-rank diagnostics.
- `probe_interpolation.csv`: sweep from task-label probes to isotropic probes.
- `summary_errors.png`, `rank_curves.png`, `additivity.png`,
  `actlr_comparison.png`, `probe_interpolation.png`: quick-look plots.
- `summary.txt`: compact textual summary of the run.
