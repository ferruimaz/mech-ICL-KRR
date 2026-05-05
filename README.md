# Mechanistic ICL as Kernel Ridge Regression

Code for the paper **"Operator-Level Evidence for In-Context Kernel Ridge Regression"**.

The final repository is organized around three paper experiments: a final-state operator
certificate, a prediction-risk rank and layerwise availability diagnostic, and a causal direction
removal test.

## Repository Structure

```text
src/
  model.py                    ICL transformer
  data.py                     Linear episode generation
  train.py                    Training CLI
  experiment_utils/           Shared final experiment utilities

experiments/
  experiment_1_operator_certificate/
  experiment_2_rank_emergence/
  experiment_3_causal_surgery/
  run_final_suite.py

checkpoints/final/
  manifest.json               Final checkpoint inventory and SHA256 hashes

paper/
  operator_exposition_structured_rewrite.tex
  operator_exposition_structured_rewrite.pdf
```

## Setup

```bash
conda env create -f environment.yml
conda activate mech-icl-krr
```

The regenerated final runs in this cleanup used `/opt/homebrew/bin/python3` on CPU.

## Final Experiments

Run the full final suite from the repo root:

```bash
python -m experiments.run_final_suite --mode full --clean --device cpu
```

For a quick wiring check:

```bash
python -m experiments.run_final_suite --mode smoke --clean --device cpu
```

The final suite writes results under the three experiment folders:

```text
experiments/experiment_1_operator_certificate/results/
experiments/experiment_2_rank_emergence/results/
experiments/experiment_3_causal_surgery/results/
experiments/experiment_3_causal_surgery/results_complement_ablation/
```

## Checkpoints

Use `checkpoints/final/manifest.json` as the source of truth for the final checkpoint set. It records
the original source filename, architecture, training regime, SHA256 hash, and experiment use for
each final checkpoint.
