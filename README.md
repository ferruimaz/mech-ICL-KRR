# Mechanistic ICL as Kernel Ridge Regression

Code for the paper **"Operator-Level Evidence for In-Context Kernel Ridge Regression"**.

We show that a transformer trained on in-context linear regression implements kernel ridge regression (KRR) in a mechanistically precise sense: its final-layer activations span a low-dimensional frozen subspace that induces a KRR-faithful linear operator, and the high-leverage directions of that operator are causally used by the model.

---

## Repository structure

```
src/                          Core implementation
  model.py                    ICL transformer (residual-only, no LayerNorm)
  data.py                     Episode generation with spectral curriculum
  train.py                    Training loop and CLI

experiments/
  shared/support.py           Shared utilities (model loading, kernels, diagnostics)
  exp1_operator_certificate/  Exp 1 – operator-Galerkin certificate
  exp2_budget_closure/        Exp 2 – native budget and rank threshold
  exp3_causal_surgery/        Exp 3 – causal leverage intervention

checkpoints/                  Trained model weights
environment.yml               Conda environment (Python 3.13, PyTorch ≥ 2.7)
```

---

## Setup

```bash
conda env create -f environment.yml
conda activate mech-icl-krr
```

---

## Training

```bash
python src/train.py --d_x 5 --n_layers 8 --train_steps 10000 --save_path checkpoints/model_L8.pt
```

Key options: `--d_x`, `--n_layers`, `--n_heads`, `--d_model`, `--train_steps`.

---

## Experiments

Run from the repo root. Each experiment writes results (CSV, PNG, TXT) to its own `results/` subdirectory.

**Experiment 1 – Operator-Galerkin certificate**
```bash
python -m experiments.exp1_operator_certificate.run
```

**Experiment 2 – Native budget and rank threshold**
```bash
python -m experiments.exp2_budget_closure.run --suite
```

**Experiment 3 – Galerkin-leverage causal surgery**
```bash
python -m experiments.exp3_causal_surgery.run \
  --checkpoint model_L8.pt \
  --d-x 5 --d-model 128 --n-layers 8 --n-heads 4 \
  --episodes 32 --n-ctx 47 --n-tgt 16 \
  --n-causal 32 --k-remove-list 1,2,4 --layer-rule sweep
```

---

## Checkpoints

| File | Config | Purpose |
|---|---|---|
| `model_L8.pt` | d_x=5, D=128, L=8, H=4 | Main model |
| `model_w512_L8_dx20_h16_n200.pt` | d_x=20, D=512, L=8, H=16 | Wide model |
| `model_gp_mixed.pt` | d_x=5, D=128, L=8, H=4 | GP-kernel trained |
| `model_L{2,4,6,12}.pt` | depth sweep | Exp 1 & 2 depth ablation |
| `model_H{1,2,8}.pt` | head sweep | Exp 2 head ablation |
| `model_dx{3,5,8,10,15}.pt` | dimension sweep | Exp 2 rank threshold |
