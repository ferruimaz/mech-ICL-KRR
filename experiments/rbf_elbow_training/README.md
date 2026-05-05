# RBF Elbow Training Experiment

This folder keeps the target-rich RBF retraining experiment self-contained.

The goal is to train the same `d_x=5`, `d_model=128`, `L=8`, `H=4`
transformer on the fixed RBF task at the sample-size elbow:

- kernel: RBF, lengthscale `ell=3`, signal variance `1`
- noise: `sigma2=0.1`
- context points: `n_ctx=47`
- training target points: default `n_tgt=64`
- diagnostics: `n_tgt=64` and `n_tgt=128`, rank cap `47`

The scripts intentionally reuse the existing model and experiment code. They do
not modify the checkpoint registries in the original experiment files.

Typical run:

```bash
conda run -n 2t-beta-np python experiments/rbf_elbow_training/train_rbf_elbow.py \
  --results-dir experiments/rbf_elbow_training/results/nctx47_ntgt64_seed42 \
  --init-checkpoint checkpoints/model_rbf_fixed_l3.pt \
  --train-steps 5000 --batch-size 16 --lr 1e-4 --lr-final 2e-5
```

Then run the diagnostics on the saved checkpoint:

```bash
conda run -n 2t-beta-np python experiments/exp2_validated_rank/rbf_final_basis.py \
  --results-dir experiments/rbf_elbow_training/results/nctx47_ntgt64_seed42/final_basis_ntgt64 \
  --checkpoint "$(pwd)/experiments/rbf_elbow_training/results/nctx47_ntgt64_seed42/checkpoint_final.pt" \
  --name rbf_elbow_nctx47_ntgt64 --device cpu --episodes 16 \
  --d-x 5 --d-model 128 --n-layers 8 --n-heads 4 \
  --n-ctx 47 --n-tgt 64 --sigma2 0.1 \
  --kernel-lengthscale 3.0 --kernel-signal-var 1.0 \
  --rank-tau 0.01 --tau-sv 0.001 --curve-r-max 47 \
  --n-build 32 --n-eval 32
```

Use `run_causal_diagnostic.py` for the validated-rank causal diagnostic because
the upstream script has a fixed RBF checkpoint registry.

