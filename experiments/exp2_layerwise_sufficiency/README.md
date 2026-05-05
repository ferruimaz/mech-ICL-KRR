# Experiment 2 Layerwise Sufficiency

This is a separate Experiment 2 variant for testing layerwise emergence of a
KRR-relevant source subspace.

For each layer `l`, the script builds candidate context-side spans from:

- `response`: finite-difference hidden responses `D_y H_l`
- `raw`: raw context hidden states `H_l`
- `combined`: the union of both

Inside each candidate span, directions are ordered by a KRR-targeted criterion:
the prefix `Q_{l,k}` maximizes captured energy of `T A^{1/2}` within that span.
The resulting curves answer whether layer `l` contains a sufficient KRR source
subspace, measured by excess prediction risk over the KRR posterior risk.

This experiment is intentionally not a causal ablation test. It is an oracle
containment/sufficiency diagnostic that can be followed by subspace-level causal
validation.
