# d_x=40 scale-aware gated transformer checkpoint

This directory contains a trained in-context ridge-regression transformer checkpoint and the minimal source files needed to instantiate it.

## Files

- `minimal_scale_canonical_single_dx40_tau16_am2p0_2p0_inv_rank_bs16_ga1_150000_s122.best.pt`: PyTorch `state_dict` for the trained model.
- `training.log`: training log for the run that produced the checkpoint.
- `model.py`: model definition used to load the checkpoint.
- `data.py`: data-generation code for the synthetic ridge-regression episodes.
- `train.py`: training script/config reference.

Checkpoint SHA256:

```text
97e49656e65ea72faf46d8b548d26f82c552500cb8c60903aa4c920d5105c334
```

## Summary

This is a scale-aware softmax transformer trained for in-context ridge regression in dimension

\[
d_x=40.
\]

The model has:

\[
L=8,\qquad d_{\rm model}=128,\qquad H=4
\]

with no LayerNorm, target-to-target attention masking, empirical scale canonicalization, and learned per-layer residual gates driven by a scalar context-scale estimate.

The checkpoint is:

```text
minimal_scale_canonical_single_dx40_tau16_am2p0_2p0_inv_rank_bs16_ga1_150000_s122.best.pt
```

It was trained for 150,000 optimizer steps. The best fixed-validation score occurred at the final step:

\[
\mathrm{MSE}/\mathrm{MSE}_{\rm ridge}=1.638.
\]

The fixed validation bank contained 30 batches from the same training distribution.

## Ridge-regression task

Each episode samples a covariance matrix \(\Sigma\in\mathbb R^{40\times 40}\), context examples, and target examples. For context points,

\[
x_i\sim\mathcal N(0,\Sigma),
\qquad
\beta\sim\mathcal N(0,I_{40}),
\qquad
y_i=x_i^\top\beta+\epsilon_i,
\qquad
\epsilon_i\sim\mathcal N(0,\sigma^2),
\]

with

\[
\sigma^2=0.1.
\]

For target points,

\[
x_j^{\rm tgt}\sim\mathcal N(0,\Sigma),
\qquad
y_j^{\rm tgt}= (x_j^{\rm tgt})^\top\beta .
\]

The model is trained to predict the noiseless target values \(y_j^{\rm tgt}\).

For each batch, a single eigenvalue spectrum is sampled and all examples in that batch use that same spectrum. A random orthogonal basis is sampled per episode/batch element, so the eigenvalues are shared within the batch but the covariance eigenbasis is not fixed.

## Spectral quantities

Let

\[
\lambda_1\geq\lambda_2\geq\cdots\geq \lambda_{40}\geq0
\]

be the eigenvalues of \(\Sigma\). The ridge effective dimension is

\[
\tau(\Sigma)
=
\operatorname{tr}\!\left[\Sigma(\Sigma+\sigma^2I)^{-1}\right]
=
\sum_{i=1}^{40}\frac{\lambda_i}{\lambda_i+\sigma^2}.
\]

The explicit scale coordinate used by the sampler is

\[
a=\log_{10}\lambda_1.
\]

Training was constrained to

\[
a\in[-2,2],
\qquad
\tau(\Sigma)\in[10^{-3},16].
\]

## Training spectrum distribution

The training sampler is `scale_tau_direct` with two minimal spectral branches, sampled uniformly between branches.

First sample

\[
a\sim {\rm Uniform}[-2,2],
\qquad
\lambda_1=10^a.
\]

Then sample a branch and choose its shape parameter so that \(\tau(\Sigma)\leq16\), by sampling uniformly over the branch-specific feasible \(\tau\)-interval.

### Smooth branch

The smooth branch has exponential decay in log-eigenvalue:

\[
\lambda_i
=
10^{a-s(i-1)/(d_x-1)},
\qquad
i=1,\ldots,d_x,
\]

where

\[
s\in[0,8].
\]

For a sampled \(a\), the code computes the feasible interval of effective dimensions attained as \(s\) varies over \([0,8]\), intersects it with \([10^{-3},16]\), samples a target \(\tau_\star\) uniformly from that interval, and solves for \(s\) by bisection so that

\[
\tau(\lambda(a,s))=\tau_\star.
\]

### Step branch

The step branch has \(r\) large eigenvalues and a flat tail:

\[
\lambda_i=
\begin{cases}
10^a, & i\leq r,\\
10^{a-t}, & i>r,
\end{cases}
\]

where

\[
r\in\{1,2,\ldots,16\},
\qquad
t\in[1,8].
\]

The rank is sampled with probability proportional to \(1/r\):

\[
\mathbb P(r=k)\propto \frac{1}{k}.
\]

For a sampled \(a,r\), the code computes the feasible interval of effective dimensions attained as \(t\) varies over \([1,8]\), intersects it with \([10^{-3},16]\), samples \(\tau_\star\) uniformly from that interval, and solves for \(t\) analytically so that

\[
\tau(\lambda(a,r,t))=\tau_\star.
\]

## Context and target sizes

The number of target points is fixed:

\[
n_{\rm tgt}=8.
\]

The number of context points depends on the sampled effective dimension:

\[
n_{\rm ctx}
=
\operatorname{round}(\gamma\,\tau(\Sigma)),
\]

where

\[
\gamma\in\{4,8,12\}
\]

is sampled during training.

## Transformer input format

Each token is in \(\mathbb R^{d_x+2}=\mathbb R^{42}\).

Context token:

\[
z_i^{\rm ctx}
=
(x_i,\ y_i,\ 0).
\]

Target token:

\[
z_j^{\rm tgt}
=
(x_j^{\rm tgt},\ 0,\ 1).
\]

The final coordinate marks target tokens.

## Base transformer architecture

The model embeds each token as

\[
h_i^{(0)} = W_{\rm emb} z_i + b_{\rm emb}
\in\mathbb R^{128}.
\]

There are \(L=8\) transformer blocks. Each block has standard multi-head softmax self-attention followed by a two-layer GELU MLP. There is no LayerNorm.

For layer \(\ell\), ignoring scale gates for a moment,

\[
\widetilde h^{(\ell)}
=
h^{(\ell)}
+ \operatorname{MHA}_\ell(h^{(\ell)}),
\]

\[
h^{(\ell+1)}
=
\widetilde h^{(\ell)}
+ \operatorname{FFN}_\ell(\widetilde h^{(\ell)}).
\]

The attention block uses \(H=4\) heads of width \(32\). Attention logits are

\[
\frac{QK^\top}{\sqrt{32}}.
\]

Target-key attention is masked: no query token can attend to target key/value tokens. Thus both context and target queries attend only to context key/value tokens. This includes target-to-target masking and also prevents context tokens from reading target-token information.

The FFN has hidden width

\[
2d_{\rm model}=256,
\]

and uses GELU:

\[
\operatorname{FFN}(h)=W_2\,{\rm GELU}(W_1h+b_1)+b_2.
\]

The output head is linear:

\[
\hat y_j = w_{\rm out}^\top h_j^{(8)} + b_{\rm out}
\]

at target positions.

## Empirical scale canonicalization

The model estimates an episode-level input scale from context inputs only:

\[
\widehat s
=
\frac{1}{n_{\rm ctx}d_x}
\sum_{i=1}^{n_{\rm ctx}}\lVert x_i\rVert_2^2.
\]

The log-scale input to the gate controller is

\[
z_s
=
\operatorname{clip}(\log_{10}\widehat s,\,-8,\,8).
\]

With `scale_canonical=True`, the model canonicalizes inputs by

\[
x_i \leftarrow \frac{x_i}{\sqrt{\widehat s}},
\qquad
x_j^{\rm tgt}\leftarrow \frac{x_j^{\rm tgt}}{\sqrt{\widehat s}}.
\]

With `scale_y=True`, context responses are also canonicalized:

\[
y_i \leftarrow \frac{y_i}{\sqrt{\widehat s}}.
\]

The model predicts in canonicalized units and then rescales the output:

\[
\hat y_j^{\rm final}
=
\sqrt{\widehat s}\,\hat y_j^{\rm canonical}.
\]

The true spectrum, Gram matrix, ridge solution, and \(\tau\) are not provided to the model.

## Learned scale gates

The scale controller is a learned MLP receiving only the scalar \(z_s\). It has hidden width 16:

\[
u(z_s)
=
W_2\tanh(W_1 z_s+b_1)+b_2
\in\mathbb R^{2L}.
\]

The final linear layer of this MLP was initialized to zero, so at initialization all gates are exactly 1.

The raw outputs are converted into bounded positive gates:

\[
g_j(z_s)
=
\exp\left(\log B\cdot\tanh(u_j(z_s))\right),
\]

with

\[
B=3.
\]

Thus each gate lies in

\[
g_j(z_s)\in[1/3,3].
\]

The \(2L=16\) gates are interpreted as one attention residual gate and one FFN residual gate per layer:

\[
g_\ell^{\rm attn}(z_s),
\qquad
g_\ell^{\rm ffn}(z_s),
\qquad
\ell=1,\ldots,8.
\]

The gated residual updates are

\[
\widetilde h^{(\ell)}
=
h^{(\ell)}
+
g_\ell^{\rm attn}(z_s)\,
\operatorname{MHA}_\ell(h^{(\ell)}),
\]

\[
h^{(\ell+1)}
=
\widetilde h^{(\ell)}
+
g_\ell^{\rm ffn}(z_s)\,
\operatorname{FFN}_\ell(\widetilde h^{(\ell)}).
\]

The gates are per-episode scalars, shared across tokens and hidden dimensions within a layer. They modulate residual step sizes rather than injecting a full vector side-channel.

## Training details

Training ran on CPU with synthetic data generated on CPU.

Main hyperparameters:

```text
d_x = 40
d_model = 128
n_layers = 8
n_heads = 4
ffn_mult = 2
mask_tgt_tgt = true
scale_canonical = true
scale_y = true
scale_controller = layer_gates
scale_gate_hidden = 16
scale_gate_bound = 3.0
scale_log_clip = 8.0
spectrum_family = minimal
minimal_sampling_scheme = scale_tau_direct
tau window = [0.001, 16.0]
log10(lambda_1) window = [-2.0, 2.0]
gamma values = 4,8,12
n_tgt = 8
batch_size = 16
grad_accum_steps = 1
train_steps = 150000
optimizer = AdamW
learning rate = 5e-5
min learning rate = 1e-5
warmup_steps = 1000
weight_decay = 1e-4
grad_clip = 1.0
eval_every = 2000
eval_batches = 10
fixed_validation_bank = true
validation_bank_size = 30
seed = 122
```

The learning-rate schedule used linear warmup for 1000 steps followed by cosine annealing with floor \(10^{-5}\).

## Loading the checkpoint

Place this directory on the Python path, or run the following from inside it:

```python
import torch
from model import ICLTransformer

model = ICLTransformer(
    d_x=40,
    d_model=128,
    n_layers=8,
    n_heads=4,
    ffn_mult=2,
    mask_tgt_tgt=True,
    scale_canonical=True,
    scale_stat="mean_x2",
    scale_eps=1e-8,
    scale_y=True,
    scale_controller="layer_gates",
    scale_gate_hidden=16,
    scale_gate_bound=3.0,
    scale_log_clip=8.0,
    scale_conditioner="none",
)

state = torch.load(
    "minimal_scale_canonical_single_dx40_tau16_am2p0_2p0_inv_rank_bs16_ga1_150000_s122.best.pt",
    map_location="cpu",
)
model.load_state_dict(state)
model.eval()
```

The forward signature is:

```python
preds = model(x_ctx, y_ctx, x_tgt)
```

where

```text
x_ctx: shape (batch, n_ctx, 40)
y_ctx: shape (batch, n_ctx)
x_tgt: shape (batch, n_tgt, 40)
preds: shape (batch, n_tgt)
```

## Notes and caveats

This checkpoint is the best current \(d_x=40\) scale-aware gated model trained on the \(a\in[-2,2]\), \(\tau\le16\) curriculum. It is not a fully scale-free ridge solver. Empirically, it improves trainability and robustness relative to weaker scale side-channels, but high scale and high effective dimension remain difficult.

The checkpoint file is a raw PyTorch `state_dict`, not a full serialized training object. The architecture arguments above must be used when loading it.
