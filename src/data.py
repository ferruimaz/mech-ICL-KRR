"""
Episode generation for in-context linear regression with spectral curriculum.

Data generation pipeline (per episode):
  1. Pick a spectral profile (polynomial / exponential / step) at random
  2. Sample profile parameters and build eigenvalue spectrum Λ
  3. Draw U from Gaussian orthogonal ensemble (Gaussian matrix → QR)
  4. x ~ N(0, UΛU^T) via direct square root: x = z · √Λ · U^T
  5. β ~ N(0, I_d),  ε ~ N(0, σ²),  y = x^T β + ε

Spectral profiles:
  polynomial:  λ_i = c · i^{-α}         (Matérn-like smooth decay)
  exponential: λ_i = c · exp(-α(i-1))   (RBF-like steep decay)
  step:        λ_i = c_hi (i≤r), c_lo   (spiked / finite-rank)
"""

import torch


# ── Spectral profiles ────────────────────────────────────────────────────

def _polynomial_eigenvalues(d):
    """λ_i = c · i^{-α}.  α ~ U[1, 3], c ~ U[1, 10]."""
    alpha = 1.0 + 2.0 * torch.rand(1).item()
    c = 1.0 + 9.0 * torch.rand(1).item()
    idx = torch.arange(1, d + 1, dtype=torch.float32)
    return c * idx.pow(-alpha)


def _exponential_eigenvalues(d):
    """λ_i = c · exp(-α(i-1)).  α ~ U[0.5, 2], c ~ U[1, 10]."""
    alpha = 0.5 + 1.5 * torch.rand(1).item()
    c = 1.0 + 9.0 * torch.rand(1).item()
    idx = torch.arange(d, dtype=torch.float32)
    return c * torch.exp(-alpha * idx)


def _step_eigenvalues(d, sigma2):
    """λ_i = c_hi (i ≤ r), c_lo (i > r).  c_lo ≤ σ² so tail is in the noise floor."""
    r = torch.randint(1, d, (1,)).item()  # r ∈ {1, ..., d-1}
    c_hi = 2.0 + 8.0 * torch.rand(1).item()
    c_lo = 0.01 + (sigma2 - 0.01) * torch.rand(1).item()
    eig = torch.full((d,), c_lo)
    eig[:r] = c_hi
    return eig


PROFILES = {
    "polynomial": _polynomial_eigenvalues,
    "exponential": _exponential_eigenvalues,
    # step needs sigma2, handled separately
}
PROFILE_NAMES = ["polynomial", "exponential", "step"]


def sample_eigenvalues(d, sigma2, profile=None):
    """Sample one eigenvalue spectrum. If profile is None, pick at random."""
    name = profile or PROFILE_NAMES[torch.randint(len(PROFILE_NAMES), (1,)).item()]
    if name == "step":
        return _step_eigenvalues(d, sigma2)
    return PROFILES[name](d)


# ── Batch generation ─────────────────────────────────────────────────────

def _build_batch(eigenvalues, batch_size, d, n_ctx, n_tgt, sigma2, device):
    """Shared batch construction from a pre-computed eigenvalue tensor (B, d)."""
    A = torch.randn(batch_size, d, d)
    Q, R = torch.linalg.qr(A)
    sign = torch.sign(torch.diagonal(R, dim1=-2, dim2=-1))
    U = Q * sign.unsqueeze(-2)

    n = n_ctx + n_tgt
    sqrt_lam = eigenvalues.sqrt().unsqueeze(1)
    z = torch.randn(batch_size, n, d)
    x = (z * sqrt_lam) @ U.transpose(-2, -1)

    beta = torch.randn(batch_size, d)
    f = (x * beta.unsqueeze(1)).sum(-1)

    y_ctx = f[:, :n_ctx] + torch.randn(batch_size, n_ctx) * (sigma2 ** 0.5)
    y_tgt = f[:, n_ctx:]

    return (x[:, :n_ctx].to(device), y_ctx.to(device),
            x[:, n_ctx:].to(device), y_tgt.to(device),
            eigenvalues.to(device))


def sample_batch(batch_size, d, n_ctx, n_tgt, sigma2, device="cpu"):
    """Sample a batch with random spectral profiles (polynomial / exponential / step).

    Returns: (x_ctx, y_ctx, x_tgt, y_tgt, eigenvalues)
    """
    eigenvalues = torch.stack([sample_eigenvalues(d, sigma2) for _ in range(batch_size)])
    return _build_batch(eigenvalues, batch_size, d, n_ctx, n_tgt, sigma2, device)


def sample_batch_eigenvalues(batch_size, d, n_ctx, n_tgt, sigma2, eigenvalue_fn, device="cpu"):
    """Like sample_batch but uses a custom eigenvalue function.

    eigenvalue_fn(d) → Tensor of shape (d,) returning the eigenvalue spectrum.
    """
    eigenvalues = torch.stack([eigenvalue_fn(d) for _ in range(batch_size)])
    return _build_batch(eigenvalues, batch_size, d, n_ctx, n_tgt, sigma2, device)


def sample_batch_profile(batch_size, d, n_ctx, n_tgt, sigma2, profile, device="cpu"):
    """Like sample_batch but forces a specific spectral profile for all episodes."""
    eigenvalues = torch.stack([sample_eigenvalues(d, sigma2, profile=profile)
                               for _ in range(batch_size)])
    return _build_batch(eigenvalues, batch_size, d, n_ctx, n_tgt, sigma2, device)
