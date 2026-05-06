"""
Episode generation for in-context linear regression.

There are two sampler families.

1. ``legacy`` reproduces the original polynomial / exponential / step mixture.
2. ``minimal`` keeps the curriculum intentionally small:
   - ``minimal_smooth``: one log-span parameter interpolating from uniform to
     smooth exponential decay
   - ``minimal_step``: one active-rank parameter plus one tail-depth parameter

For the minimal family we support both the older tau-primary curricula and a
direct sampler that keeps ``log10(lambda_1)`` explicit while drawing ``tau``
uniformly over the branch-specific feasible interval.
"""

import torch


# ── Legacy spectral profiles ─────────────────────────────────────────────

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
    r = torch.randint(1, d, (1,)).item()
    c_hi = 2.0 + 8.0 * torch.rand(1).item()
    c_lo = 0.01 + (sigma2 - 0.01) * torch.rand(1).item()
    eig = torch.full((d,), c_lo)
    eig[:r] = c_hi
    return eig


LEGACY_PROFILES = {
    "polynomial": _polynomial_eigenvalues,
    "exponential": _exponential_eigenvalues,
}
PROFILE_NAMES = ["polynomial", "exponential", "step"]


def sample_eigenvalues(d, sigma2, profile=None):
    """Sample one legacy eigenvalue spectrum."""
    name = profile or PROFILE_NAMES[torch.randint(len(PROFILE_NAMES), (1,)).item()]
    if name == "step":
        return _step_eigenvalues(d, sigma2)
    return LEGACY_PROFILES[name](d)


# ── Minimal bounded-d_eff profiles ──────────────────────────────────────

MINIMAL_PROFILE_NAMES = ["minimal_smooth", "minimal_step"]


def effective_dimension(eigenvalues, sigma2):
    """τ(Λ) = Σ_i λ_i / (λ_i + σ²)."""
    return (eigenvalues / (eigenvalues + sigma2)).sum(dim=-1)


def _sample_step_rank(d, rank_values, rank_distribution):
    """Sample a valid step rank under the requested proposal law."""
    valid_ranks = [r for r in rank_values if 1 <= r < d]
    if not valid_ranks:
        raise ValueError(f"step rank values must lie in [1, {d-1}]")
    law = str(rank_distribution).strip().lower()
    if law == "uniform":
        return valid_ranks[torch.randint(len(valid_ranks), (1,)).item()]
    if law in {"inv_rank", "inverse_rank", "one_over_r"}:
        weights = torch.tensor([1.0 / float(r) for r in valid_ranks], dtype=torch.float32)
        probs = weights / weights.sum()
        idx = torch.multinomial(probs, 1).item()
        return valid_ranks[idx]
    raise ValueError(f"unknown step rank distribution: {rank_distribution}")


def _sample_log10_scale(log10_lambda1_min, log10_lambda1_max, distribution="uniform", power=2.0):
    """Sample log10(lambda_1) on a bounded interval.

    ``large_power`` keeps the same support but biases toward the upper endpoint:
    if U is uniform on [0,1], use U**(1/power), with density proportional to
    u**(power-1). ``power=1`` recovers uniform sampling.
    """
    lo = float(log10_lambda1_min)
    hi = float(log10_lambda1_max)
    u = torch.rand(1).item()
    law = str(distribution).strip().lower()
    if law in {"uniform", "flat"}:
        v = u
    elif law in {"large_power", "upper_power", "high_power", "large", "high"}:
        v = u ** (1.0 / max(float(power), 1e-8))
    else:
        raise ValueError(f"unknown scale distribution: {distribution}")
    return lo + (hi - lo) * v


def _minimal_smooth_normalized(d, span_min, span_max):
    """Normalized smooth spectrum with λ_1 = 1 and log-span in [span_min, span_max]."""
    span = span_min + (span_max - span_min) * torch.rand(1).item()
    idx = torch.arange(d, dtype=torch.float32)
    denom = max(1, d - 1)
    return torch.pow(10.0, -span * idx / denom)


def _minimal_step_normalized(d, rank_values, depth_min, depth_max, rank_distribution="uniform"):
    """Normalized step spectrum with λ_1 = ... = λ_r = 1 and tail 10^{-depth}."""
    r = _sample_step_rank(d, rank_values, rank_distribution)
    depth = depth_min + (depth_max - depth_min) * torch.rand(1).item()
    tail = 10.0 ** (-depth)
    eig = torch.full((d,), tail, dtype=torch.float32)
    eig[:r] = 1.0
    return eig


def _smooth_spectrum_from_scale_span(d, log10_lambda1, span):
    """Smooth spectrum with explicit log10(lambda_1) and total log-span."""
    idx = torch.arange(d, dtype=torch.float32)
    denom = max(1, d - 1)
    return torch.pow(10.0, log10_lambda1 - span * idx / denom)


def _step_spectrum_from_scale_rank_depth(d, log10_lambda1, rank, depth):
    """Step spectrum with explicit log10(lambda_1), rank, and tail depth."""
    tail = 10.0 ** (log10_lambda1 - depth)
    eig = torch.full((d,), tail, dtype=torch.float32)
    eig[:rank] = 10.0 ** log10_lambda1
    return eig


def _minimal_profile_name(profile):
    """Sample or validate the minimal profile name without sampling its shape."""
    if profile is None:
        return MINIMAL_PROFILE_NAMES[torch.randint(len(MINIMAL_PROFILE_NAMES), (1,)).item()]
    if profile not in MINIMAL_PROFILE_NAMES:
        raise ValueError(f"unknown minimal profile: {profile}")
    return profile


def _solve_log10_scale_for_tau(shape, sigma2, tau_target, low=-6.0, high=6.0, steps=60):
    """
    Given normalized shapes with max entry 1, solve for log10(scale) such that
    τ(scale * shape) = tau_target.
    """
    if not (0.0 < float(tau_target) < shape.shape[-1]):
        raise ValueError(f"tau_target must lie in (0, {shape.shape[-1]})")
    lo = torch.full((shape.shape[0],), float(low), dtype=shape.dtype)
    hi = torch.full((shape.shape[0],), float(high), dtype=shape.dtype)
    tau_target = torch.full((shape.shape[0],), float(tau_target), dtype=shape.dtype)
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        scaled = shape * torch.pow(10.0, mid).unsqueeze(-1)
        tau_mid = effective_dimension(scaled, sigma2)
        lo = torch.where(tau_mid < tau_target, mid, lo)
        hi = torch.where(tau_mid >= tau_target, mid, hi)
    return 0.5 * (lo + hi)


def _solve_smooth_span_for_tau(
    d,
    sigma2,
    log10_lambda1,
    tau_target,
    span_low,
    span_high,
    steps=60,
):
    """
    Solve for the smooth log-span s with tau(lambda(a, s)) = tau_target.

    tau is monotone decreasing in span, so a simple scalar bisection is enough.
    """
    lo = float(span_low)
    hi = float(span_high)
    target = float(tau_target)
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        eig = _smooth_spectrum_from_scale_span(d, log10_lambda1, mid)
        tau_mid = float(effective_dimension(eig.unsqueeze(0), sigma2)[0])
        if tau_mid > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _solve_step_depth_for_tau(d, sigma2, log10_lambda1, rank, tau_target):
    """
    Solve for the step tail depth t with tau(lambda(a, r, t)) = tau_target.

    The step branch admits a closed-form inversion for the tail contribution.
    """
    top = rank * (10.0 ** log10_lambda1) / ((10.0 ** log10_lambda1) + sigma2)
    residual = float(tau_target) - float(top)
    tail_dims = d - int(rank)
    if tail_dims <= 0:
        raise ValueError("step branch requires rank < d")
    if residual <= 0.0:
        return float("inf")
    frac = residual / float(tail_dims)
    frac = min(max(frac, 1e-12), 1.0 - 1e-12)
    tail_lambda = sigma2 * frac / (1.0 - frac)
    return float(log10_lambda1 - torch.log10(torch.tensor(tail_lambda)).item())


def _sample_minimal_normalized_shape(
    d,
    profile,
    smooth_span_min,
    smooth_span_max,
    step_rank_values,
    step_depth_min,
    step_depth_max,
    step_rank_distribution="uniform",
):
    """Sample one normalized minimal spectrum with λ_1 = 1."""
    name = profile or MINIMAL_PROFILE_NAMES[torch.randint(len(MINIMAL_PROFILE_NAMES), (1,)).item()]
    if name == "minimal_smooth":
        shape = _minimal_smooth_normalized(d, smooth_span_min, smooth_span_max)
    elif name == "minimal_step":
        shape = _minimal_step_normalized(
            d,
            step_rank_values,
            step_depth_min,
            step_depth_max,
            rank_distribution=step_rank_distribution,
        )
    else:
        raise ValueError(f"unknown minimal profile: {name}")
    return name, shape


def _sample_minimal_batch_spectrum(
    d,
    sigma2,
    profile,
    sampling_scheme,
    tau_min,
    tau_max,
    log10_lambda1_min,
    log10_lambda1_max,
    rejection_attempts,
    smooth_span_min,
    smooth_span_max,
    step_rank_values,
    step_depth_min,
    step_depth_max,
    step_rank_distribution="uniform",
    scale_distribution="uniform",
    scale_distribution_power=2.0,
    tau_target=None,
):
    """
    Sample one batch-level minimal spectrum and return
    (profile_name, eigenvalues[d], tau_actual, log10_lambda1).
    """
    for _ in range(max(1, int(rejection_attempts))):
        if sampling_scheme == "scale_tau_direct":
            name = _minimal_profile_name(profile)
            log10_scale = _sample_log10_scale(
                log10_lambda1_min,
                log10_lambda1_max,
                distribution=scale_distribution,
                power=scale_distribution_power,
            )

            if name == "minimal_smooth":
                tau_at_span_min = float(
                    effective_dimension(
                        _smooth_spectrum_from_scale_span(d, log10_scale, smooth_span_min).unsqueeze(0),
                        sigma2,
                    )[0]
                )
                tau_at_span_max = float(
                    effective_dimension(
                        _smooth_spectrum_from_scale_span(d, log10_scale, smooth_span_max).unsqueeze(0),
                        sigma2,
                    )[0]
                )
                feasible_low = max(float(tau_min), tau_at_span_max)
                feasible_high = min(float(tau_max), tau_at_span_min)
                if feasible_low > feasible_high:
                    continue
                target = feasible_low + (feasible_high - feasible_low) * torch.rand(1).item()
                span = _solve_smooth_span_for_tau(
                    d=d,
                    sigma2=sigma2,
                    log10_lambda1=log10_scale,
                    tau_target=target,
                    span_low=smooth_span_min,
                    span_high=smooth_span_max,
                )
                eig = _smooth_spectrum_from_scale_span(d, log10_scale, span)
                tau_actual = float(effective_dimension(eig.unsqueeze(0), sigma2)[0])
                return name, eig, tau_actual, log10_scale

            rank = _sample_step_rank(d, step_rank_values, step_rank_distribution)
            tau_at_depth_min = float(
                effective_dimension(
                    _step_spectrum_from_scale_rank_depth(d, log10_scale, rank, step_depth_min).unsqueeze(0),
                    sigma2,
                )[0]
            )
            tau_at_depth_max = float(
                effective_dimension(
                    _step_spectrum_from_scale_rank_depth(d, log10_scale, rank, step_depth_max).unsqueeze(0),
                    sigma2,
                )[0]
            )
            feasible_low = max(float(tau_min), tau_at_depth_max)
            feasible_high = min(float(tau_max), tau_at_depth_min)
            if feasible_low > feasible_high:
                continue
            target = feasible_low + (feasible_high - feasible_low) * torch.rand(1).item()
            depth = _solve_step_depth_for_tau(
                d=d,
                sigma2=sigma2,
                log10_lambda1=log10_scale,
                rank=rank,
                tau_target=target,
            )
            depth = min(max(depth, float(step_depth_min)), float(step_depth_max))
            eig = _step_spectrum_from_scale_rank_depth(d, log10_scale, rank, depth)
            tau_actual = float(effective_dimension(eig.unsqueeze(0), sigma2)[0])
            return name, eig, tau_actual, log10_scale

        name, shape = _sample_minimal_normalized_shape(
            d=d,
            profile=profile,
            smooth_span_min=smooth_span_min,
            smooth_span_max=smooth_span_max,
            step_rank_values=step_rank_values,
            step_depth_min=step_depth_min,
            step_depth_max=step_depth_max,
            step_rank_distribution=step_rank_distribution,
        )

        if sampling_scheme == "tau_exact":
            if tau_target is None:
                raise ValueError("tau_exact requires tau_target")
            target = float(tau_target)
            log10_scale = float(_solve_log10_scale_for_tau(shape.unsqueeze(0), sigma2, target)[0])
            eig = shape * (10.0 ** log10_scale)
            tau_actual = float(effective_dimension(eig.unsqueeze(0), sigma2)[0])
            return name, eig, tau_actual, log10_scale

        if sampling_scheme == "scale_uniform_reject_tau":
            log10_scale = _sample_log10_scale(
                log10_lambda1_min,
                log10_lambda1_max,
                distribution=scale_distribution,
                power=scale_distribution_power,
            )
            eig = shape * (10.0 ** log10_scale)
            tau_actual = float(effective_dimension(eig.unsqueeze(0), sigma2)[0])
            if tau_actual <= tau_max:
                return name, eig, tau_actual, log10_scale
            continue

        if sampling_scheme == "tau_uniform_reject_scale":
            target = float(tau_target) if tau_target is not None else (
                tau_min + (tau_max - tau_min) * torch.rand(1).item()
            )
            log10_scale = float(_solve_log10_scale_for_tau(shape.unsqueeze(0), sigma2, target)[0])
            if log10_lambda1_min <= log10_scale <= log10_lambda1_max:
                eig = shape * (10.0 ** log10_scale)
                tau_actual = float(effective_dimension(eig.unsqueeze(0), sigma2)[0])
                return name, eig, tau_actual, log10_scale
            continue

        raise ValueError(f"unknown minimal sampling_scheme: {sampling_scheme}")

    raise RuntimeError(
        "failed to sample a minimal spectrum within the requested constraints; "
        "try relaxing tau or scale bounds"
    )


# ── Shared batch construction ───────────────────────────────────────────

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

    return (
        x[:, :n_ctx].to(device),
        y_ctx.to(device),
        x[:, n_ctx:].to(device),
        y_tgt.to(device),
        eigenvalues.to(device),
    )


# ── Public batch samplers ───────────────────────────────────────────────

def sample_batch(batch_size, d, n_ctx, n_tgt, sigma2, device="cpu"):
    """Sample a legacy batch with random polynomial / exponential / step spectra."""
    eigenvalues = torch.stack([sample_eigenvalues(d, sigma2) for _ in range(batch_size)])
    return _build_batch(eigenvalues, batch_size, d, n_ctx, n_tgt, sigma2, device)


def sample_batch_eigenvalues(batch_size, d, n_ctx, n_tgt, sigma2, eigenvalue_fn, device="cpu"):
    """Like sample_batch but uses a custom eigenvalue function."""
    eigenvalues = torch.stack([eigenvalue_fn(d) for _ in range(batch_size)])
    return _build_batch(eigenvalues, batch_size, d, n_ctx, n_tgt, sigma2, device)


def sample_batch_profile(batch_size, d, n_ctx, n_tgt, sigma2, profile, device="cpu"):
    """Like sample_batch but forces one legacy spectral profile for all episodes."""
    eigenvalues = torch.stack([sample_eigenvalues(d, sigma2, profile=profile) for _ in range(batch_size)])
    return _build_batch(eigenvalues, batch_size, d, n_ctx, n_tgt, sigma2, device)


def sample_batch_minimal(
    batch_size,
    d,
    n_tgt,
    sigma2,
    tau_target,
    gamma,
    profile=None,
    device="cpu",
    sampling_scheme="tau_exact",
    tau_min=1e-3,
    tau_max=3.0,
    log10_lambda1_min=-1.0,
    log10_lambda1_max=1.0,
    rejection_attempts=256,
    smooth_span_min=0.0,
    smooth_span_max=3.0,
    step_rank_values=None,
    step_depth_min=1.0,
    step_depth_max=3.0,
    step_rank_distribution="uniform",
    scale_distribution="uniform",
    scale_distribution_power=2.0,
):
    """
    Sample a minimal bounded-d_eff batch.

    One covariance is sampled per batch, and the batch contains independent data
    realizations from that covariance. This keeps n_ctx tied to the actual batch
    effective dimension.
    """
    step_rank_values = step_rank_values or list(range(1, d))
    _, eig, tau_actual, _ = _sample_minimal_batch_spectrum(
        d=d,
        sigma2=sigma2,
        profile=profile,
        sampling_scheme=sampling_scheme,
        tau_min=tau_min,
        tau_max=tau_max,
        log10_lambda1_min=log10_lambda1_min,
        log10_lambda1_max=log10_lambda1_max,
        rejection_attempts=rejection_attempts,
        smooth_span_min=smooth_span_min,
        smooth_span_max=smooth_span_max,
        step_rank_values=step_rank_values,
        step_depth_min=step_depth_min,
        step_depth_max=step_depth_max,
        step_rank_distribution=step_rank_distribution,
        scale_distribution=scale_distribution,
        scale_distribution_power=scale_distribution_power,
        tau_target=tau_target,
    )
    eigenvalues = eig.unsqueeze(0).repeat(batch_size, 1)
    n_ctx = max(1, int(round(float(gamma) * float(tau_actual))))
    return _build_batch(eigenvalues, batch_size, d, n_ctx, n_tgt, sigma2, device)


def sample_batch_minimal_same_shape_multiscale(
    batch_size,
    d,
    n_tgt,
    sigma2,
    gamma,
    k=3,
    profile=None,
    device="cpu",
    tau_min=1e-3,
    tau_max=3.0,
    log10_lambda1_min=-1.0,
    log10_lambda1_max=1.0,
    rejection_attempts=256,
    smooth_span_min=0.0,
    smooth_span_max=3.0,
    step_rank_values=None,
    step_depth_min=1.0,
    step_depth_max=3.0,
    step_rank_distribution="uniform",
    scale_distribution="uniform",
    scale_distribution_power=2.0,
):
    """
    Sample K sibling batches sharing one normalized spectral shape.

    The global scales are iid uniform over the requested scale window. To avoid
    scale clipping or boundary pile-up, feasibility is imposed at the shape
    level: the sampled normalized shape must satisfy the tau constraints at the
    scale-window endpoints. The model is not given tau or sigma; these are only
    used to define the synthetic training distribution.
    """
    step_rank_values = step_rank_values or list(range(1, d))
    k = max(1, int(k))
    for _ in range(max(1, int(rejection_attempts))):
        _, shape = _sample_minimal_normalized_shape(
            d=d,
            profile=profile,
            smooth_span_min=smooth_span_min,
            smooth_span_max=smooth_span_max,
            step_rank_values=step_rank_values,
            step_depth_min=step_depth_min,
            step_depth_max=step_depth_max,
            step_rank_distribution=step_rank_distribution,
        )
        eig_low = shape * (10.0 ** float(log10_lambda1_min))
        eig_high = shape * (10.0 ** float(log10_lambda1_max))
        tau_low = float(effective_dimension(eig_low.unsqueeze(0), sigma2)[0])
        tau_high = float(effective_dimension(eig_high.unsqueeze(0), sigma2)[0])
        if tau_low < float(tau_min) or tau_high > float(tau_max):
            continue

        batches = []
        for _ in range(k):
            log10_scale = _sample_log10_scale(
                log10_lambda1_min,
                log10_lambda1_max,
                distribution=scale_distribution,
                power=scale_distribution_power,
            )
            eig = shape * (10.0 ** log10_scale)
            tau_actual = float(effective_dimension(eig.unsqueeze(0), sigma2)[0])
            eigenvalues = eig.unsqueeze(0).repeat(batch_size, 1)
            n_ctx = max(1, int(round(float(gamma) * tau_actual)))
            batches.append(_build_batch(eigenvalues, batch_size, d, n_ctx, n_tgt, sigma2, device))
        return batches

    raise RuntimeError(
        "failed to sample a same-shape multiscale minimal batch; "
        "try relaxing tau bounds, scale bounds, or shape span/depth"
    )
