"""Shared support layer for the operator-Galerkin paper experiments."""

import math
import os
import sys

import numpy as np
import torch

SUPPORT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(SUPPORT_DIR, "..", ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")

sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SRC_DIR)

from model import ICLTransformer
from data import sample_batch_eigenvalues
from train import Config, _ridge_predictions, set_seed, get_device


CHECKPOINT_DIR = os.path.join(REPO_ROOT, "checkpoints")


def flat_eigenvalues(d, c_lo=1.0, c_hi=10.0):
    c = c_lo + (c_hi - c_lo) * torch.rand(1).item()
    return torch.full((d,), c)


def sample_batch(batch_size, d, n_ctx, n_tgt, sigma2, device="cpu"):
    return sample_batch_eigenvalues(
        batch_size, d, n_ctx, n_tgt, sigma2, flat_eigenvalues, device
    )


def load_model(path, cfg, device, n_layers=None):
    nl = n_layers or cfg.n_layers
    model = ICLTransformer(
        d_x=cfg.d_x,
        d_model=cfg.d_model,
        n_layers=nl,
        n_heads=cfg.n_heads,
        ffn_mult=cfg.ffn_mult,
    ).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


def dual_variables(x_ctx, y_ctx, sigma2):
    K = x_ctx @ x_ctx.transpose(-2, -1)
    n_ctx = K.shape[-1]
    I = torch.eye(n_ctx, device=K.device).unsqueeze(0)
    return torch.linalg.solve(K + sigma2 * I, y_ctx.unsqueeze(-1)).squeeze(-1)


def ridge_system(x_ctx, y_ctx, sigma2):
    d = x_ctx.shape[-1]
    XtX = x_ctx.transpose(-2, -1) @ x_ctx
    Xty = (x_ctx.transpose(-2, -1) @ y_ctx.unsqueeze(-1)).squeeze(-1)
    I = torch.eye(d, device=x_ctx.device).unsqueeze(0)
    return XtX + sigma2 * I, Xty


def results_path(filename, subdir=None):
    root = os.path.join(RESULTS_DIR, subdir) if subdir else RESULTS_DIR
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, filename)


def checkpoint_path(filename):
    return os.path.join(CHECKPOINT_DIR, filename)


def squared_distances(X1, X2):
    sq1 = (X1 ** 2).sum(-1, keepdim=True)
    sq2 = (X2 ** 2).sum(-1, keepdim=True)
    cross = X1 @ X2.transpose(-2, -1)
    return (sq1 + sq2.transpose(-2, -1) - 2 * cross).clamp(min=0.0)


def _broadcast_param(p, batch_size):
    if isinstance(p, (int, float)):
        return p
    if p.dim() == 0:
        return p.item()
    return p.view(batch_size, 1, 1)


def rbf_kernel(X1, X2, lengthscale, signal_var=1.0):
    batch_size = X1.shape[0]
    ls = _broadcast_param(lengthscale, batch_size)
    sv = _broadcast_param(signal_var, batch_size)
    dist_sq = squared_distances(X1, X2)
    return sv * torch.exp(-dist_sq / (2.0 * ls ** 2))


def matern_kernel(X1, X2, lengthscale, signal_var=1.0, nu=1.5):
    batch_size = X1.shape[0]
    ls = _broadcast_param(lengthscale, batch_size)
    sv = _broadcast_param(signal_var, batch_size)
    dist = squared_distances(X1, X2).clamp(min=1e-20).sqrt()
    r = dist / ls

    if nu == 0.5:
        return sv * torch.exp(-r)
    if nu == 1.5:
        sqrt3_r = math.sqrt(3.0) * r
        return sv * (1.0 + sqrt3_r) * torch.exp(-sqrt3_r)
    if nu == 2.5:
        sqrt5_r = math.sqrt(5.0) * r
        return sv * (1.0 + sqrt5_r + 5.0 * r ** 2 / 3.0) * torch.exp(-sqrt5_r)
    raise ValueError(f"Unsupported Matern nu={nu}")


def linear_kernel(X1, X2, _lengthscale, signal_var=1.0):
    batch_size = X1.shape[0]
    sv = _broadcast_param(signal_var, batch_size)
    return sv * (X1 @ X2.transpose(-2, -1))


KERNEL_REGISTRY = {
    "rbf": rbf_kernel,
    "linear": linear_kernel,
    "matern12": lambda X1, X2, ls, sv=1.0: matern_kernel(X1, X2, ls, sv, nu=0.5),
    "matern32": lambda X1, X2, ls, sv=1.0: matern_kernel(X1, X2, ls, sv, nu=1.5),
    "matern52": lambda X1, X2, ls, sv=1.0: matern_kernel(X1, X2, ls, sv, nu=2.5),
}


def compute_kernel(X1, X2, kernel_type, lengthscale, signal_var=1.0):
    if kernel_type not in KERNEL_REGISTRY:
        raise ValueError(f"Unknown kernel: {kernel_type}")
    return KERNEL_REGISTRY[kernel_type](X1, X2, lengthscale, signal_var)


KERNEL_TYPES = ["rbf", "matern32", "matern52"]


def sample_kernel_params(kernel_type=None):
    if kernel_type is None:
        kernel_type = KERNEL_TYPES[torch.randint(len(KERNEL_TYPES), (1,)).item()]
    log_ls = torch.empty(1).uniform_(torch.tensor(0.3).log(), torch.tensor(5.0).log())
    lengthscale = log_ls.exp().item()
    signal_var = 1.0 + 4.0 * torch.rand(1).item()
    return kernel_type, lengthscale, signal_var


def sample_gp_batch(batch_size, d, n_ctx, n_tgt, sigma2, kernel_type=None, device="cpu"):
    total = n_ctx + n_tgt
    X = torch.randn(batch_size, total, d)

    kernel_types = []
    lengthscales = torch.empty(batch_size)
    signal_vars = torch.empty(batch_size)
    for b in range(batch_size):
        kt, ls, sv = sample_kernel_params(kernel_type)
        kernel_types.append(kt)
        lengthscales[b] = ls
        signal_vars[b] = sv

    f = torch.zeros(batch_size, total)
    for kt in set(kernel_types):
        mask = [i for i, t in enumerate(kernel_types) if t == kt]
        if not mask:
            continue
        idx = torch.tensor(mask)
        X_sub = X[idx]
        ls_sub = lengthscales[idx]
        sv_sub = signal_vars[idx]
        K_full = compute_kernel(X_sub, X_sub, kt, ls_sub, sv_sub)
        jitter = 1e-5 * torch.eye(total).unsqueeze(0)
        L = torch.linalg.cholesky(K_full + jitter)
        z = torch.randn(len(mask), total, 1)
        f[idx] = (L @ z).squeeze(-1)

    x_ctx = X[:, :n_ctx, :]
    x_tgt = X[:, n_ctx:, :]
    y_ctx = f[:, :n_ctx] + torch.randn(batch_size, n_ctx) * (sigma2 ** 0.5)
    y_tgt = f[:, n_ctx:]
    meta = {
        "kernel_types": kernel_types,
        "lengthscales": lengthscales,
        "signal_vars": signal_vars,
    }
    return x_ctx.to(device), y_ctx.to(device), x_tgt.to(device), y_tgt.to(device), meta


MODELS = {
    "standard": {
        "checkpoint": "final/linear_baseline_dx5_L8.pt",
        "d_x": 5,
        "d_model": 128,
        "n_heads": 4,
        "label": r"$d_x\!=\!5$, $D\!=\!128$",
    },
}

ACTIVE_EIG_PROJ_RTOL = 1e-8
ACTIVE_EIG_EIG_RTOL = 1e-12
PACKET_BASIS_RANK_RTOL = 1e-12


def sample_episode_batch(model_key, cfg, device, return_target_kernel=False):
    mcfg = MODELS[model_key]
    if mcfg.get("kernel_type") == "gp":
        x_ctx, y_ctx, x_tgt, y_tgt, meta = sample_gp_batch(
            cfg.batch_size, cfg.d_x, cfg.n_ctx, cfg.n_tgt, cfg.sigma2, device=device
        )
        K_list = []
        Kt_list = []
        for ep in range(cfg.batch_size):
            kt = meta["kernel_types"][ep]
            ls = meta["lengthscales"][ep : ep + 1]
            sv = meta["signal_vars"][ep : ep + 1]
            xc = x_ctx[ep : ep + 1].cpu()
            xt = x_tgt[ep : ep + 1].cpu()
            K_ep = compute_kernel(xc, xc, kt, ls, sv).squeeze(0)
            Kt_ep = compute_kernel(xt, xc, kt, ls, sv).squeeze(0)
            K_list.append(K_ep)
            Kt_list.append(Kt_ep)
        K_batch = torch.stack(K_list).double()
        if return_target_kernel:
            return x_ctx, y_ctx, x_tgt, y_tgt, K_batch, torch.stack(Kt_list).double()
        return x_ctx, y_ctx, x_tgt, y_tgt, K_batch

    x_ctx, y_ctx, x_tgt, y_tgt, _ = sample_batch(
        cfg.batch_size, cfg.d_x, cfg.n_ctx, cfg.n_tgt, cfg.sigma2, device
    )
    x_ctx_cpu = x_ctx.cpu().double()
    x_tgt_cpu = x_tgt.cpu().double()
    K_batch = x_ctx_cpu @ x_ctx_cpu.transpose(-2, -1)
    if return_target_kernel:
        Kt_batch = x_tgt_cpu @ x_ctx_cpu.transpose(-2, -1)
        return x_ctx, y_ctx, x_tgt, y_tgt, K_batch, Kt_batch
    return x_ctx, y_ctx, x_tgt, y_tgt, K_batch


def eigendecompose_active(K, y, threshold=ACTIVE_EIG_PROJ_RTOL, eig_rtol=ACTIVE_EIG_EIG_RTOL):
    eigvals, V = torch.linalg.eigh(K)
    eig_scale = eigvals.abs().max().clamp(min=1.0)
    zero_tol = eig_rtol * eig_scale
    distinct_tol = eig_rtol * eig_scale
    proj_tol = threshold * y.norm()

    grouped_eigvals = []
    grouped_vecs = []
    grouped_proj = []

    i = 0
    while i < len(eigvals):
        j = i + 1
        while j < len(eigvals) and torch.abs(eigvals[j] - eigvals[i]) <= distinct_tol:
            j += 1

        coeffs = V[:, i:j].T @ y
        proj_norm = coeffs.norm()
        if proj_norm > proj_tol:
            mu = eigvals[i:j].mean()
            if torch.abs(mu) <= zero_tol:
                mu = torch.zeros_like(mu)
            grouped_eigvals.append(mu)
            grouped_vecs.append(V[:, i:j] @ (coeffs / proj_norm))
            grouped_proj.append(proj_norm)
        i = j

    if not grouped_vecs:
        return (
            torch.zeros(0, dtype=K.dtype, device=K.device),
            torch.zeros(K.shape[0], 0, dtype=K.dtype, device=K.device),
            torch.zeros(0, dtype=K.dtype, device=K.device),
        )

    return (
        torch.stack(grouped_eigvals),
        torch.stack(grouped_vecs, dim=1),
        torch.stack(grouped_proj),
    )


def _local_packet_cost(mu, sigma2, proj, degree):
    n = len(mu)
    if n == 0 or n <= degree + 1:
        return 0.0

    rho = 1.0 / (mu + sigma2)
    w = (mu + sigma2) * proj ** 2
    mu_center = mu.mean()
    mu_scale = (mu - mu_center).abs().max().clamp(min=1e-10)
    mu_norm = (mu - mu_center) / mu_scale
    V = torch.vander(mu_norm, N=degree + 1, increasing=True)
    sw = w.sqrt()
    WV = sw.unsqueeze(1) * V
    Wrho = sw * rho
    fitted = (V @ torch.linalg.lstsq(WV, Wrho.unsqueeze(-1)).solution).squeeze(-1)
    return (w * (rho - fitted) ** 2).sum().item()


def oracle_consecutive_partition(eigvals, sigma2, proj, q, degree):
    nu = len(eigvals)
    if nu == 0:
        return [[] for _ in range(q)]
    if q >= nu:
        return [[i] for i in range(nu)] + [[] for _ in range(q - nu)]

    cost = {}
    for i in range(nu):
        for j in range(i, nu):
            idx = list(range(i, j + 1))
            cost[(i, j)] = _local_packet_cost(eigvals[idx], sigma2, proj[idx], degree)

    inf = float("inf")
    dp = [[inf] * (nu + 1) for _ in range(q + 1)]
    split = [[0] * (nu + 1) for _ in range(q + 1)]
    dp[0][0] = 0.0

    for a in range(1, q + 1):
        for i in range(a, nu + 1):
            for j in range(a - 1, i):
                val = dp[a - 1][j] + cost[(j, i - 1)]
                if val < dp[a][i]:
                    dp[a][i] = val
                    split[a][i] = j

    packets = []
    i = nu
    for a in range(q, 0, -1):
        j = split[a][i]
        packets.append(list(range(j, i)))
        i = j
    packets.reverse()
    return packets


def compute_kappa(eigvals, sigma2, proj, packets, degree):
    rho = 1.0 / (eigvals + sigma2)
    w = (eigvals + sigma2) * proj ** 2
    alpha_A_sq = (w * rho ** 2).sum().item()
    if alpha_A_sq < 1e-30:
        return 0.0
    total = sum(
        _local_packet_cost(eigvals[pkt], sigma2, proj[pkt], degree)
        for pkt in packets
        if pkt
    )
    return np.sqrt(max(total / alpha_A_sq, 0.0))


def _normalized_vandermonde(mu, degree):
    mu_center = mu.mean()
    mu_scale = (mu - mu_center).abs().max().clamp(min=1e-12)
    mu_norm = (mu - mu_center) / mu_scale
    return torch.vander(mu_norm, N=degree + 1, increasing=True)


def packetized_krylov_basis_active(eigvals, vecs, proj, packets, degree, rtol=PACKET_BASIS_RANK_RTOL):
    nu = len(eigvals)
    if nu == 0:
        return torch.zeros(vecs.shape[0], 0, dtype=vecs.dtype, device=vecs.device)

    coeff_blocks = []
    for pkt in packets:
        if not pkt:
            continue
        idx = torch.tensor(pkt, dtype=torch.long, device=eigvals.device)
        mu_pkt = eigvals[idx]
        proj_pkt = proj[idx]
        design = proj_pkt.unsqueeze(1) * _normalized_vandermonde(mu_pkt, degree)
        U_pkt, S_pkt, _ = torch.linalg.svd(design, full_matrices=False)
        if len(S_pkt) == 0 or S_pkt[0] <= 0:
            continue
        keep = S_pkt > rtol * S_pkt[0]
        if not keep.any():
            continue
        coeff = torch.zeros((nu, int(keep.sum().item())), dtype=vecs.dtype, device=vecs.device)
        coeff[idx] = U_pkt[:, keep].to(vecs.dtype)
        coeff_blocks.append(coeff)

    if not coeff_blocks:
        return torch.zeros(vecs.shape[0], 0, dtype=vecs.dtype, device=vecs.device)

    return vecs @ torch.cat(coeff_blocks, dim=1)


def packetized_krylov_basis(K, packet_vecs, degree, eps=1e-10):
    all_basis = []
    for u in packet_vecs:
        norm = u.norm()
        if norm < eps:
            continue
        q = u / norm
        chain = [q]
        for _ in range(degree):
            w = K @ chain[-1]
            for b in chain:
                w = w - (b @ w) * b
            norm = w.norm()
            if norm < eps:
                break
            chain.append(w / norm)
        all_basis.extend(chain)

    if not all_basis:
        return torch.zeros(K.shape[0], 0, dtype=K.dtype, device=K.device)

    raw = torch.stack(all_basis, dim=1)
    Q, R = torch.linalg.qr(raw, mode="reduced")
    diag = R.diag().abs()
    keep = diag > eps * diag.max()
    return Q[:, keep]


def single_packet_kappa(eigvals, sigma2, proj, degree):
    nu = len(eigvals)
    if nu == 0:
        return 0.0
    return compute_kappa(eigvals, sigma2, proj, [list(range(nu))], degree)


def multi_packet_kappa_consecutive(eigvals, sigma2, proj, q, degree):
    nu = len(eigvals)
    if nu == 0:
        return 0.0
    q_eff = min(max(q, 1), nu)
    packets = oracle_consecutive_partition(eigvals, sigma2, proj, q_eff, degree)
    return compute_kappa(eigvals, sigma2, proj, packets, degree)


def _to_device(batch, device):
    x_ctx, y_ctx, x_tgt, y_tgt, K_batch, Kt_batch = batch
    return x_ctx.to(device), y_ctx.to(device), x_tgt.to(device), y_tgt.to(device), K_batch, Kt_batch


def last_layer_states(decomp):
    n_layers = len(decomp["attn_deltas"])
    h = decomp["h0"].cpu().double()
    for ell in range(n_layers - 1):
        h = h + decomp["attn_deltas"][ell].cpu().double() + decomp["ffn_deltas"][ell].cpu().double()
    post_attn = h + decomp["attn_deltas"][-1].cpu().double()
    attn_last = decomp["attentions"][-1].cpu().double()
    return h, post_attn, attn_last


def fit_effective_readout(model, batches, cfg, device):
    features = []
    targets = []
    with torch.no_grad():
        for batch in batches:
            x_ctx, y_ctx, x_tgt, _, _, _ = _to_device(batch, device)
            y_model = model(x_ctx, y_ctx, x_tgt).cpu().double()
            decomp = model.decompose(x_ctx, y_ctx, x_tgt)
            _, post_attn, _ = last_layer_states(decomp)
            z_tgt = post_attn[:, cfg.n_ctx :, :].reshape(-1, cfg.d_model)
            features.append(z_tgt)
            targets.append(y_model.reshape(-1, 1))

    X = torch.cat(features, dim=0)
    y = torch.cat(targets, dim=0)
    X_aug = torch.cat([X, torch.ones(X.shape[0], 1, dtype=X.dtype)], dim=1)
    sol = torch.linalg.lstsq(X_aug, y).solution.squeeze(-1)
    return sol[:-1], float(sol[-1].item())


def extract_ov_readout_vectors(model, u_vec):
    layer = model.layers[-1]
    W_qkv = layer.qkv.weight.detach().cpu().double()
    W_O = layer.wo.weight.detach().cpu().double()
    d_model = W_O.shape[0]
    dk = layer.dk
    W_V = W_qkv[2 * d_model :, :]

    u_row = u_vec.reshape(1, -1)
    per_head = []
    for h in range(layer.h):
        W_V_h = W_V[h * dk : (h + 1) * dk, :]
        W_O_h = W_O[:, h * dk : (h + 1) * dk]
        per_head.append((u_row @ W_O_h @ W_V_h).squeeze(0))
    return per_head


__all__ = [
    "CHECKPOINT_DIR",
    "Config",
    "KERNEL_TYPES",
    "MODELS",
    "RESULTS_DIR",
    "_ridge_predictions",
    "checkpoint_path",
    "compute_kernel",
    "compute_kappa",
    "dual_variables",
    "eigendecompose_active",
    "extract_ov_readout_vectors",
    "fit_effective_readout",
    "flat_eigenvalues",
    "get_device",
    "last_layer_states",
    "load_model",
    "multi_packet_kappa_consecutive",
    "oracle_consecutive_partition",
    "packetized_krylov_basis",
    "packetized_krylov_basis_active",
    "results_path",
    "ridge_system",
    "sample_batch",
    "sample_episode_batch",
    "sample_gp_batch",
    "set_seed",
    "single_packet_kappa",
]
