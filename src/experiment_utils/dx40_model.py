"""
Transformer for in-context linear regression.

Architecture:
  - Input projection: linear (d+2) → d_model
  - Transformer body: multi-head self-attention + FFN, residual connections, NO LayerNorm
  - Output head: linear d_model → 1, applied at target positions
  - Optional scale-canonical front-end with a tiny bottleneck gate controller
"""

import math
import torch
import torch.nn as nn


class ScaleController(nn.Module):
    """Tiny scalar controller for scale-dependent residual gates.

    The last layer is zero-initialized, so every gate is exactly 1 at
    initialization. This makes the controller an opt-in adaptation mechanism
    rather than a hidden change to the base no-LN transformer.
    """

    def __init__(self, n_layers, hidden_dim=16, gate_bound=3.0):
        super().__init__()
        self.n_layers = int(n_layers)
        self.gate_bound = float(gate_bound)
        hidden_dim = max(1, int(hidden_dim))
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2 * self.n_layers),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, log10_s_hat):
        raw = self.net(log10_s_hat.unsqueeze(-1))
        if self.gate_bound <= 1.0:
            return torch.ones_like(raw).view(-1, self.n_layers, 2)
        log_bound = math.log(self.gate_bound)
        gates = torch.exp(log_bound * torch.tanh(raw))
        return gates.view(-1, self.n_layers, 2)


class ScaleEmbedding(nn.Module):
    """Small typed side-channel for a global episode scale scalar."""

    def __init__(self, out_dim, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, int(hidden_dim)),
            nn.Tanh(),
            nn.Linear(int(hidden_dim), int(out_dim)),
        )

    def forward(self, scale_scalar):
        return self.net(scale_scalar.unsqueeze(-1))


class FinalScaleAdapter(nn.Module):
    """Low-rank scale-conditioned correction to the final readout."""

    def __init__(self, d_model, rank=8, hidden_dim=16):
        super().__init__()
        rank = max(1, int(rank))
        self.features = nn.Linear(d_model, rank, bias=False)
        self.coeffs = nn.Sequential(
            nn.Linear(1, int(hidden_dim)),
            nn.Tanh(),
            nn.Linear(int(hidden_dim), rank + 1),
        )
        # Start as the base readout exactly; scale becomes an opt-in adapter.
        nn.init.zeros_(self.coeffs[-1].weight)
        nn.init.zeros_(self.coeffs[-1].bias)

    def forward(self, h_tgt, scale_scalar):
        params = self.coeffs(scale_scalar.unsqueeze(-1))
        coeffs = params[:, :-1]
        bias = params[:, -1]
        correction = (self.features(h_tgt) * coeffs.unsqueeze(1)).sum(-1)
        return correction + bias.unsqueeze(-1)


class SelfAttnBlock(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        ffn_mult=2,
        mask_tgt_tgt=False,
        finite_attn_mask=False,
        attn_logit_clip=30.0,
        scale_ffn_dim=0,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dk = d_model // n_heads
        self.mask_tgt_tgt = mask_tgt_tgt
        self.finite_attn_mask = finite_attn_mask
        self.attn_logit_clip = float(attn_logit_clip)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Linear(ffn_mult * d_model, d_model),
        )
        self.scale_ffn_bias = None
        if int(scale_ffn_dim) > 0:
            self.scale_ffn_bias = nn.Linear(int(scale_ffn_dim), ffn_mult * d_model, bias=False)
            nn.init.zeros_(self.scale_ffn_bias.weight)

    def forward(self, x, return_attn=False, knockout=None, n_ctx=None,
                skip_ffn=False, zero_heads=None, return_residuals=False,
                uniform_attn=False, attn_gate=None, ffn_gate=None,
                scale_embedding=None):
        B, S, D = x.shape
        h, dk = self.h, self.dk
        qkv = self.qkv(x).view(B, S, 3, h, dk).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        logits = q @ k.transpose(-2, -1) / math.sqrt(dk)
        # Keep raw no-LN attention numerically stable on long-context runs.
        if self.finite_attn_mask:
            logits = logits.clamp(min=-self.attn_logit_clip, max=self.attn_logit_clip)
            mask_value = -self.attn_logit_clip
        else:
            logits = logits.clamp(max=self.attn_logit_clip)
            mask_value = float("-inf")
        if self.mask_tgt_tgt and n_ctx is not None and n_ctx < S:
            logits[:, :, :, n_ctx:] = mask_value
        attn = torch.softmax(logits, dim=-1)
        if uniform_attn:
            attn = torch.ones_like(attn) / S
        if zero_heads is not None:
            for h_idx in zero_heads:
                attn[:, h_idx, :, :] = 0
        if knockout and n_ctx is not None:
            _blocks = {
                "ctx_ctx": (slice(None, n_ctx), slice(None, n_ctx)),
                "ctx_tgt": (slice(None, n_ctx), slice(n_ctx, None)),
                "tgt_ctx": (slice(n_ctx, None), slice(None, n_ctx)),
                "tgt_tgt": (slice(n_ctx, None), slice(n_ctx, None)),
            }
            for block in knockout:
                r, c = _blocks[block]
                attn[:, :, r, c] = 0
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        out = (attn @ v).transpose(1, 2).reshape(B, S, D)
        attn_delta = self.wo(out)
        if attn_gate is not None:
            attn_delta = attn_delta * attn_gate.view(B, 1, 1)
        x = x + attn_delta
        if not skip_ffn:
            if self.scale_ffn_bias is None or scale_embedding is None:
                ffn_delta = self.ff(x)
            else:
                ffn_hidden = self.ff[0](x)
                ffn_hidden = ffn_hidden + self.scale_ffn_bias(scale_embedding).unsqueeze(1)
                ffn_delta = self.ff[2](self.ff[1](ffn_hidden))
        else:
            ffn_delta = torch.zeros_like(x)
        if not skip_ffn and ffn_gate is not None:
            ffn_delta = ffn_delta * ffn_gate.view(B, 1, 1)
        x = x + ffn_delta if not skip_ffn else x
        if return_residuals:
            return x, attn, attn_delta, ffn_delta
        if return_attn:
            return x, attn  # attn: (B, H, S, S)
        return x


class ICLTransformer(nn.Module):
    """
    Transformer for in-context linear regression.

    Token format:
      context: (x_i ∈ R^d, y_i, 0)  →  R^{d+2}
      target:  (x_i ∈ R^d, 0,   1)  →  R^{d+2}

    Forward pass returns predictions at all target positions: (B, n_targets).
    """

    def __init__(
        self,
        d_x,
        d_model,
        n_layers,
        n_heads,
        ffn_mult=2,
        mask_tgt_tgt=False,
        finite_attn_mask=False,
        attn_logit_clip=30.0,
        scale_canonical=False,
        scale_stat="mean_x2",
        scale_eps=1e-8,
        scale_y=True,
        scale_controller="none",
        scale_gate_hidden=16,
        scale_gate_bound=3.0,
        scale_log_clip=8.0,
        scale_conditioner="none",
        scale_condition_dim=8,
        scale_condition_hidden=16,
    ):
        super().__init__()
        self.d_x = d_x
        self.mask_tgt_tgt = mask_tgt_tgt
        self.scale_canonical = bool(scale_canonical)
        self.scale_stat = str(scale_stat)
        self.scale_eps = float(scale_eps)
        self.scale_y = bool(scale_y)
        self.scale_controller_type = str(scale_controller).lower()
        self.scale_log_clip = float(scale_log_clip)
        self.scale_conditioner_type = str(scale_conditioner).lower()
        self.scale_condition_dim = int(scale_condition_dim)
        self.embed = nn.Linear(d_x + 2, d_model)
        scale_ffn_dim = self.scale_condition_dim if self.scale_conditioner_type == "ffn_bias" else 0
        self.layers = nn.ModuleList(
            [
                SelfAttnBlock(
                    d_model,
                    n_heads,
                    ffn_mult,
                    mask_tgt_tgt=mask_tgt_tgt,
                    finite_attn_mask=finite_attn_mask,
                    attn_logit_clip=attn_logit_clip,
                    scale_ffn_dim=scale_ffn_dim,
                )
                for _ in range(n_layers)
            ]
        )
        self.head = nn.Linear(d_model, 1)

        if self.scale_controller_type == "none":
            self.scale_controller = None
        elif self.scale_controller_type == "layer_gates":
            self.scale_controller = ScaleController(
                n_layers=n_layers,
                hidden_dim=scale_gate_hidden,
                gate_bound=scale_gate_bound,
            )
        else:
            raise ValueError(f"unknown scale_controller: {scale_controller}")

        if self.scale_conditioner_type == "none":
            self.scale_embedding = None
            self.final_scale_adapter = None
        elif self.scale_conditioner_type == "ffn_bias":
            self.scale_embedding = ScaleEmbedding(
                out_dim=scale_condition_dim,
                hidden_dim=scale_condition_hidden,
            )
            self.final_scale_adapter = None
        elif self.scale_conditioner_type == "final_adapter":
            self.scale_embedding = None
            self.final_scale_adapter = FinalScaleAdapter(
                d_model=d_model,
                rank=scale_condition_dim,
                hidden_dim=scale_condition_hidden,
            )
        else:
            raise ValueError(f"unknown scale_conditioner: {scale_conditioner}")

    def _uses_scale_features(self):
        return self.scale_canonical or self.scale_controller is not None or self.scale_conditioner_type != "none"

    def _estimate_s_hat(self, x_ctx):
        if self.scale_stat != "mean_x2":
            raise ValueError(f"unknown scale_stat: {self.scale_stat}")
        return x_ctx.pow(2).mean(dim=(1, 2)).clamp_min(self.scale_eps)

    def _prepare_scale(self, x_ctx, y_ctx, x_tgt):
        if not self._uses_scale_features():
            return x_ctx, y_ctx, x_tgt, None, None

        s_hat = self._estimate_s_hat(x_ctx)
        sqrt_s_hat = torch.sqrt(s_hat).view(-1, 1, 1)
        raw_log10_s_hat = torch.log10(s_hat)
        log10_s_hat = raw_log10_s_hat.clamp(
            min=-self.scale_log_clip,
            max=self.scale_log_clip,
        )
        scale_info = {
            "s_hat": s_hat,
            "sqrt_s_hat": sqrt_s_hat.view(-1),
            "raw_log10_s_hat": raw_log10_s_hat,
            "log10_s_hat": log10_s_hat,
            "log10_inv_s_hat": -log10_s_hat,
        }

        output_scale = None
        if self.scale_canonical:
            x_ctx = x_ctx / sqrt_s_hat
            x_tgt = x_tgt / sqrt_s_hat
            if self.scale_y:
                y_scale = sqrt_s_hat.squeeze(-1)
                y_ctx = y_ctx / y_scale
                output_scale = y_scale

        return x_ctx, y_ctx, x_tgt, output_scale, scale_info

    def _compute_gates(self, scale_info):
        if self.scale_controller is None:
            return None, None
        gates = self.scale_controller(scale_info["log10_s_hat"])
        return gates[:, :, 0], gates[:, :, 1]

    def _compute_scale_embedding(self, scale_info):
        if self.scale_embedding is None:
            return None
        return self.scale_embedding(scale_info["log10_inv_s_hat"])

    def _apply_final_scale_adapter(self, base_preds, h_tgt, scale_info):
        if self.final_scale_adapter is None or scale_info is None:
            return base_preds
        return base_preds + self.final_scale_adapter(h_tgt, scale_info["log10_inv_s_hat"])

    @staticmethod
    def _gate_at(gates, layer_idx):
        if gates is None:
            return None
        return gates[:, layer_idx]

    @staticmethod
    def _rescale_preds(preds, output_scale):
        if output_scale is None:
            return preds
        return preds * output_scale

    def forward(self, x_ctx, y_ctx, x_tgt, return_internals=False,
                knockout_spec=None, ffn_knockout_layers=None,
                head_knockout_spec=None, uniform_attn_layers=None):
        """
        x_ctx: (B, n_ctx, d)
        y_ctx: (B, n_ctx)
        x_tgt: (B, n_tgt, d)
        knockout_spec: dict mapping layer index → set of blocks to zero
        ffn_knockout_layers: set of layer indices where FFN is skipped
        head_knockout_spec: dict mapping layer index → set of head indices to zero
        uniform_attn_layers: set of layer indices where attention is replaced with 1/S

        Returns: (B, n_tgt)  or  ((B, n_tgt), internals_dict) if return_internals=True
        """
        B, n_ctx, d = x_ctx.shape
        n_tgt = x_tgt.shape[1]
        x_ctx, y_ctx, x_tgt, output_scale, scale_info = self._prepare_scale(
            x_ctx, y_ctx, x_tgt
        )
        attn_gates, ffn_gates = self._compute_gates(scale_info) if scale_info else (None, None)
        scale_embedding = self._compute_scale_embedding(scale_info) if scale_info else None

        # context tokens: [x_i, y_i, 0]
        ctx = torch.cat([
            x_ctx,
            y_ctx.unsqueeze(-1),
            torch.zeros(B, n_ctx, 1, device=x_ctx.device, dtype=x_ctx.dtype),
        ], dim=-1)

        # target tokens: [x_i, 0, 1]
        tgt = torch.cat([
            x_tgt,
            torch.zeros(B, n_tgt, 1, device=x_tgt.device, dtype=x_tgt.dtype),
            torch.ones(B, n_tgt, 1, device=x_tgt.device, dtype=x_tgt.dtype),
        ], dim=-1)

        h = self.embed(torch.cat([ctx, tgt], dim=1))  # (B, n_ctx+n_tgt, d_model)

        if return_internals:
            internals = {"layer_preds": [], "attentions": [],
                         "hidden_states": [], "ctx_hidden_states": []}
            if scale_info is not None:
                internals["scale"] = {k: v.detach() for k, v in scale_info.items()}
            if attn_gates is not None:
                internals["attn_gates"] = []
                internals["ffn_gates"] = []
            for i, layer in enumerate(self.layers):
                ko = knockout_spec.get(i) if knockout_spec else None
                sf = ffn_knockout_layers and i in ffn_knockout_layers
                zh = head_knockout_spec.get(i) if head_knockout_spec else None
                ua = uniform_attn_layers and i in uniform_attn_layers
                h, attn = layer(h, return_attn=True, knockout=ko, n_ctx=n_ctx,
                                skip_ffn=sf, zero_heads=zh, uniform_attn=ua,
                                attn_gate=self._gate_at(attn_gates, i),
                                ffn_gate=self._gate_at(ffn_gates, i),
                                scale_embedding=scale_embedding)
                layer_pred = self.head(h[:, n_ctx:, :]).squeeze(-1)  # (B, n_tgt)
                layer_pred = self._apply_final_scale_adapter(layer_pred, h[:, n_ctx:, :], scale_info)
                layer_pred = self._rescale_preds(layer_pred, output_scale)
                internals["layer_preds"].append(layer_pred)
                internals["attentions"].append(attn)
                internals["hidden_states"].append(h[:, n_ctx:, :].detach())
                internals["ctx_hidden_states"].append(h[:, :n_ctx, :].detach())
                if attn_gates is not None:
                    internals["attn_gates"].append(attn_gates[:, i].detach())
                    internals["ffn_gates"].append(ffn_gates[:, i].detach())
            final_preds = internals["layer_preds"][-1]
            return final_preds, internals

        for i, layer in enumerate(self.layers):
            ko = knockout_spec.get(i) if knockout_spec else None
            sf = ffn_knockout_layers and i in ffn_knockout_layers
            zh = head_knockout_spec.get(i) if head_knockout_spec else None
            ua = uniform_attn_layers and i in uniform_attn_layers
            h = layer(h, knockout=ko, n_ctx=n_ctx, skip_ffn=sf,
                      zero_heads=zh, uniform_attn=ua,
                      attn_gate=self._gate_at(attn_gates, i),
                      ffn_gate=self._gate_at(ffn_gates, i),
                      scale_embedding=scale_embedding)

        # predictions at target positions
        h_tgt = h[:, n_ctx:, :]  # (B, n_tgt, d_model)
        preds = self.head(h_tgt).squeeze(-1)  # (B, n_tgt)
        preds = self._apply_final_scale_adapter(preds, h_tgt, scale_info)
        return self._rescale_preds(preds, output_scale)

    @torch.no_grad()
    def decompose(self, x_ctx, y_ctx, x_tgt):
        """Return per-layer residual stream decomposition.

        Returns dict with:
            h0:          (B, S, d_model)  embedding output
            attn_deltas: list of (B, S, d_model) per layer
            ffn_deltas:  list of (B, S, d_model) per layer
            attentions:  list of (B, H, S, S) per layer
        """
        B, n_ctx, d = x_ctx.shape
        n_tgt = x_tgt.shape[1]
        x_ctx, y_ctx, x_tgt, output_scale, scale_info = self._prepare_scale(
            x_ctx, y_ctx, x_tgt
        )
        attn_gates, ffn_gates = self._compute_gates(scale_info) if scale_info else (None, None)
        scale_embedding = self._compute_scale_embedding(scale_info) if scale_info else None
        ctx = torch.cat([x_ctx, y_ctx.unsqueeze(-1),
                         torch.zeros(B, n_ctx, 1, device=x_ctx.device, dtype=x_ctx.dtype)], dim=-1)
        tgt = torch.cat([x_tgt, torch.zeros(B, n_tgt, 1, device=x_tgt.device, dtype=x_tgt.dtype),
                         torch.ones(B, n_tgt, 1, device=x_tgt.device, dtype=x_tgt.dtype)], dim=-1)
        h = self.embed(torch.cat([ctx, tgt], dim=1))
        result = {"h0": h.detach(), "attn_deltas": [], "ffn_deltas": [], "attentions": []}
        if scale_info is not None:
            result["scale"] = {k: v.detach() for k, v in scale_info.items()}
        if output_scale is not None:
            result["output_scale"] = output_scale.detach()
        if attn_gates is not None:
            result["attn_gates"] = []
            result["ffn_gates"] = []
        for i, layer in enumerate(self.layers):
            h, attn, attn_d, ffn_d = layer(
                h,
                return_residuals=True,
                n_ctx=n_ctx,
                attn_gate=self._gate_at(attn_gates, i),
                ffn_gate=self._gate_at(ffn_gates, i),
                scale_embedding=scale_embedding,
            )
            result["attn_deltas"].append(attn_d.detach())
            result["ffn_deltas"].append(ffn_d.detach())
            result["attentions"].append(attn.detach())
            if attn_gates is not None:
                result["attn_gates"].append(attn_gates[:, i].detach())
                result["ffn_gates"].append(ffn_gates[:, i].detach())
        return result
