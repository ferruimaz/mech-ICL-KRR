"""
Transformer for in-context linear regression.

Architecture:
  - Input projection: linear (d+2) → d_model
  - Transformer body: multi-head self-attention + FFN, residual connections, NO LayerNorm
  - Output head: linear d_model → 1, applied at target positions
"""

import math
import torch
import torch.nn as nn


class SelfAttnBlock(nn.Module):
    def __init__(self, d_model, n_heads, ffn_mult=2):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dk = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Linear(ffn_mult * d_model, d_model),
        )

    def forward(self, x, return_attn=False, knockout=None, n_ctx=None,
                skip_ffn=False, zero_heads=None, return_residuals=False,
                uniform_attn=False):
        B, S, D = x.shape
        h, dk = self.h, self.dk
        qkv = self.qkv(x).view(B, S, 3, h, dk).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(dk), dim=-1)
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
        x = x + attn_delta
        ffn_delta = self.ff(x) if not skip_ffn else torch.zeros_like(x)
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

    def __init__(self, d_x, d_model, n_layers, n_heads, ffn_mult=2):
        super().__init__()
        self.d_x = d_x
        self.embed = nn.Linear(d_x + 2, d_model)
        self.layers = nn.ModuleList(
            [SelfAttnBlock(d_model, n_heads, ffn_mult) for _ in range(n_layers)]
        )
        self.head = nn.Linear(d_model, 1)

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

        # context tokens: [x_i, y_i, 0]
        ctx = torch.cat([
            x_ctx,
            y_ctx.unsqueeze(-1),
            torch.zeros(B, n_ctx, 1, device=x_ctx.device),
        ], dim=-1)

        # target tokens: [x_i, 0, 1]
        tgt = torch.cat([
            x_tgt,
            torch.zeros(B, n_tgt, 1, device=x_tgt.device),
            torch.ones(B, n_tgt, 1, device=x_tgt.device),
        ], dim=-1)

        h = self.embed(torch.cat([ctx, tgt], dim=1))  # (B, n_ctx+n_tgt, d_model)

        if return_internals:
            internals = {"layer_preds": [], "attentions": [],
                         "hidden_states": [], "ctx_hidden_states": []}
            for i, layer in enumerate(self.layers):
                ko = knockout_spec.get(i) if knockout_spec else None
                sf = ffn_knockout_layers and i in ffn_knockout_layers
                zh = head_knockout_spec.get(i) if head_knockout_spec else None
                ua = uniform_attn_layers and i in uniform_attn_layers
                h, attn = layer(h, return_attn=True, knockout=ko, n_ctx=n_ctx,
                                skip_ffn=sf, zero_heads=zh, uniform_attn=ua)
                layer_pred = self.head(h[:, n_ctx:, :]).squeeze(-1)  # (B, n_tgt)
                internals["layer_preds"].append(layer_pred)
                internals["attentions"].append(attn)
                internals["hidden_states"].append(h[:, n_ctx:, :].detach())
                internals["ctx_hidden_states"].append(h[:, :n_ctx, :].detach())
            final_preds = internals["layer_preds"][-1]
            return final_preds, internals

        for i, layer in enumerate(self.layers):
            ko = knockout_spec.get(i) if knockout_spec else None
            sf = ffn_knockout_layers and i in ffn_knockout_layers
            zh = head_knockout_spec.get(i) if head_knockout_spec else None
            ua = uniform_attn_layers and i in uniform_attn_layers
            if ko or sf or zh or ua:
                h = layer(h, knockout=ko, n_ctx=n_ctx, skip_ffn=sf,
                          zero_heads=zh, uniform_attn=ua)
            else:
                h = layer(h)

        # predictions at target positions
        h_tgt = h[:, n_ctx:, :]  # (B, n_tgt, d_model)
        return self.head(h_tgt).squeeze(-1)  # (B, n_tgt)

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
        ctx = torch.cat([x_ctx, y_ctx.unsqueeze(-1),
                         torch.zeros(B, n_ctx, 1, device=x_ctx.device)], dim=-1)
        tgt = torch.cat([x_tgt, torch.zeros(B, n_tgt, 1, device=x_tgt.device),
                         torch.ones(B, n_tgt, 1, device=x_tgt.device)], dim=-1)
        h = self.embed(torch.cat([ctx, tgt], dim=1))
        result = {"h0": h.detach(), "attn_deltas": [], "ffn_deltas": [], "attentions": []}
        for layer in self.layers:
            h, attn, attn_d, ffn_d = layer(h, return_residuals=True)
            result["attn_deltas"].append(attn_d.detach())
            result["ffn_deltas"].append(ffn_d.detach())
            result["attentions"].append(attn.detach())
        return result
