"""
"Dynamical Systems Theory Behind a Hierarchical Reasoning Model"
(Es'kin & Smorkalov, 2026)

Using stablemax instead of softmax
for numerical stability inside the reasoning model.

  - StableMax3/5 polynomial softmax approximations
  - SelfAttention module using stablemax
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# StableMax variants (Section 3.2.3, Eqs 35-40)
# ============================================================

def _stablemax_s(x: torch.Tensor) -> torch.Tensor:
    """Original StableMax base function s(x), Eq 36."""
    pos = 1.0 + x
    neg = 1.0 / (1.0 - x)
    return torch.where(x >= 0, pos, neg)


def _stablemax3_s(x: torch.Tensor) -> torch.Tensor:
    """StableMax3 base function s_3(x), Eq 38.
    3rd-order Taylor approx of exp(x)."""
    pos = 1.0 + x * (1.0 + 0.5 * x * (1.0 + x / 3.0))
    neg = 1.0 / (1.0 - x * (1.0 - 0.5 * x * (1.0 - x / 3.0)))
    return torch.where(x >= 0, pos, neg)


def _stablemax5_s(x: torch.Tensor) -> torch.Tensor:
    """StableMax5 base function s_5(x), Eq 40.
    5th-order Taylor approx of exp(x)."""
    pos = 1.0 + x * (1.0 + 0.5 * x * (1.0 + x * (1.0 + x / 5.0) / 4.0 / 3.0))
    neg_inner = 1.0 - x * (1.0 - 0.5 * x * (1.0 - x * (1.0 - x / 5.0) / 4.0 / 3.0))
    return torch.where(x >= 0, pos, 1.0 / neg_inner)


def stablemax(logits: torch.Tensor, variant: str = "3", dim: int = -1) -> torch.Tensor:
    """StableMax normalization (replaces softmax). Eqs 35, 37, 39."""
    # Subtract max for numerical stability (same trick as softmax)
    logits = logits - logits.max(dim=dim, keepdim=True).values
    if variant == "1":
        s = _stablemax_s(logits)
    elif variant == "3":
        s = _stablemax3_s(logits)
    elif variant == "5":
        s = _stablemax5_s(logits)
    else:
        raise ValueError(f"Unknown StableMax variant: {variant}")
    return s / s.sum(dim=dim, keepdim=True)


def stablemax_cross_entropy(logits: torch.Tensor, targets: torch.Tensor,
                            variant: str = "3") -> torch.Tensor:
    """Cross-entropy using StableMax instead of softmax."""
    probs = stablemax(logits, variant=variant)
    # Clamp for log stability
    log_probs = torch.log(probs.clamp(min=1e-10))
    return F.nll_loss(log_probs.view(-1, logits.size(-1)),
                      targets.view(-1), reduction="mean")


# ============================================================
# MLP-Mixer layer (token-mixing alternative to attention)
# ============================================================

class MLPMixer(nn.Module):
    """MLP-mixer: token-mixing MLP applied across the sequence dimension."""
    def __init__(self, seq_len: int, dim: int, expansion: int = 4):
        super().__init__()
        self.token_mix = nn.Sequential(
            nn.Linear(seq_len, seq_len * expansion),
            nn.Tanh(),
            nn.Linear(seq_len * expansion, seq_len),
        )

    def forward(self, x):
        # x: (B, S, D) → transpose → mix tokens → transpose back
        return self.token_mix(x.transpose(-1, -2)).transpose(-1, -2)


# ============================================================
# Self-Attention with optional StableMax
# ============================================================

class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4,
                 use_stablemax: str = "none"):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.use_stablemax = use_stablemax

    def forward(self, x, pos):
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # 3 x (B, H, S, D_h)
        q = pos(q)
        k = pos(k)

        if self.use_stablemax != "none":
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = stablemax(attn, variant=self.use_stablemax, dim=-1)
            out = attn @ v
        else:
            out = F.scaled_dot_product_attention(q, k, v)

        out = out.transpose(1, 2).reshape(B, S, D)
        return self.out_proj(out)


