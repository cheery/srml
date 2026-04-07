"""
Deep Delta Learning (DDL) — PyTorch Implementation
====================================================
Based on: "Deep Delta Learning" (Zhang et al., 2026)
arXiv:2601.00417

Implements the Delta Residual block that generalizes the identity shortcut
connection to a learnable rank-1 perturbation:

    A(X) = I - β(X) k(X) k(X)^T

    X_{l+1} = A(X_l) X_l + β(X_l) k(X_l) v(X_l)^T

Supports both d_v=1 (scalar, drop-in replacement for standard residual)
and d_v>1 (expanded matrix-valued hidden state).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=4096, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (T, dim/2)
        return freqs.cos(), freqs.sin()


def apply_rotary_pos_emb(x, cos, sin):
    """x: (B, n_heads, T, head_dim). cos/sin: (T, head_dim/2)"""
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    cos = cos[:x.shape[-2]].unsqueeze(0).unsqueeze(0)  # (1,1,T,d2)
    sin = sin[:x.shape[-2]].unsqueeze(0).unsqueeze(0)
    out = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return out


# ---------------------------------------------------------------------------
# Core DDL components
# ---------------------------------------------------------------------------

class DeltaGate(nn.Module):
    """
    Computes β(X) ∈ (0, 2) via  β = 2 · σ(Linear(c))  or
    β = 2 · σ(Linear(tanh(Linear(c))))  where c = RMSNorm(x_in).

    Initializes bias so that β starts near `beta_init`.
    """
    def __init__(self, d_model, hidden_size=0, beta_init=0.5, eps=1e-6):
        super().__init__()
        self.norm = RMSNorm(d_model, eps=eps)
        if hidden_size > 0:
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden_size, bias=False),
                nn.Tanh(),
                nn.Linear(hidden_size, 1, bias=True),
            )
            # init output bias
            nn.init.zeros_(self.net[0].weight)
            nn.init.zeros_(self.net[2].weight)
            self.net[2].bias.data.fill_(math.log(beta_init / 2.0 / (1.0 - beta_init / 2.0 + 1e-8)))
        else:
            self.net = nn.Linear(d_model, 1, bias=True)
            nn.init.zeros_(self.net.weight)
            self.net.bias.data.fill_(math.log(beta_init / 2.0 / (1.0 - beta_init / 2.0 + 1e-8)))

    def forward(self, x):
        """x: (B, T, d) or (B, T, d, d_v) → β: (B, T, 1)"""
        if x.dim() == 4:
            # expanded state: pool over value channels
            x = x.mean(dim=-1)
        c = self.norm(x)
        logit = self.net(c)  # (B, T, 1)
        # compute in float for numerical stability
        beta = 2.0 * torch.sigmoid(logit.float()).to(x.dtype)
        return beta


class DeltaResidualScalar(nn.Module):
    """
    DDL residual block for d_v = 1 (scalar value regime).

    Replaces:  x_{l+1} = x_l + F(x_l)
    With:      x_{l+1} = x_l + β_l (v_l - k_l^T x_l) k_l

    Uses k-Map: the sublayer output F(x_ctx) IS the unnormalized k direction.
    The scalar value v_l = σ(w_v^T x_l) is a separate linear projection.
    """
    def __init__(self, d_model, beta_init=0.5, beta_hidden_size=0, eps_k=1e-6):
        super().__init__()
        self.eps_k = eps_k
        self.k_scale = 1.0 / math.sqrt(d_model)
        self.gate = DeltaGate(d_model, hidden_size=beta_hidden_size, beta_init=beta_init)
        # value projection: scalar v = σ(w_v^T x_in)
        self.w_v = nn.Linear(d_model, 1, bias=False)

    def forward(self, x, sublayer_output):
        """
        x: (B, T, d) — current hidden state
        sublayer_output: (B, T, d) — output of attention or MLP sublayer (= k_tilde)

        Returns: updated x of shape (B, T, d)
        """
        # k direction: normalize sublayer output via RMSNorm + scale
        k_tilde = sublayer_output
        # RMS norm to get unit direction, then scale by 1/sqrt(d)
        k_hat = F.normalize(k_tilde.float(), dim=-1, eps=self.eps_k).to(x.dtype)
        k = k_hat * self.k_scale  # (B, T, d)

        # β gate
        beta = self.gate(x)  # (B, T, 1)

        # scalar value
        v = torch.sigmoid(self.w_v(x))  # (B, T, 1)

        # projection of x onto k: k^T x  (dot product per token)
        kTx = (k * x).sum(dim=-1, keepdim=True)  # (B, T, 1)

        # delta update: x + β * (v - k^T x) * k
        update = beta * (v - kTx) * k  # (B, T, d)
        return x + update


class CausalDepthwiseConv1d(nn.Module):
    """Causal depthwise conv along the time axis."""
    def __init__(self, channels, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=channels,
            bias=False,
        )

    def forward(self, x):
        """x: (B, T, C) → (B, T, C)"""
        x = x.transpose(1, 2)  # (B, C, T)
        x = self.conv(x)
        if self.kernel_size > 1:
            x = x[..., :-(self.kernel_size - 1)]
        return x.transpose(1, 2)


class ResidualShortConvCompressor(nn.Module):
    """
    Compress expanded state (B, T, d, d_v) → (B, T, d) for sublayer input.
    Uses causal depthwise conv along T, then learned weighted pooling over d_v.
    """
    def __init__(self, d_model, d_v, shortconv_kernel_size=4):
        super().__init__()
        self.d_v = d_v
        # depthwise conv over time with d*d_v channels
        self.conv = CausalDepthwiseConv1d(d_model * d_v, shortconv_kernel_size)
        # learned read vector w_p
        self.w_p = nn.Parameter(torch.ones(d_v) / d_v)

    def forward(self, X):
        """X: (B, T, d, d_v) → x_in: (B, T, d)"""
        B, T, d, dv = X.shape
        # flatten last two dims for conv
        h = X.reshape(B, T, d * dv)
        h = self.conv(h)  # (B, T, d*d_v)
        h = h.reshape(B, T, d, dv)
        # weighted pool over value channels
        x_in = (h * self.w_p.view(1, 1, 1, dv)).sum(dim=-1)  # (B, T, d)
        return x_in


class DeltaResidualExpanded(nn.Module):
    """
    DDL residual block for d_v > 1 (expanded matrix-valued state).

    State X ∈ R^{d × d_v}. Update:
        X_{l+1} = X_l + β_l k_l (v_l^T - k_l^T X_l)

    Uses k-Map: sublayer output h_l = k_tilde.
    Value v_l = W_v @ x_in  where x_in is the compressed input.
    """
    def __init__(self, d_model, d_v, beta_init=0.5, beta_hidden_size=0,
                 shortconv_kernel_size=4, eps_k=1e-6):
        super().__init__()
        self.d_model = d_model
        self.d_v = d_v
        self.eps_k = eps_k
        self.k_scale = 1.0 / math.sqrt(d_model)

        # compressor: (B,T,d,d_v) → (B,T,d)
        self.compressor = ResidualShortConvCompressor(d_model, d_v, shortconv_kernel_size)

        # gate operates on compressed input
        self.gate = DeltaGate(d_model, hidden_size=beta_hidden_size, beta_init=beta_init)

        # value projection: x_in (d) → v (d_v)
        self.W_v = nn.Linear(d_model, d_v, bias=False)

    def compress(self, X):
        """Compress expanded state for sublayer input."""
        return self.compressor(X)

    def forward(self, X, sublayer_output, x_in):
        """
        X: (B, T, d, d_v) — expanded hidden state
        sublayer_output: (B, T, d) — output of sublayer (= k_tilde = h_l)
        x_in: (B, T, d) — compressed input (from self.compress)

        Returns: updated X of shape (B, T, d, d_v)
        """
        # k direction
        k_tilde = sublayer_output
        k_hat = F.normalize(k_tilde.float(), dim=-1, eps=self.eps_k).to(X.dtype)
        k = k_hat * self.k_scale  # (B, T, d)

        # β gate from compressed input
        beta = self.gate(x_in)  # (B, T, 1)

        # value vector
        v = self.W_v(x_in)  # (B, T, d_v)

        # k^T X: project each value column onto k → (B, T, d_v)
        # k: (B,T,d), X: (B,T,d,d_v) → kTX = einsum('btd,btdv->btv')
        kTX = torch.einsum('btd,btdv->btv', k, X)

        # correction: v^T - k^T X  → (B, T, d_v)
        correction = v - kTX  # (B, T, d_v)

        # rank-1 update: β * k ⊗ correction
        # k: (B,T,d,1) * correction: (B,T,1,d_v) → (B,T,d,d_v)
        update = beta.unsqueeze(-1) * k.unsqueeze(-1) * correction.unsqueeze(-2)

        return X + update


# ---------------------------------------------------------------------------
# Transformer sublayers
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len=4096):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # QK normalization for stability
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)

    def forward(self, x, mask=None):
        B, T, d = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # QK norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE
        cos, sin = self.rotary(T, x.device)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        # scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        if mask is None:
            mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
        attn = attn + mask.unsqueeze(0).unsqueeze(0)
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, T, d)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            # standard 8/3 expansion, rounded to multiple of 64
            hidden_dim = int(2 * (4 * d_model) / 3)
            hidden_dim = 64 * ((hidden_dim + 63) // 64)
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ---------------------------------------------------------------------------
# DDL Transformer blocks
# ---------------------------------------------------------------------------

class DDLTransformerBlock(nn.Module):
    """
    Single transformer block with DDL residual connections (d_v = 1).

    Pre-norm architecture:
        x_ctx = RMSNorm(x)
        h = Sublayer(x_ctx)   (attention or MLP)
        x_next = DeltaRes(x, h)
    """
    def __init__(self, d_model, n_heads, max_seq_len=4096,
                 beta_init=0.5, beta_hidden_size=0):
        super().__init__()
        # Attention sublayer
        self.attn_norm = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, max_seq_len)
        self.attn_delta = DeltaResidualScalar(d_model, beta_init=beta_init,
                                               beta_hidden_size=beta_hidden_size)

        # MLP sublayer
        self.mlp_norm = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model)
        self.mlp_delta = DeltaResidualScalar(d_model, beta_init=beta_init,
                                              beta_hidden_size=beta_hidden_size)

    def forward(self, x, mask=None):
        # Attention with DDL residual
        h_attn = self.attn(self.attn_norm(x), mask=mask)
        x = self.attn_delta(x, h_attn)

        # MLP with DDL residual
        h_mlp = self.mlp(self.mlp_norm(x))
        x = self.mlp_delta(x, h_mlp)

        return x


class DDLTransformerBlockExpanded(nn.Module):
    """
    Single transformer block with DDL residual connections (d_v > 1).

    Uses Compress-Process-Expand protocol:
    1. Compress: X (B,T,d,d_v) → x_in (B,T,d)
    2. Process:  h = Sublayer(RMSNorm(x_in))
    3. Expand:   X_next = DeltaResExpanded(X, h, x_in)
    """
    def __init__(self, d_model, d_v, n_heads, max_seq_len=4096,
                 beta_init=0.5, beta_hidden_size=0, shortconv_kernel_size=4):
        super().__init__()
        # Attention sublayer
        self.attn_norm = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, max_seq_len)
        self.attn_delta = DeltaResidualExpanded(
            d_model, d_v, beta_init=beta_init,
            beta_hidden_size=beta_hidden_size,
            shortconv_kernel_size=shortconv_kernel_size,
        )

        # MLP sublayer
        self.mlp_norm = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model)
        self.mlp_delta = DeltaResidualExpanded(
            d_model, d_v, beta_init=beta_init,
            beta_hidden_size=beta_hidden_size,
            shortconv_kernel_size=shortconv_kernel_size,
        )

    def forward(self, X, mask=None):
        """X: (B, T, d, d_v)"""
        # Attention with DDL residual
        x_in = self.attn_delta.compress(X)          # (B,T,d)
        h_attn = self.attn(self.attn_norm(x_in), mask=mask)
        X = self.attn_delta(X, h_attn, x_in)

        # MLP with DDL residual
        x_in = self.mlp_delta.compress(X)            # (B,T,d)
        h_mlp = self.mlp(self.mlp_norm(x_in))
        X = self.mlp_delta(X, h_mlp, x_in)

        return X


# ---------------------------------------------------------------------------
# Full DDL Transformer models
# ---------------------------------------------------------------------------

class DDLTransformer(nn.Module):
    """
    DDL Transformer language model with d_v = 1 (drop-in residual replacement).
    """
    def __init__(self, vocab_size, d_model=768, n_layers=12, n_heads=6,
                 max_seq_len=1024, beta_init=0.5, beta_hidden_size=0):
        super().__init__()
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            DDLTransformerBlock(d_model, n_heads, max_seq_len,
                                beta_init=beta_init,
                                beta_hidden_size=beta_hidden_size)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # weight tying
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        idx: (B, T) token indices
        targets: (B, T) optional targets for loss computation
        """
        B, T = idx.shape
        x = self.tok_emb(idx)  # (B, T, d)

        mask = torch.triu(torch.full((T, T), float('-inf'), device=idx.device), diagonal=1)
        for block in self.blocks:
            x = block(x, mask=mask)

        x = self.norm(x)
        logits = self.head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss


class DDLTransformerExpanded(nn.Module):
    """
    DDL Transformer language model with d_v > 1 (expanded matrix-valued state).
    """
    def __init__(self, vocab_size, d_model=768, d_v=4, n_layers=12, n_heads=6,
                 max_seq_len=1024, beta_init=0.5, beta_hidden_size=0,
                 shortconv_kernel_size=4):
        super().__init__()
        self.d_model = d_model
        self.d_v = d_v
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            DDLTransformerBlockExpanded(
                d_model, d_v, n_heads, max_seq_len,
                beta_init=beta_init,
                beta_hidden_size=beta_hidden_size,
                shortconv_kernel_size=shortconv_kernel_size,
            )
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # weight tying
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        idx: (B, T) token indices
        targets: (B, T) optional targets for loss computation
        """
        B, T = idx.shape
        x_emb = self.tok_emb(idx)  # (B, T, d)

        # expand: repeat embedding across d_v channels
        # X_0 = x_emb @ 1_{d_v}^T → (B, T, d, d_v)
        X = x_emb.unsqueeze(-1).expand(B, T, self.d_model, self.d_v)
        # need contiguous for conv operations
        X = X.contiguous()

        mask = torch.triu(torch.full((T, T), float('-inf'), device=idx.device), diagonal=1)
        for block in self.blocks:
            X = block(X, mask=mask)

        # compress final state: pool over d_v
        x_out = X.mean(dim=-1)  # (B, T, d)
        x_out = self.final_norm(x_out)
        logits = self.head(x_out)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss


# ---------------------------------------------------------------------------
# Convenience constructors matching paper configurations
# ---------------------------------------------------------------------------

def ddl_small(vocab_size, d_v=1, max_seq_len=1024, **kwargs):
    """124M param model (Table 4: 12 layers, 6 heads, d=768, head_dim=128)."""
    if d_v == 1:
        return DDLTransformer(vocab_size, d_model=768, n_layers=12, n_heads=6,
                              max_seq_len=max_seq_len, **kwargs)
    else:
        return DDLTransformerExpanded(vocab_size, d_model=768, d_v=d_v,
                                      n_layers=12, n_heads=6,
                                      max_seq_len=max_seq_len, **kwargs)


def ddl_medium(vocab_size, d_v=1, max_seq_len=1024, **kwargs):
    """353M param model (Table 4: 24 layers, 8 heads, d=1024, head_dim=128)."""
    if d_v == 1:
        return DDLTransformer(vocab_size, d_model=1024, n_layers=24, n_heads=8,
                              max_seq_len=max_seq_len, **kwargs)
    else:
        return DDLTransformerExpanded(vocab_size, d_model=1024, d_v=d_v,
                                      n_layers=24, n_heads=8,
                                      max_seq_len=max_seq_len, **kwargs)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 32000

    print("=== DDL Transformer (d_v=1, small) ===")
    model = ddl_small(vocab_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params / 1e6:.1f}M")

    idx = torch.randint(0, vocab_size, (2, 128), device=device)
    targets = torch.randint(0, vocab_size, (2, 128), device=device)
    logits, loss = model(idx, targets)
    print(f"Logits shape: {logits.shape}, Loss: {loss.item():.4f}")

    print("\n=== DDL Transformer (d_v=4, small) ===")
    model_exp = ddl_small(vocab_size, d_v=4).to(device)
    n_params = sum(p.numel() for p in model_exp.parameters())
    print(f"Parameters: {n_params / 1e6:.1f}M")

    logits, loss = model_exp(idx, targets)
    print(f"Logits shape: {logits.shape}, Loss: {loss.item():.4f}")

    print("\nDone!")
