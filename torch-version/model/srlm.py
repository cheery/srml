import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from .scatter import scatter

@dataclass
class SRLMConfig:
    vocab_size:            int
    context_length:        int
    d_model:               int
    n_priors:              int
    n_posteriors:          int
    n_heads:               int
    d_frequency_embedding: int = 256
    N:                     int = 2
    T:                     int = 4

class BlockAttnRes(nn.Module):
    """Block attention residual: weighted combination over all accumulated blocks."""

    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.RMSNorm(d_model)
        self.proj = nn.Linear(d_model, 1, bias=False)

    def forward(self, blocks: list[torch.Tensor]) -> torch.Tensor:
        V = torch.stack(blocks)           # (N, B, L, D)
        K = self.norm(V)
        logits = torch.einsum('d, n b t d -> n b t', self.proj.weight.squeeze(), K)
        h = torch.einsum('n b t, n b t d -> b t d', logits.softmax(0), V)
        return h


class SRLM(nn.Module):
    def __init__(self, cfg: SRLMConfig):
        super().__init__()
        self.cfg = cfg
        self.input     = InputLayer(cfg)
        self.prior     = nn.ModuleList([
            DiTBlock(cfg.d_model, cfg.d_model, cfg.n_heads, mlp_ratio=4.0)
            for _ in range(cfg.n_priors)
        ])
        self.main      = HRM(cfg)
        self.posterior = nn.ModuleList([
            DiTBlock(cfg.d_model, cfg.d_model, cfg.n_heads, mlp_ratio=4.0)
            for _ in range(cfg.n_posteriors)
        ])
        self.block_res = BlockAttnRes(cfg.d_model)
        self.norm      = AdaLN(cfg.d_model)
        self.output    = OutputLayer(cfg)

    def forward(self, z, x, sigma):
        q, p, t = self.input(x, sigma)
        blocks = [q]

        for layer in self.prior:
            h = self.block_res(blocks)
            y = layer(h, p, t)
            blocks.append(y)

        h = self.block_res(blocks)
        z, y = self.main(z, h, p, t)
        blocks.append(y)

        for layer in self.posterior:
            h = self.block_res(blocks)
            y = layer(h, p, t)
            blocks.append(y)

        h = self.block_res(blocks)
        y = self.norm(h, t)
        return z, self.output(x, y, sigma)

class HRM(nn.Module):
    def __init__(self, cfg: SRLMConfig):
        super().__init__()
        self.N    = cfg.N
        self.T    = cfg.T
        self.fast = FastLayer(cfg)
        self.slow = SlowLayer(cfg)

    def forward(self, z, x, p, t):
        zH, zL = z

        with torch.no_grad():
            for i in range(self.N * self.T - 1):
                zL = self.fast(zH, zL, x, p, t)
                if (i + 1) % self.T == 0:
                    zH = self.slow(zH, zL, p, t)

        zL = self.fast(zH, zL, x, p, t)
        zH = self.slow(zH, zL, p, t)
        return (zH, zL), zH

class FastLayer(nn.Module):
    def __init__(self, cfg: SRLMConfig):
        super().__init__()
        self.normH = AdaLN(cfg.d_model)
        self.normL = AdaLN(cfg.d_model)
        self.normX = AdaLN(cfg.d_model)
        self.inj   = nn.Linear(cfg.d_model * 3, cfg.d_model)
        self.s5d   = DiTBlock(cfg.d_model, cfg.d_model, num_heads=cfg.n_heads, mlp_ratio=4.0)

    def forward(self, zH: torch.Tensor, zL: torch.Tensor, x: torch.Tensor, p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        zH = self.normH(zH, t)
        zL = self.normL(zL, t)
        x  = self.normX(x, t)
        x = torch.cat([zH, zL, x], dim=-1)  # (B, L, 3*D)
        x = self.inj(x)                      # (B, L, D)
        x = self.s5d(x, p, t)
        return x


class SlowLayer(nn.Module):
    def __init__(self, cfg: SRLMConfig):
        super().__init__()
        self.normH = AdaLN(cfg.d_model)
        self.normL = AdaLN(cfg.d_model)
        self.inj   = nn.Linear(cfg.d_model * 2, cfg.d_model)
        self.s5d   = DiTBlock(cfg.d_model, cfg.d_model, num_heads=cfg.n_heads, mlp_ratio=4.0)

    def forward(self, zH: torch.Tensor, zL: torch.Tensor, p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        zH = self.normH(zH, t)
        zL = self.normL(zL, t)
        x = torch.cat([zH, zL], dim=-1)  # (B, L, 2*D)
        x = self.inj(x)
        x = self.s5d(x, p, t)
        return x


def _sinusoidal_pos_emb(length: int, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
    pos   = torch.arange(length, dtype=torch.float32)
    args  = pos[:, None] * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (length, dim)


class InputLayer(nn.Module):
    def __init__(self, cfg: SRLMConfig):
        super().__init__()
        self.input_emb    = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.timestep_emb = TimestepEmbedder(cfg.d_model, cfg.d_frequency_embedding)
        # Fixed sinusoidal position encoding — not learnable so it can't collapse
        # when training on position-agnostic data (all positions same target token).
        #self.pos_emb    = nn.Embedding(cfg.context_length, cfg.d_model)
        self.register_buffer("pos_emb", _sinusoidal_pos_emb(cfg.context_length, cfg.d_model))

    def forward(self, x: torch.Tensor, sigma: torch.Tensor):
        y = self.input_emb(x)                              # (B, L, D)
        sigma_emb = F.silu(self.timestep_emb(sigma))       # (B, D)
        p = self.pos_emb[:x.shape[1]]                      # (L, D), fixed
        return y, p, sigma_emb


class OutputLayer(nn.Module):
    def __init__(self, cfg: SRLMConfig):
        super().__init__()
        #self.ff   = FeedForward(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model*4, bias=True),
            nn.GELU(approximate="tanh"), # Or nn.SiLU()
            nn.Linear(cfg.d_model*4, cfg.d_model, bias=True),
        )
        self.proj = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, x: torch.Tensor, y: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        #y = self.ff(y)
        y = self.mlp(y)
            #y = F.dropout(y, p=0.1, training=True)
        y = self.proj(y)
        return scatter(x, y, sigma)


class FeedForward(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.pre = nn.Linear(dim, 4 * dim)
        # zero init on output projection — starts as identity passthrough
        self.pos = nn.Linear(4 * dim, dim)
        nn.init.zeros_(self.pos.weight)
        nn.init.zeros_(self.pos.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pos(F.gelu(self.pre(x)))


class AdaLN(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.proj = nn.Linear(d_model, 2 * d_model)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D), cond: (B, D)
        scale, shift = self.proj(cond).chunk(2, dim=-1)  # each (B, D)
        return self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiTBlock(nn.Module):
    """
    A standard DiT block with adaLN-Zero modulation.
    """
    def __init__(self, hidden_dim: int, embedding_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Normalization layers (using LayerNorm without elementwise affine,
        # as adaLN provides the affine parameters)
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)

        # Modulation network to generate adaLN parameters
        self.modulation = AdaLNModulation(embedding_dim, hidden_dim)
        self.num_heads = num_heads
        self.qkv = nn.Linear(embedding_dim, 3 * hidden_dim, bias=True)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        # Attention layer
        #self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

        # MLP layer
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"), # Or nn.SiLU()
            nn.Linear(mlp_hidden_dim, hidden_dim, bias=True),
        )

    def forward(self, x, p, c):
        # x shape: [batch_size, num_patches, hidden_dim]
        # c shape: [batch_size, embedding_dim]

        # Calculate modulation parameters (shift, scale, gate for MSA and MLP)
        # Each param shape: [batch_size, 1, hidden_dim] after unsqueeze
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = [
            m.unsqueeze(1) for m in self.modulation(c)
        ]

        x += p
        # --- Apply adaLN-Zero before MHSA ---
        x_norm1 = self.norm1(x)
        # Modulate: Scale, Shift
        x_modulated1 = x_norm1 * (1 + scale_msa) + shift_msa
        # Apply attention
        B,L,H = x_modulated1.shape
        q, k, v = self.qkv(x_modulated1).chunk(3, dim=-1)
        q = q.view(B, L, self.num_heads, H // self.num_heads).transpose(1, 2)
        k = k.view(B, L, self.num_heads, H // self.num_heads).transpose(1, 2)
        v = v.view(B, L, self.num_heads, H // self.num_heads).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(q, k, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, H)
        attn_output = self.out_proj(attn_output)
        #attn_output, _ = self.attn(x_modulated1, x_modulated1, x_modulated1)
        # Apply gating and add residual connection
        x = x + gate_msa * attn_output

        # --- Apply adaLN-Zero before MLP ---
        x_norm2 = self.norm2(x)
        # Modulate: Scale, Shift
        x_modulated2 = x_norm2 * (1 + scale_mlp) + shift_mlp
        # Apply MLP
        mlp_output = self.mlp(x_modulated2)
        # Apply gating and add residual connection
        x = x + gate_mlp * mlp_output

        # Output shape: [batch_size, num_patches, hidden_dim]
        return x

class AdaLNModulation(nn.Module):
    """
    Calculates shift, scale, and gate parameters from conditioning embeddings.
    Initializes the final linear layer's weights and biases to zero.
    """
    def __init__(self, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.silu = nn.SiLU()
        # Initialize the linear layer with zeros for the "Zero" aspect
        self.linear = nn.Linear(embedding_dim, 6 * hidden_dim, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, c):
        # c shape: [batch_size, embedding_dim]
        mod_params = self.linear(self.silu(c))
        # mod_params shape: [batch_size, 6 * hidden_dim]
        # Split into 6 parts for shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        # Each part has shape [batch_size, hidden_dim]
        # Add unsqueeze(1) to make them broadcastable with token dim: [batch_size, 1, hidden_dim]
        return mod_params.chunk(6, dim=1)

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.pre = nn.Linear(frequency_embedding_size, hidden_size)
        self.pos = nn.Linear(hidden_size, hidden_size)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = timestep_embedding(t, self.frequency_embedding_size)
        return self.pos(F.silu(self.pre(t_freq)))


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) *
        torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]           # (B, half)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, dim)
    #if dim % 2:
    #    embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def make_z(batch_size: int, seq_len: int, d_model: int, device=None) -> tuple:
    """Create zero initial HRM state."""
    z = torch.zeros(batch_size, seq_len, d_model, device=device)
    return z, z.clone()
