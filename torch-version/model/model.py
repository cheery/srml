"""
SRLM -- Straightforward reasoning language model
        (Or something like that...)

SRLM is an energy-based diffusion language model
that contains a G-Mem and CMM as an optional layer.

The memory carries context across text segments during
training and sampling, helping the model to keep coherence.

CMM is there in hopes of improving reasoning capabilities
in the model.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Literal, Callable, Any, Tuple
from .attnres import BlockDivider, BlockAttnResOp

@dataclass
class GMemConfig:
    memory_dim:            int = 256
    num_slots:             int = 1024

@dataclass
class PonderConfig:
    N_H:                   int = 3
    N_L:                   int = 6
    noise_sigma:           float = 0.01

@dataclass
class SRLMConfig:
    vocab_size:            int
    max_context_length:    int
    hidden_dim:            int
    num_heads:             int = 8
    front_layers:          int
    mid_layers:            list[PonderConfig | GMemConfig]
    back_layers:           int
    dropout:               float = 0.2
    block_size:            int = 1 # Full AttnRes at this point.




class SRLM(nn.Module):
    def __init__(self, cfg: SRLMConfig):
        super().__init__()
        self.cfg = cfg
        self.input = InputLayer()
        self.front_layers = nn.ModuleList([
            init_layer(cfg)
            for _ in range(cfg.front_layers)
        ])
        self.mid_layers = nn.ModuleList([
            init_middle_layer(cfg, which)
            for which in cfg.mid_layers
        ])
        self.back_layers = nn.ModuleList([
            init_layer(cfg)
            for _ in range(cfg.back_layers)
        ])
        self.out_proj = nn.Linear()

    def forward(self, x):
        return x

#    @torch.no_grad()
#    def encode_document(self, tokens: torch.Tensor) -> torch.Tensor:
#        """Encode a document into a compressed memory block.
#
#        Runs tokens through input + prior layers (at sigma=0, i.e. clean text),
#        then pools to a fixed-size representation that matches (B, L, D) shape
#        for compatibility with the block list.
#
#        Args:
#            tokens: (B, L_doc) int tensor — document token IDs.
#                    Long documents are split into context_length chunks,
#                    encoded separately, and averaged.
#        Returns:
#            memory: (B, L, D) — compressed document representation,
#                    where L = cfg.context_length
#        """
#        B = tokens.shape[0]
#        L = self.cfg.context_length
#        L_doc = tokens.shape[1]
#        sigma = torch.zeros(B, device=tokens.device)
#
#        if L_doc <= L:
#            if L_doc < L:
#                tokens = F.pad(tokens, (0, L - L_doc), value=0)
#            h, _, _, _, _ = self._encode_prior(tokens, sigma)
#            return h
#
#        # Long doc: split into chunks, encode each, average
#        chunks = [tokens[:, i:i+L] for i in range(0, L_doc, L)]
#        if chunks[-1].shape[1] < L:
#            chunks[-1] = F.pad(chunks[-1], (0, L - chunks[-1].shape[1]), value=0)
#        encoded = []
#        for chunk in chunks:
#            h, _, _, _, _ = self._encode_prior(chunk, sigma)
#            encoded.append(h)
#        return torch.stack(encoded).mean(dim=0)  # (B, L, D)
#
#    def _encode_prior(self, tokens, sigma, memories=None):
#        """Run input + prior layers, return aggregated output via mid_res."""
#        q, cos, sin, t = self.input(tokens, sigma)
#        bd = BlockDivider.top(q, block_size=1)
#        #if memories is not None:
#        #    blocks.extend(memories)
#        for layer in self.prior:
#            bd = layer(bd, cos, sin, t)
#        h = self.mid_res(bd)
#        bd = bd.div(force=True)
#        return h, bd, cos, sin, t
#
#    def front(self, x, sigma, memories=None):
#        """Encode input through prior layers with optional memory context.
#
#        Memories are injected into the block list so prior layers and HRM
#        both have access to stored context.
#        """
#        h, bd, cos, sin, t = self._encode_prior(x, sigma, memories)
#        return h, bd, cos, sin, t, sigma
#
#    def recur(self, st, ix):
#        """Run just HRM + Q_head. Cheap — call this in adaptive loops.
#
#        Returns:
#            st: new HRM state (detached)
#            y: HRM output (B, L, D)
#            q: (B, 1) halting signal
#        """
#        h, bd, cos, sin, t, sigma = ix
#        st, y = self.main(st, h, cos, sin, t)
#        q = self.Q_head(y.mean(dim=1)) # TODO: um... explain? why 'mean' there?
#        return st, y, q
#
#    def head(self, y, ix):
#        """Run posterior layers + output. Expensive — call once after recur() loop.
#
#        Returns:
#            log_score: (B, L, V) output logits
#            aux_loss: scalar (always 0 now — routing removed)
#        """
#        _, bd, cos, sin, t, sigma = ix
#        bd = bd(y)
#
#        for layer in self.posterior:
#            bd = layer(bd, cos, sin, t)
#
#        h = self.final_res(bd)
#        return self.output(h, t, sigma), torch.zeros((), device=h.device)
#
#    def step(self, st, ix):
#        """Convenience: recur() + head() in one call.
#
#        Returns:
#            st: new HRM state (detached)
#            log_score: (B, L, V) output logits
#            q: (B, 1) halting signal
#            aux_loss: routing entropy loss
#        """
#        st, y, q = self.recur(st, ix)
#        log_score, aux_loss = self.head(y, ix)
#        return st, log_score, q, aux_loss
#
#    def short(self, x, sigma_bar):
#        h, bd, cos, sin, t = self._encode_prior(x, sigma_bar)
#        y = torch.zeros_like(h)
#        log_score, _ = self.head(y, (h, bd, cos, sin, t, sigma_bar))
#        return log_score
#
#
#    def sideways(self, st, x, sigma_bar):
#        ix = self.front(x, sigma_bar, memories=None)
#        st, log_score, q, aux_loss = self.step(st, ix)
#        return st, log_score
#
#    def forward(self, x, sigma_bar):
#        """Standard API: front() + step() in one call.
#
#        Args:
#            z: HRM state tuple (y, z)
#            x: (B, L) input token IDs
#            sigma: (B,) noise level
#            memories: optional list of memory tensors
#        Returns:
#            z: new HRM state
#            log_score: (B, L, V) output logits
#            aux_loss: scalar auxiliary loss
#        """
#        z = make_z(x.shape[0], x.shape[1], self.cfg.d_model, device=x.device)
#        ix = self.front(x, sigma_bar, memories=None)
#        st, log_score, q, aux_loss = self.step(z, ix)
#        return log_score
#
#class HRM(nn.Module):
#    """Hierarchical Recurrent Memory — v2 (TRM-inspired).
#
#    Uses a single shared DiTBlock for both fast and slow recursion roles,
#    following TRM's finding that one network generalizes better than two.
#
#    Fast step:  zL = block(zL + zH + x, p, t)   — refine with input
#    Slow step:  zH = block(zH + zL, p, t)        — consolidate
#
#    All iterations except the final cycle run under no_grad.
#    """
#    def __init__(self, cfg: SRLMConfig):
#        super().__init__()
#        self.N = cfg.N
#        self.T = cfg.T
#        self.block = DiTBlock(cfg.d_model, cfg.d_model, num_heads=cfg.n_heads,
#                              mlp_ratio=4.0, dropout=cfg.dropout)
#        self.norm_y = AdaLN(cfg.d_model)
#        self.norm_z = AdaLN(cfg.d_model)
#
#    def latent_recursion(self, x, y, z, cos, sin, t, n):
#        for i in range(n):
#            z = self.block(x + y + z, cos, sin, t)
#        y = self.block(y + z, cos, sin, t)
#        return y, z
#
#    def forward(self, st, x, cos, sin, t):
#        y, z = self.norm_y(st[0], t), self.norm_z(st[1], t)
#
#        with torch.no_grad():
#            for j in range(self.T - 1):
#                y, z = self.latent_recursion(x, y, z, cos, sin, t, self.N)
#        y, z = self.latent_recursion(x, y, z, cos, sin, t, self.N)
#        return (y.detach(), z.detach()), y
#
#def _sinusoidal_pos_emb(length: int, dim: int) -> torch.Tensor:
#    half = dim // 2
#    freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
#    pos   = torch.arange(length, dtype=torch.float32)
#    args  = pos[:, None] * freqs[None]
#    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (length, dim)
#
#
#class InputLayer(nn.Module):
#    def __init__(self, cfg: SRLMConfig):
#        super().__init__()
#        self.input_emb    = nn.Embedding(cfg.vocab_size, cfg.d_model)
#        self.timestep_emb = TimestepEmbedder(cfg.d_model, cfg.d_frequency_embedding)
#        # Fixed sinusoidal position encoding — not learnable so it can't collapse
#        # when training on position-agnostic data (all positions same target token).
#        #self.pos_emb    = nn.Embedding(cfg.context_length, cfg.d_model)
#        #self.register_buffer("pos_emb", _sinusoidal_pos_emb(cfg.context_length, cfg.d_model))
#        # Precompute rotary frequencies
#        head_dim = cfg.d_model // cfg.n_heads
#        half = head_dim // 2
#        freqs = 1.0 / (10000.0 ** (torch.arange(half).float() / half))
#        pos = torch.arange(cfg.context_length).float()
#        angles = pos[:, None] * freqs[None, :]
#        self.register_buffer("rot_cos", torch.cos(angles), persistent=False)
#        self.register_buffer("rot_sin", torch.sin(angles), persistent=False)
#
#
#    def forward(self, x: torch.Tensor, sigma: torch.Tensor):
#        L = x.shape[-1]
#        y = self.input_emb(x)                              # (B, L, D)
#        sigma_emb = self.timestep_emb(sigma)       # (B, D)
#        #y += self.pos_emb[:x.shape[1]]                      # (L, D), fixed
#        cos = self.rot_cos[:L][None, None, :, :]                  # (1,1,L,half)
#        sin = self.rot_sin[:L][None, None, :, :]
#        return y, cos, sin, sigma_emb
#
#def _apply_rotary(x, cos, sin):
#    """Apply rotary embedding to x: (B, H, L, D)."""
#    d = x.shape[-1] // 2
#    x1, x2 = x[..., :d], x[..., d:]
#    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
#
#
#class OutputLayer(nn.Module):
#    def __init__(self, cfg: SRLMConfig):
#        super().__init__()
#        self.outlet = cfg.outlet
#        self.norm = nn.LayerNorm(cfg.d_model)
#        self.proj = nn.Linear(cfg.d_model, cfg.vocab_size)
#        nn.init.zeros_(self.proj.weight)
#        nn.init.constant_(self.proj.bias, -6.0)
#
#    def forward(self, y: torch.Tensor, t: torch.Tensor, sigma_bar: torch.Tensor) -> torch.Tensor:
#        y = self.norm(y)
#        y = self.proj(y)
#        return self.outlet(y, sigma_bar)
#
#class FeedForward(nn.Module):
#    def __init__(self, dim: int):
#        super().__init__()
#        self.pre = nn.Linear(dim, 4 * dim)
#        # zero init on output projection — starts as identity passthrough
#        self.pos = nn.Linear(4 * dim, dim)
#        nn.init.zeros_(self.pos.weight)
#        nn.init.zeros_(self.pos.bias)
#
#    def forward(self, x: torch.Tensor) -> torch.Tensor:
#        return self.pos(F.gelu(self.pre(x)))
#
#
#class AdaLN(nn.Module):
#    def __init__(self, d_model: int):
#        super().__init__()
#        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
#        self.proj = nn.Linear(d_model, 2 * d_model)
#
#    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
#        # x: (B, L, D), cond: (B, D)
#        scale, shift = self.proj(cond).chunk(2, dim=-1)  # each (B, D)
#        return self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
#
#class AttnResDiTBlock(nn.Module):
#    """
#    A AttnRes DiT block with adaLN-Zero modulation.
#    """
#    def __init__(self, hidden_dim: int, embedding_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
#        super().__init__()
#        self.attn_res = BlockAttnResOp(hidden_dim)
#        self.mlp_res = BlockAttnResOp(hidden_dim)
#
#        # Normalization layers (using LayerNorm without elementwise affine,
#        # as adaLN provides the affine parameters)
#        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
#        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
#
#        # Modulation network to generate adaLN parameters
#        self.modulation = AdaLNModulation(embedding_dim, hidden_dim)
#        self.attn = SelfAttention(hidden_dim, num_heads)
#
#        # MLP layer
#        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
#        self.mlp = nn.Sequential(
#            nn.Linear(hidden_dim, mlp_hidden_dim, bias=True),
#            nn.GELU(approximate="tanh"),
#            nn.Linear(mlp_hidden_dim, hidden_dim, bias=True),
#        )
#
#        self.drop_attn = nn.Dropout(dropout)
#        self.drop_mlp  = nn.Dropout(dropout)
#
#    def forward(self, bd, cos, sin, c):
#        # x shape: [batch_size, num_patches, hidden_dim]
#        # c shape: [batch_size, embedding_dim]
#
#        # Calculate modulation parameters (shift, scale, gate for MSA and MLP)
#        # Each param shape: [batch_size, 1, hidden_dim] after unsqueeze
#        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = [
#            m.unsqueeze(1) for m in self.modulation(c)
#        ]
#
#        h = self.attn_res(bd)
#        bd = bd.div()
#        # --- Apply adaLN-Zero before MHSA ---
#        h_norm1 = self.norm1(h)
#        # Modulate: Scale, Shift
#        h_modulated1 = h_norm1 * (1 + scale_msa) + shift_msa
#        # Apply attention
#        attn_output = self.attn(h_modulated1, cos, sin)
#
#        # Apply gating, dropout, and add residual connection
#        bd = bd(gate_msa * self.drop_attn(attn_output))
#
#        h = self.mlp_res(bd)
#        bd = bd.div()
#        # --- Apply adaLN-Zero before MLP ---
#        h_norm2 = self.norm2(h)
#        # Modulate: Scale, Shift
#        h_modulated2 = h_norm2 * (1 + scale_mlp) + shift_mlp
#        # Apply MLP
#        mlp_output = self.mlp(h_modulated2)
#        # Apply gating, dropout, and add residual connection
#        bd = bd(gate_mlp * self.drop_mlp(mlp_output))
#
#        # Output shape: [batch_size, num_patches, hidden_dim]
#        return bd
#
#class DiTBlock(nn.Module):
#    """
#    A standard DiT block with adaLN-Zero modulation.
#    """
#    def __init__(self, hidden_dim: int, embedding_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
#        super().__init__()
#        self.attn = SelfAttention(hidden_dim, num_heads)
#
#        # Normalization layers (using LayerNorm without elementwise affine,
#        # as adaLN provides the affine parameters)
#        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
#        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
#
#        # Modulation network to generate adaLN parameters
#        self.modulation = AdaLNModulation(embedding_dim, hidden_dim)
#        # MLP layer
#        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
#        self.mlp = nn.Sequential(
#            nn.Linear(hidden_dim, mlp_hidden_dim, bias=True),
#            nn.GELU(),#approximate="tanh"),
#            nn.Linear(mlp_hidden_dim, hidden_dim, bias=True),
#        )
#
#        self.drop_attn = nn.Dropout(dropout)
#        self.drop_mlp  = nn.Dropout(dropout)
#
#    def forward(self, x, cos, sin, c):
#        # x shape: [batch_size, num_patches, hidden_dim]
#        # c shape: [batch_size, embedding_dim]
#
#        # Calculate modulation parameters (shift, scale, gate for MSA and MLP)
#        # Each param shape: [batch_size, 1, hidden_dim] after unsqueeze
#        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = [
#            m.unsqueeze(1) for m in self.modulation(c)
#        ]
#
#        # --- Apply adaLN-Zero before MHSA ---
#        x_norm1 = self.norm1(x)
#        # Modulate: Scale, Shift
#        x_modulated1 = x_norm1 * (1 + scale_msa) + shift_msa
#        # Apply attention
#        attn_output = self.attn(x_modulated1, cos, sin)
#        # Apply gating, dropout, and add residual connection
#        x = x + gate_msa * self.drop_attn(attn_output)
#
#        # --- Apply adaLN-Zero before MLP ---
#        x_norm2 = self.norm2(x)
#        # Modulate: Scale, Shift
#        x_modulated2 = x_norm2 * (1 + scale_mlp) + shift_mlp
#        # Apply MLP
#        mlp_output = self.mlp(x_modulated2)
#        # Apply gating, dropout, and add residual connection
#        x = x + gate_mlp * self.drop_mlp(mlp_output)
#
#        # Output shape: [batch_size, num_patches, hidden_dim]
#        return x
#
#class SelfAttention(nn.Module):
#    def __init__(self, dim, num_heads):
#        super().__init__()
#        self.num_heads = num_heads
#        self.head_dim = dim // num_heads
#        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
#        self.out_proj = nn.Linear(dim, dim)
#
#    def forward(self, x, cos, sin):
#        B,L,H = x.shape
#        #q, k, v = self.qkv(x_modulated1).chunk(3, dim=-1)
#        #q = q.view(B, L, self.num_heads, H // self.num_heads).transpose(1, 2)
#        #k = k.view(B, L, self.num_heads, H // self.num_heads).transpose(1, 2)
#        #v = v.view(B, L, self.num_heads, H // self.num_heads).transpose(1, 2)
#        #q = _apply_rotary(q, cos, sin)
#        #k = _apply_rotary(k, cos, sin)
#        #attn_output = F.scaled_dot_product_attention(q, k, v)
#        #attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, H)
#        #attn_output = self.out_proj(attn_output)
#
#        B, L, _ = x.shape
#        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
#        q, k, v = qkv.permute(2, 0, 3, 1, 4)          # 3 × (B,H,L,D)
#        q = _apply_rotary(q, cos, sin)
#        k = _apply_rotary(k, cos, sin)
#        out = F.scaled_dot_product_attention(q, k, v)   # (B,H,L,D)
#        out = out.transpose(1, 2).reshape(B, L, -1)
#        return self.out_proj(out)
#
#class AdaLNModulation(nn.Module):
#    """
#    Calculates shift, scale, and gate parameters from conditioning embeddings.
#    Initializes the final linear layer's weights and biases to zero.
#    """
#    def __init__(self, embedding_dim: int, hidden_dim: int):
#        super().__init__()
#        self.silu = nn.SiLU()
#        # Initialize the linear layer with zeros for the "Zero" aspect
#        self.linear = nn.Linear(embedding_dim, 6 * hidden_dim, bias=True)
#        nn.init.zeros_(self.linear.weight)
#        nn.init.zeros_(self.linear.bias)
#
#    def forward(self, c):
#        # c shape: [batch_size, embedding_dim]
#        mod_params = self.linear(self.silu(c))
#        # mod_params shape: [batch_size, 6 * hidden_dim]
#        # Split into 6 parts for shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
#        # Each part has shape [batch_size, hidden_dim]
#        # Add unsqueeze(1) to make them broadcastable with token dim: [batch_size, 1, hidden_dim]
#        return mod_params.chunk(6, dim=1)
#
#class TimestepEmbedder(nn.Module):
#    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
#        super().__init__()
#        self.frequency_embedding_size = frequency_embedding_size
#        self.pre = nn.Linear(frequency_embedding_size, hidden_size)
#        self.pos = nn.Linear(hidden_size, hidden_size)
#
#    def forward(self, t: torch.Tensor) -> torch.Tensor:
#        t_freq = timestep_embedding(t, self.frequency_embedding_size)
#        return self.pos(F.silu(self.pre(t_freq)))
#
#
#def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
#    half = dim // 2
#    freqs = torch.exp(
#        -math.log(max_period) *
#        torch.arange(half, dtype=torch.float32, device=t.device) / half
#    )
#    args = t[:, None].float() * freqs[None]           # (B, half)
#    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, dim)
#    #if dim % 2:
#    #    embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
#    return embedding
#
#
#def make_z(batch_size: int, seq_len: int, d_model: int, device=None) -> tuple:
#    """Create zero initial HRM state."""
#    z = torch.zeros(batch_size, seq_len, d_model, device=device)
#    return z, z.clone()
#
## ============================================================
## This is left here for completeness.
## ============================================================
## Score Transformer  (Section 5.1, Appendix C.2)
## ============================================================
## DiT-style encoder-only transformer (Peebles & Xie, 2023):
##   - adaLN-zero time conditioning on σ̄(t)  (not t itself)
##   - rotary positional embeddings  (Su et al., 2021)
##   - separate input embedding and output projection matrices
##   - output exponentiated for positivity; scaled by (e^σ̄ − 1)
##     for absorb  (Appendix C.2)
#
#def _apply_rotary(x, cos, sin):
#    """Apply rotary embedding to x: (B, H, L, D)."""
#    d = x.shape[-1] // 2
#    x1, x2 = x[..., :d], x[..., d:]
#    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
#
#
#class _SelfAttention(nn.Module):
#    def __init__(self, dim, num_heads):
#        super().__init__()
#        self.num_heads = num_heads
#        self.head_dim = dim // num_heads
#        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
#        self.out_proj = nn.Linear(dim, dim)
#
#    def forward(self, x, cos, sin):
#        B, L, _ = x.shape
#        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
#        q, k, v = qkv.permute(2, 0, 3, 1, 4)          # 3 × (B,H,L,D)
#        q = _apply_rotary(q, cos, sin)
#        k = _apply_rotary(k, cos, sin)
#        out = F.scaled_dot_product_attention(q, k, v)   # (B,H,L,D)
#        out = out.transpose(1, 2).reshape(B, L, -1)
#        return self.out_proj(out)
#
#
#class _AdaLNBlock(nn.Module):
#    """Transformer block with adaLN-zero conditioning (DiT)."""
#    def __init__(self, dim, num_heads, mlp_ratio=4):
#        super().__init__()
#        self.res1 = BlockAttnResOp(dim)
#        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
#        self.attn = _SelfAttention(dim, num_heads)
#        self.res2 = BlockAttnResOp(dim)
#        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
#        self.mlp = nn.Sequential(
#            nn.Linear(dim, dim * mlp_ratio),
#            nn.GELU(),
#            nn.Linear(dim * mlp_ratio, dim))
#        # 6 modulation parameters: (γ1, β1, α1, γ2, β2, α2)
#        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
#        nn.init.zeros_(self.adaln[-1].weight)
#        nn.init.zeros_(self.adaln[-1].bias)
#
#    def forward(self, bd, c, cos, sin):
#        g1, b1, a1, g2, b2, a2 = self.adaln(c).unsqueeze(1).chunk(6, dim=-1)
#
#        x = self.res1(bd)
#        bd = bd.div()
#        h = self.norm1(x) * (1 + g1) + b1
#        bd = bd(a1 * self.attn(h, cos, sin))
#
#        x = self.res2(bd)
#        bd = bd.div()
#        h = self.norm2(x) * (1 + g2) + b2
#        bd = bd(a2 * self.mlp(h))
#        return bd
#
#
#class ScoreTransformer(nn.Module):
#    """
#    Small DiT-style score network for SEDD (Section 5.1, Appendix C.2).
#
#    The network is conditioned on σ̄(t) (not t) via sinusoidal
#    embeddings fed through adaLN-zero.  Outputs are exponentiated
#    for positivity and optionally scaled by (e^σ̄ − 1) for absorb.
#    """
#    def __init__(self, n: int, max_len: int, dim: int = 256,
#                 num_heads: int = 4, num_layers: int = 4,
#                 mode: Literal["absorb", "uniform"] = "absorb"):
#        super().__init__()
#        self.n = n
#        self.dim = dim
#        self.mode = mode
#
#        self.tok_embed = nn.Embedding(n, dim)
#        self.out_proj = nn.Linear(dim, n)
#        # Initialise output bias negative so exp(logit) starts small
#        nn.init.zeros_(self.out_proj.weight)
#        nn.init.constant_(self.out_proj.bias, -6.0)
#
#        self.time_mlp = nn.Sequential(
#            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
#        self.blocks = nn.ModuleList(
#            [_AdaLNBlock(dim, num_heads) for _ in range(num_layers)])
#        self.final_res = BlockAttnResOp(dim)
#        self.final_norm = nn.LayerNorm(dim)
#
#        # Precompute rotary frequencies
#        head_dim = dim // num_heads
#        half = head_dim // 2
#        freqs = 1.0 / (10000.0 ** (torch.arange(half).float() / half))
#        pos = torch.arange(max_len).float()
#        angles = pos[:, None] * freqs[None, :]
#        self.register_buffer("rot_cos", torch.cos(angles), persistent=False)
#        self.register_buffer("rot_sin", torch.sin(angles), persistent=False)
#
#    def _time_embed(self, sigma_bar):
#        half = self.dim // 2
#        freqs = torch.exp(
#            -math.log(10000.0) * torch.arange(half, device=sigma_bar.device) / half)
#        args = sigma_bar[:, None] * freqs[None, :]
#        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
#        return self.time_mlp(emb)                                 # (B, dim)
#
#    def forward(self, xt: torch.Tensor, sigma_bar: torch.Tensor):
#        """
#        Returns **log-scores** (not scores).
#
#        log s_θ = logits + log(e^σ̄ − 1)   for absorb
#        log s_θ = logits                    for uniform
#        """
#        B, L = xt.shape
#        h = self.tok_embed(xt)                                    # (B,L,D)
#        c = self._time_embed(sigma_bar)                           # (B,D)
#        cos = self.rot_cos[:L][None, None, :, :]                  # (1,1,L,half)
#        sin = self.rot_sin[:L][None, None, :, :]
#        bd = BlockDivider.top(h, block_size=1)
#
#        for block in self.blocks:
#            bd = block(bd, c, cos, sin)
#
#        h = self.final_res(bd)
#        h = self.final_norm(h)
#        log_scores = self.out_proj(h)                             # (B,L,n)
#
#        # Absorb scaling in log-space: log(e^σ̄ − 1), stable (Appendix C.2)
#        return Outlet()(log_scores, sigma_bar)
