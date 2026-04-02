import torch
import torch.nn as nn
import torch.nn.functional as F

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, num_heads, max_context_length):
        super().__init__()
        # Rotary position embeddings
        head_dim = dim // num_heads
        half = head_dim // 2
        freqs = 1.0 / (10000.0 ** (torch.arange(half).float() / half))
        pos = torch.arange(max_context_length).float()
        angles = pos[:, None] * freqs[None, :]
        self.register_buffer("rot_cos", torch.cos(angles), persistent=False)
        self.register_buffer("rot_sin", torch.sin(angles), persistent=False)

    def forward(self, seq_len):
        cos = self.rot_cos[:seq_len][None, None, :, :]
        sin = self.rot_sin[:seq_len][None, None, :, :]
        return Rotor(cos, sin)

class Rotor:
    def __init__(self, cos, sin):
        self.cos = cos
        self.sin = sin

    def __call__(self, x):
        cos = self.cos
        sin = self.sin
        d = x.shape[-1] // 2
        x1, x2 = x[..., :d], x[..., d:]
        return torch.cat([x1 * cos - x2 * sin,
                          x2 * cos + x1 * sin], dim=-1)
