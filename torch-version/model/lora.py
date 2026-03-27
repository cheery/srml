"""
LoRA (Low-Rank Adaptation) for SRLM.

Wraps existing nn.Linear layers with low-rank adapters.
Base weights are frozen; only the small A/B matrices train.
Can be merged back into base weights for zero-overhead inference.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with LoRA adapter."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0,
                 dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scaling = alpha / rank

        d_in = base.in_features
        d_out = base.out_features

        # Freeze base weights
        base.weight.requires_grad_(False)
        if base.bias is not None:
            base.bias.requires_grad_(False)

        # Low-rank adapter matrices
        self.lora_A = nn.Parameter(torch.empty(rank, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.merged = False

    def forward(self, x):
        if self.merged:
            return self.base(x)
        return self.base(x) + (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling

    def merge(self):
        """Merge LoRA weights into base for zero-overhead inference."""
        if not self.merged:
            self.base.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def unmerge(self):
        """Unmerge LoRA weights from base for continued training."""
        if self.merged:
            self.base.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
            self.merged = False


def apply_lora(model, rank=8, alpha=16.0, dropout=0.0, targets=None):
    """Apply LoRA adapters to selected linear layers in model.

    Args:
        model: nn.Module to modify in-place
        rank: LoRA rank
        alpha: LoRA scaling factor
        dropout: dropout on LoRA path
        targets: list of attribute name substrings to target
                 (default: attention Q/K/V and output projections)
    Returns:
        list of LoRALinear modules that were inserted
    """
    if targets is None:
        targets = ['qkv', 'out_proj']

    lora_layers = []

    def _replace_linear(parent, name, linear):
        lora = LoRALinear(linear, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, name, lora)
        lora_layers.append(lora)

    for module_name, module in model.named_modules():
        for attr_name in targets:
            if hasattr(module, attr_name):
                layer = getattr(module, attr_name)
                if isinstance(layer, nn.Linear):
                    _replace_linear(module, attr_name, layer)

    return lora_layers


def lora_parameters(model):
    """Return only the LoRA parameters (for optimizer)."""
    params = []
    for m in model.modules():
        if isinstance(m, LoRALinear):
            params.extend([m.lora_A, m.lora_B])
    return params


def merge_lora(model):
    """Merge all LoRA adapters into base weights."""
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.merge()


def unmerge_lora(model):
    """Unmerge all LoRA adapters from base weights."""
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.unmerge()


def remove_lora(model):
    """Remove LoRA adapters, restoring original nn.Linear layers (after merge)."""
    for module in model.modules():
        for name in list(vars(module).keys()):
            attr = getattr(module, name)
            if isinstance(attr, LoRALinear):
                attr.merge()
                setattr(module, name, attr.base)
