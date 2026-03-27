"""
Exponential Moving Average for model parameters.

Maintains shadow weights updated as:
    shadow = mu * shadow + (1 - mu) * param

Used by TRM for training stability during deep supervision.
For eval/sampling, swap in EMA weights; for training, swap back.
"""

import copy
import torch
import torch.nn as nn


class EMA:
    def __init__(self, model: nn.Module, mu: float = 0.999):
        self.mu = mu
        self.shadow = {}
        self.backup = {}
        self.register(model)

    def register(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        """Update shadow weights toward current model weights."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].lerp_(param.data, 1.0 - self.mu)

    def apply(self, model: nn.Module):
        """Swap EMA weights into model (for eval). Call restore() after."""
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """Restore original weights after eval with EMA."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return {'mu': self.mu, 'shadow': self.shadow}

    def load_state_dict(self, state_dict):
        self.mu = state_dict['mu']
        self.shadow = state_dict['shadow']
