import torch
import torch.nn as nn

from s5 import S5

class S5Dual(nn.Module):
    """Bidirectional S5: fwd(x) + flip(bwd(flip(x)))."""

    def __init__(self, d_model: int, d_state: int):
        super().__init__()
        self.fwd = S5(d_model, d_state)
        self.bwd = S5(d_model, d_state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flip = x.flip(dims=[1])
        return self.fwd(x) + self.bwd(x_flip).flip(dims=[1])
