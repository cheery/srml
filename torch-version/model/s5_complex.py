"""
s5.py — S5 SSM layer with complex arithmetic and parallel scan.

Input/output shapes: (B, L, D)

Set USE_PARALLEL_SCAN = True and wrap model with torch.compile for
best performance. Falls back to sequential scan otherwise.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._higher_order_ops.associative_scan import associative_scan

USE_PARALLEL_SCAN = True


class S5Dual(nn.Module):
    """Bidirectional S5: fwd(x) + flip(bwd(flip(x)))."""

    def __init__(self, d_model: int, d_state: int):
        super().__init__()
        self.fwd = S5(d_model, d_state)
        self.bwd = S5(d_model, d_state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flip = x.flip(dims=[1])
        return self.fwd(x) + self.bwd(x_flip).flip(dims=[1])


class S5(nn.Module):
    """Discretized diagonal SSM + GELU.

    Parameters stored as float32 real/imag pairs.
    Complex tensors are constructed only during the forward pass.
    """

    def __init__(self, d_model: int, d_state: int):
        super().__init__()
        H, P = d_model, d_state
        self.H = H
        self.P = P

        self.log_real  = nn.Parameter(torch.randn(P) * 0.5)
        self.imag      = nn.Parameter(torch.randn(P) * 1.0)

        std_b = 1.0 / H ** 0.5
        self.B_real = nn.Parameter(torch.empty(P, H).normal_(std=std_b))
        self.B_imag = nn.Parameter(torch.empty(P, H).normal_(std=std_b))

        std_c = 1.0 / P ** 0.5
        self.C_real = nn.Parameter(torch.empty(H, P).normal_(std=std_c))
        self.C_imag = nn.Parameter(torch.empty(H, P).normal_(std=std_c))

        self.D         = nn.Parameter(torch.ones(H))
        self.log_Delta = nn.Parameter(torch.zeros(P))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Build complex tensors from float32 parameter pairs
        Lambda  = torch.complex(-self.log_real.exp(), self.imag)
        B_tilde = torch.complex(self.B_real, self.B_imag)
        C_tilde = torch.complex(self.C_real, self.C_imag)
        Delta   = self.log_Delta.exp()

        Lambda_bar, B_bar = discretize(Lambda, B_tilde, Delta)
        y = apply_ssm(Lambda_bar, B_bar, C_tilde, self.D, x)
        return F.gelu(y)


def discretize(
    Lambda:  torch.Tensor,  # (P,) complex64
    B_tilde: torch.Tensor,  # (P, H) complex64
    Delta:   torch.Tensor,  # (P,) float32
) -> tuple[torch.Tensor, torch.Tensor]:
    Lambda_bar = torch.exp(Lambda * Delta)
    denom = Lambda.abs().pow(2).clamp(min=1e-12)
    Lambda_safe = torch.where(Lambda.abs() < 1e-6,
                              torch.full_like(Lambda, 1e-6),
                              Lambda)
    B_bar = (1.0 / Lambda_safe * (Lambda_bar - 1.0)).unsqueeze(-1) * B_tilde
    return Lambda_bar, B_bar


def binary_operator(element_i, element_j):
    A_i, Bu_i = element_i
    A_j, Bu_j = element_j
    return A_j * A_i, A_j * Bu_i + Bu_j


def apply_ssm(
    Lambda_bar: torch.Tensor,  # (P,) complex64
    B_bar:      torch.Tensor,  # (P, H) complex64
    C_tilde:    torch.Tensor,  # (H, P) complex64
    D:          torch.Tensor,  # (H,) float32
    x:          torch.Tensor,  # (B, L, H) float32
) -> torch.Tensor:             # (B, L, H) float32
    B, L, H = x.shape
    P = Lambda_bar.shape[0]

    # Bu: (B, L, P) complex
    Bu = x.to(torch.complex64) @ B_bar.t()

    if USE_PARALLEL_SCAN:
        La = Lambda_bar.unsqueeze(0).unsqueeze(0).expand(B, L, P).contiguous()
        _, xs = associative_scan(binary_operator, (La, Bu), dim=1)
    else:
        xs_list = []
        state = torch.zeros(B, P, dtype=torch.complex64, device=x.device)
        for t in range(L):
            state = Lambda_bar * state + Bu[:, t, :]
            xs_list.append(state)
        xs = torch.stack(xs_list, dim=1)  # (B, L, P)

    # Re(C @ xs) + D * x
    ys = (xs @ C_tilde.t()).real + D * x
    return ys
