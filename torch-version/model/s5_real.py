"""
s5.py — S5 SSM layer in pure float32 PyTorch.

No complex tensors anywhere. Complex arithmetic is done manually
as paired real/imaginary float tensors throughout.

Input/output shapes: (B, L, D)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._higher_order_ops.associative_scan import associative_scan

USE_PARALLEL_SCAN = False

# ---------------------------------------------------------------------------
# S5Dual: bidirectional wrapper
# ---------------------------------------------------------------------------

class S5Dual(nn.Module):
    """Bidirectional S5: fwd(x) + flip(bwd(flip(x))).
    Input/output: (B, L, D)
    """
    def __init__(self, d_model: int, d_state: int):
        super().__init__()
        self.fwd = S5(d_model, d_state)
        self.bwd = S5(d_model, d_state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flip = x.flip(dims=[1])
        return self.fwd(x) + self.bwd(x_flip).flip(dims=[1])


# ---------------------------------------------------------------------------
# S5: single SSM layer + GELU, fully real arithmetic
# ---------------------------------------------------------------------------

class S5(nn.Module):
    """Discretized diagonal SSM + GELU, all float32.

    State:  x[t] = Lambda * x[t-1] + B * u[t]
    Output: y[t] = Re(C * x[t]) + D * u[t]

    Lambda is parameterized as -exp(log_real) + i*imag to enforce stability.
    All complex arithmetic uses paired real/imag float tensors.
    """

    def __init__(self, d_model: int, d_state: int):
        super().__init__()
        H, P = d_model, d_state
        self.H = H
        self.P = P

        # Lambda: Re = -exp(log_real) < 0 always (stable)
        self.log_real  = nn.Parameter(torch.randn(P) * 0.5)
        self.imag      = nn.Parameter(torch.randn(P) * 1.0)

        # B_tilde (P, H): real and imag parts
        std_b = 1.0 / H ** 0.5
        self.B_real = nn.Parameter(torch.empty(P, H).normal_(std=std_b))
        self.B_imag = nn.Parameter(torch.empty(P, H).normal_(std=std_b))

        # C_tilde (H, P): real and imag parts
        std_c = 1.0 / P ** 0.5
        self.C_real = nn.Parameter(torch.empty(H, P).normal_(std=std_c))
        self.C_imag = nn.Parameter(torch.empty(H, P).normal_(std=std_c))

        # D: skip connection (init 1), log_Delta: step size (init 0 → Delta=1)
        self.D         = nn.Parameter(torch.ones(H))
        self.log_Delta = nn.Parameter(torch.zeros(P))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, H)
        Lbar_r, Lbar_i, B_bar_r, B_bar_i = discretize(
            self.log_real, self.imag, self.log_Delta,
            self.B_real, self.B_imag,
        )
        y = apply_ssm(
            Lbar_r, Lbar_i,
            B_bar_r, B_bar_i,
            self.C_real, self.C_imag,
            self.D, x,
        )
        return F.gelu(y)


# ---------------------------------------------------------------------------
# discretize: ZOH, fully real float32
# ---------------------------------------------------------------------------

def discretize(
    log_real:  torch.Tensor,  # (P,)
    imag:      torch.Tensor,  # (P,)
    log_Delta: torch.Tensor,  # (P,)
    B_real:    torch.Tensor,  # (P, H)
    B_imag:    torch.Tensor,  # (P, H)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """ZOH discretization in pure float32.

    Lambda     = -exp(log_real) + i*imag
    Lambda_bar = exp(Lambda * Delta)
    B_bar      = (1/Lambda) * (Lambda_bar - 1) * B_tilde

    Returns: Lbar_r, Lbar_i, B_bar_r, B_bar_i  — all (P,) or (P, H)
    """
    Lr    = -log_real.exp()    # (P,)
    Li    = imag               # (P,)
    Delta = log_Delta.exp()    # (P,)

    # Lambda_bar = exp(Lambda * Delta)
    # = exp(Lr*Delta) * (cos(Li*Delta) + i*sin(Li*Delta))
    ad     = Lr * Delta
    bd     = Li * Delta
    exp_ad = torch.exp(ad)
    Lbar_r = exp_ad * torch.cos(bd)
    Lbar_i = exp_ad * torch.sin(bd)

    # 1/Lambda = conj(Lambda) / |Lambda|^2
    denom  = (Lr * Lr + Li * Li).clamp(min=1e-12)
    inv_Lr =  Lr / denom
    inv_Li = -Li / denom

    # coeff = (1/Lambda) * (Lambda_bar - 1)
    diff_r  = Lbar_r - 1.0
    diff_i  = Lbar_i
    coeff_r = inv_Lr * diff_r - inv_Li * diff_i
    coeff_i = inv_Lr * diff_i + inv_Li * diff_r

    # B_bar = coeff * B_tilde
    cr = coeff_r.unsqueeze(-1)  # (P, 1)
    ci = coeff_i.unsqueeze(-1)
    B_bar_r = cr * B_real - ci * B_imag   # (P, H)
    B_bar_i = cr * B_imag + ci * B_real   # (P, H)

    return Lbar_r, Lbar_i, B_bar_r, B_bar_i


# ---------------------------------------------------------------------------
# Parallel scan operator — all real float32
# ---------------------------------------------------------------------------

def binary_operator(element_i, element_j):
    """Associative operator for parallel scan.

    Each element: (A_real, A_imag, x_real, x_imag)
    A_out = A_j * A_i
    x_out = A_j * x_i + x_j
    """
    Ar_i, Ai_i, xr_i, xi_i = element_i
    Ar_j, Ai_j, xr_j, xi_j = element_j
    Ar_out = Ar_j * Ar_i - Ai_j * Ai_i
    Ai_out = Ar_j * Ai_i + Ai_j * Ar_i
    xr_out = Ar_j * xr_i - Ai_j * xi_i + xr_j
    xi_out = Ar_j * xi_i + Ai_j * xr_i + xi_j
    return Ar_out, Ai_out, xr_out, xi_out

# ---------------------------------------------------------------------------
# apply_ssm: sequential (default) or parallel scan
# ---------------------------------------------------------------------------

def apply_ssm(
    Lbar_r:  torch.Tensor,  # (P,)
    Lbar_i:  torch.Tensor,  # (P,)
    B_bar_r: torch.Tensor,  # (P, H)
    B_bar_i: torch.Tensor,  # (P, H)
    C_real:  torch.Tensor,  # (H, P)
    C_imag:  torch.Tensor,  # (H, P)
    D:       torch.Tensor,  # (H,)
    x:       torch.Tensor,  # (B, L, H)
) -> torch.Tensor:          # (B, L, H)
    B, L, H = x.shape
    P = Lbar_r.shape[0]

    # Bu = B_bar^T @ u: (B, L, P)
    Bu_r = x @ B_bar_r.t()
    Bu_i = x @ B_bar_i.t()

    if USE_PARALLEL_SCAN:
        Lr_e = Lbar_r.unsqueeze(0).unsqueeze(0).expand(B, L, P).contiguous()
        Li_e = Lbar_i.unsqueeze(0).unsqueeze(0).expand(B, L, P).contiguous()
        _, _, xs_r, xs_i = associative_scan(
            binary_operator,
            (Lr_e, Li_e, Bu_r, Bu_i),
            dim=1,
            combine_mode="pointwise"
        )
    else:
        # Sequential scan — O(L), no compilation required
        xs_r = torch.zeros(B, P, device=x.device, dtype=x.dtype)
        xs_i = torch.zeros_like(xs_r)
        outs_r: list[torch.Tensor] = []
        outs_i: list[torch.Tensor] = []
        for t in range(L):
            new_r = Lbar_r * xs_r - Lbar_i * xs_i + Bu_r[:, t, :]
            new_i = Lbar_r * xs_i + Lbar_i * xs_r + Bu_i[:, t, :]
            xs_r, xs_i = new_r, new_i
            outs_r.append(xs_r)
            outs_i.append(xs_i)
        xs_r = torch.stack(outs_r, dim=1)  # (B, L, P)
        xs_i = torch.stack(outs_i, dim=1)

    # Re(C @ xs) = C_r @ xs_r - C_i @ xs_i
    ys = xs_r @ C_real.t() - xs_i @ C_imag.t()  # (B, L, H)
    return ys + D * x
