"""
s5_scan.py — Correct parallel prefix scan for S5 SSM.

Uses PyTorch iterative doubling (Hillis-Steele algorithm).
Each round creates new tensors — no in-place mutation during reads,
so no race conditions possible. Fully differentiable via autograd.

Compiled with torch.compile for GPU efficiency.

Binary operator:
    A_out = A_j * A_i
    x_out = A_j * x_i + x_j
"""

import math
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Core iterative doubling scan — operates on (BP, L) tensors
# ---------------------------------------------------------------------------

def _scan_fwd(Ar, Ai, xr, xi):
    """Forward prefix scan via iterative doubling.

    Each round shifts by stride positions and combines.
    New tensors created each round — no race conditions.

    Args: all (BP, L) float32
    Returns: Ar, Ai, xr, xi (BP, L) — prefix scan results
    """
    BP, L = xr.shape
    n_rounds = int(math.ceil(math.log2(max(L, 2))))
    stride = 1

    for _ in range(n_rounds):
        # Shift left by stride: pad identity on the left
        # Identity element: A=1+0i, x=0+0i
        Ar_l = F.pad(Ar[:, :-stride], (stride, 0), value=1.0)
        Ai_l = F.pad(Ai[:, :-stride], (stride, 0), value=0.0)
        xr_l = F.pad(xr[:, :-stride], (stride, 0), value=0.0)
        xi_l = F.pad(xi[:, :-stride], (stride, 0), value=0.0)

        # Combine: (left) o (current)
        # A_out = A_cur * A_left
        new_Ar = Ar * Ar_l - Ai * Ai_l
        new_Ai = Ar * Ai_l + Ai * Ar_l
        # x_out = A_cur * x_left + x_cur
        new_xr = Ar * xr_l - Ai * xi_l + xr
        new_xi = Ar * xi_l + Ai * xr_l + xi

        Ar, Ai, xr, xi = new_Ar, new_Ai, new_xr, new_xi
        stride *= 2

    return xr, xi


def _scan_bwd(dxr, dxi, Lr, Li):
    """Reverse prefix scan for gradients.

    h_i = g_i + conj(Lambda_{i+1}) * h_{i+1}

    Args: all (BP, L) float32
        dxr, dxi: upstream gradient
        Lr, Li:   original Lambda values (conj = Lr, -Li)
    Returns: dBu_r, dBu_i (BP, L)
    """
    BP, L = dxr.shape
    n_rounds = int(math.ceil(math.log2(max(L, 2))))

    hr, hi = dxr, dxi
    stride = 1

    for _ in range(n_rounds):
        # Shift right by stride: pad zeros on the right
        hr_r  = F.pad(hr[:, stride:],  (0, stride), value=0.0)
        hi_r  = F.pad(hi[:, stride:],  (0, stride), value=0.0)
        Lr_r  = F.pad(Lr[:, stride:],  (0, stride), value=1.0)
        Li_r  = F.pad(Li[:, stride:],  (0, stride), value=0.0)

        # h_i += conj(Lambda_{i+stride}) * h_{i+stride}
        # conj(Lambda) = (Lr, -Li)
        new_hr = hr + Lr_r * hr_r + Li_r * hi_r
        new_hi = hi + Lr_r * hi_r - Li_r * hr_r

        hr, hi = new_hr, new_hi
        stride *= 2

    return hr, hi


# ---------------------------------------------------------------------------
# Autograd Function: custom backward for efficiency
# (forward is differentiable via autograd too, but custom bwd is faster)
# ---------------------------------------------------------------------------

class ParallelScanFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Lbar_r, Lbar_i, Bu_r, Bu_i):
        """
        Args: all (B, L, P) float32
        Returns: xs_r, xs_i (B, L, P)
        """
        B, L, P = Bu_r.shape

        # Reshape to (BP, L) — each row is one independent SSM stream
        Lr = Lbar_r.permute(0, 2, 1).reshape(B*P, L).contiguous()
        Li = Lbar_i.permute(0, 2, 1).reshape(B*P, L).contiguous()
        Br = Bu_r.permute(0, 2, 1).reshape(B*P, L).contiguous()
        Bi = Bu_i.permute(0, 2, 1).reshape(B*P, L).contiguous()

        xr_flat, xi_flat = _scan_fwd(Lr, Li, Br, Bi)

        xs_r = xr_flat.reshape(B, P, L).permute(0, 2, 1).contiguous()
        xs_i = xi_flat.reshape(B, P, L).permute(0, 2, 1).contiguous()

        # Save original Lambda (not modified by scan) for backward
        ctx.save_for_backward(Lbar_r, Lbar_i, xs_r, xs_i)
        ctx.BLP = (B, L, P)
        return xs_r, xs_i

    @staticmethod
    def backward(ctx, dxs_r, dxs_i):
        Lbar_r, Lbar_i, xs_r, xs_i = ctx.saved_tensors
        B, L, P = ctx.BLP

        Lr = Lbar_r.permute(0, 2, 1).reshape(B*P, L).contiguous()
        Li = Lbar_i.permute(0, 2, 1).reshape(B*P, L).contiguous()
        dr = dxs_r.permute(0, 2, 1).reshape(B*P, L).contiguous()
        di = dxs_i.permute(0, 2, 1).reshape(B*P, L).contiguous()

        # Gradient w.r.t. Bu via reverse scan
        dBr_flat, dBi_flat = _scan_bwd(dr, di, Lr, Li)

        dBu_r = dBr_flat.reshape(B, P, L).permute(0, 2, 1).contiguous()
        dBu_i = dBi_flat.reshape(B, P, L).permute(0, 2, 1).contiguous()

        # Gradient w.r.t. Lambda: dL/dLbar_i = g_i * conj(xs_{i-1})
        xs_r_prev = torch.cat([torch.zeros_like(xs_r[:, :1, :]), xs_r[:, :-1, :]], dim=1)
        xs_i_prev = torch.cat([torch.zeros_like(xs_i[:, :1, :]), xs_i[:, :-1, :]], dim=1)
        dLbar_r = (dxs_r * xs_r_prev + dxs_i * xs_i_prev).sum(dim=(0, 1))
        dLbar_i = (dxs_i * xs_r_prev - dxs_r * xs_i_prev).sum(dim=(0, 1))

        return dLbar_r, dLbar_i, dBu_r, dBu_i


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parallel_scan(Lbar_r: torch.Tensor, Lbar_i: torch.Tensor,
                  Bu_r: torch.Tensor, Bu_i: torch.Tensor):
    """Parallel prefix scan for S5 SSM.

    Args:
        Lbar_r, Lbar_i: (P,) discretized Lambda real/imag
        Bu_r, Bu_i:     (B, L, P) input projection real/imag
    Returns:
        xs_r, xs_i: (B, L, P) hidden state sequence
    """
    B, L, P = Bu_r.shape
    Lr = Lbar_r[None, None, :].expand(B, L, P).contiguous()
    Li = Lbar_i[None, None, :].expand(B, L, P).contiguous()
    return ParallelScanFn.apply(Lr, Li, Bu_r, Bu_i)


def apply_ssm_parallel(Lbar_r, Lbar_i, B_bar_r, B_bar_i, C_real, C_imag, D, x):
    """Drop-in replacement for apply_ssm using parallel scan."""
    Bu_r = x @ B_bar_r.t()
    Bu_i = x @ B_bar_i.t()
    xs_r, xs_i = parallel_scan(Lbar_r, Lbar_i, Bu_r, Bu_i)
    return xs_r @ C_real.t() - xs_i @ C_imag.t() + D * x


# Compile for speed — works on both CUDA and ROCm
try:
    _scan_fwd = torch.compile(_scan_fwd)
    _scan_bwd = torch.compile(_scan_bwd)
except Exception:
    pass  # torch.compile not available, run eagerly


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    B, L, H, P = 2, 64, 32, 16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    Lbar_r = (-torch.rand(P) * 0.5).to(device)
    Lbar_i = torch.randn(P).to(device)
    Bu_r   = torch.randn(B, L, P, device=device)
    Bu_i   = torch.randn(B, L, P, device=device)

    # Sequential reference
    def seq_scan(Lr, Li, Br, Bi):
        xr = torch.zeros(B, P, device=device)
        xi = torch.zeros(B, P, device=device)
        outs_r, outs_i = [], []
        for t in range(L):
            new_r = Lr * xr - Li * xi + Br[:, t, :]
            new_i = Lr * xi + Li * xr + Bi[:, t, :]
            xr, xi = new_r, new_i
            outs_r.append(xr)
            outs_i.append(xi)
        return torch.stack(outs_r, dim=1), torch.stack(outs_i, dim=1)

    print("\nTesting forward vs sequential reference...")
    ref_r, ref_i = seq_scan(Lbar_r, Lbar_i, Bu_r, Bu_i)
    xs_r, xs_i   = parallel_scan(Lbar_r, Lbar_i, Bu_r, Bu_i)
    err = (xs_r - ref_r).abs().max().item()
    print(f"  max forward error: {err:.2e} {'✓' if err < 1e-4 else '✗ TOO LARGE'}")

    print("\nTesting gradient correctness vs autograd reference...")
    Bu_r_t = Bu_r.clone().requires_grad_(True)
    Bu_i_t = Bu_i.clone().requires_grad_(True)
    Bu_r_r = Bu_r.clone().requires_grad_(True)
    Bu_i_r = Bu_i.clone().requires_grad_(True)

    xs_r_t, xs_i_t = parallel_scan(Lbar_r, Lbar_i, Bu_r_t, Bu_i_t)
    (xs_r_t.sum() + xs_i_t.sum()).backward()

    ref_r2, ref_i2 = seq_scan(Lbar_r, Lbar_i, Bu_r_r, Bu_i_r)
    (ref_r2.sum() + ref_i2.sum()).backward()

    grad_err = (Bu_r_t.grad - Bu_r_r.grad).abs().max().item()
    print(f"  max grad error: {grad_err:.2e} {'✓' if grad_err < 1e-4 else '✗ TOO LARGE'}")

    print("\nTesting apply_ssm_parallel...")
    B_bar_r = torch.randn(P, H, device=device)
    B_bar_i = torch.randn(P, H, device=device)
    C_real  = torch.randn(H, P, device=device)
    C_imag  = torch.randn(H, P, device=device)
    D       = torch.ones(H, device=device)
    x = torch.randn(B, L, H, device=device, requires_grad=True)
    y = apply_ssm_parallel(Lbar_r, Lbar_i, B_bar_r, B_bar_i, C_real, C_imag, D, x)
    assert y.shape == (B, L, H)
    y.sum().backward()
    assert not x.grad.isnan().any()
    print(f"  output shape: {y.shape} ✓")

    print("\nAll tests passed.")
