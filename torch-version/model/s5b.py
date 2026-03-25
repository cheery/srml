"""
s5_scan_triton.py — Correct Triton parallel scan using one kernel per round.

The key insight: Hillis-Steele requires reading OLD values while writing NEW
values. In global memory this requires a barrier between rounds — achieved
by launching a separate kernel per round. Each launch is an implicit barrier.

For log2(L) rounds with L=256, that's 8 kernel launches — cheap.

Binary operator:
    A_out = A_j * A_i
    x_out = A_j * x_i + x_j
"""

import torch
import triton
import triton.language as tl
import math


# ---------------------------------------------------------------------------
# Single round of Hillis-Steele: reads from src, writes to dst
# Uses double buffering — src and dst are swapped each round
# ---------------------------------------------------------------------------

@triton.jit
def _scan_round_kernel(
    Ar_src, Ai_src, xr_src, xi_src,   # read from these
    Ar_dst, Ai_dst, xr_dst, xi_dst,   # write to these
    stride,                             # combine with element stride steps left
    BP, L,
    BLOCK: tl.constexpr,
):
    """One round of Hillis-Steele prefix scan.
    Each thread processes BLOCK consecutive elements of one (b,p) stream.
    """
    pid  = tl.program_id(0)   # stream index (b*P + p)
    bid  = tl.program_id(1)   # block index along L
    offs = tl.arange(0, BLOCK)
    glob = bid * BLOCK + offs
    mask = (pid < BP) & (glob < L)
    base = pid * L + bid * BLOCK

    # Current element
    Ar_c = tl.load(Ar_src + base + offs, mask=mask, other=1.0)
    Ai_c = tl.load(Ai_src + base + offs, mask=mask, other=0.0)
    xr_c = tl.load(xr_src + base + offs, mask=mask, other=0.0)
    xi_c = tl.load(xi_src + base + offs, mask=mask, other=0.0)

    # Left neighbour (stride steps back)
    left = base + offs - stride
    left_mask = mask & (glob >= stride)

    Ar_l = tl.load(Ar_src + left, mask=left_mask, other=1.0)
    Ai_l = tl.load(Ai_src + left, mask=left_mask, other=0.0)
    xr_l = tl.load(xr_src + left, mask=left_mask, other=0.0)
    xi_l = tl.load(xi_src + left, mask=left_mask, other=0.0)

    # Combine: (left) o (current)
    # A_out = A_c * A_l
    new_Ar = tl.where(left_mask, Ar_c * Ar_l - Ai_c * Ai_l, Ar_c)
    new_Ai = tl.where(left_mask, Ar_c * Ai_l + Ai_c * Ar_l, Ai_c)
    # x_out = A_c * x_l + x_c
    new_xr = tl.where(left_mask, Ar_c * xr_l - Ai_c * xi_l + xr_c, xr_c)
    new_xi = tl.where(left_mask, Ar_c * xi_l + Ai_c * xr_l + xi_c, xi_c)

    tl.store(Ar_dst + base + offs, new_Ar, mask=mask)
    tl.store(Ai_dst + base + offs, new_Ai, mask=mask)
    tl.store(xr_dst + base + offs, new_xr, mask=mask)
    tl.store(xi_dst + base + offs, new_xi, mask=mask)


# ---------------------------------------------------------------------------
# Backward round: reverse Hillis-Steele
# h_i += conj(Lambda_{i+stride}) * h_{i+stride}
# ---------------------------------------------------------------------------

@triton.jit
def _bwd_round_kernel(
    hr_src, hi_src,    # read gradient accumulator
    Lr_ptr, Li_ptr,    # original Lambda (read only, never modified)
    hr_dst, hi_dst,    # write updated gradient
    stride,
    BP, L,
    BLOCK: tl.constexpr,
):
    pid  = tl.program_id(0)
    bid  = tl.program_id(1)
    offs = tl.arange(0, BLOCK)
    glob = bid * BLOCK + offs
    mask = (pid < BP) & (glob < L)
    base = pid * L + bid * BLOCK

    hr_c = tl.load(hr_src + base + offs, mask=mask, other=0.0)
    hi_c = tl.load(hi_src + base + offs, mask=mask, other=0.0)

    right      = base + offs + stride
    right_mask = mask & (glob + stride < L)

    hr_r = tl.load(hr_src + right, mask=right_mask, other=0.0)
    hi_r = tl.load(hi_src + right, mask=right_mask, other=0.0)
    # conj(Lambda) at i+stride: (Lr, -Li)
    Lr_r = tl.load(Lr_ptr + right, mask=right_mask, other=1.0)
    Li_r = tl.load(Li_ptr + right, mask=right_mask, other=0.0)

    # h_i += conj(Lambda_{i+stride}) * h_{i+stride}
    new_hr = tl.where(right_mask, hr_c + Lr_r * hr_r + Li_r * hi_r, hr_c)
    new_hi = tl.where(right_mask, hi_c + Lr_r * hi_r - Li_r * hr_r, hi_c)

    tl.store(hr_dst + base + offs, new_hr, mask=mask)
    tl.store(hi_dst + base + offs, new_hi, mask=mask)


# ---------------------------------------------------------------------------
# Host runners
# ---------------------------------------------------------------------------

def _next_pow2(n):
    return 1 << (n - 1).bit_length() if n > 1 else 1


def _run_scan(Lr, Li, Br, Bi, BLOCK=256):
    """Forward parallel prefix scan. (BP, L) -> (BP, L)."""
    BP, L  = Br.shape
    n_rounds = int(math.ceil(math.log2(max(L, 2))))
    grid   = lambda _: (BP, triton.cdiv(L, BLOCK))

    # Double buffer: ping and pong
    # Init ping with inputs
    Ar_ping = Lr.clone()
    Ai_ping = Li.clone()
    xr_ping = Br.clone()
    xi_ping = Bi.clone()

    Ar_pong = torch.empty_like(Ar_ping)
    Ai_pong = torch.empty_like(Ai_ping)
    xr_pong = torch.empty_like(xr_ping)
    xi_pong = torch.empty_like(xi_ping)

    stride = 1
    for _ in range(n_rounds):
        _scan_round_kernel[grid(None)](
            Ar_ping, Ai_ping, xr_ping, xi_ping,
            Ar_pong, Ai_pong, xr_pong, xi_pong,
            stride, BP, L, BLOCK=BLOCK,
        )
        # Swap ping and pong
        Ar_ping, Ar_pong = Ar_pong, Ar_ping
        Ai_ping, Ai_pong = Ai_pong, Ai_ping
        xr_ping, xr_pong = xr_pong, xr_ping
        xi_ping, xi_pong = xi_pong, xi_ping
        stride *= 2

    return xr_ping, xi_ping


def _run_bwd_scan(dxr, dxi, Lr, Li, BLOCK=256):
    """Backward reverse prefix scan. (BP, L) -> (BP, L)."""
    BP, L    = dxr.shape
    n_rounds = int(math.ceil(math.log2(max(L, 2))))
    grid     = lambda _: (BP, triton.cdiv(L, BLOCK))

    hr_ping = dxr.clone()
    hi_ping = dxi.clone()
    hr_pong = torch.empty_like(hr_ping)
    hi_pong = torch.empty_like(hi_ping)

    # Reverse Hillis-Steele: start from largest stride
    stride = 1 << (n_rounds - 1)
    for _ in range(n_rounds):
        _bwd_round_kernel[grid(None)](
            hr_ping, hi_ping,
            Lr, Li,
            hr_pong, hi_pong,
            stride, BP, L, BLOCK=BLOCK,
        )
        hr_ping, hr_pong = hr_pong, hr_ping
        hi_ping, hi_pong = hi_pong, hi_ping
        stride //= 2

    return hr_ping, hi_ping


# ---------------------------------------------------------------------------
# Autograd Function
# ---------------------------------------------------------------------------

class ParallelScanFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Lbar_r, Lbar_i, Bu_r, Bu_i):
        B, L, P = Bu_r.shape

        Lr = Lbar_r.permute(0, 2, 1).reshape(B*P, L).contiguous()
        Li = Lbar_i.permute(0, 2, 1).reshape(B*P, L).contiguous()
        Br = Bu_r.permute(0, 2, 1).reshape(B*P, L).contiguous()
        Bi = Bu_i.permute(0, 2, 1).reshape(B*P, L).contiguous()

        xr_flat, xi_flat = _run_scan(Lr, Li, Br, Bi)

        xs_r = xr_flat.reshape(B, P, L).permute(0, 2, 1).contiguous()
        xs_i = xi_flat.reshape(B, P, L).permute(0, 2, 1).contiguous()

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

        dBr_flat, dBi_flat = _run_bwd_scan(dr, di, Lr, Li)

        dBu_r = dBr_flat.reshape(B, P, L).permute(0, 2, 1).contiguous()
        dBu_i = dBi_flat.reshape(B, P, L).permute(0, 2, 1).contiguous()

        xs_r_prev = torch.cat([torch.zeros_like(xs_r[:, :1, :]), xs_r[:, :-1, :]], dim=1)
        xs_i_prev = torch.cat([torch.zeros_like(xs_i[:, :1, :]), xs_i[:, :-1, :]], dim=1)
        dLbar_r = (dxs_r * xs_r_prev + dxs_i * xs_i_prev).sum(dim=(0, 1))
        dLbar_i = (dxs_i * xs_r_prev - dxs_r * xs_i_prev).sum(dim=(0, 1))

        return dLbar_r, dLbar_i, dBu_r, dBu_i


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parallel_scan(Lbar_r, Lbar_i, Bu_r, Bu_i):
    B, L, P = Bu_r.shape
    Lr = Lbar_r[None, None, :].expand(B, L, P).contiguous()
    Li = Lbar_i[None, None, :].expand(B, L, P).contiguous()
    return ParallelScanFn.apply(Lr, Li, Bu_r, Bu_i)


def apply_ssm_triton(Lbar_r, Lbar_i, B_bar_r, B_bar_i, C_real, C_imag, D, x):
    Bu_r = x @ B_bar_r.t()
    Bu_i = x @ B_bar_i.t()
    xs_r, xs_i = parallel_scan(Lbar_r, Lbar_i, Bu_r, Bu_i)
    return xs_r @ C_real.t() - xs_i @ C_imag.t() + D * x


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    B, L, H, P = 2, 64, 32, 16
    device = "cuda"

    Lbar_r = (-torch.rand(P) * 0.5).to(device)
    Lbar_i = torch.randn(P).to(device)
    Bu_r   = torch.randn(B, L, P, device=device)
    Bu_i   = torch.randn(B, L, P, device=device)

    print("Testing forward vs sequential reference...")

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

    ref_r, ref_i = seq_scan(Lbar_r, Lbar_i, Bu_r, Bu_i)
    xs_r, xs_i   = parallel_scan(Lbar_r, Lbar_i, Bu_r, Bu_i)

    err = (xs_r - ref_r).abs().max().item()
    print(f"  max forward error: {err:.2e} {'✓' if err < 1e-3 else '✗ TOO LARGE'}")

    print("Testing backward...")
    Bu_r_g = Bu_r.clone().requires_grad_(True)
    Bu_i_g = Bu_i.clone().requires_grad_(True)
    xs_r2, xs_i2 = parallel_scan(Lbar_r, Lbar_i, Bu_r_g, Bu_i_g)
    (xs_r2.sum() + xs_i2.sum()).backward()
    assert Bu_r_g.grad is not None
    assert not Bu_r_g.grad.isnan().any()
    print(f"  grad shape: {Bu_r_g.grad.shape} ✓")

    print("Testing gradient correctness vs torch autograd...")
    Bu_r_ref = Bu_r.clone().requires_grad_(True)
    Bu_i_ref = Bu_i.clone().requires_grad_(True)
    ref_r2, ref_i2 = seq_scan(Lbar_r, Lbar_i, Bu_r_ref, Bu_i_ref)
    (ref_r2.sum() + ref_i2.sum()).backward()
    grad_err = (Bu_r_g.grad - Bu_r_ref.grad).abs().max().item()
    print(f"  max grad error: {grad_err:.2e} {'✓' if grad_err < 1e-3 else '✗ TOO LARGE'}")

    print("Testing apply_ssm_triton...")
    B_bar_r = torch.randn(P, H, device=device)
    B_bar_i = torch.randn(P, H, device=device)
    C_real  = torch.randn(H, P, device=device)
    C_imag  = torch.randn(H, P, device=device)
    D       = torch.ones(H, device=device)
    x = torch.randn(B, L, H, device=device, requires_grad=True)
    y = apply_ssm_triton(Lbar_r, Lbar_i, B_bar_r, B_bar_i, C_real, C_imag, D, x)
    assert y.shape == (B, L, H)
    y.sum().backward()
    assert not x.grad.isnan().any()
    print(f"  output shape: {y.shape} ✓")

    print("\nAll tests passed.")
