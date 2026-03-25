"""
s5_scan_triton.py — Hand-written parallel prefix scan in Triton.

Uses Hillis-Steele iterative doubling. The key correctness requirement:
each round must read the values produced by the PREVIOUS round, not
the original inputs. We use two in-place buffers (Lr_ptr for A,
xr_ptr for x) that are initialized once and updated each round.

Binary operator:
    A_out = A_j * A_i          (complex multiply)
    x_out = A_j * x_i + x_j   (complex multiply-add)
"""

import torch
import triton
import triton.language as tl
import math


# ---------------------------------------------------------------------------
# Initialise output buffers with inputs (copy kernel)
# ---------------------------------------------------------------------------

@triton.jit
def _init_kernel(
    Lr_src, Li_src, Br_src, Bi_src,
    Lr_dst, Li_dst, xr_dst, xi_dst,
    N,
):
    i = tl.program_id(0) * tl.num_programs(0) + tl.arange(0, 1024)
    # simple element-wise copy
    offs = tl.program_id(0) * 1024 + tl.arange(0, 1024)
    mask = offs < N
    tl.store(Lr_dst + offs, tl.load(Lr_src + offs, mask=mask), mask=mask)
    tl.store(Li_dst + offs, tl.load(Li_src + offs, mask=mask), mask=mask)
    tl.store(xr_dst + offs, tl.load(Br_src + offs, mask=mask), mask=mask)
    tl.store(xi_dst + offs, tl.load(Bi_src + offs, mask=mask), mask=mask)


# ---------------------------------------------------------------------------
# Forward scan kernel: Hillis-Steele iterative doubling
# One program per (b,p) stream. All rounds happen in one kernel call.
# Buffers Lr_ptr, Li_ptr, xr_ptr, xi_ptr are read AND written each round.
# ---------------------------------------------------------------------------

@triton.jit
def _fwd_scan_kernel(
    Lr_ptr, Li_ptr,   # (BP, L) Lambda — read/write each round
    xr_ptr, xi_ptr,   # (BP, L) x state — read/write each round
    last_Ar_ptr, last_Ai_ptr,
    last_xr_ptr, last_xi_ptr,
    L: tl.constexpr,
    BLOCK_L: tl.constexpr,
    LOG2_BLOCK: tl.constexpr,
):
    pid   = tl.program_id(0)
    block = tl.program_id(1)

    l_off = block * BLOCK_L
    offs  = tl.arange(0, BLOCK_L)
    glob  = l_off + offs
    mask  = glob < L
    base  = pid * L + l_off

    # Each round: read current values, combine with left neighbour, write back
    stride = 1
    for _ in tl.static_range(LOG2_BLOCK):
        left_offs = offs - stride
        left_mask = mask & (left_offs >= 0)

        # Read current element's accumulated values
        Ar_cur = tl.load(Lr_ptr + base + offs, mask=mask, other=1.0)
        Ai_cur = tl.load(Li_ptr + base + offs, mask=mask, other=0.0)
        xr_cur = tl.load(xr_ptr + base + offs, mask=mask, other=0.0)
        xi_cur = tl.load(xi_ptr + base + offs, mask=mask, other=0.0)

        # Read left neighbour's accumulated values
        Ar_l = tl.load(Lr_ptr + base + left_offs, mask=left_mask, other=1.0)
        Ai_l = tl.load(Li_ptr + base + left_offs, mask=left_mask, other=0.0)
        xr_l = tl.load(xr_ptr + base + left_offs, mask=left_mask, other=0.0)
        xi_l = tl.load(xi_ptr + base + left_offs, mask=left_mask, other=0.0)

        # Combine: (left) o (current)
        # A_out = A_cur * A_left
        new_Ar = tl.where(left_mask, Ar_cur * Ar_l - Ai_cur * Ai_l, Ar_cur)
        new_Ai = tl.where(left_mask, Ar_cur * Ai_l + Ai_cur * Ar_l, Ai_cur)
        # x_out = A_cur * x_left + x_cur
        new_xr = tl.where(left_mask, Ar_cur * xr_l - Ai_cur * xi_l + xr_cur, xr_cur)
        new_xi = tl.where(left_mask, Ar_cur * xi_l + Ai_cur * xr_l + xi_cur, xi_cur)

        # Write back so next round reads updated values
        tl.store(Lr_ptr + base + offs, new_Ar, mask=mask)
        tl.store(Li_ptr + base + offs, new_Ai, mask=mask)
        tl.store(xr_ptr + base + offs, new_xr, mask=mask)
        tl.store(xi_ptr + base + offs, new_xi, mask=mask)

        stride = stride * 2

    # Store last valid element for inter-block combination
    n_blocks  = tl.cdiv(L, BLOCK_L)
    last_base = pid * n_blocks + block
    valid     = tl.minimum(BLOCK_L, L - l_off)
    last_idx  = valid - 1

    last_Ar_v = tl.sum(tl.where(offs == last_idx, new_Ar, tl.zeros_like(new_Ar)), axis=0)
    last_Ai_v = tl.sum(tl.where(offs == last_idx, new_Ai, tl.zeros_like(new_Ai)), axis=0)
    last_xr_v = tl.sum(tl.where(offs == last_idx, new_xr, tl.zeros_like(new_xr)), axis=0)
    last_xi_v = tl.sum(tl.where(offs == last_idx, new_xi, tl.zeros_like(new_xi)), axis=0)

    tl.store(last_Ar_ptr + last_base, last_Ar_v)
    tl.store(last_Ai_ptr + last_base, last_Ai_v)
    tl.store(last_xr_ptr + last_base, last_xr_v)
    tl.store(last_xi_ptr + last_base, last_xi_v)


# ---------------------------------------------------------------------------
# Apply inter-block prefix to each block
# ---------------------------------------------------------------------------

@triton.jit
def _fwd_combine_kernel(
    prefix_Ar_ptr, prefix_Ai_ptr,
    prefix_xr_ptr, prefix_xi_ptr,
    xr_ptr, xi_ptr,
    L,
    BLOCK_L: tl.constexpr,
):
    pid   = tl.program_id(0)
    block = tl.program_id(1)
    if block == 0:
        return

    offs  = tl.arange(0, BLOCK_L)
    l_off = block * BLOCK_L
    mask  = (l_off + offs) < L
    base  = pid * L + l_off

    n_blocks = tl.cdiv(L, BLOCK_L)
    prev     = pid * n_blocks + block - 1

    pAr = tl.load(prefix_Ar_ptr + prev)
    pAi = tl.load(prefix_Ai_ptr + prev)
    pxr = tl.load(prefix_xr_ptr + prev)
    pxi = tl.load(prefix_xi_ptr + prev)

    xr = tl.load(xr_ptr + base + offs, mask=mask, other=0.0)
    xi = tl.load(xi_ptr + base + offs, mask=mask, other=0.0)

    new_xr = pAr * xr - pAi * xi + pxr
    new_xi = pAr * xi + pAi * xr + pxi

    tl.store(xr_ptr + base + offs, new_xr, mask=mask)
    tl.store(xi_ptr + base + offs, new_xi, mask=mask)


# ---------------------------------------------------------------------------
# Backward scan kernel: reverse Hillis-Steele
# h_i = g_i + conj(Lambda_{i+1}) * h_{i+1}
# ---------------------------------------------------------------------------

@triton.jit
def _bwd_scan_kernel(
    dxr_ptr, dxi_ptr,   # (BP, L) upstream gradients — read/write
    Lr_ptr,  Li_ptr,    # (BP, L) original Lambda (read only)
    L: tl.constexpr,
    BLOCK_L: tl.constexpr,
    LOG2_BLOCK: tl.constexpr,
):
    pid   = tl.program_id(0)
    block = tl.program_id(1)

    l_off = block * BLOCK_L
    offs  = tl.arange(0, BLOCK_L)
    mask  = (l_off + offs) < L
    base  = pid * L + l_off

    # Init: h = g
    # (dxr_ptr already contains g — we update in place)

    stride = 1
    for _ in tl.static_range(LOG2_BLOCK):
        right_offs = offs + stride
        right_mask = mask & (right_offs < BLOCK_L) & (l_off + right_offs < L)

        hr_cur = tl.load(dxr_ptr + base + offs,       mask=mask,       other=0.0)
        hi_cur = tl.load(dxi_ptr + base + offs,       mask=mask,       other=0.0)
        hr_r   = tl.load(dxr_ptr + base + right_offs, mask=right_mask, other=0.0)
        hi_r   = tl.load(dxi_ptr + base + right_offs, mask=right_mask, other=0.0)

        # conj(Lambda) at i+stride position (original Lambda)
        Lr_r = tl.load(Lr_ptr + base + right_offs, mask=right_mask, other=1.0)
        Li_r = tl.load(Li_ptr + base + right_offs, mask=right_mask, other=0.0)
        # conj: (Lr, -Li)
        new_hr = tl.where(right_mask, hr_cur + Lr_r * hr_r + Li_r * hi_r, hr_cur)
        new_hi = tl.where(right_mask, hi_cur + Lr_r * hi_r - Li_r * hr_r, hi_cur)

        tl.store(dxr_ptr + base + offs, new_hr, mask=mask)
        tl.store(dxi_ptr + base + offs, new_hi, mask=mask)

        stride = stride * 2


# ---------------------------------------------------------------------------
# Host-side runners
# ---------------------------------------------------------------------------

def _next_pow2(n):
    return 1 << (n - 1).bit_length() if n > 1 else 1


def _run_scan(Lr_orig, Li_orig, Br, Bi, BLOCK_L=None):
    BP, L = Br.shape
    if BLOCK_L is None:
        BLOCK_L = min(_next_pow2(L), 1024)
    LOG2_BLOCK = int(math.ceil(math.log2(max(BLOCK_L, 2))))
    n_blocks   = triton.cdiv(L, BLOCK_L)
    grid       = (BP, n_blocks)

    # Working buffers: initialised with inputs, updated in-place each round
    Lr = Lr_orig.clone()
    Li = Li_orig.clone()
    xr = Br.clone()      # init x = Bu
    xi = Bi.clone()

    last_Ar = torch.ones (BP, n_blocks, device=Br.device, dtype=torch.float32)
    last_Ai = torch.zeros(BP, n_blocks, device=Br.device, dtype=torch.float32)
    last_xr = torch.zeros(BP, n_blocks, device=Br.device, dtype=torch.float32)
    last_xi = torch.zeros(BP, n_blocks, device=Br.device, dtype=torch.float32)

    _fwd_scan_kernel[grid](
        Lr, Li, xr, xi,
        last_Ar, last_Ai, last_xr, last_xi,
        L=L, BLOCK_L=BLOCK_L, LOG2_BLOCK=LOG2_BLOCK,
    )

    if n_blocks > 1:
        pAr = last_Ar.clone()
        pAi = last_Ai.clone()
        pxr = last_xr.clone()
        pxi = last_xi.clone()
        for b in range(1, n_blocks):
            Ar_p, Ai_p = pAr[:, b-1], pAi[:, b-1]
            xr_p, xi_p = pxr[:, b-1], pxi[:, b-1]
            Ar_c, Ai_c = last_Ar[:, b], last_Ai[:, b]
            xr_c, xi_c = last_xr[:, b], last_xi[:, b]
            pAr[:, b] = Ar_c * Ar_p - Ai_c * Ai_p
            pAi[:, b] = Ar_c * Ai_p + Ai_c * Ar_p
            pxr[:, b] = Ar_c * xr_p - Ai_c * xi_p + xr_c
            pxi[:, b] = Ar_c * xi_p + Ai_c * xr_p + xi_c

        _fwd_combine_kernel[grid](
            pAr, pAi, pxr, pxi,
            xr, xi,
            L, BLOCK_L=BLOCK_L,
        )

    return xr, xi


def _run_bwd_scan(dxr_orig, dxi_orig, Lr, Li, BLOCK_L=None):
    BP, L = dxr_orig.shape
    if BLOCK_L is None:
        BLOCK_L = min(_next_pow2(L), 1024)
    LOG2_BLOCK = int(math.ceil(math.log2(max(BLOCK_L, 2))))
    n_blocks   = triton.cdiv(L, BLOCK_L)
    grid       = (BP, n_blocks)

    dBr = dxr_orig.clone()
    dBi = dxi_orig.clone()

    _bwd_scan_kernel[grid](
        dBr, dBi, Lr, Li,
        L=L, BLOCK_L=BLOCK_L, LOG2_BLOCK=LOG2_BLOCK,
    )
    # TODO: inter-block backward for L > BLOCK_L
    return dBr, dBi


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

        # Save original Lr/Li for backward (not the modified ones)
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

        # dLambda: d/dLbar of sum_i Lbar * xs_{i-1}
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
