"""
G-Mem + EDLM: Gated Latent Memory for Masked Diffusion Language Models
=======================================================================

Bolts the G-MemLLM latent memory bank onto the EDLM denoising transformer.
The memory carries context across text segments during both training and
sampling, helping the diffusion model maintain long-range coherence.

Architecture:
  DenoisingTransformer (frozen or trainable)
    → hidden states (B, L, dim)
    → LatentMemoryBank retrieves + enhances hidden states
    → enhanced hidden states → out_proj → logits

The memory bank is updated (consolidated) at each forward pass, and the
updated memory state is returned for use in the next segment/step.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Any
from pathlib import Path

# Import from the EDLM implementation
import sys
sys.path.insert(0, str(Path(__file__).parent.resolve()))
sys.path.insert(0, str((Path(__file__).parent / "../../model").resolve()))

from edlm import (
    DenoisingTransformer, EnergyModel, MDLMLoss, NCELoss,
    LogLinearSchedule, mask_tokens, sample, Sampler, SamplingStepper,
    SamplingStep,
    load_kalevala, create_dataloader, as_text, nce_loss,
    estimate_nll,
)
from gmem import LatentMemoryBank
from cmm import RecursiveBlock, RMSNorm
from ema import EMA


class MemAugmentedDenoiser(nn.Module):
    """
    Denoising transformer with memory bank at the output (original design).
    Memory sits between final hidden states and output projection.
    """

    def __init__(self, denoiser: DenoisingTransformer,
                 num_slots: int = 64,
                 memory_dim: int = 128,
                 num_heads: int = 4,
                 freeze_denoiser: bool = False):
        super().__init__()
        self.denoiser = denoiser
        self.memory_bank = LatentMemoryBank(
            hidden_dim=denoiser.dim,
            memory_dim=memory_dim,
            num_slots=num_slots,
            num_heads=num_heads,
        )
        self.n_vocab = denoiser.n_vocab
        self.mask_id = denoiser.mask_id
        self.dim = denoiser.dim

        if freeze_denoiser:
            for p in self.denoiser.parameters():
                p.requires_grad = False

    def forward(self, xt, t, memory=None):
        hidden = self.denoiser.get_hidden(xt, t)
        enhanced, updated_memory, importance_scores = self.memory_bank(
            hidden, memory)
        logits = self.denoiser.out_proj(enhanced)
        return logits, updated_memory, importance_scores

    def get_hidden(self, xt, t, memory=None):
        hidden = self.denoiser.get_hidden(xt, t)
        enhanced, _, _ = self.memory_bank(hidden, memory)
        return enhanced


class MidLayerMemDenoiser(nn.Module):
    """
    Denoising transformer with memory bank injected mid-network.

    Architecture (4 layers, memory after layer 2):
      tok_embed(xt) + time_embed(t)
        → Block 0 → Block 1           [early layers]
        → LatentMemoryBank             [retrieve + enhance + consolidate]
        → Block 2 → Block 3           [late layers]
        → final_norm → out_proj → logits

    The late layers can transform and integrate the memory signal through
    attention and MLPs, giving the network real capacity to use the memory
    rather than just linearly mixing it before the output projection.

    The memory bank:
      - READS from mid-layer representations (richer than final-layer)
      - WRITES consolidated information back into the stream
      - Late layers then attend over memory-enhanced representations
    """

    def __init__(self, denoiser: DenoisingTransformer,
                 inject_after: int = 2,
                 num_slots: int = 64,
                 memory_dim: int = 128,
                 num_heads: int = 4,
                 freeze_denoiser: bool = False):
        super().__init__()
        self.denoiser = denoiser
        self.inject_after = inject_after
        self.memory_bank = LatentMemoryBank(
            hidden_dim=denoiser.dim,
            memory_dim=memory_dim,
            num_slots=num_slots,
            num_heads=num_heads,
        )
        self.n_vocab = denoiser.n_vocab
        self.mask_id = denoiser.mask_id
        self.dim = denoiser.dim

        num_blocks = len(denoiser.blocks)
        assert 0 < inject_after < num_blocks, \
            f"inject_after={inject_after} must be in (0, {num_blocks})"

        if freeze_denoiser:
            for p in self.denoiser.parameters():
                p.requires_grad = False

    def _forward_split(self, xt, t, memory=None):
        """Run blocks in two halves with memory injection in between."""
        B, L = xt.shape
        d = self.denoiser

        h = d.tok_embed(xt)
        c = d._time_embed(t)
        cos = d.rot_cos[:L][None, None, :, :]
        sin = d.rot_sin[:L][None, None, :, :]

        # Early layers (0 .. inject_after-1)
        for block in d.blocks[:self.inject_after]:
            h = block(h, c, cos, sin)

        # Memory bank: retrieve, enhance, consolidate
        enhanced, updated_memory, importance_scores = self.memory_bank(
            h, memory)

        # Late layers (inject_after .. end)
        h = enhanced
        for block in d.blocks[self.inject_after:]:
            h = block(h, c, cos, sin)

        h = d.final_norm(h)
        return h, updated_memory, importance_scores

    def forward(self, xt, t, memory=None):
        h, updated_memory, importance_scores = self._forward_split(
            xt, t, memory)
        logits = self.denoiser.out_proj(h)
        return logits, updated_memory, importance_scores

    def get_hidden(self, xt, t, memory=None):
        h, _, _ = self._forward_split(xt, t, memory)
        return h


class PonderBlock(nn.Module):
    """
    CMM-style recursive refinement block for use inside a larger network.

    A small shared transformer block applied N_iter times with post-norm
    residuals and tanh activation to guarantee contraction.

    Unlike the full CMM which has separate L/H modules and deep supervision,
    this is a single drop-in layer: input → N iterations → output.
    Optional additive noise (NSDE) during training for robustness.
    """
    def __init__(self, dim: int, seq_len: int, N_iter: int = 4,
                 num_heads: int = 4, mlp_ratio: int = 4,
                 noise_sigma: float = 0.0):
        super().__init__()
        self.N_iter = N_iter
        self.noise_sigma = noise_sigma

        # Single shared block applied recursively
        # Use attention since this sits inside an LLM-like architecture
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.Tanh(),  # bounded activation for contraction
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply N_iter recursive refinement steps."""
        h = x
        for _ in range(self.N_iter):
            # Self-attention + residual + post-norm
            attn_out, _ = self.attn(h, h, h)
            h = self.norm1(h + attn_out)
            # MLP + residual + post-norm
            h = self.norm2(h + self.mlp(h))
            # Optional NSDE noise
            if self.noise_sigma > 0 and self.training:
                h = h + torch.randn_like(h) * self.noise_sigma
        return h


class MemPonderDenoiser(nn.Module):
    """
    Denoising transformer with memory bank + recursive pondering.

    Architecture (4 layers, memory + ponder after layer 2):
      tok_embed(xt) + time_embed(t)
        → Block 0 → Block 1               [early layers: feature extraction]
        → LatentMemoryBank                 [retrieve context from memory]
        → PonderBlock (N_iter iterations)  [reason about retrieved memory]
        → Block 2 → Block 3               [late layers: integrate reasoning]
        → final_norm → out_proj → logits

    The pondering block iteratively refines the memory-enhanced
    representations. This gives the model "thinking time" to integrate
    retrieved context before the late layers process it — unlike a single
    linear gate which is all the original G-MemLLM provides.
    """

    def __init__(self, denoiser: DenoisingTransformer,
                 inject_after: int = 2,
                 num_slots: int = 64,
                 memory_dim: int = 128,
                 num_heads: int = 4,
                 N_iter: int = 4,
                 ponder_noise: float = 0.0,
                 freeze_denoiser: bool = False):
        super().__init__()
        self.denoiser = denoiser
        self.inject_after = inject_after
        self.memory_bank = LatentMemoryBank(
            hidden_dim=denoiser.dim,
            memory_dim=memory_dim,
            num_slots=num_slots,
            num_heads=num_heads,
        )
        self.ponder = PonderBlock(
            dim=denoiser.dim,
            seq_len=512,  # not used by attention variant
            N_iter=N_iter,
            num_heads=num_heads,
            noise_sigma=ponder_noise,
        )
        self.n_vocab = denoiser.n_vocab
        self.mask_id = denoiser.mask_id
        self.dim = denoiser.dim

        num_blocks = len(denoiser.blocks)
        assert 0 < inject_after < num_blocks

        if freeze_denoiser:
            for p in self.denoiser.parameters():
                p.requires_grad = False

    def _forward_split(self, xt, t, memory=None):
        B, L = xt.shape
        d = self.denoiser

        h = d.tok_embed(xt)
        c = d._time_embed(t)
        cos = d.rot_cos[:L][None, None, :, :]
        sin = d.rot_sin[:L][None, None, :, :]

        # Early layers
        for block in d.blocks[:self.inject_after]:
            h = block(h, c, cos, sin)

        # Memory: retrieve + enhance + consolidate
        h, updated_memory, importance_scores = self.memory_bank(h, memory)

        # Ponder: iterative refinement over memory-enhanced representations
        h = self.ponder(h)

        # Late layers
        for block in d.blocks[self.inject_after:]:
            h = block(h, c, cos, sin)

        h = d.final_norm(h)
        return h, updated_memory, importance_scores

    def forward(self, xt, t, memory=None):
        h, updated_memory, importance_scores = self._forward_split(
            xt, t, memory)
        logits = self.denoiser.out_proj(h)
        return logits, updated_memory, importance_scores

    def get_hidden(self, xt, t, memory=None):
        h, _, _ = self._forward_split(xt, t, memory)
        return h


# ============================================================
# Loss wrapper that includes memory regularization
# ============================================================

@dataclass
class MemMDLMLoss:
    """MDLM loss + memory sparsity/entropy regularization."""
    schedule: Any
    mask_id: int
    lambda_sparsity: float = 0.01
    lambda_entropy: float = 0.01
    t_min: float = 1e-4

    def perturb(self, x0, t=None, answer_mask=None):
        inner = MDLMLoss(self.schedule, self.mask_id, self.t_min)
        return inner.perturb(x0, t, answer_mask)

    def __call__(self, logits, x0, is_masked, importance_scores):
        # Base MDLM loss
        inner = MDLMLoss(self.schedule, self.mask_id, self.t_min)
        base_loss = inner(logits, x0, is_masked)

        # Memory regularization
        l_sparsity = importance_scores.abs().mean()
        p = F.softmax(importance_scores, dim=-1)
        l_entropy = (p * p.log()).sum(dim=-1).mean()

        mem_loss = self.lambda_sparsity * l_sparsity + self.lambda_entropy * l_entropy
        return base_loss + mem_loss, base_loss, mem_loss


# ============================================================
# Sampling with memory
# ============================================================

@torch.no_grad()
def sample_with_memory(
    mem_denoiser: MemAugmentedDenoiser,
    schedule,
    batch_size,
    seq_len,
    num_steps=256,
    energy_model=None,
    k=8,
    window_w=0.2,
    device=torch.device("cpu"),
    initial_memory=None,
):
    """
    Sample from the memory-augmented denoiser.

    The memory is updated at each denoising step, allowing the model
    to accumulate and refine its understanding as tokens are revealed.
    """
    sampler = Sampler(schedule, mem_denoiser.mask_id, mem_denoiser.n_vocab)
    xt, stepper = sampler(batch_size, seq_len, device, num_steps,
                          energy_model=energy_model, k=k, window_w=window_w)

    memory = initial_memory

    for s in stepper:
        logits, memory, _ = mem_denoiser(xt, s.t, memory=memory)
        xt = s.sample(xt, logits)

    return xt, memory


@torch.no_grad()
def sample_long_with_memory(
    mem_denoiser: MemAugmentedDenoiser,
    schedule,
    num_segments,
    seq_len,
    num_steps=128,
    device=torch.device("cpu"),
):
    """
    Generate a long sequence by sampling segment-by-segment,
    carrying memory forward. This is where the memory bank shines —
    each segment's generation is informed by all previous segments.
    """
    memory = None
    all_segments = []

    for i in range(num_segments):
        xt, memory = sample_with_memory(
            mem_denoiser, schedule,
            batch_size=1, seq_len=seq_len,
            num_steps=num_steps,
            device=device,
            initial_memory=memory,
        )
        all_segments.append(xt)

    return torch.cat(all_segments, dim=1)  # (1, num_segments * seq_len)


# ============================================================
# Training
# ============================================================

def evaluate_and_sample(mem_denoiser, denoiser, schedule, long_loader,
                        seq_len, segments_per_group, sample_steps, device):
    """Shared evaluation: samples + perplexity comparison."""
    mem_denoiser.eval()

    # Single-segment samples
    print("\n--- Base MDLM (no memory) ---")
    denoiser.eval()
    s = sample(denoiser, schedule, batch_size=4, seq_len=seq_len,
               num_steps=sample_steps, device=device)
    for i in range(4):
        print(f"  {i+1}: {repr(as_text(s[i]))}")

    print("\n--- G-Mem Denoiser (single segment, fresh memory) ---")
    s, _ = sample_with_memory(mem_denoiser, schedule,
                              batch_size=4, seq_len=seq_len,
                              num_steps=sample_steps, device=device)
    for i in range(4):
        print(f"  {i+1}: {repr(as_text(s[i]))}")

    # Multi-segment: this is where memory should help
    print(f"\n--- G-Mem Denoiser ({segments_per_group} segments with carried memory) ---")
    long_sample = sample_long_with_memory(
        mem_denoiser, schedule,
        num_segments=segments_per_group, seq_len=seq_len,
        num_steps=sample_steps, device=device)
    print(f"  {repr(as_text(long_sample[0]))}")

    # Perplexity comparison on held-out segments
    print("\n--- Perplexity comparison (multi-segment) ---")
    x0_long = next(long_loader).to(device)
    memory = None
    print(f"{'Segment':>8} | {'Base CE':>8} | {'Mem CE':>8} | {'Diff':>8}")
    print("-" * 42)
    for seg_i in range(segments_per_group):
        x0 = x0_long[:, seg_i * seq_len : (seg_i + 1) * seq_len]

        mdlm_l = MDLMLoss(schedule, denoiser.mask_id)
        xt, t, is_masked = mdlm_l.perturb(x0)
        with torch.no_grad():
            base_logits = denoiser(xt, t)
        base_ce = mdlm_l(base_logits, x0, is_masked).item()

        with torch.no_grad():
            mem_logits, memory, _ = mem_denoiser(xt, t, memory=memory)
        mem_ce = MDLMLoss(schedule, denoiser.mask_id)(mem_logits, x0, is_masked).item()

        diff = base_ce - mem_ce
        print(f"{seg_i+1:>8} | {base_ce:>8.4f} | {mem_ce:>8.4f} | {diff:>+8.4f}")


def train_two_phase():
    """Original paper approach: train base denoiser first, then bolt on memory."""
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Mode: TWO-PHASE (freeze denoiser, then train memory)")

    n_vocab, seq_len, batch_size = 256, 128, 32
    dim, num_heads, num_layers = 256, 4, 4
    mdlm_steps, mdlm_lr = 10000, 3e-4
    mem_steps, mem_lr = 5000, 3e-4
    num_slots, memory_dim = 64, 128
    segments_per_group = 4
    sample_steps = 128

    text = load_kalevala()
    dataloader = create_dataloader(text, batch_size=batch_size, length=seq_len, stride=64)
    long_dataloader = create_dataloader(
        text, batch_size=batch_size,
        length=seq_len * segments_per_group, stride=seq_len)
    schedule = LogLinearSchedule(eps=1e-3)

    def infinite_loader(dl):
        while True:
            yield from dl

    # === Phase 1: Train base denoiser ===
    print("=" * 60)
    print("Phase 1: Training MDLM Denoiser")
    print("=" * 60)

    denoiser = DenoisingTransformer(
        n_vocab, seq_len * segments_per_group,
        dim, num_heads, num_layers).to(device)
    print(f"Denoiser params: {sum(p.numel() for p in denoiser.parameters()):,}")

    ema_den = EMA(denoiser, mu=0.999)
    opt = torch.optim.AdamW(denoiser.parameters(), lr=mdlm_lr)
    loader = infinite_loader(dataloader)

    ema_lv = None
    for step in range(1, mdlm_steps + 1):
        x0 = next(loader).to(device)
        denoiser.train()
        opt.zero_grad()
        mdlm_loss = MDLMLoss(schedule, denoiser.mask_id)
        xt, t, is_masked = mdlm_loss.perturb(x0)
        logits = denoiser(xt, t)
        loss = mdlm_loss(logits, x0, is_masked)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
        opt.step()
        ema_den.update(denoiser)
        lv = loss.item()
        ema_lv = lv if ema_lv is None else 0.99 * ema_lv + 0.01 * lv
        if step % 500 == 0:
            print(f"  step {step:5d}  loss {lv:.4f}  ema {ema_lv:.4f}")

    ema_den.apply(denoiser)
    denoiser.eval()

    # === Phase 2: Train memory (denoiser frozen) ===
    print("\n" + "=" * 60)
    print("Phase 2: Training Memory-Augmented Denoiser (denoiser frozen)")
    print("=" * 60)

    mem_denoiser = MemAugmentedDenoiser(
        denoiser, num_slots=num_slots, memory_dim=memory_dim,
        num_heads=num_heads, freeze_denoiser=True).to(device)

    mem_params = sum(p.numel() for p in mem_denoiser.memory_bank.parameters())
    total_params = sum(p.numel() for p in mem_denoiser.parameters())
    print(f"Memory bank params: {mem_params:,} / {total_params:,} "
          f"({100*mem_params/total_params:.2f}%)")

    opt_mem = torch.optim.AdamW(mem_denoiser.memory_bank.parameters(), lr=mem_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt_mem, T_max=mem_steps)
    long_loader = infinite_loader(long_dataloader)
    loss_fn = MemMDLMLoss(schedule, denoiser.mask_id)

    ema_lv = None
    for step in range(1, mem_steps + 1):
        x0_long = next(long_loader).to(device)
        mem_denoiser.train()
        memory = None
        total_loss = 0.0

        for seg_i in range(segments_per_group):
            x0 = x0_long[:, seg_i * seq_len : (seg_i + 1) * seq_len]
            xt, t, is_masked = loss_fn.perturb(x0)
            logits, memory, importance_scores = mem_denoiser(xt, t, memory=memory)
            loss, _, _ = loss_fn(logits, x0, is_masked, importance_scores)
            total_loss = total_loss + loss
            memory = memory.detach()

        total_loss = total_loss / segments_per_group
        opt_mem.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(mem_denoiser.memory_bank.parameters(), 1.0)
        opt_mem.step()
        scheduler.step()

        lv = total_loss.item()
        ema_lv = lv if ema_lv is None else 0.99 * ema_lv + 0.01 * lv
        if step % 250 == 0:
            print(f"  step {step:5d}  loss {lv:.4f}  ema {ema_lv:.4f}  "
                  f"lr {scheduler.get_last_lr()[0]:.2e}")

    # === Evaluate ===
    print("\n" + "=" * 60)
    print("Evaluation")
    print("=" * 60)
    long_loader = infinite_loader(long_dataloader)
    evaluate_and_sample(mem_denoiser, denoiser, schedule, long_loader,
                        seq_len, segments_per_group, sample_steps, device)

    torch.save(mem_denoiser.memory_bank.state_dict(), "gmem_diffusion_twophase.pt")
    print("\nSaved to gmem_diffusion_twophase.pt")


def train_joint(placement="mid"):
    """Joint training: denoiser + memory bank trained together from scratch."""
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Mode: JOINT (denoiser + memory trained together, placement={placement})")

    n_vocab, seq_len, batch_size = 256, 128, 32
    dim, num_heads, num_layers = 256, 4, 4
    total_steps = 10000
    lr = 3e-4
    num_slots, memory_dim = 64, 128
    segments_per_group = 4
    sample_steps = 128

    text = load_kalevala()
    long_dataloader = create_dataloader(
        text, batch_size=batch_size,
        length=seq_len * segments_per_group, stride=seq_len)
    schedule = LogLinearSchedule(eps=1e-3)

    def infinite_loader(dl):
        while True:
            yield from dl

    # Build memory-augmented denoiser from scratch (nothing frozen)
    denoiser = DenoisingTransformer(
        n_vocab, seq_len * segments_per_group,
        dim, num_heads, num_layers).to(device)

    if placement == "mid":
        inject_after = num_layers // 2  # after layer 2 of 4
        mem_denoiser = MidLayerMemDenoiser(
            denoiser, inject_after=inject_after,
            num_slots=num_slots, memory_dim=memory_dim,
            num_heads=num_heads, freeze_denoiser=False).to(device)
    elif placement == "ponder":
        inject_after = num_layers // 2
        mem_denoiser = MemPonderDenoiser(
            denoiser, inject_after=inject_after,
            num_slots=num_slots, memory_dim=memory_dim,
            num_heads=num_heads, N_iter=4, ponder_noise=0.01,
            freeze_denoiser=False).to(device)
    else:
        mem_denoiser = MemAugmentedDenoiser(
            denoiser, num_slots=num_slots, memory_dim=memory_dim,
            num_heads=num_heads, freeze_denoiser=False).to(device)

    total_params = sum(p.numel() for p in mem_denoiser.parameters())
    mem_params = sum(p.numel() for p in mem_denoiser.memory_bank.parameters())
    den_params = sum(p.numel() for p in denoiser.parameters())
    print(f"Total params: {total_params:,} "
          f"(denoiser {den_params:,} + memory {mem_params:,})")

    # Also train a bare denoiser as baseline (same arch, same data)
    baseline = DenoisingTransformer(
        n_vocab, seq_len * segments_per_group,
        dim, num_heads, num_layers).to(device)
    baseline.load_state_dict(denoiser.state_dict())  # same init

    # Single optimizer for everything
    opt = torch.optim.AdamW(mem_denoiser.parameters(), lr=lr)
    opt_base = torch.optim.AdamW(baseline.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    scheduler_base = torch.optim.lr_scheduler.CosineAnnealingLR(opt_base, T_max=total_steps)

    long_loader = infinite_loader(long_dataloader)
    loss_fn = MemMDLMLoss(schedule, denoiser.mask_id)

    print("=" * 60)
    print(f"Training jointly for {total_steps} steps")
    print(f"  {segments_per_group} segments x {seq_len} tokens per step")
    print("=" * 60)

    ema_lv = None
    ema_base_lv = None
    for step in range(1, total_steps + 1):
        x0_long = next(long_loader).to(device)

        # --- Train memory-augmented model on multi-segment ---
        mem_denoiser.train()
        memory = None
        total_loss = 0.0

        # Share the same noise across both models for fair comparison
        all_xt, all_t, all_masked = [], [], []
        for seg_i in range(segments_per_group):
            x0 = x0_long[:, seg_i * seq_len : (seg_i + 1) * seq_len]
            xt, t, is_masked = loss_fn.perturb(x0)
            all_xt.append(xt)
            all_t.append(t)
            all_masked.append(is_masked)

            logits, memory, importance_scores = mem_denoiser(xt, t, memory=memory)
            loss, _, _ = loss_fn(logits, x0, is_masked, importance_scores)
            total_loss = total_loss + loss
            memory = memory.detach()

        total_loss = total_loss / segments_per_group
        opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(mem_denoiser.parameters(), 1.0)
        opt.step()
        scheduler.step()

        # --- Train baseline on same segments (no memory) ---
        baseline.train()
        base_total = 0.0
        base_mdlm = MDLMLoss(schedule, baseline.mask_id)
        for seg_i in range(segments_per_group):
            x0 = x0_long[:, seg_i * seq_len : (seg_i + 1) * seq_len]
            logits_b = baseline(all_xt[seg_i], all_t[seg_i])
            loss_b = base_mdlm(logits_b, x0, all_masked[seg_i])
            base_total = base_total + loss_b
        base_total = base_total / segments_per_group
        opt_base.zero_grad()
        base_total.backward()
        torch.nn.utils.clip_grad_norm_(baseline.parameters(), 1.0)
        opt_base.step()
        scheduler_base.step()

        lv = total_loss.item()
        blv = base_total.item()
        ema_lv = lv if ema_lv is None else 0.99 * ema_lv + 0.01 * lv
        ema_base_lv = blv if ema_base_lv is None else 0.99 * ema_base_lv + 0.01 * blv

        if step % 50 == 0:
            print(f"  step {step:5d}  mem {lv:.4f} (ema {ema_lv:.4f})  "
                  f"base {blv:.4f} (ema {ema_base_lv:.4f})  "
                  f"gap {ema_base_lv - ema_lv:+.4f}  "
                  f"lr {scheduler.get_last_lr()[0]:.2e}")

    # === Evaluate ===
    print("\n" + "=" * 60)
    print("Evaluation (joint-trained mem vs baseline)")
    print("=" * 60)
    long_loader = infinite_loader(long_dataloader)
    evaluate_and_sample(mem_denoiser, baseline, schedule, long_loader,
                        seq_len, segments_per_group, sample_steps, device)

    torch.save({
        "mem_denoiser": mem_denoiser.state_dict(),
        "baseline": baseline.state_dict(),
    }, "gmem_diffusion_joint.pt")
    print("\nSaved to gmem_diffusion_joint.pt")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["joint", "twophase"], default="joint",
                        help="joint: train everything together; "
                             "twophase: train denoiser first, then memory")
    parser.add_argument("--placement", choices=["mid", "output", "ponder"],
                        default="mid",
                        help="mid: memory after layer 2; "
                             "output: memory after all layers; "
                             "ponder: memory + CMM recursive refinement after layer 2")
    args = parser.parse_args()

    if args.mode == "joint":
        train_joint(placement=args.placement)
    else:
        train_two_phase()
