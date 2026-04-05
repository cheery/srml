"""
Energy-Based Diffusion Language Model (EDLM)
=============================================
PyTorch implementation of:
"Energy-Based Diffusion Language Models for Text Generation"
(Xu, Geffner, Kreis, Nie, Xu, Leskovec, Ermon, Vahdat — ICLR 2025)

Implements:
  - Masked Diffusion Language Model (MDLM) as base denoiser p_θ
  - Residual Energy-Based Model E_φ  (Eq 7)
  - NCE training for energy function (Algorithm 2, Eq 10)
  - Denoising via Importance Sampling (Algorithm 1)

The key innovation is a residual EBM at the full sequence level
that corrects the factorized denoiser's per-token independence:

  p_{θ,φ}(x₀|x_t) = μ_θ(x₀|x_t) · exp(−E_φ(x₀, x_t, t)) / Z_φ(x_t)

where μ_θ is the pretrained diffusion model and E_φ captures
inter-token correlations via an energy function.
"""

from dataclasses import dataclass
from typing import Any
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .ema import EMA

directory = Path(__file__).parent


# ============================================================
# Data Loading
# ============================================================

def load_kalevala():
    filename = (directory / "../../../data/kalevala.plain.txt").resolve()
    with filename.open("r", encoding="utf-8") as fd:
        text = fd.read().replace("\n", " ")
    return text


class TextDataset(Dataset):
    def __init__(self, text, length, stride):
        self.chunks = []
        data = text.encode("utf-8")
        raw = torch.frombuffer(bytearray(data), dtype=torch.uint8).long()
        for i in range(0, len(raw) - length, stride):
            self.chunks.append(raw[i:i + length])

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]


def create_dataloader(text, batch_size=32, length=128, stride=64,
                      shuffle=True, drop_last=True):
    dataset = TextDataset(text, length, stride)
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=shuffle, drop_last=drop_last)


def as_text(t):
    return t.cpu().to(torch.uint8).numpy().tobytes().decode("utf-8", errors="replace")


# ============================================================
# Noise Schedule  (Masked Diffusion)
# ============================================================
# α_t ∈ [0, 1] is the probability each token is retained.
# Equivalent to SEDD absorb with σ̄(t) = −log(α_t).

class LogLinearSchedule:
    r"""
    α_t = 1 − (1−ε)·t

    At t=0: α ≈ 1  (clean data).
    At t=1: α ≈ ε  (almost fully masked).
    """
    def __init__(self, eps=1e-3):
        self.eps = eps

    def alpha(self, t):
        """Token retention probability at time t."""
        return 1.0 - (1.0 - self.eps) * t


# ============================================================
# Forward Process  q(x_t | x₀) — Eq (2) with π = m
# ============================================================

def mask_tokens(x0, alpha_t, mask_id):
    """
    Independently mask each token.

    Each position stays as x₀ with probability α_t,
    or becomes MASK with probability 1 − α_t.

    Args:
        x0:      (B, L) clean token ids
        alpha_t: scalar or (B,) retention probabilities
        mask_id: integer id for the MASK token
    Returns:
        xt:        (B, L)  noised tokens
        keep_mask: (B, L)  bool, True where token was kept
    """
    B, L = x0.shape
    if isinstance(alpha_t, (int, float)):
        a = alpha_t
    else:
        a = alpha_t[:, None] if alpha_t.dim() == 1 else alpha_t

    keep_mask = torch.rand(B, L, device=x0.device) < a
    xt = x0.clone()
    xt[~keep_mask] = mask_id
    return xt, keep_mask


# ============================================================
# Transformer Architecture  (Bidirectional, DiT-style adaLN)
# ============================================================

def _apply_rotary(x, cos, sin):
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos - x2 * sin,
                      x2 * cos + x1 * sin], dim=-1)


class _SelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, cos, sin):
        B, L, _ = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        q = _apply_rotary(q, cos, sin)
        k = _apply_rotary(k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, L, -1)
        return self.out_proj(out)


class _AdaLNBlock(nn.Module):
    """Transformer block with adaLN-zero conditioning (DiT-style)."""
    def __init__(self, dim, num_heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = _SelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim))
        # 6 modulation params: (γ₁, β₁, α₁, γ₂, β₂, α₂)
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaln[-1].weight)
        nn.init.zeros_(self.adaln[-1].bias)

    def forward(self, x, c, cos, sin):
        g1, b1, a1, g2, b2, a2 = self.adaln(c).unsqueeze(1).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + g1) + b1
        x = x + a1 * self.attn(h, cos, sin)
        h = self.norm2(x) * (1 + g2) + b2
        x = x + a2 * self.mlp(h)
        return x


class DenoisingTransformer(nn.Module):
    """
    MDLM-style bidirectional x₀ predictor  (Eq 5–6).

    Takes x_t (with MASK tokens) and time t, predicts logits over
    the clean vocabulary at every position.  The denoiser factorises
    independently per token:

      p_θ(x₀|x_t) = Π_i  softmax(f_θ(x_t, t))_i
    """
    def __init__(self, n_vocab, max_len, dim=256,
                 num_heads=4, num_layers=4):
        super().__init__()
        self.n_vocab = n_vocab
        self.mask_id = n_vocab            # MASK = extra token
        self.dim = dim

        self.tok_embed = nn.Embedding(n_vocab + 1, dim)   # +1 for MASK
        self.out_proj = nn.Linear(dim, n_vocab)            # clean vocab only

        self.time_mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList(
            [_AdaLNBlock(dim, num_heads) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(dim)

        # Rotary position embeddings
        head_dim = dim // num_heads
        half = head_dim // 2
        freqs = 1.0 / (10000.0 ** (torch.arange(half).float() / half))
        pos = torch.arange(max_len).float()
        angles = pos[:, None] * freqs[None, :]
        self.register_buffer("rot_cos", torch.cos(angles), persistent=False)
        self.register_buffer("rot_sin", torch.sin(angles), persistent=False)

    def _time_embed(self, t):
        """Sinusoidal time embedding → MLP."""
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.time_mlp(emb)

    def get_hidden(self, x, t):
        """Final-layer hidden states (B, L, dim)."""
        B, L = x.shape
        h = self.tok_embed(x)
        c = self._time_embed(t)
        cos = self.rot_cos[:L][None, None, :, :]
        sin = self.rot_sin[:L][None, None, :, :]
        for block in self.blocks:
            h = block(h, c, cos, sin)
        return self.final_norm(h)

    def forward(self, xt, t):
        """Logits (B, L, n_vocab) for x₀ prediction."""
        return self.out_proj(self.get_hidden(xt, t))

class EnergyModelBase(nn.Module):
    """
    Residual energy function  E_φ(x₀, x_t, t)  — Eq (7).

    Architecture (Section 5.1 / Appendix C.1):
      1. Bidirectional transformer backbone (initialised from the
         pretrained MDLM denoiser).
      2. Mean-pool the final-layer token representations.
      3. Project to a single scalar energy.

    Low energy  → coherent / realistic x₀ proposal.
    High energy → incoherent / unlikely x₀ proposal.
    """
    def __init__(self, dim):
        super().__init__()
        self.energy_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1))
        # Start near zero so initial energies are small
        nn.init.zeros_(self.energy_head[-1].weight)
        nn.init.zeros_(self.energy_head[-1].bias)

    def outlet(self, h):
        pooled = h.mean(dim=1)                        # (B, dim)
        return self.energy_head(pooled).squeeze(-1)   # (B,)


class EnergyModel(EnergyModelBase):
    def __init__(self, mk_denoiser, dim=256):
        super().__init__(dim)
        self.backbone = mk_denoiser()

    def init_from_denoiser(self, denoiser):
        """Copy pretrained MDLM weights into the backbone."""
        self.backbone.load_state_dict(denoiser.state_dict())

    def forward(self, x0, t):
        """
        E_φ(x₀, x_t, t) → (B,) scalar energies.

        The candidate x₀ is fed through the backbone (conditioned on t);
        since carry-over means x₀ = x_t at unmasked positions, the model
        implicitly has access to x_t through x₀ itself.
        """
        h = self.backbone.get_hidden(x0, t)          # (B, L, dim)
        return self.outlet(h)


# ============================================================
# MDLM Denoiser Training Loss
# ============================================================
@dataclass
class MDLMLoss:
    """
    Cross-entropy on masked positions, uniformly sampled t.

    For the log-linear schedule the per-position NELBO weight
    −α′_t / (1−α_t) = 1/t cancels with the expected number of
    masked tokens (1−α_t) = (1−ε)t, so uniform-weight CE over
    masked positions is an unbiased estimator (up to a constant).
    """
    schedule : Any
    mask_id : int
    t_min : float = 1e-4
    def perturb(self, x0, t=None, answer_mask=None):
        """
        Args:
            answer_mask: (B, L) bool — True for answer positions
        Only answer positions (answer_mask == True) are noised and
        contribute to the loss.  Question positions stay clean,
        matching conditional sampling at inference.
        """
        B, L = x0.shape
        device = x0.device
        t_min = self.t_min

        if t is None:
            t = torch.rand(B, device=device) * (1.0 - t_min) + t_min
        alpha_t = self.schedule.alpha(t)

        xt, keep_mask = mask_tokens(x0, alpha_t, self.mask_id)
        is_masked = ~keep_mask                            # (B, L)
        if answer_mask is not None:
            xt[~answer_mask] = x0[~answer_mask]             # restore questions
            # loss only at masked answer positions
            #is_masked = (xt == self.mask_id) & answer_mask
            is_masked &= answer_mask
        return xt, t, is_masked

    def __call__(self, logits, x0, is_masked):
        n_masked = is_masked.float().sum()
        if n_masked == 0:
            return torch.tensor(0.0, device=x0.device, requires_grad=True)

        ce = F.cross_entropy(
            logits.transpose(1, 2), x0.long(), reduction='none')  # (B, L)
        return (ce * is_masked.float()).sum() / n_masked

    def per_sample(self, logits, x0, is_masked):
        """Per-sample mean CE over masked positions. Returns (B,) tensor."""
        ce = F.cross_entropy(
            logits.transpose(1, 2), x0.long(), reduction='none')  # (B, L)
        masked_ce = ce * is_masked.float()                         # (B, L)
        n_per = is_masked.float().sum(dim=1).clamp(min=1)          # (B,)
        return masked_ce.sum(dim=1) / n_per                        # (B,)

    @classmethod
    def _example_of_use(cls, denoiser, x0):
        mdlm_loss = cls(schedule, denoiser.mask_id, t_min=1e-4)
        xt, t, is_masked = mdlm_loss.perturb(x0, answer_mask=None)
        logits = denoiser(xt, t)                         # (B, L, n_vocab)
        return mdlm_loss(logits, x0, is_masked)

# ============================================================
# NCE Training Loss  — Algorithm 2, Eq (10)
# ============================================================

@dataclass
class NCELoss:
    """
    Noise Contrastive Estimation loss for the energy function.

    Positive:  x₊ = x₀                  (true clean data)
    Negative:  x₋ ~ p_θ(x̃₀ | x_t)     (sampled from frozen denoiser)

    Binary classification objective (Eq 10):
      L = E[ −log σ(−E(x₊)) − log σ(E(x₋)) ]
        = E[ softplus(E(x₊)) + softplus(−E(x₋)) ]

    If answer_mask is provided, only answer positions are noised
    (question positions stay clean), matching conditional training.
    """
    schedule : Any
    mask_id : int
    n_vocab : int
    t_min : float = 1e-4
    def perturb(self, x0, t=None, answer_mask=None):
        B, L = x0.shape
        device = x0.device
        mask_id = self.mask_id
        t_min = self.t_min
        if t is None:
            t = torch.rand(B, device=device) * (1.0 - t_min) + t_min
        alpha_t = self.schedule.alpha(t)
        # ---- forward process ----
        xt, _ = mask_tokens(x0, alpha_t, mask_id)
        if answer_mask is not None:
            xt[~answer_mask] = x0[~answer_mask]          # restore questions
        is_masked = (xt == mask_id)
        return xt, t, is_masked

    @torch.no_grad()
    def sample_neg(self, x0, logits, is_masked):
        B, L = logits.shape[0:2]
        probs = F.softmax(logits, dim=-1)                  # (B, L, n)
        sampled = torch.multinomial(
            probs.reshape(-1, self.n_vocab), 1).reshape(B, L)
        x_neg = x0.clone()
        x_neg[is_masked] = sampled[is_masked]
        return x_neg

    def __call__(self, e_pos, e_neg):
        # ---- NCE binary cross-entropy ----
        loss = F.softplus(e_pos).mean() + F.softplus(-e_neg).mean()
        return loss

def nce_loss(energy_model, denoiser, x0, schedule,
             answer_mask=None, t_min=1e-4):
    nce_loss = NCELoss(schedule, denoiser.mask_id, denoiser.n_vocab)

    xt, t, is_masked = nce_loss.perturb(x0, answer_mask=answer_mask)
    # ---- positive: true x₀ ----
    x_pos = x0
    # ---- negative: sample from factorised denoiser, carry-over ----
    with torch.no_grad():
        logits = denoiser(xt, t)
    x_neg = nce_loss.sample_neg(x0, logits, is_masked)

    # ---- energies ----
    loss = nce_loss(
        e_pos = energy_model(x_pos, t),                    # (B,)
        e_neg = energy_model(x_neg, t))                    # (B,)
    return loss

# ============================================================
# Sampling  — Algorithm 1: Denoising via Importance Sampling
# ============================================================
@dataclass
class Sampler:
    """
    Algorithm 1 from the paper.

    Reverses from t = 1 (all MASK) to t = 0 (clean) in N steps.
    At each step the denoiser proposes x₀ candidates; if within the
    importance sampling window [1−w, 1] and an energy model is
    provided, k candidates are drawn and resampled by exp(−E).
    """
    schedule : Any
    mask_id : int
    n_vocab : int

    def __call__(self, batch_size, seq_len, device, num_steps=256):
        # τ₁ > τ₂ > … > τ_N  (from ≈1 to 0)
        taus = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
        # x_{τ₁} ← m  (fully masked)
        xt = torch.full((batch_size, seq_len), self.mask_id,
                        dtype=torch.long, device=device)
        return xt, SamplingStepper(self, taus, batch_size, seq_len, num_steps,
                                   device)

@dataclass
class SamplingStepper:
    sampler    : Sampler
    taus       : torch.Tensor
    batch_size : int
    seq_len    : int
    num_steps  : int
    device     : torch.device

    def __iter__(self):
        for step in range(self.num_steps):
            tau_n = self.taus[step].item()
            tau_next = self.taus[step + 1].item()
            t_vec = torch.full((self.batch_size,), tau_n, device=self.device)
            yield SamplingStep(self, step, tau_n, tau_next, t_vec)

@dataclass
class SamplingStep:
    stepper  : SamplingStepper
    step     : int
    tau_n    : float
    tau_next : float
    t        : torch.Tensor

    def propose_x0(self, xt, logits):
        """Sample a single x₀ candidate from denoiser logits.

        Fills masked positions with samples from softmax(logits),
        keeps unmasked positions unchanged.

        Returns:
            x0: (B, L) proposed clean tokens
        """
        mask_id = self.stepper.sampler.mask_id
        n = self.stepper.sampler.n_vocab
        B, L = xt.shape
        probs = F.softmax(logits, dim=-1)
        s = torch.multinomial(probs.reshape(-1, n), 1).reshape(B, L)
        x0 = xt.clone()
        x0[xt == mask_id] = s[xt == mask_id]
        return x0

    def propose_x0_k(self, xt, logits, k):
        """Sample k x₀ candidates from denoiser logits.

        Returns:
            candidates: (k, B, L) proposed clean tokens
        """
        mask_id = self.stepper.sampler.mask_id
        n = self.stepper.sampler.n_vocab
        B, L = xt.shape
        probs = F.softmax(logits, dim=-1)
        flat_probs = probs.reshape(-1, n)
        is_masked = (xt == mask_id)
        candidates = []
        for _ in range(k):
            s = torch.multinomial(flat_probs, 1).reshape(B, L)
            cand = xt.clone()
            cand[is_masked] = s[is_masked]
            candidates.append(cand)
        return torch.stack(candidates, dim=0)

    @staticmethod
    def select_by_energy(candidates, energies):
        """Resample one x₀ per batch element, weighted by exp(-E).

        Args:
            candidates: (k, B, L)
            energies:   (k, B) scalar energies per candidate

        Returns:
            x0: (B, L) selected candidates
        """
        k, B = energies.shape
        device = energies.device
        weights = F.softmax(-energies, dim=0)            # (k, B)
        idx = torch.multinomial(weights.t(), 1).squeeze(-1)  # (B,)
        return candidates[idx, torch.arange(B, device=device)]

    def reverse_step(self, xt, x0):
        """Reverse diffusion step: q(x_{τ_{n+1}} | x_{τ_n}, x₀) — Eq (4).

        Unmasks a fraction of masked positions, replacing them with
        values from x0. At the last step (tau_next=0), accepts x0 fully.

        Args:
            xt: (B, L) current noised tokens
            x0: (B, L) proposed clean tokens

        Returns:
            xt_new: (B, L) less-noised tokens
        """
        stepper = self.stepper
        mask_id = stepper.sampler.mask_id
        device = stepper.device
        B, L = xt.shape
        is_masked = (xt == mask_id)

        if self.tau_next > 0:
            alpha_t = stepper.sampler.schedule.alpha(
                torch.tensor(self.tau_n, device=device))
            alpha_s = stepper.sampler.schedule.alpha(
                torch.tensor(self.tau_next, device=device))
            p_unmask = ((alpha_s - alpha_t)
                        / (1.0 - alpha_t)).clamp(0, 1)
            unmask = torch.rand(B, L, device=device) < p_unmask
            xt_new = xt.clone()
            xt_new[is_masked & unmask] = x0[is_masked & unmask]
            return xt_new
        else:
            return x0

    def sample(self, xt, logits):
        """Convenience: propose_x0 + reverse_step in one call."""
        return self.reverse_step(xt, self.propose_x0(xt, logits))


@torch.no_grad()
def sample(
    denoiser,
    schedule,
    batch_size,
    seq_len,
    num_steps=256,
    energy_model=None,
    k=8,
    window_w=0.2,
    device=torch.device("cpu"),
):
    """
    Algorithm 1: Denoising via Importance Sampling.

    Args:
        denoiser:     MDLM denoiser p_θ
        schedule:     noise schedule
        batch_size:   number of sequences to generate
        seq_len:      sequence length
        num_steps:    number of denoising timesteps N
        energy_model: residual energy E_φ (None → pure MDLM)
        k:            importance sampling size
        window_w:     importance sampling window length
        device:       compute device
    Returns:
        (batch_size, seq_len) generated token ids
    """
    sampler = Sampler(schedule, denoiser.mask_id, denoiser.n_vocab)
    xt, stepper = sampler(batch_size, seq_len, device, num_steps)

    for s in stepper:
        logits = denoiser(xt, s.t)

        if energy_model is not None and s.tau_n >= 1.0 - window_w:
            candidates = s.propose_x0_k(xt, logits, k)       # (k, B, L)
            c_flat = candidates.reshape(k * batch_size, seq_len)
            t_flat = s.t.unsqueeze(0).expand(k, -1).reshape(k * batch_size)
            energies = energy_model(c_flat, t_flat).reshape(k, batch_size)
            x0 = s.select_by_energy(candidates, energies)
        else:
            x0 = s.propose_x0(xt, logits)

        xt = s.reverse_step(xt, x0)
    return xt


# ============================================================
# Perplexity / NLL Evaluation  (Section 4.3, Eq 12–14)
# ============================================================
# Kept here for completeness, but not used yet.

@torch.no_grad()
def estimate_nll(
    denoiser, x0, schedule, energy_model=None,
    num_t=200, num_z_samples=16, t_min=1e-4,
):
    """
    Monte-Carlo estimate of the NELBO / NLL upper bound.

    Without energy model: standard MDLM diffusion ELBO.
    With energy model: EDLM bound using the partition function
    estimator from Theorem 1.

    Returns per-sample NLL in nats, shape (B,).
    """
    B, L = x0.shape
    device = x0.device
    mask_id = denoiser.mask_id
    n = denoiser.n_vocab

    ts = torch.rand(num_t, device=device) * (1.0 - t_min) + t_min
    total = torch.zeros(B, device=device)

    for ti in ts:
        t_batch = ti.expand(B)
        alpha_t = schedule.alpha(t_batch)

        xt, _ = mask_tokens(x0, alpha_t, mask_id)
        is_masked = (xt == mask_id)

        logits = denoiser(xt, t_batch)
        log_probs = F.log_softmax(logits, dim=-1)

        # log p_θ(x₀|x_t) at masked positions
        lp = (log_probs.gather(2, x0.unsqueeze(-1)).squeeze(-1)
              * is_masked.float()).sum(-1)                  # (B,)

        if energy_model is not None:
            e_x0 = energy_model(x0, t_batch)

            # estimate log Z_φ via samples from p_θ
            probs = F.softmax(logits, dim=-1)
            es = []
            for _ in range(num_z_samples):
                s = torch.multinomial(
                    probs.reshape(-1, n), 1).reshape(B, L)
                xs = x0.clone()
                xs[is_masked] = s[is_masked]
                es.append(energy_model(xs, t_batch))
            e_stack = torch.stack(es, dim=0)               # (nz, B)
            log_z = (torch.logsumexp(-e_stack, dim=0)
                     - math.log(num_z_samples))

            contrib = lp - e_x0 - log_z
        else:
            contrib = lp

        # NELBO weight: −α′/(1−α) = 1/t for log-linear
        weight = 1.0 / ti.clamp(min=t_min)
        total += weight * (-contrib)

    nll = total / num_t                                    # (B,) nats
    return nll


# ============================================================
# Training
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- hyperparameters ----
    n_vocab    = 256        # byte-level encoding
    seq_len    = 128
    batch_size = 32
    dim        = 256
    num_heads  = 4
    num_layers = 4

    mdlm_steps = 10000      # phase 1: train denoiser
    mdlm_lr    = 3e-4

    nce_steps  = 8000       # phase 2: train energy (NCE)
    nce_lr     = 1e-4

    sample_steps  = 128     # sampling
    importance_k  = 8
    importance_w  = 0.2

    # ---- data ----
    text = load_kalevala()
    dataloader = create_dataloader(
        text, batch_size=batch_size, length=seq_len, stride=64)
    schedule = LogLinearSchedule(eps=1e-3)

    def infinite_loader(dl):
        while True:
            yield from dl

    loader = infinite_loader(dataloader)

    # ==========================================================
    # Phase 1: Train MDLM Denoiser
    # ==========================================================
    print("=" * 60)
    print("Phase 1: Training MDLM Denoiser")
    print("=" * 60)

    denoiser = DenoisingTransformer(
        n_vocab, seq_len, dim, num_heads, num_layers).to(device)
    print(f"Denoiser params: "
          f"{sum(p.numel() for p in denoiser.parameters()):,}")

    ema_den = EMA(denoiser, mu=0.999)
    opt = torch.optim.AdamW(denoiser.parameters(), lr=mdlm_lr)

    ema_lv = None
    for step in range(1, mdlm_steps + 1):
        x0 = next(loader).to(device)
        denoiser.train()
        opt.zero_grad()

        mdlm_loss = MDLMLoss(schedule, denoiser.mask_id)
        xt, t, is_masked = mdlm_loss.perturb(x0, answer_mask=None)
        logits = denoiser(xt, t)                         # (B, L, n_vocab)
        loss = mdlm_loss(logits, x0, is_masked)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
        opt.step()
        ema_den.update(denoiser)

        lv = loss.item()
        ema_lv = lv if ema_lv is None else 0.99 * ema_lv + 0.01 * lv

        if step % 100 == 0:
            print(f"  step {step:5d}  loss {lv:.4f}  ema {ema_lv:.4f}")

        if step % 2000 == 1:
            denoiser.eval()
            ema_den.apply(denoiser)
            s = sample(denoiser, schedule, batch_size=2,
                       seq_len=seq_len, num_steps=64, device=device)
            for i in range(2):
                print(f"  MDLM sample {i+1}: "
                      f"{repr(as_text(s[i][:80]))}")
            ema_den.restore(denoiser)

    # switch to EMA weights for denoiser
    ema_den.apply(denoiser)
    denoiser.eval()
    for p in denoiser.parameters():
        p.requires_grad = False

    # ==========================================================
    # Phase 2: Train Energy Model (NCE)
    # ==========================================================
    print("\n" + "=" * 60)
    print("Phase 2: Training Energy Model (NCE)")
    print("=" * 60)

    energy = EnergyModel(
        (lambda: DenoisingTransformer(
            n_vocab, seq_len, dim, num_heads, num_layers)),
        dim=dim).to(device)
    energy.init_from_denoiser(denoiser)
    print(f"Energy model params: "
          f"{sum(p.numel() for p in energy.parameters()):,}")

    ema_eng = EMA(energy, mu=0.999)
    opt_e = torch.optim.AdamW(energy.parameters(), lr=nce_lr)

    loader = infinite_loader(dataloader)
    ema_lv = None
    for step in range(1, nce_steps + 1):
        x0 = next(loader).to(device)
        energy.train()
        opt_e.zero_grad()

        loss = nce_loss(energy, denoiser, x0, schedule)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(energy.parameters(), 1.0)
        opt_e.step()
        ema_eng.update(energy)

        lv = loss.item()
        ema_lv = lv if ema_lv is None else 0.99 * ema_lv + 0.01 * lv

        if step % 100 == 0:
            print(f"  step {step:5d}  nce_loss {lv:.4f}  "
                  f"ema {ema_lv:.4f}")

    ema_eng.apply(energy)
    energy.eval()

    # ==========================================================
    # Phase 3: Sampling comparison
    # ==========================================================
    print("\n" + "=" * 60)
    print("Phase 3: Sampling")
    print("=" * 60)

    print("\n--- MDLM samples (no energy correction) ---")
    for _ in range(100):
        s = sample(denoiser, schedule, batch_size=4, seq_len=seq_len,
                   num_steps=sample_steps, device=device)
        for i in range(4):
            print(f"  {i+1}: {repr(as_text(s[i][:100]))}")

    print(f"\n--- EDLM samples (k={importance_k}, w={importance_w}) ---")
    for _ in range(100):
        s = sample(denoiser, schedule, batch_size=4, seq_len=seq_len,
                   num_steps=sample_steps, energy_model=energy,
                   k=importance_k, window_w=importance_w, device=device)
        for i in range(4):
            print(f"  {i+1}: {repr(as_text(s[i][:100]))}")

    # ---- optional: NLL evaluation ----
    print("\n--- NLL evaluation (first batch) ---")
    x0 = next(infinite_loader(dataloader)).to(device)

    nll_mdlm = estimate_nll(denoiser, x0, schedule, num_t=50)
    print(f"  MDLM  NLL:  {nll_mdlm.mean().item():.2f} nats/seq  "
          f"({nll_mdlm.mean().item() / seq_len:.4f} nats/token)")

    nll_edlm = estimate_nll(
        denoiser, x0, schedule, energy_model=energy, num_t=50)
    print(f"  EDLM  NLL:  {nll_edlm.mean().item():.2f} nats/seq  "
          f"({nll_edlm.mean().item() / seq_len:.4f} nats/token)")
