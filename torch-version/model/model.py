"""
SRLM -- Straightforward reasoning language model
        (Or something like that...)

SRLM is an energy-based diffusion language model
that contains a G-Mem and CMM as an optional layer.

The memory carries context across text segments during
training and sampling, helping the model to keep coherence.

CMM is there in hopes of improving reasoning capabilities
in the model.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Literal, Callable, Any, Tuple
from .attnres import BlockDivider, BlockAttnResOp
from .edlm import (
        MDLMLoss, NCELoss, mask_tokens, Sampler, SamplingStep,
        LogLinearSchedule, EnergyModelBase,
)
from .gmem import (
        LatentMemoryTerminal, LatentMemoryIn,
        LatentMemoryBank, MemoryLoss
)
from .cmm import (
    PonderBlock, equilibrium_loss, routh_hurwitz_stable_loss,
    routh_hurwitz_unstable_loss, repulsion_loss,
)
from .rotary import RotaryPositionEmbedding

@dataclass
class GMemConfig:
    memory_dim:            int = 256
    num_slots:             int = 1024

@dataclass
class PonderConfig:
    N_super:               int = 4
    N_H:                   int = 3
    N_L:                   int = 6
    noise_sigma:           float = 0.01
    noise_type:            str = "additive"
    use_stablemax:         str = "3"
    use_attention:         bool = True

@dataclass
class SRLMConfig:
    gmem:                  GMemConfig
    ponder:                PonderConfig
    vocab_size:            int = 256
    max_context_length:    int = 128
    hidden_dim:            int = 256
    num_heads:             int = 8
    mlp_ratio:             int = 4
    front_layers:          int = 2
    back_layers:           int = 2
    dropout:               float = 0.2

class SRLMEnergyModel(EnergyModelBase):
    def __init__(self, cfg):
        super().__init__(cfg.hidden_dim)
        self.denoiser = SRLMDenoiser(cfg)

    def init_from_denoiser(self, denoiser):
        self.denoiser.load_state_dict(denoiser.state_dict())

    def forward(self, x0, t, memory=None):
        h, importance_scores = self.denoiser.get_hidden(x0, t, memory)
        return self.outlet(h)

class SRLMDenoiser(nn.Module):
    def __init__(self, cfg: SRLMConfig):
        super().__init__()
        self.cfg = cfg
        self.input = MaskedInputLayer(cfg.vocab_size,
                                      cfg.hidden_dim,
                                      cfg.num_heads,
                                      cfg.max_context_length)
        self.front_layers = nn.ModuleList([
            init_layer(cfg)
            for _ in range(cfg.front_layers)
        ])
        self.latent_memory_in = LatentMemoryIn(
                cfg.hidden_dim,
                cfg.gmem_memory_dim,
                cfg.num_heads)
        self.latent_memory = LatentMemoryBank(
                    cfg.hidden_dim,
                    cfg.gmem.memory_dim,
                    cfg.num_heads)
        self.ponder = PonderBlock(
            cfg.hidden_dim,
            cfg.max_context_length,
            num_heads=cfg.num_heads,
            N_H=cfg.ponder.N_H,
            N_L=cfg.ponder.N_L,
            noise_sigma=cfg.ponder.noise_sigma,
            noise_type=cfg.ponder.noise_type,
            use_attention=cfg.ponder.use_attention,
            use_stablemax=cfg.ponder.use_stablemax,
        )
        self.back_layers = nn.ModuleList([
            init_layer(cfg)
            for _ in range(cfg.back_layers)
        ])
        self.out_proj = nn.Linear(cfg.hidden_dim, cfg.vocab_size)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, -6.0)

    def get_front(self, xt, t):
        h, c, p_emb = self.input(xt, t)
        for layer in self.front_layers:
            h = layer(h, c, p_emb)
            h += torch.randn_like(h) * 0.01
        return h, c, p_emb

    def init_memory(self, batch_size, device):
        """Create zero-initialized memory state."""
        return torch.zeros(batch_size, self.cfg.gmem.num_slots,
                           self.cfg.gmem.memory_dim, device=device)

    def init_states(self, h):
        z_H = h.clone()
        z_L = torch.zeros_like(h)
        return z_H, z_L

    def roll(self, h, c, p_emb, z_H, z_L, memory=None):
        z_H, z_L, q_values = self.ponder(h, c, p_emb, z_H, z_L)
        if memory is None:
            memory = self.init_memory(h.shape[0], h.device)
        z_H, memory, importance_scores = self.latent_memory(z_H, memory)
        return z_H, z_L, q_values, memory, importance_scores

    def init_t(self, xt):
        return torch.ones((xt.shape[0],), device=xt.device) * 2.0 # TODO: consider whether 1.0 should go here.

    def pioneer(self, xt, t=None, memory=None):
        if t is None:
            t = self.init_t(xt)
        h, c, p_emb = self.get_front(xt, t)
        z_H, z_L = self.init_states(h)
        for _ in range(self.cfg.ponder.N_super):
            z_H, z_L, q_values, memory, importance_scores = self.roll(h, c, p_emb, z_H, z_L, memory=None)
            q_mean = q_values.mean(0)
            if q_mean[0] > q_mean[1]:
                return memory
        return memory

    def get_back(self, h, c, p_emb):
        for layer in self.back_layers:
            h = layer(h, c, p_emb)
            h += torch.randn_like(h) * 0.01
        return h

    def get_behind(self, h, c, p_emb):
        h = self.get_back(h, c, p_emb)
        return self.out_proj(h)

    def get_hidden(self, z, xt, t, memory=None):
        h, c, p_emb = self.get_front(xt, t)
        if memory is None:
            memory = self.init_memory(h.shape[0], h.device)
        h, importance_scores = self.latent_memory_in(h, memory)
        h = self.get_back(h, c, p_emb)
        return h, importance_scores

    def forward(self, xt, t, memory=None):
        h, importance_scores = self.get_hidden(xt, t, memory)
        return self.out_proj(h)

def init_layer(cfg):
    return Block(cfg.hidden_dim, cfg.num_heads, cfg.mlp_ratio, cfg.dropout)

class BaseInputLayer(nn.Module):
    def __init__(self, dim, num_heads, max_context_length):
        super().__init__()
        self.dim = dim
        self.pos_embed = RotaryPositionEmbedding(dim,
                                                 num_heads,
                                                 max_context_length)

class MaskedInputLayer(BaseInputLayer):
    def __init__(self, vocab_size, dim, num_heads, max_context_length):
        super().__init__(dim, num_heads, max_context_length)
        self.tok_embed = nn.Embedding(vocab_size + 1, dim)  # +1 for MASK
        self.time_mlp = nn.Sequential(
                nn.Linear(dim, dim),
                nn.SiLU(),
                nn.Linear(dim, dim))

    def _time_embed(self, t):
        """Sinusoidal time embedding -> MLP."""
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.time_mlp(emb)

    def forward(self, x, t):
        B, L = x.shape
        h = self.tok_embed(x)
        c = self._time_embed(t)
        pos = self.pos_embed(L)
        return h, c, pos

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = SelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim))
        # 6 modulation params: (g1, b1, a1, g2, b2, a2)
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaln[-1].weight)
        nn.init.zeros_(self.adaln[-1].bias)

        self.drop_attn = nn.Dropout(dropout)
        self.drop_mlp  = nn.Dropout(dropout)

    def forward(self, y, c, p_emb):
        g1, b1, a1, g2, b2, a2 = self.adaln(c).unsqueeze(1).chunk(6, dim=-1)
        h = self.norm1(y) * (1 + g1) + b1
        y = y + a1 * self.drop_attn(self.attn(h, p_emb))
        h = self.norm2(y) * (1 + g2) + b2
        y = y + a2 * self.drop_mlp(self.mlp(h))
        return y

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, h, pos):
        B, L, _ = h.shape
        qkv = self.qkv(h).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        q = pos(q)
        k = pos(k)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, L, -1)
        return self.out_proj(out)


# ============================================================
# Training losses
# ============================================================

def mdlm_loss(denoiser, x0, schedule, memory=None,
              answer_mask=None, t_min=1e-4):
    """
    MDLM denoiser loss: cross-entropy on masked positions.

    Returns:
        loss, memory, importance_scores
    """
    mask_id = denoiser.cfg.vocab_size
    loss_fn = MDLMLoss(schedule, mask_id, t_min=t_min)
    xt, t, is_masked = loss_fn.perturb(x0, answer_mask=answer_mask)
    h, importance_scores = denoiser.get_hidden(xt, t, memory)
    logits = denoiser.out_proj(h)
    loss = loss_fn(logits, x0, is_masked)
    return loss, memory, importance_scores


def nce_loss(energy_model, denoiser, x0, schedule,
             memory=None, answer_mask=None, t_min=1e-4):
    """
    NCE loss for the energy model.
    Denoiser is frozen (no_grad); only energy_model gets gradients.

    Returns:
        loss, memory, importance_scores
    """
    n_vocab = denoiser.cfg.vocab_size
    mask_id = n_vocab
    loss_fn = NCELoss(schedule, mask_id, n_vocab, t_min=t_min)

    xt, t, is_masked = loss_fn.perturb(x0, answer_mask=answer_mask)

    # Negative sample from frozen denoiser
    with torch.no_grad():
        logits = denoiser(xt, t, memory)
    x_neg = loss_fn.sample_neg(x0, logits, is_masked)

    # Energies -- positive (true x0) and negative (denoiser sample)
    e_pos = energy_model(x0, t, memory)
    e_neg = energy_model(x_neg, t, memory)

    loss = loss_fn(e_pos, e_neg)
    return loss, memory, importance_scores


# ============================================================
# Sampling
# ============================================================

@torch.no_grad()
def sample(denoiser, schedule, batch_size, seq_len, num_steps=256,
           energy_model=None, k=8, window_w=0.2,
           memory=None,
           device=torch.device("cpu")):
    """
    MDLM / EDLM sampling with memory

    Args:
        denoiser:      SRLMDenoiser
        schedule:      LogLinearSchedule
        energy_model:  SRLMEnergyModel or None (pure MDLM if None)
        k:             importance sampling candidates (only with energy_model)
        window_w:      importance sampling window [1-w, 1]
        memory:        initial G-Mem state or None

    Returns:
        xt:      generated tokens (batch_size, seq_len)
        memory:  final memory state
    """
    n_vocab = denoiser.cfg.vocab_size
    mask_id = n_vocab
    sampler = Sampler(schedule, mask_id, n_vocab)
    xt, stepper = sampler(batch_size, seq_len, device, num_steps)
    z = None

    for s in stepper:
        z, logits, memory, _ = denoiser(z, xt, s.t, memory)
        if energy_model is not None and s.tau_n >= 1.0 - window_w:
            candidates = s.propose_x0_k(xt, logits, k)       # (k, B, L)
            c_flat = candidates.reshape(k * batch_size, seq_len)
            t_flat = s.t.unsqueeze(0).expand(k, -1).reshape(k * batch_size)
            if memory is not None:
                mem_flat = memory.unsqueeze(0).expand(k, -1, -1, -1) \
                                 .reshape(k * batch_size, *memory.shape[1:])
            else:
                mem_flat = None
            energies = energy_model(c_flat, t_flat, mem_flat)
            energies = energies.reshape(k, batch_size)
            x0 = s.select_by_energy(candidates, energies)
        else:
            x0 = s.propose_x0(xt, logits)

        xt = s.reverse_step(xt, x0)

    return xt, memory


# ============================================================
# Ponder Training (deep supervision through SRLM)
# ============================================================

@dataclass
class PonderTrainer:
    """
    Deep supervision trainer for SRLMDenoiser (with integrated PonderBlock).

    Per segment:
      1. Fresh perturbation of x0
      2. Front layers + memory read + PonderBlock + back layers -> logits
      3. MDLM loss + auxiliary CMM losses (equilibrium, RH, repulsion, ACT)
      4. backward(), detach z and memory

    The PonderBlock's z_H/z_L state threads across segments, giving
    the reasoning module iterative refinement with gradient signal at
    each step. ACT halt uses soft masked-position accuracy.
    """
    denoiser: SRLMDenoiser
    schedule: Any
    N_super: int = 16
    M_min: int = 2           # minimum segments before ACT can halt
    eps_M_min: float = 0.1   # probability of randomizing M_min higher
    lambda_LM: float = 1.0
    lambda_BCE: float = 0.1
    lambda_mem: float = 0.01
    lambda_rep: float = 0.1
    lambda_equil: float = 0.1
    lambda_RH_stable: float = 0.1

    def train_step(
        self,
        x0: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        memory: Optional[torch.Tensor] = None,
        answer_mask: Optional[torch.Tensor] = None,
        w: Optional[torch.Tensor] = None,
        t_min: float = 1e-4,
    ) -> tuple[dict[str, float], torch.Tensor]:
        """
        Args:
            x0:        clean tokens (denoiser target, answer included)
            optimizer: shared optimizer for ponder + denoiser
            memory:    G-Mem state or None
            answer_mask: (B, L) bool -- True at answer positions to noise
            ponder_x0: separate clean tokens for ponder (e.g. question only).
                       If None, ponder sees x0.
            t_min:     minimum diffusion time

        Returns:
            losses:  dict of loss values for logging
            memory:  updated memory state (detached)
        """
        denoiser = self.denoiser
        mask_id = denoiser.cfg.vocab_size
        device = x0.device

        loss_fn = MDLMLoss(self.schedule, mask_id, t_min=t_min)

        # Randomized minimum segments (CMM paper Section 2.3.2):
        # with probability eps, sample M_min from {M_min,...,N_super},
        # otherwise use M_min as-is
        if torch.rand(1).item() < self.eps_M_min:
            min_seg = torch.randint(self.M_min, self.N_super + 1, ()).item()
        else:
            min_seg = self.M_min

        x_in = x0.clone()
        x_in[answer_mask] = mask_id
        t_in = self.init_t(x_in)
        h, c, p_emb = self.get_front(x_in, t_in)
        z_H, z_L = self.init_states(h)
        
        all_losses = {k: 0.0 for k in [
            "LM", "BCE", "mem", "rep", "equil", "RH_stable", "total"
        ]}

        for seg in range(self.N_super):
            h, c, p_emb = self.get_front(x_in, t_in)
            z_H, z_L, q_values, memory, importance_scores = denoiser.roll(h, c, p_emb, z_H, z_L, memory=None)

            xt, t, is_masked = loss_fn.perturb(x0, answer_mask=answer_mask)

            losses = {}
            # Fresh perturbation each segment -- different t, different masks
            xt, t, is_masked = loss_fn.perturb(x0, answer_mask=answer_mask)
            h, importance_scores2 = denoiser.get_front(xt, t, memory)
            logits = denoiser.out_proj(h)
            losses["LM"] = loss_fn(logits, x0, is_masked)

            # ACT halt loss -- soft accuracy on masked positions
            # (TRM-style: halt BCE only, no continue target)
            with torch.no_grad():
                token_correct = (logits.argmax(-1) == x0).float()
                if is_masked.any():
                    masked_correct = (token_correct * is_masked.float()).sum(-1)
                    masked_count = is_masked.float().sum(-1).clamp(min=1)
                    accuracy = masked_correct / masked_count
                else:
                    accuracy = token_correct.mean(-1)
            target = torch.stack([accuracy, 1.0 - accuracy], dim=-1)
            losses["BCE"] = F.binary_cross_entropy(q_values, target)

            # Memory regularization
            mem_loss_fn = MemoryLoss()
            losses["mem"] = mem_loss_fn(importance_scores) + mem_loss_fn(importance_scores2)

            losses["rep"] = repulsion_loss(z_H_new)

            # Expensive losses only on last segment
            is_last = (seg == self.N_super - 1)
            if is_last:
                block = denoiser.ponder.block
                losses["equil"] = equilibrium_loss(z_H_new, z_L_new, block, c, p_emb)
                losses["RH_stable"] = routh_hurwitz_stable_loss(
                    z_H_new + z_L_new, block, c, p_emb)
            else:
                losses["equil"] = torch.tensor(0.0, device=device)
                losses["RH_stable"] = torch.tensor(0.0, device=device)

            total = (
                self.lambda_LM * losses["LM"]
                + self.lambda_BCE * losses["BCE"]
                + self.lambda_mem * losses["mem"]
                + self.lambda_rep * losses["rep"]
                + self.lambda_equil * losses["equil"]
                + self.lambda_RH_stable * losses["RH_stable"]
            )

            total.backward()

            # Detach latent states for next segment (deep supervision)
            z_H = z_H_new.detach()
            z_L = z_L_new.detach()
            memory = memory.detach()

            for k, v in losses.items():
                all_losses[k] += v.item()
            all_losses["total"] += total.item()

            # ACT early stopping: halt when Q_halt > Q_continue
            # after running at least min_seg segments
            if seg >= min_seg - 1:
                with torch.no_grad():
                    q_mean = q_values.mean(0)
                    if q_mean[0] > q_mean[1]:
                        break

        # Scale all gradients by 1/n_run so effective LR is independent
        # of how many segments actually ran (all params accumulate from N segments)
        n_run = seg + 1
        for p in denoiser.parameters():
            if p.grad is not None:
                p.grad /= n_run

        # Have the callee call these and denoiser.train(), more diverse usecases this way.
        #torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
        #optimizer.step()
        #optimizer.zero_grad()

        memory = memory.detach()
        return {k: v / n_run for k, v in all_losses.items()}, memory
