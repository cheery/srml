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
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Literal, Callable, Any, Tuple
from .attnres import BlockDivider, BlockAttnResOp
from .ddl import DeltaResidualExpanded
from .edlm import (
        MDLMLoss, NCELoss, mask_tokens, Sampler, SamplingStep,
        LogLinearSchedule, EnergyModelBase,
)
from .gmem import (
        LatentMemoryTerminal,
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
    d_v:                   int = 4

class SRLMEnergyModel(EnergyModelBase):
    def __init__(self, cfg):
        super().__init__(cfg.hidden_dim)
        self.denoiser = SRLMDenoiser(cfg)

    def init_from_denoiser(self, denoiser):
        self.denoiser.load_state_dict(denoiser.state_dict())

    def forward(self, x0, z_H=None):
        h = self.denoiser.get_hidden(x0, z_H)
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
            d_v=cfg.d_v,
        )
        self.back_layers = nn.ModuleList([
            init_layer(cfg)
            for _ in range(cfg.back_layers)
        ])
        self.out_proj = nn.Linear(cfg.hidden_dim, cfg.vocab_size)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, -6.0)

    def get_front(self, xt):
        h, p_emb = self.input(xt)
        # Expand to (B, T, d, d_v) for DDL blocks
        h = h.unsqueeze(-1).expand(*h.shape, self.cfg.d_v).contiguous()
        for layer in self.front_layers:
            h = layer(h, p_emb)
            if self.training:
                h += torch.randn_like(h) * 0.01
        return h, p_emb

    def init_memory(self, batch_size, device):
        """Create zero-initialized memory state."""
        return torch.zeros(batch_size, self.cfg.gmem.num_slots,
                           self.cfg.gmem.memory_dim, device=device)

    def init_states(self, h):
        z_H = h.clone()
        z_L = torch.zeros_like(h)
        return z_H, z_L

    def roll(self, h, p_emb, z_H, z_L, memory=None):
        z_H, z_L, q_values = self.ponder(h, p_emb, z_H, z_L)
        if memory is None:
            memory = self.init_memory(h.shape[0], h.device)
        # Compress for memory interface, then apply delta back to expanded state
        z_H_comp = z_H.mean(dim=-1)
        z_H_new, memory, importance_scores = self.latent_memory(z_H_comp, memory)
        z_H = z_H + (z_H_new - z_H_comp).unsqueeze(-1)
        return z_H, z_L, q_values, memory, importance_scores

    def pioneer(self, xt, memory=None):
        h, p_emb = self.get_front(xt)
        z_H, z_L = self.init_states(h)
        for _ in range(self.cfg.ponder.N_super):
            z_H, z_L, q_values, memory, importance_scores = self.roll(h, p_emb, z_H, z_L, memory=None)
            q_mean = q_values.mean(0)
            if q_mean[0] > q_mean[1]:
                return z_H, memory
        return z_H, memory

    def get_back(self, h, p_emb):
        for layer in self.back_layers:
            h = layer(h, p_emb)
            if self.training:
                h += torch.randn_like(h) * 0.01
        return h

    def get_behind(self, h, p_emb):
        h = self.get_back(h, p_emb)
        return self.out_proj(h.mean(dim=-1))

    def get_hidden(self, xt, z_H=None):
        h, p_emb = self.get_front(xt)
        if z_H is not None:
            h = self.get_back(h + z_H, p_emb)
        else:
            h = self.get_back(h, p_emb)
        return h.mean(dim=-1)

    def forward(self, xt, z_H=None):
        h = self.get_hidden(xt, z_H)
        return self.out_proj(h)

def init_layer(cfg):
    return Block(cfg.hidden_dim, cfg.num_heads, cfg.mlp_ratio, cfg.dropout, d_v=cfg.d_v)

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

    def forward(self, x):
        B, L = x.shape
        h = self.tok_embed(x)
        pos = self.pos_embed(L)
        return h, pos

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, dropout=0.0, d_v=4):
        super().__init__()
        self.attn = SelfAttention(dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim))
        #self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        #self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.rms1 = nn.RMSNorm(dim)
        self.rms2 = nn.RMSNorm(dim)

        self.drop_attn = nn.Dropout(dropout)
        self.drop_mlp  = nn.Dropout(dropout)

        # DDL expanded residuals
        self.attn_delta = DeltaResidualExpanded(dim, d_v)
        self.mlp_delta = DeltaResidualExpanded(dim, d_v)

    def forward(self, Y, p_emb):
        # Y: (B, T, d, d_v)
        x_in = self.attn_delta.compress(Y)
        sublayer_out = self.drop_attn(self.attn(x_in, p_emb))
        Y = self.attn_delta(Y, sublayer_out, x_in)
        # Post-norm on each value channel
        Y = self.rms1(Y.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)

        x_in = self.mlp_delta.compress(Y)
        sublayer_out = self.drop_mlp(self.mlp(x_in))
        Y = self.mlp_delta(Y, sublayer_out, x_in)
        Y = self.rms2(Y.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)

        return Y

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

def mdlm_loss(denoiser, x0, schedule, z_H=None,
              answer_mask=None, t_min=1e-4):
    """
    MDLM denoiser loss: cross-entropy on masked positions.

    Returns:
        loss
    """
    mask_id = denoiser.cfg.vocab_size
    loss_fn = MDLMLoss(schedule, mask_id, t_min=t_min)
    xt, t, is_masked = loss_fn.perturb(x0, answer_mask=answer_mask)
    logits = denoiser(xt, z_H)
    loss = loss_fn(logits, x0, is_masked)
    return loss


def nce_loss(energy_model, denoiser, x0, schedule,
             z_H=None, answer_mask=None, t_min=1e-4):
    """
    NCE loss for the energy model.
    Denoiser is frozen (no_grad); only energy_model gets gradients.

    Returns:
        loss
    """
    n_vocab = denoiser.cfg.vocab_size
    mask_id = n_vocab
    loss_fn = NCELoss(schedule, mask_id, n_vocab, t_min=t_min)

    xt, t, is_masked = loss_fn.perturb(x0, answer_mask=answer_mask)

    # Negative sample from frozen denoiser
    with torch.no_grad():
        logits = denoiser(xt, z_H)
    x_neg = loss_fn.sample_neg(x0, logits, is_masked)

    # Energies -- positive (true x0) and negative (denoiser sample)
    e_pos = energy_model(x0, z_H)
    e_neg = energy_model(x_neg, z_H)

    loss = loss_fn(e_pos, e_neg)
    return loss


# ============================================================
# Sampling
# ============================================================

@torch.no_grad()
def sample(denoiser, schedule, batch_size, seq_len, num_steps=256,
           energy_model=None, k=8, window_w=0.2,
           z_H=None,
           device=torch.device("cpu")):
    """
    MDLM / EDLM sampling with optional ponder state

    Args:
        denoiser:      SRLMDenoiser
        schedule:      LogLinearSchedule
        energy_model:  SRLMEnergyModel or None (pure MDLM if None)
        k:             importance sampling candidates (only with energy_model)
        window_w:      importance sampling window [1-w, 1]
        z_H:           ponder state from pioneer, or None

    Returns:
        xt:      generated tokens (batch_size, seq_len)
    """
    n_vocab = denoiser.cfg.vocab_size
    mask_id = n_vocab
    sampler = Sampler(schedule, mask_id, n_vocab)
    xt, stepper = sampler(batch_size, seq_len, device, num_steps)

    for s in stepper:
        logits = denoiser(xt, z_H)
        if energy_model is not None and s.tau_n >= 1.0 - window_w:
            candidates = s.propose_x0_k(xt, logits, k)       # (k, B, L)
            c_flat = candidates.reshape(k * batch_size, seq_len)
            if z_H is not None:
                zH_flat = z_H.unsqueeze(0).expand(k, -1, -1, -1) \
                              .reshape(k * batch_size, *z_H.shape[1:])
            else:
                zH_flat = None
            energies = energy_model(c_flat, zH_flat)
            energies = energies.reshape(k, batch_size)
            x0 = s.select_by_energy(candidates, energies)
        else:
            x0 = s.propose_x0(xt, logits)

        xt = s.reverse_step(xt, x0)

    return xt


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
        memory: Optional[torch.Tensor] = None,
        answer_mask: Optional[torch.Tensor] = None,
        t_min: float = 1e-4,
    ) -> tuple[dict[str, float], torch.Tensor]:
        """
        Args:
            x0:          clean tokens (denoiser target, answer included)
            memory:      G-Mem state or None
            answer_mask: (B, L) bool -- True at answer positions to noise
            t_min:       minimum diffusion time

        Returns:
            losses:  dict of loss values for logging
            memory:  updated memory state (detached)
        """
        state = self.init_train_state(x0, memory, answer_mask, t_min)
        for seg in range(self.N_super):
            state.roll()
            if state.compute_losses(seg):
                break
        return state.finish()

    def init_train_state(self, x0, memory, answer_mask, t_min):
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
        h, p_emb = denoiser.get_front(x_in)
        z_H, z_L = denoiser.init_states(h)

        return PonderTrainerState(
            trainer=self,
            loss_fn=loss_fn,
            min_seg=min_seg,
            x0=x0,
            x_in=x_in,
            answer_mask=answer_mask,
            memory=memory,
            z_H=z_H,
            z_L=z_L,
            seg=0,
            all_losses={k: 0.0 for k in [
                "LM", "BCE", "mem", "rep", "equil", "RH_stable", "total"
            ]},
        )

class PonderTrainerState:
    def __init__(self, trainer, loss_fn, min_seg,
                 x0, x_in, answer_mask, memory,
                 z_H, z_L, seg, all_losses):
        self.trainer = trainer
        self.loss_fn = loss_fn
        self.min_seg = min_seg
        self.x0 = x0
        self.x_in = x_in
        self.answer_mask = answer_mask
        self.memory = memory
        self.z_H = z_H
        self.z_L = z_L
        self.seg = seg
        self.all_losses = all_losses
        # Set after roll()
        self.q_values = None
        self.importance_scores = None
        self.p_emb = None
        # Set before compute_common_losses (by compute_losses or externally)
        self.logits = None
        self.is_masked = None

    def roll(self):
        denoiser = self.trainer.denoiser
        h, p_emb = denoiser.get_front(self.x_in)
        self.z_H, self.z_L, self.q_values, self.memory, self.importance_scores = \
            denoiser.roll(h, p_emb, self.z_H, self.z_L, self.memory)
        self.p_emb = p_emb

    def compute_losses(self, seg):
        trainer = self.trainer
        denoiser = trainer.denoiser

        # Fresh perturbation each segment -- different t, different masks
        xt, t, is_masked = self.loss_fn.perturb(self.x0, answer_mask=self.answer_mask)
        logits = denoiser(xt, self.z_H)
        lm_loss = self.loss_fn(logits, self.x0, is_masked)

        # Store for compute_common_losses (BCE needs logits + is_masked)
        self.logits = logits
        self.is_masked = is_masked

        return self.compute_common_losses(seg, lm_loss)

    def compute_common_losses(self, seg, lm_loss):
        trainer = self.trainer
        denoiser = trainer.denoiser
        device = self.x0.device

        losses = {}
        losses["LM"] = lm_loss

        # ACT halt loss -- soft accuracy on masked positions
        with torch.no_grad():
            token_correct = (self.logits.argmax(-1) == self.x0).float()
            if self.is_masked.any():
                masked_correct = (token_correct * self.is_masked.float()).sum(-1)
                masked_count = self.is_masked.float().sum(-1).clamp(min=1)
                accuracy = masked_correct / masked_count
            else:
                accuracy = token_correct.mean(-1)
        target = torch.stack([accuracy, 1.0 - accuracy], dim=-1)
        losses["BCE"] = F.binary_cross_entropy(self.q_values, target)

        # Memory regularization
        mem_loss_fn = MemoryLoss()
        losses["mem"] = mem_loss_fn(self.importance_scores)

        losses["rep"] = repulsion_loss(self.z_H)

        # Expensive losses only on last segment
        is_last = (seg == trainer.N_super - 1)
        if is_last:
            block = denoiser.ponder.block
            losses["equil"] = equilibrium_loss(self.z_H, self.z_L, block, self.p_emb)
            losses["RH_stable"] = routh_hurwitz_stable_loss(
                self.z_H + self.z_L, block, self.p_emb)
        else:
            losses["equil"] = torch.tensor(0.0, device=device)
            losses["RH_stable"] = torch.tensor(0.0, device=device)

        total = (
            trainer.lambda_LM * losses["LM"]
            + trainer.lambda_BCE * losses["BCE"]
            + trainer.lambda_mem * losses["mem"]
            + trainer.lambda_rep * losses["rep"]
            + trainer.lambda_equil * losses["equil"]
            + trainer.lambda_RH_stable * losses["RH_stable"]
        )

        total.backward()

        # Detach latent states for next segment (deep supervision)
        self.z_H = self.z_H.detach()
        self.z_L = self.z_L.detach()
        self.memory = self.memory.detach()
        self.seg = seg

        for k, v in losses.items():
            self.all_losses[k] += v.item()
        self.all_losses["total"] += total.item()

        # ACT early stopping: halt when Q_halt > Q_continue
        # after running at least min_seg segments
        if seg >= self.min_seg - 1:
            with torch.no_grad():
                q_mean = self.q_values.mean(0)
                if q_mean[0] > q_mean[1]:
                    return True
        return False

    def finish(self):
        denoiser = self.trainer.denoiser
        n_run = self.seg + 1
        for p in denoiser.parameters():
            if p.grad is not None:
                p.grad /= n_run

        memory = self.memory.detach()
        return {k: v / n_run for k, v in self.all_losses.items()}, memory
