"""
Contraction Mapping Model (CMM)
===============================
"Dynamical Systems Theory Behind a Hierarchical Reasoning Model"
(Es'kin & Smorkalov, 2026)

A TRM-style recursive reasoning architecture with:
  - Shared recursive transformer block (L-module = H-module)
  - Contraction mapping via equilibrium + Routh-Hurwitz stability losses
  - Hyperspherical repulsion loss to prevent representational collapse
  - StableMax3/5 polynomial softmax approximations
  - Neural SDE noise injection for robust latent trajectories
  - Adaptive loss balancing via AlgGradNorm
  - Deep supervision with ACT (adaptive computational time)
  - EMA weight averaging

Architecture (Fig 2, TRM variant):
  x → InputEmbedding → x̃
  For N_H high-level cycles:
    For N_L low-level steps:
      ẑ_L = F(ẑ_H + ẑ_L + x̃; θ)     [L-module, Eq 4]
    ẑ_H = F(ẑ_H + ẑ_L; θ)             [H-module, Eq 5]
  y = OutputEmbedding(ẑ_H)

The single module F consists of two transformer layers (Attention/MLP-mixer + MLP)
with RMS norm and residual connections, using tanh activation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional


# ============================================================
# StableMax variants (Section 3.2.3, Eqs 35-40)
# ============================================================

def _stablemax_s(x: torch.Tensor) -> torch.Tensor:
    """Original StableMax base function s(x), Eq 36."""
    pos = 1.0 + x
    neg = 1.0 / (1.0 - x)
    return torch.where(x >= 0, pos, neg)


def _stablemax3_s(x: torch.Tensor) -> torch.Tensor:
    """StableMax3 base function s_3(x), Eq 38.
    3rd-order Taylor approx of exp(x)."""
    pos = 1.0 + x * (1.0 + 0.5 * x * (1.0 + x / 3.0))
    neg = 1.0 / (1.0 - x * (1.0 - 0.5 * x * (1.0 - x / 3.0)))
    return torch.where(x >= 0, pos, neg)


def _stablemax5_s(x: torch.Tensor) -> torch.Tensor:
    """StableMax5 base function s_5(x), Eq 40.
    5th-order Taylor approx of exp(x)."""
    pos = 1.0 + x * (1.0 + 0.5 * x * (1.0 + x * (1.0 + x / 5.0) / 4.0 / 3.0))
    neg_inner = 1.0 - x * (1.0 - 0.5 * x * (1.0 - x * (1.0 - x / 5.0) / 4.0 / 3.0))
    return torch.where(x >= 0, pos, 1.0 / neg_inner)


def stablemax(logits: torch.Tensor, variant: str = "3", dim: int = -1) -> torch.Tensor:
    """StableMax normalization (replaces softmax). Eqs 35, 37, 39."""
    # Subtract max for numerical stability (same trick as softmax)
    logits = logits - logits.max(dim=dim, keepdim=True).values
    if variant == "1":
        s = _stablemax_s(logits)
    elif variant == "3":
        s = _stablemax3_s(logits)
    elif variant == "5":
        s = _stablemax5_s(logits)
    else:
        raise ValueError(f"Unknown StableMax variant: {variant}")
    return s / s.sum(dim=dim, keepdim=True)


def stablemax_cross_entropy(logits: torch.Tensor, targets: torch.Tensor,
                            variant: str = "3") -> torch.Tensor:
    """Cross-entropy using StableMax instead of softmax."""
    probs = stablemax(logits, variant=variant)
    # Clamp for log stability
    log_probs = torch.log(probs.clamp(min=1e-10))
    return F.nll_loss(log_probs.view(-1, logits.size(-1)),
                      targets.view(-1), reduction="mean")


# ============================================================
# RMS Normalization
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


# ============================================================
# MLP-Mixer layer (token-mixing alternative to attention)
# ============================================================

class MLPMixer(nn.Module):
    """MLP-mixer: token-mixing MLP applied across the sequence dimension."""
    def __init__(self, seq_len: int, dim: int, expansion: int = 4):
        super().__init__()
        self.token_mix = nn.Sequential(
            nn.Linear(seq_len, seq_len * expansion),
            nn.Tanh(),
            nn.Linear(seq_len * expansion, seq_len),
        )

    def forward(self, x):
        # x: (B, S, D) → transpose → mix tokens → transpose back
        return self.token_mix(x.transpose(-1, -2)).transpose(-1, -2)


# ============================================================
# Self-Attention with optional StableMax
# ============================================================

class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4,
                 use_stablemax: str = "none"):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.use_stablemax = use_stablemax

    def forward(self, x):
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # 3 x (B, H, S, D_h)

        if self.use_stablemax != "none":
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = stablemax(attn, variant=self.use_stablemax, dim=-1)
            out = attn @ v
        else:
            out = F.scaled_dot_product_attention(q, k, v)

        out = out.transpose(1, 2).reshape(B, S, D)
        return self.out_proj(out)


# ============================================================
# Recursive Transformer Block (Section 3.6, Fig 2)
# ============================================================

class RecursiveBlock(nn.Module):
    """
    Single transformer block used recursively as both L-module and H-module.

    Structure (bottom-up per Fig 2, post-norm):
      x → Attention/MLP-mixer → + x (residual) → RMS Norm
        → MLP → + prev (residual) → RMS Norm
    Applied ×2 (two sub-layers stacked).

    Post-norm is critical: it constrains the state magnitude after each
    residual addition, preventing representation blowup during deep recursion.
    Uses tanh activation for contraction mapping properties (Section 3.6).
    """
    def __init__(self, dim: int, seq_len: int, num_heads: int = 4,
                 mlp_ratio: int = 4, use_attention: bool = False,
                 use_stablemax: str = "none"):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(2):  # ×2 sub-layers
            layer = nn.ModuleDict({
                "norm1": RMSNorm(dim),  # post-norm after mixer + residual
                "norm2": RMSNorm(dim),  # post-norm after MLP + residual
                "mlp": nn.Sequential(
                    nn.Linear(dim, dim * mlp_ratio),
                    nn.Tanh(),
                    nn.Linear(dim * mlp_ratio, dim),
                ),
            })
            if use_attention:
                layer["mixer"] = SelfAttention(dim, num_heads, use_stablemax)
            else:
                layer["mixer"] = MLPMixer(seq_len, dim)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer["norm1"](x + layer["mixer"](x))   # post-norm
            x = layer["norm2"](x + layer["mlp"](x))     # post-norm
        return x


# ============================================================
# Q-Head for ACT (Section 2.3.2, Eq 8)
# ============================================================

class QHead(nn.Module):
    """
    Predicts Q-values for halt/continue actions.
    Q̂^(m) = σ(θ_Q^T · ẑ_H^(m·N_L·N_H))

    Output: (q_halt, q_continue) after sigmoid.
    """
    def __init__(self, dim: int, seq_len: int):
        super().__init__()
        self.proj = nn.Linear(dim * seq_len, 2)

    def forward(self, z_H: torch.Tensor) -> torch.Tensor:
        """z_H: (B, S, D) → (B, 2) Q-values after sigmoid."""
        B = z_H.size(0)
        flat = z_H.reshape(B, -1)
        return torch.sigmoid(self.proj(flat))


# ============================================================
# CMM Model
# ============================================================

class CMM(nn.Module):
    """
    Contraction Mapping Model.

    Args:
        n_vocab: vocabulary size (input and output)
        seq_len: sequence length
        dim: hidden dimension D
        num_heads: attention heads (if use_attention=True)
        N_H: high-level recursion steps
        N_L: low-level recursion steps per H-step
        mlp_ratio: MLP expansion ratio
        use_attention: use self-attention (True) or MLP-mixer (False)
        use_stablemax: "none", "1", "3", or "5"
        noise_sigma: σ for NSDE noise injection (0 = deterministic ODE)
        noise_type: "additive" or "multiplicative"
        identical_layers: if True, use weight-sharing within the block too
    """
    def __init__(
        self,
        n_vocab: int,
        seq_len: int,
        dim: int = 512,
        num_heads: int = 4,
        N_H: int = 3,
        N_L: int = 6,
        mlp_ratio: int = 4,
        use_attention: bool = False,
        use_stablemax: str = "3",
        noise_sigma: float = 0.0,
        noise_type: str = "additive",
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.seq_len = seq_len
        self.dim = dim
        self.N_H = N_H
        self.N_L = N_L
        self.noise_sigma = noise_sigma
        self.noise_type = noise_type
        self.use_stablemax = use_stablemax

        # Input/output embeddings
        self.input_embed = nn.Embedding(n_vocab, dim)
        self.output_embed = nn.Linear(dim, n_vocab)

        # Single shared recursive block (TRM: F_L ≡ F_H)
        self.block = RecursiveBlock(
            dim, seq_len, num_heads, mlp_ratio,
            use_attention, use_stablemax,
        )

        # Q-head for ACT
        self.q_head = QHead(dim, seq_len)

    def _inject_noise(self, z: torch.Tensor) -> torch.Tensor:
        """NSDE noise injection (Eqs 42-43)."""
        if self.noise_sigma <= 0 or not self.training:
            return z
        noise = torch.randn_like(z) * self.noise_sigma
        if self.noise_type == "additive":
            return z + noise  # Eq 42
        else:
            return z * (1.0 + noise)  # Eq 43

    def forward_reasoning(
        self,
        x_emb: torch.Tensor,
        z_H: Optional[torch.Tensor] = None,
        z_L: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run N_H × N_L recursive reasoning steps (Eqs 4-5).

        Args:
            x_emb: embedded input x̃ (B, S, D)
            z_H: initial high-level state (B, S, D) or None → x̃
            z_L: initial low-level state (B, S, D) or None → zeros

        Returns:
            z_H: final high-level state
            z_L: final low-level state
        """
        B, S, D = x_emb.shape

        if z_H is None:
            z_H = x_emb.clone()
        if z_L is None:
            z_L = torch.zeros_like(x_emb)

        step = 0
        for h in range(self.N_H):
            for l in range(self.N_L):
                # L-module: ẑ_L = F(ẑ_H + ẑ_L + x̃; θ)  [Eq 4]
                z_L = self.block(z_H + z_L + x_emb)
                z_L = self._inject_noise(z_L)
                step += 1

            # H-module: ẑ_H = F(ẑ_H + ẑ_L; θ)  [Eq 5]
            z_H = self.block(z_H + z_L)
            z_H = self._inject_noise(z_H)

        return z_H, z_L

    def forward(
        self,
        x: torch.Tensor,
        z_H: Optional[torch.Tensor] = None,
        z_L: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Full forward pass.

        Args:
            x: input token ids (B, S)
            z_H, z_L: optional initial latent states (for continued reasoning)

        Returns:
            dict with: logits, z_H, z_L, q_values
        """
        x_emb = self.input_embed(x)
        z_H, z_L = self.forward_reasoning(x_emb, z_H, z_L)

        logits = self.output_embed(z_H)  # (B, S, n_vocab)
        q_values = self.q_head(z_H)      # (B, 2)

        return {
            "logits": logits,
            "z_H": z_H,
            "z_L": z_L,
            "q_values": q_values,
        }


# ============================================================
# Auxiliary Losses (Section 3.2)
# ============================================================

def equilibrium_loss(z_H: torch.Tensor, z_L: torch.Tensor,
                     block: RecursiveBlock) -> torch.Tensor:
    """
    Equilibrium loss (Eq 30):
    L_equil = [ẑ_H - F̂_H(ẑ_H + ẑ_L; θ_H)]²

    Encourages z_H to be a fixed point of the H-module.
    The block input is detached to avoid double gradient paths
    that cause instability.
    """
    inp = (z_H + z_L).detach()
    target = block(inp)
    return F.mse_loss(z_H, target)


def routh_hurwitz_stable_loss(z: torch.Tensor,
                              block: RecursiveBlock) -> torch.Tensor:
    """
    RH stability loss (Eq 31):
    L_RH_stable = [ReLU(1/K Σ J_ii - 1)]²

    Encourages the mean diagonal Jacobian < 1 (contraction) at the
    stable equilibrium point ẑ_H = ŷ.

    Uses finite-difference approximation of the Jacobian diagonal.
    """
    z_req = z.detach().requires_grad_(True)
    Fz = block(z_req)
    # Approximate trace of Jacobian via random projection (Hutchinson's trick)
    v = torch.randn_like(z_req)
    Fz_v = (Fz * v).sum()
    grad_v = torch.autograd.grad(Fz_v, z_req, create_graph=True)[0]
    # E[v^T J v] = trace(J), normalized by K = total elements per sample
    K = z.shape[1] * z.shape[2]  # S * D
    trace_est = (grad_v * v).sum(dim=(1, 2)) / K  # (B,) ≈ mean(J_ii)
    return F.relu(trace_est - 1.0).pow(2).mean()


def routh_hurwitz_unstable_loss(z: torch.Tensor,
                                block: RecursiveBlock) -> torch.Tensor:
    """
    RH unstable loss (Eq 32):
    L_RH_unstable = [ReLU(1 - 1/K Σ J_ii)]²

    Encourages the mean diagonal Jacobian > 1 (expansion) at the
    unstable equilibrium point ẑ_H = x̃ (repeller).
    """
    z_req = z.detach().requires_grad_(True)
    Fz = block(z_req)
    v = torch.randn_like(z_req)
    Fz_v = (Fz * v).sum()
    grad_v = torch.autograd.grad(Fz_v, z_req, create_graph=True)[0]
    K = z.shape[1] * z.shape[2]
    trace_est = (grad_v * v).sum(dim=(1, 2)) / K
    return F.relu(1.0 - trace_est).pow(2).mean()


def repulsion_loss(z: torch.Tensor) -> torch.Tensor:
    """
    Hyperspherical repulsion loss (Eq 34):
    L_rep = 1/(B(B-1)) Σ_i Σ_{j≠i} ⟨u_i, u_j⟩²

    where u_i = z_i / ||z_i|| are L2-normalized flattened representations.
    Prevents representational collapse by pushing samples apart.
    """
    B = z.size(0)
    if B < 2:
        return torch.tensor(0.0, device=z.device)

    # Flatten and normalize
    flat = z.reshape(B, -1)  # (B, S*D)
    u = F.normalize(flat, dim=-1)  # (B, S*D)

    # Gram matrix of cosine similarities
    gram = u @ u.t()  # (B, B)

    # Mask out diagonal, square, and average
    mask = 1.0 - torch.eye(B, device=z.device)
    return (gram.pow(2) * mask).sum() / (B * (B - 1))


# ============================================================
# AlgGradNorm — Adaptive Loss Balancing (Section 3.4.1)
# ============================================================

class AlgGradNorm:
    """
    Algebraic Gradient Normalization for multi-objective loss balancing.
    (Eqs 45-50)

    Dynamically adjusts loss weights so gradients from different loss
    components have similar magnitudes.
    """
    def __init__(self, loss_names: list[str], alpha: float = 1.5,
                 rho: float = 0.95, T_reset: int = 1000):
        self.loss_names = loss_names
        self.N = len(loss_names)
        self.alpha = alpha
        self.rho = rho
        self.T_reset = T_reset

        # Weights λ_n, initialized uniformly
        self.weights = {name: 1.0 for name in loss_names}
        # Initial loss values L_n(0) for relative progress
        self.initial_losses = {name: None for name in loss_names}
        self.step_count = 0

    def update(self, losses: dict[str, float],
               grad_norms: dict[str, float]):
        """
        Update weights based on current losses and their gradient norms.

        Args:
            losses: {name: loss_value} for each component
            grad_norms: {name: ||∇_w (λ_n L_n)||_2} for each component
        """
        self.step_count += 1

        # Soft reset initial losses periodically (Eq 46 note)
        if self.step_count % self.T_reset == 1:
            for name in self.loss_names:
                self.initial_losses[name] = None

        # Set initial losses if needed
        for name in self.loss_names:
            if self.initial_losses[name] is None:
                self.initial_losses[name] = max(losses[name], 1e-8)

        # Relative inverse training rates r_n (Eq 46)
        L_tilde = {}
        for name in self.loss_names:
            L_tilde[name] = losses[name] / self.initial_losses[name]
        mean_L_tilde = sum(L_tilde.values()) / self.N
        r = {name: L_tilde[name] / max(mean_L_tilde, 1e-8)
             for name in self.loss_names}

        # Average gradient norm (Eq 45)
        G_bar = sum(grad_norms.values()) / max(self.N, 1)

        # Target gradient norms (Eq 47)
        G_target = {name: G_bar * (r[name] ** self.alpha)
                     for name in self.loss_names}

        # Update weights (Eq 48)
        lambda_temp = {}
        for name in self.loss_names:
            ratio = G_target[name] / max(grad_norms[name], 1e-8)
            ratio = max(0.1, min(10.0, ratio))  # clamp
            lambda_temp[name] = self.weights[name] * ratio

        # Renormalize (Eq 49)
        total = sum(lambda_temp.values())
        for name in self.loss_names:
            lambda_hat = lambda_temp[name] * self.N / max(total, 1e-8)
            # EMA smoothing (Eq 50)
            self.weights[name] = (self.rho * self.weights[name]
                                  + (1 - self.rho) * lambda_hat)

    def get_weights(self) -> dict[str, float]:
        return dict(self.weights)


# ============================================================
# Deep Supervision Training (Section 2.3, Fig 4)
# ============================================================

@dataclass
class CMMTrainer:
    """
    Deep supervision trainer for CMM.

    Runs N_super supervision segments, each performing a full
    N_H × N_L reasoning pass. The latent state is detached between
    segments (one-step gradient approximation). ACT Q-head learns
    when to halt.
    """
    model: CMM
    N_super: int = 16
    use_stablemax: str = "3"
    lambda_LM: float = 1.0
    lambda_BCE: float = 0.5
    lambda_rep_x: float = 1e3
    lambda_rep_z: float = 1e3
    lambda_equil_x: float = 1.0
    lambda_equil_z: float = 1.0
    lambda_RH_stable_z: float = 1e4
    lambda_RH_unstable_x: float = 10.0
    use_alg_grad_norm: bool = False
    grad_norm: Optional[AlgGradNorm] = field(default=None, init=False)

    def __post_init__(self):
        if self.use_alg_grad_norm:
            self.grad_norm = AlgGradNorm([
                "LM", "BCE", "rep_x", "rep_z",
                "equil_x", "equil_z",
                "RH_stable_z", "RH_unstable_x",
            ])

    def compute_losses(
        self,
        result: dict[str, torch.Tensor],
        y_true: torch.Tensor,
        x_emb: torch.Tensor,
        compute_expensive: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Compute loss components for one supervision segment.

        Args:
            compute_expensive: if False, skip RH and equilibrium losses
                (which require extra forward/backward passes through the block).
        """
        logits = result["logits"]
        z_H = result["z_H"]
        z_L = result["z_L"]
        q_values = result["q_values"]
        block = self.model.block
        device = logits.device

        losses = {}

        # Primary task loss (Eq 7) — always computed
        losses["LM"] = stablemax_cross_entropy(
            logits, y_true, variant=self.use_stablemax)

        # ACT halt loss (Eq 11) — cheap
        correct = (logits.argmax(-1) == y_true).all(dim=-1).float()
        target_halt = correct
        target_continue = torch.zeros_like(correct)
        target = torch.stack([target_halt, target_continue], dim=-1)
        losses["BCE"] = F.binary_cross_entropy(q_values, target)

        # Repulsion losses (Eq 34) — cheap (just a gram matrix)
        losses["rep_x"] = repulsion_loss(x_emb)
        losses["rep_z"] = repulsion_loss(z_H)

        if compute_expensive:
            # Equilibrium losses (Eq 30) — one extra block forward
            losses["equil_z"] = equilibrium_loss(z_H, z_L, block)
            losses["equil_x"] = equilibrium_loss(x_emb, torch.zeros_like(x_emb), block)

            # RH stability losses (Eqs 31-32) — block forward + autograd
            losses["RH_stable_z"] = routh_hurwitz_stable_loss(z_H + z_L, block)
            losses["RH_unstable_x"] = routh_hurwitz_unstable_loss(x_emb, block)
        else:
            zero = torch.tensor(0.0, device=device)
            losses["equil_z"] = zero
            losses["equil_x"] = zero
            losses["RH_stable_z"] = zero
            losses["RH_unstable_x"] = zero

        return losses

    def weighted_loss(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        """Combine losses with weights (Eq 44)."""
        if self.use_alg_grad_norm and self.grad_norm is not None:
            weights = self.grad_norm.get_weights()
        else:
            weights = {
                "LM": self.lambda_LM,
                "BCE": self.lambda_BCE,
                "rep_x": self.lambda_rep_x,
                "rep_z": self.lambda_rep_z,
                "equil_x": self.lambda_equil_x,
                "equil_z": self.lambda_equil_z,
                "RH_stable_z": self.lambda_RH_stable_z,
                "RH_unstable_x": self.lambda_RH_unstable_x,
            }

        total = sum(weights.get(k, 0) * v for k, v in losses.items()
                    if k in weights)
        return total

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        N_accum: int = 1,
    ) -> dict[str, float]:
        """
        One deep supervision training step.

        Runs N_super segments with gradient accumulation over N_accum
        consecutive segments before each optimizer step.

        Args:
            x: input (B, S) token ids
            y: target (B, S) token ids
            optimizer: the optimizer
            N_accum: gradient accumulation steps (= N_super for full accum)

        Returns:
            dict of loss values for logging
        """
        self.model.train()
        x_emb = self.model.input_embed(x)

        # Initialize latent states (Section 4.4: initialized by x̃ and 0)
        z_H = x_emb.clone()
        z_L = torch.zeros_like(x_emb)

        all_losses = {k: 0.0 for k in [
            "LM", "BCE", "rep_x", "rep_z",
            "equil_x", "equil_z", "RH_stable_z", "RH_unstable_x", "total"
        ]}
        accum_count = 0

        for seg in range(self.N_super):
            x_emb = self.model.input_embed(x)
            # Forward reasoning pass
            z_H_new, z_L_new = self.model.forward_reasoning(x_emb, z_H, z_L)

            result = {
                "logits": self.model.output_embed(z_H_new),
                "z_H": z_H_new,
                "z_L": z_L_new,
                "q_values": self.model.q_head(z_H_new),
            }

            # Only compute expensive losses (equilibrium, RH) on last segment
            # where z_H should actually be near the fixed point
            is_last = (seg == self.N_super - 1)
            losses = self.compute_losses(result, y, x_emb,
                                         compute_expensive=is_last)
            total = self.weighted_loss(losses) / N_accum

            total.backward()
            accum_count += 1

            # Accumulate and step
            if accum_count >= N_accum:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                # Update AlgGradNorm if enabled
                if self.use_alg_grad_norm and self.grad_norm is not None:
                    loss_vals = {k: v.item() for k, v in losses.items()}
                    # Approximate gradient norms (use loss magnitude as proxy)
                    gnorms = {k: abs(v) * self.grad_norm.weights.get(k, 1.0)
                              for k, v in loss_vals.items()}
                    self.grad_norm.update(loss_vals, gnorms)

                accum_count = 0

            # Detach states for next segment (one-step grad approx)
            z_H = z_H_new.detach()
            z_L = z_L_new.detach()

            # Log losses
            for k, v in losses.items():
                all_losses[k] += v.item()
            all_losses["total"] += total.item() * N_accum

            # ACT early stopping: halt if Q_halt > Q_continue
            with torch.no_grad():
                q = result["q_values"].mean(0)
                if q[0] > q[1] and seg >= 1:
                    break

        # Average over segments actually run
        n_run = seg + 1
        return {k: v / n_run for k, v in all_losses.items()}


# ============================================================
# EMA (Exponential Moving Average)
# ============================================================

class EMA:
    """Simple EMA for model weights."""
    def __init__(self, model: nn.Module, mu: float = 0.999):
        self.mu = mu
        self.shadow = {name: p.data.clone()
                       for name, p in model.named_parameters()}

    def update(self, model: nn.Module):
        for name, p in model.named_parameters():
            self.shadow[name].mul_(self.mu).add_(p.data, alpha=1 - self.mu)

    def apply(self, model: nn.Module):
        self.backup = {name: p.data.clone()
                       for name, p in model.named_parameters()}
        for name, p in model.named_parameters():
            p.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        for name, p in model.named_parameters():
            p.data.copy_(self.backup[name])


# ============================================================
# Demo / Test
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Small test config
    n_vocab = 10
    seq_len = 16
    dim = 64
    batch_size = 8

    model = CMM(
        n_vocab=n_vocab,
        seq_len=seq_len,
        dim=dim,
        N_H=2,
        N_L=4,
        use_attention=False,
        use_stablemax="3",
        noise_sigma=0.01,
        noise_type="additive",
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {params:,}")

    # Dummy data: identity mapping (predict input)
    x = torch.randint(0, n_vocab, (batch_size, seq_len), device=device)
    y = x.clone()

    # Test forward
    result = model(x)
    print(f"Logits: {result['logits'].shape}")
    print(f"Q-values: {result['q_values'][:2]}")

    # Test training
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = CMMTrainer(
        model=model,
        N_super=4,
        use_stablemax="3",
    )

    print("\nTraining (identity task)...")
    for step in range(500):
        x = torch.randint(0, n_vocab, (batch_size, seq_len), device=device)
        y = x.clone()
        losses = trainer.train_step(x, y, optimizer, N_accum=2)
        if (step + 1) % 10 == 0:
            acc = (model(x)["logits"].argmax(-1) == y).float().mean()
            print(f"  step {step+1:3d}  "
                  f"LM {losses['LM']:.4f}  "
                  f"equil_z {losses['equil_z']:.4f}  "
                  f"rep_z {losses['rep_z']:.4f}  "
                  f"acc {acc:.3f}")

    print("\nDone!")
