"""
GRPO (Group Relative Policy Optimization) for discrete diffusion.

Adapted from GRPO-Zero for SRLM/SEDD:
- Generate K candidate completions via diffusion sampling
- Score each with a verifier (reward function)
- Compute group-relative advantages
- Backprop SEDD loss weighted by advantage

The key insight: instead of autoregressive log_prob * advantage,
we use sedd_score_entropy * advantage. Good samples get upweighted
in the denoising objective.
"""

import torch
import torch.nn.functional as F
from .catsample import sample_categorical


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def arithmetic_reward(tokens):
    """Check if an arithmetic expression N+M=Y is correct.

    Args:
        tokens: 1-D int tensor of token IDs (byte values)
    Returns:
        float reward: 1.0 if correct, 0.0 if wrong/unparseable
    """
    try:
        text = bytes(tokens.clamp(0, 255).cpu().tolist()).decode("utf-8", errors="replace").strip()
        # Find the pattern: N+M=Y
        if '=' not in text or '+' not in text:
            return 0.0
        lhs, rhs = text.split('=', 1)
        # rhs might have trailing spaces/garbage — take leading digits
        rhs = rhs.strip()
        rhs_digits = ''
        for ch in rhs:
            if ch.isdigit():
                rhs_digits += ch
            else:
                break
        if not rhs_digits:
            return 0.0
        parts = lhs.split('+', 1)
        if len(parts) != 2:
            return 0.0
        n = int(parts[0].strip())
        m = int(parts[1].strip())
        y = int(rhs_digits)
        correct = n + m
        if y == correct:
            return 1.0
        # Partial reward: 0.1 for producing any number after =,
        # plus up to 0.9 for closeness
        error = abs(y - correct)
        closeness = max(0.0, 1.0 - error / max(correct, 1))
        return 0.1 + 0.9 * closeness
    except (ValueError, OverflowError):
        return 0.0


def sudoku_reward(generated_tokens, puzzle_tokens=None):
    """Score a sudoku solution by constraint satisfaction.

    Checks row, column, and box uniqueness (27 constraints).
    Returns fraction of constraints satisfied (0.0 to 1.0).

    Args:
        generated_tokens: 1-D int tensor (89 bytes: 9 rows + 8 newlines)
        puzzle_tokens:    unused (kept for API compatibility)
    Returns:
        float reward: 0.0 to 1.0
    """
    try:
        text = bytes(generated_tokens[:89].clamp(0, 255).cpu().tolist()).decode("utf-8", errors="replace")
        lines = text.split('\n')
        if len(lines) != 9:
            return 0.0
        grid = []
        for line in lines:
            row = [int(ch) for ch in line[:9] if ch.isdigit()]
            if len(row) != 9:
                return 0.0
            grid.append(row)

        score = 0.0
        checks = 0
        # Row uniqueness
        for r in range(9):
            digits = [d for d in grid[r] if 1 <= d <= 9]
            checks += 1
            if len(digits) == len(set(digits)) == 9:
                score += 1.0
        # Column uniqueness
        for c in range(9):
            digits = [grid[r][c] for r in range(9) if 1 <= grid[r][c] <= 9]
            checks += 1
            if len(digits) == len(set(digits)) == 9:
                score += 1.0
        # Box uniqueness
        for br in range(3):
            for bc in range(3):
                digits = []
                for r in range(br*3, br*3+3):
                    for c in range(bc*3, bc*3+3):
                        if 1 <= grid[r][c] <= 9:
                            digits.append(grid[r][c])
                checks += 1
                if len(digits) == len(set(digits)) == 9:
                    score += 1.0
        return score / checks  # 0.0 to 1.0
    except (ValueError, IndexError):
        return 0.0


# ---------------------------------------------------------------------------
# GRPO step for diffusion
# ---------------------------------------------------------------------------

def grpo_step(
    model,
    optimizer,
    loss_fn,
    sampler,
    graph,
    noise,
    prompt_batch,        # (B, L) int tensor — prompts with masks
    clean_batch,         # (B, L) int tensor — ground truth tokens
    reward_fn,           # callable(generated_tokens) -> float
    z,                   # HRM state
    device,
    verbose=False,       # print prompts, samples, rewards
    K=4,                 # candidates per prompt
    sampling_steps=50,   # diffusion steps for generation
    grad_clip=0.1,
):
    """One GRPO update step.

    1. Generate K candidates per prompt via diffusion
    2. Score each with reward_fn
    3. Compute group-relative advantages
    4. Weighted SEDD loss update

    Args:
        model: SRLM model
        optimizer: optimizer
        loss_fn: SEDD loss function (from loss_function())
        sampler: Sampler instance
        graph: AbsorbingGraph
        noise: LogLinearNoise
        prompt_batch: (B, L) — prompts (visible tokens + MASK_TOKEN for blanks)
        clean_batch:  (B, L) — full ground truth
        reward_fn: callable(1-D tokens) -> float reward
        z: HRM state tuple
        device: torch device
        K: number of candidates per prompt
        sampling_steps: diffusion sampling steps
        grad_clip: gradient clipping norm
    Returns:
        loss_value: float
        z: updated HRM state
        metrics: dict with reward stats
    """
    B, L = prompt_batch.shape
    d_model = z[0].shape[-1]
    MASK_TOKEN = graph.dim - 1  # absorbing state = last index = 256

    model.eval()

    # --- 1. Generate K candidates per prompt via diffusion ---
    # Expand prompts: (B,L) -> (B*K, L)
    prompt_expanded = prompt_batch.repeat_interleave(K, dim=0).to(device)

    # Build projector that fixes visible (non-mask) positions
    visible_mask = (prompt_expanded != MASK_TOKEN)
    visible_values = prompt_expanded.clone()

    def projector(x):
        return torch.where(visible_mask, visible_values, x)
    x = graph.sample_limit(B * K, L, device=device)
    eps = 1e-5
    timesteps = torch.linspace(1.0, eps, sampling_steps + 1, device=device)
    dt = (1 - eps) / sampling_steps

    with torch.no_grad():
        z_gen = make_z_for_grpo(B * K, L, d_model, device)
        for i in range(sampling_steps):
            t = timesteps[i].expand(B * K)
            x = projector(x)
            sigma, dsigma = noise(t)
            z_gen, log_score, _ = model(z_gen, x, sigma)
            score = log_score.exp()
            rev_rate = dt * dsigma[..., None, None] * graph.reverse_rate(x, score)
            x = graph.sample_rate(x, rev_rate)

        # Final denoising step
        x = projector(x)
        t = timesteps[-1].expand(B * K)
        sigma = noise(t)[0]
        z_gen, log_score, _ = model(z_gen, x, sigma)
        score = log_score.exp()
        stag_score = graph.staggered_score(score, sigma)
        probs = stag_score * graph.transp_transition(x, sigma)
        probs = probs[..., :-1]
        x = sample_categorical(probs)

    x = projector(x)  # ensure visible positions are correct
    generated = x  # (B*K, L)

    # --- 2. Score candidates ---
    rewards = torch.zeros(B * K, device=device)
    for i in range(B * K):
        rewards[i] = reward_fn(generated[i])

    if verbose:
        def _tok2str(t):
            return bytes(t.clamp(0, 255).cpu().tolist()).decode("utf-8", errors="replace").rstrip()
        print("    GRPO candidates:")
        for b in range(min(B, 4)):  # show up to 4 prompts
            prompt_str = _tok2str(prompt_batch[b])
            clean_str = _tok2str(clean_batch[b])
            print(f"      prompt: {repr(prompt_str):30s}  target: {repr(clean_str)}")
            for k in range(K):
                idx = b * K + k
                gen_str = _tok2str(generated[idx])
                r = rewards[idx].item()
                print(f"        k={k}: {repr(gen_str):40s} reward={r:.3f}")
        print()

    # --- 3. Group-relative advantages ---
    rewards_grouped = rewards.view(B, K)
    mean_r = rewards_grouped.mean(dim=1, keepdim=True)
    std_r = rewards_grouped.std(dim=1, keepdim=True)
    # When all candidates have same reward, advantage is 0 (no signal).
    # Use raw centered rewards as fallback when std is tiny.
    has_variance = (std_r > 1e-4).float()
    safe_std = std_r.clamp(min=1e-4)
    advantages = ((rewards_grouped - mean_r) / safe_std * has_variance).view(B * K)

    # If no group has variance (all rewards identical), fall back to
    # regular SEDD loss on the best candidate (or clean target)
    any_signal = (advantages.abs() > 1e-6).any().item()

    # --- 4. Weighted SEDD loss ---
    model.train()
    optimizer.zero_grad()

    total_loss = 0.0
    n_pos = 0

    if not any_signal:
        # No RL signal — fall back to regular SEDD on clean target
        sampling_eps = 1e-3
        t = ((1 - sampling_eps) * torch.rand(B, device=device) + sampling_eps)
        sigma, dsigma = noise(t)
        perturbed = graph.sample_transition(clean_batch.to(device), sigma[:, None])
        vis = (prompt_batch.to(device) != MASK_TOKEN)
        perturbed = torch.where(vis, clean_batch.to(device), perturbed)
        z_k = tuple(zi[:B].detach() for zi in z) if z[0].shape[0] >= B else z
        z_k, log_score, aux_loss = model(z_k, perturbed, sigma)
        loss_per_pos = graph.score_entropy(log_score, sigma[:, None], perturbed, clean_batch.to(device))
        fallback_loss = (dsigma[:, None] * loss_per_pos).sum(dim=-1).mean() + aux_loss
        fallback_loss.backward()
        total_loss = fallback_loss.item()
        n_pos = 1
    else:
        for k in range(K):
            idx = torch.arange(B, device=device) * K + k
            candidate_k = generated[idx.cpu()].to(device)   # (B, L)
            advantage_k = advantages[idx]                    # (B,)

            # SEDD loss on this candidate
            sampling_eps = 1e-3
            t = ((1 - sampling_eps) * torch.rand(B, device=device) + sampling_eps)
            sigma, dsigma = noise(t)
            perturbed = graph.sample_transition(candidate_k, sigma[:, None])

            # Keep visible positions from prompt
            vis = (prompt_batch.to(device) != MASK_TOKEN)
            perturbed = torch.where(vis, candidate_k, perturbed)

            z_k = tuple(zi[:B].detach() for zi in z) if z[0].shape[0] >= B else z
            z_k, log_score, aux_loss = model(z_k, perturbed, sigma)
            loss_per_pos = graph.score_entropy(log_score, sigma[:, None], perturbed, candidate_k)
            loss_per_sample = (dsigma[:, None] * loss_per_pos).sum(dim=-1)  # (B,)

            # Weight by advantage
            weighted_loss = (loss_per_sample * advantage_k).mean()
            weighted_loss = weighted_loss + aux_loss
            weighted_loss.backward()

            total_loss += weighted_loss.item()
            n_pos += 1

    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    # Update z from the last forward pass
    z_out = tuple(zi.detach() for zi in z_k)

    metrics = {
        'mean_reward': rewards.mean().item(),
        'max_reward': rewards.max().item(),
        'min_reward': rewards.min().item(),
        'std_reward': rewards.std().item(),
        'frac_correct': (rewards > 0.5).float().mean().item(),
    }

    return total_loss / max(n_pos, 1), z_out, metrics


def make_z_for_grpo(batch_size, seq_len, d_model, device):
    """Create fresh HRM state for GRPO sampling."""
    z = torch.zeros(batch_size, seq_len, d_model, device=device)
    return z, z.clone()
