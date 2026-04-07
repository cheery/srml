"""
DGPO-style GRPO for discrete diffusion (EDLM).

Based on "Reinforcing Diffusion Models by Direct Group Preference Optimization"
(Luo et al., 2025), adapted for masked discrete diffusion (MDLM).

Flow:
1. Generate K candidates per prompt via reverse diffusion
2. Score each with reward_fn
3. Compute group-relative z-score advantages
4. DGPO loss: sigmoid-weighted advantage * CE on masked positions
   - Reference model = frozen weights before optimization
   - Sigmoid weight adapts based on model drift from reference
"""

import torch
import torch.nn.functional as F
from .edlm import MDLMLoss, mask_tokens, Sampler
from .model import PonderTrainer

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
        if '=' not in text or '+' not in text:
            return -10000.0
        lhs, rhs = text.split('=', 1)
        rhs = rhs.strip()
        if not rhs.isdigit():
            # Penalize garbled output (e.g. '85nk', trailing junk)
            rhs_digits = ''
            for ch in rhs:
                if ch.isdigit():
                    rhs_digits += ch
                else:
                    break
            if not rhs_digits:
                return -10000.0
            # Scale down reward by fraction of clean chars
            garble_penalty = len(rhs_digits) / len(rhs)
        else:
            rhs_digits = rhs
            garble_penalty = 1.0
        parts = lhs.split('+', 1)
        if len(parts) != 2:
            return -10000.0
        n = int(parts[0].strip())
        m = int(parts[1].strip())
        y = int(rhs_digits)
        correct = n + m
        if y == correct:
            return 10 - (1.0 - garble_penalty) * 10000
        error = abs(y - correct)
        # Smooth falloff: always distinguishes closer vs farther answers
        #closeness = 1.0 / (1.0 + error)
        return -error - (1.0 - garble_penalty) * 10000
        #return (0.1 + 0.9 * closeness) * garble_penalty
    except (ValueError, OverflowError):
        return -10000.0


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
        for r in range(9):
            digits = [d for d in grid[r] if 1 <= d <= 9]
            checks += 1
            if len(digits) == len(set(digits)) == 9:
                score += 1.0
        for c in range(9):
            digits = [grid[r][c] for r in range(9) if 1 <= grid[r][c] <= 9]
            checks += 1
            if len(digits) == len(set(digits)) == 9:
                score += 1.0
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
        return score / checks
    except (ValueError, IndexError):
        return 0.0


def grpo_step(
    denoiser,
    ref_denoiser,
    optimizer,
    schedule,
    prompt_batch,
    clean_batch,
    reward_fn,
    device,
    memory=None,
    answer_mask=None,
    K=4,
    sampling_steps=50,
    grad_clip=1.0,
    epochs=1,
    clip_epsilon=0.2,
    kl_coef=0.01,
    t_min=1e-4,
    lambda_temp=5.0,
    train_steps=5,       # number of trajectory steps to train on
    T=1.0,
    beta_dgpo=None,
    verbose=True,
):
    """RL-D² RKL policy gradient for masked discrete diffusion.

    Uses on-policy single-step ratios (Section 4.1 of Ma et al., 2025).
    Each denoising step is an action in an augmented MDP. The per-step
    ratio p_θ(x0|xt,t)/p_old(x0|xt,t) is tractable, and used with
    PPO clipping.

    Key: trains on a random subset of trajectory steps (not all of
    them) to keep the gradient strong and computation bounded.
    """
    B, L = prompt_batch.shape
    mask_id = denoiser.cfg.vocab_size
    visible = (prompt_batch != mask_id).to(device)
    prompt_batch = prompt_batch.to(device)

    # --- 1. Generate K candidates, recording on-policy trajectory ---
    denoiser.eval()

    # Pioneer: let the model "think" about the prompt before generating
    with torch.no_grad():
        old_z_H, old_memory = denoiser.pioneer(prompt_batch, memory=memory)
        ref_z_H, ref_memory = ref_denoiser.pioneer(prompt_batch, memory=memory)

    prompt_expanded = prompt_batch.repeat_interleave(K, dim=0)
    visible_expanded = prompt_expanded != mask_id
    old_z_H_expanded = old_z_H.repeat_interleave(K, dim=0)
    ref_z_H_expanded = ref_z_H.repeat_interleave(K, dim=0)

    sampler = Sampler(schedule, mask_id, denoiser.cfg.vocab_size)
    xt, stepper = sampler(B * K, L, device, sampling_steps)
    xt = torch.where(visible_expanded, prompt_expanded, xt)

    trajectory = []

    with torch.no_grad():
        for s in stepper:
            is_masked = (xt == mask_id)

            logits = denoiser(xt, old_z_H_expanded)
            x0 = s.propose_x0(xt, logits/T)

            old_logp = F.log_softmax(logits/T, dim=-1)
            old_logp = torch.gather(old_logp, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)

            ref_logits = ref_denoiser(xt, ref_z_H_expanded)
            ref_logp = F.log_softmax(ref_logits/T, dim=-1)
            ref_logp = torch.gather(ref_logp, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)

            trajectory.append({
                "xt": xt.clone(),
                "x0": x0.clone(),
                "old_logp": old_logp.clone(),
                "ref_logp": ref_logp.clone(),
                "is_masked": is_masked.clone(),
            })

            xt = s.reverse_step(xt, x0)

        generated = xt

    # --- 2. Compute rewards and advantages ---
    rewards = torch.zeros(B * K, device=device)
    for i in range(B * K):
        rewards[i] = reward_fn(generated[i])

    # Clip outlier rewards so garbled outputs don't dominate z-scores
    rewards.clamp_(min=-200)

    rewards_grouped = rewards.view(B, K)
    mean_r = rewards_grouped.mean(dim=1, keepdim=True)
    std_r  = rewards_grouped.std(dim=1, keepdim=True)
    advantages = (rewards_grouped - mean_r) / (std_r + 1e-4)
    advantages = advantages.view(B * K, 1)

    def debug():
        def _tok2str(t):
            return bytes(t.clamp(0, 255).cpu().tolist()).decode("utf-8", errors="replace").rstrip()
        print("    GRPO candidates:")
        for b in range(min(B, 4)):
            prompt_str = _tok2str(prompt_batch[b])
            clean_str = _tok2str(clean_batch[b])
            print(f"      prompt: {repr(prompt_str):30s}  target: {repr(clean_str)}")
            for k in range(K):
                idx = b * K + k
                gen_str = _tok2str(generated[idx])
                r = rewards[idx].item()
                a = advantages[idx].item()
                print(f"        k={k}: {repr(gen_str):40s} reward={r:.3f} adv={a:.4f}")
        print()
    if verbose:
        debug()

    # Ponder trainer for auxiliary losses (BCE, mem, rep, equil, RH)
    clean_expanded = clean_batch.repeat_interleave(K, dim=0).to(device)
    if memory is None:
        memory_expanded = None
    else:
        memory_expanded = memory.repeat_interleave(K, dim=0)
    answer_mask_expanded = (~visible_expanded)
    trainer = PonderTrainer(denoiser=denoiser, schedule=schedule)

    # --- 3. RKL policy gradient (train on random quarter of trajectory) ---
    n_train = max(1, len(trajectory) // 12)
    train_indices = torch.randperm(len(trajectory))[:n_train].sort().values
    train_trajectory = [trajectory[i] for i in train_indices]

    full_loss = 0.0
    for epoch in range(epochs):
        total_loss = torch.tensor(0.0, device=device)

        for step_data in reversed(train_trajectory):
            optimizer.zero_grad()
            xt_s       = step_data["xt"]
            x0_s       = step_data["x0"]
            old_logp_s = step_data["old_logp"]
            ref_logp_s = step_data["ref_logp"]
            is_masked  = step_data["is_masked"]

            if not is_masked.any():
                continue

            state = trainer.init_train_state(
                x0=clean_expanded,
                memory=memory_expanded,
                answer_mask=answer_mask_expanded,
                t_min=t_min,
            )
            for seg in range(trainer.N_super):
                state.roll()

                logits_s = denoiser(xt_s, z_H=state.z_H)
                curr_logp_s = F.log_softmax(logits_s/T, dim=-1)
                curr_logp_s = torch.gather(curr_logp_s, dim=-1, index=x0_s.unsqueeze(-1)).squeeze(-1)

                # Per-token ratio vs old policy (on-policy, recorded during generation)
                ratio = torch.exp(curr_logp_s - old_logp_s)
                clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)

                # PPO clipped surrogate
                surr1 = ratio * advantages
                surr2 = clipped_ratio * advantages
                pg_loss = -torch.min(surr1, surr2)

                # KL penalty vs frozen reference model
                log_r = ref_logp_s - curr_logp_s
                kl_penalty = torch.exp(log_r) - log_r - 1

                per_token_loss = pg_loss + kl_coef * kl_penalty

                mask_f = is_masked.float()
                sample_loss = (per_token_loss * mask_f).sum(dim=-1) / mask_f.sum(dim=-1).clamp(min=1)
                step_loss = sample_loss.mean()
                total_loss = total_loss + step_loss.item()

                # Set state fields for BCE computation, then run ponder losses
                state.logits = logits_s
                state.is_masked = is_masked
                if state.compute_common_losses(seg, step_loss):
                    break
            state.finish()

            # --- Gradient flow diagnostic ---
            if verbose:
                groups = {
                    "ponder.block": denoiser.ponder.block,
                    "ponder.q_head": denoiser.ponder.q_head,
                    "front_layers": denoiser.front_layers,
                    "back_layers": denoiser.back_layers,
                    "latent_memory": denoiser.latent_memory,
                    "out_proj": denoiser.out_proj,
                }
                grad_parts = []
                for name, mod in groups.items():
                    grads = [p.grad for p in mod.parameters() if p.grad is not None]
                    if grads:
                        norm = torch.cat([g.flatten() for g in grads]).norm().item()
                    else:
                        norm = 0.0
                    grad_parts.append(f"{name}={norm:.4f}")
                print(f"    grpo grad norms: {' | '.join(grad_parts)}")

            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), grad_clip)
            optimizer.step()
        full_loss += total_loss.item()

    metrics = {
        'mean_reward': rewards.mean().item(),
        'frac_correct': (rewards > -0.5).float().mean().item(),
    }
    return full_loss / max(len(trajectory), 1), old_memory, metrics
