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
            return 0.0
        lhs, rhs = text.split('=', 1)
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


# ---------------------------------------------------------------------------
# DGPO-style GRPO step
# ---------------------------------------------------------------------------

def grpo_step(
    denoiser,
    optimizer,
    schedule,
    prompt_batch,        # (B, L) int tensor -- prompts with masks
    clean_batch,         # (B, L) int tensor -- ground truth tokens
    reward_fn,           # callable(generated_tokens) -> float
    device,
    verbose=False,
    memory=None,         # G-Mem state or None
    answer_mask=None,    # (B, L) bool -- True at answer positions
    K=4,                 # candidates per prompt
    sampling_steps=50,   # diffusion steps for generation
    grad_clip=1.0,
    epochs=5,
    beta_dgpo=1.0,       # DGPO sigmoid temperature
    t_min=1e-4,
):
    """DGPO-style GRPO update for masked discrete diffusion.

    1. Generate K candidates per prompt via reverse diffusion
    2. Score each with reward_fn
    3. Compute group-relative z-score advantages
    4. Cache reference CE loss (model before optimization)
    5. Optimize with DGPO loss: sigma(group_drift) * advantage * CE_loss
    """
    B, L = prompt_batch.shape
    n_vocab = denoiser.cfg.vocab_size
    mask_id = n_vocab  # MASK token

    prompt_batch = prompt_batch.to(device)
    clean_batch = clean_batch.to(device)

    # Visible positions: not masked in prompt
    visible = (prompt_batch != mask_id)

    denoiser.eval()

    # --- 1. Generate K candidates per prompt via diffusion ---
    prompt_expanded = prompt_batch.repeat_interleave(K, dim=0)   # (B*K, L)
    visible_expanded = prompt_expanded != mask_id

    memory_expanded = None
    if memory is not None:
        memory_expanded = memory.repeat_interleave(K, dim=0)

    sampler = Sampler(schedule, mask_id, n_vocab)
    xt, stepper = sampler(B * K, L, device, sampling_steps)
    z = None

    with torch.no_grad():
        for s in stepper:
            # Clamp visible (query) positions each step
            xt = torch.where(visible_expanded, prompt_expanded, xt)
            z, logits, memory_expanded, _ = denoiser(z, xt, s.t, memory_expanded)
            x0 = s.propose_x0(xt, logits)
            xt = s.reverse_step(xt, x0)

    # Final projection: ensure query positions are correct
    generated = torch.where(visible_expanded, prompt_expanded, xt)

    # --- 2. Score candidates with reward_fn ---
    rewards = torch.zeros(B * K, device=device)
    for i in range(B * K):
        rewards[i] = reward_fn(generated[i])

    if verbose:
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
                print(f"        k={k}: {repr(gen_str):40s} reward={r:.3f}")
        print()

    # --- 3. Group-relative advantages (z-score per group) ---
    rewards_grouped = rewards.view(B, K)
    mean_r = rewards_grouped.mean(dim=1, keepdim=True)
    std_r = rewards_grouped.std(dim=1, keepdim=True)
    advantages = (rewards_grouped - mean_r) / (std_r + 1e-4)
    advantages = advantages.view(B * K)

    # --- 4. Pre-cache reference losses and perturbations ---
    loss_fn = MDLMLoss(schedule, mask_id, t_min=t_min)
    ref_cache = []

    with torch.no_grad():
        for ep in range(epochs):
            ep_data = []
            for k in range(K):
                idx = torch.arange(B, device=device) * K + k
                candidate_k = generated[idx.cpu()].to(device)  # (B, L)

                ans_mask_k = answer_mask if answer_mask is not None else None

                xt_k, t_k, is_masked_k = loss_fn.perturb(
                    candidate_k, answer_mask=ans_mask_k)

                # Keep visible/query positions clean
                xt_k = torch.where(visible, prompt_batch, xt_k)

                _, logits_k, _, _ = denoiser(None, xt_k, t_k, memory)
                ref_loss = loss_fn(logits_k, candidate_k, is_masked_k)

                ep_data.append({
                    'xt': xt_k.clone(),
                    't': t_k.clone(),
                    'is_masked': is_masked_k.clone(),
                    'candidate': candidate_k,
                    'ref_loss': ref_loss.item(),
                })
            ref_cache.append(ep_data)

    # --- 5. DGPO optimization ---
    denoiser.train()
    total_loss = 0.0
    n_steps = 0

    for ep in range(epochs):
        optimizer.zero_grad()

        # Forward all K candidates, accumulate per-sample losses
        losses_k = []
        for k in range(K):
            c = ref_cache[ep][k]
            _, logits_k, _, _ = denoiser(None, c['xt'], c['t'], memory)
            loss_k = loss_fn(logits_k, c['candidate'], c['is_masked'])
            losses_k.append(loss_k)

        # Compute DGPO group weights per prompt (detached)
        group_terms = torch.zeros(B, device=device)
        for k in range(K):
            idx = torch.arange(B, device=device) * K + k
            delta = losses_k[k].detach() - ref_cache[ep][k]['ref_loss']
            group_terms += advantages[idx].mean() * beta_dgpo * delta / K
        group_weights = torch.sigmoid(group_terms)

        # DGPO loss: sigma(group_drift) * advantage * CE_loss
        weighted_sum = torch.tensor(0.0, device=device)
        for k in range(K):
            idx = torch.arange(B, device=device) * K + k
            adv_k = advantages[idx].mean()
            weighted_sum = weighted_sum + (group_weights.detach() * adv_k * losses_k[k]).sum()

        weighted_sum.backward()
        total_loss += weighted_sum.item()
        n_steps += 1

        torch.nn.utils.clip_grad_norm_(denoiser.parameters(), grad_clip)
        optimizer.step()

    # Detach memory for return
    memory_out = memory.detach() if memory is not None else None

    metrics = {
        'mean_reward': rewards.mean().item(),
        'max_reward': rewards.max().item(),
        'min_reward': rewards.min().item(),
        'std_reward': rewards.std().item(),
        'frac_correct': (rewards > 0.5).float().mean().item(),
    }

    return total_loss / max(n_steps, 1), memory_out, metrics
