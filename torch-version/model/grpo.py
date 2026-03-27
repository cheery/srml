"""
DGPO-style GRPO for discrete diffusion (SEDD).

Based on "Reinforcing Diffusion Models by Direct Group Preference Optimization"
(Luo et al., 2025), adapted for absorbing-state discrete diffusion.

Flow:
1. Generate K candidates per prompt via reverse diffusion
2. Score each with reward_fn
3. Compute group-relative z-score advantages
4. DGPO loss: sigmoid-weighted advantage * score_entropy
   - Reference model = frozen weights before optimization
   - Sigmoid weight adapts based on model drift from reference
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
# DGPO-style GRPO step
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
    memories=None,       # optional memory tensors for model
    K=4,                 # candidates per prompt
    sampling_steps=50,   # diffusion steps for generation
    grad_clip=1.0,
    epochs=5,
    beta_dgpo=1.0,       # DGPO sigmoid temperature
    fused=False,          # fused backward: faster but more VRAM
    max_act_steps=8,     # adaptive computation steps per diffusion step
):
    """DGPO-style GRPO update for discrete diffusion.

    1. Generate K candidates per prompt via reverse diffusion
    2. Score each with reward_fn
    3. Compute group-relative z-score advantages
    4. Cache reference score_entropy (model before optimization)
    5. Optimize with DGPO loss: σ(group_drift) * advantage * score_entropy

    The sigmoid weighting from DGPO prevents the model from drifting too
    far from the reference in a single step — it's an implicit KL constraint.
    """
    B, L = prompt_batch.shape
    d_model = z[0].shape[-1]
    MASK_TOKEN = graph.dim - 1

    prompt_batch = prompt_batch.to(device)
    clean_batch = clean_batch.to(device)
    visible = (prompt_batch != MASK_TOKEN)

    model.eval()

    # --- 1. Generate K candidates per prompt via diffusion ---
    prompt_expanded = prompt_batch.repeat_interleave(K, dim=0)
    visible_mask = (prompt_expanded != MASK_TOKEN)
    visible_values = prompt_expanded.clone()

    def projector(x):
        return torch.where(visible_mask, visible_values, x)

    memories_expanded = None
    if memories is not None:
        memories_expanded = [m.repeat_interleave(K, dim=0) for m in memories]

    x = graph.sample_limit(B * K, L, device=device)
    eps = 1e-5
    timesteps = torch.linspace(1.0, eps, sampling_steps + 1, device=device)
    dt = (1 - eps) / sampling_steps

    with torch.no_grad():
        for i in range(sampling_steps):
            t = timesteps[i].expand(B * K)
            x = projector(x)
            sigma, dsigma = noise(t)
            # Fresh z + adaptive computation via Q_head
            z_gen = make_z_for_grpo(B * K, L, d_model, device)
            ix = model.front(x, sigma, memories=memories_expanded)
            for _ in range(max_act_steps):
                z_gen, log_score, q, _ = model.step(z_gen, ix)
                if (q.squeeze(-1) > 0).all():
                    break
            score = log_score.exp()
            rev_rate = dt * dsigma[..., None, None] * graph.reverse_rate(x, score)
            x = graph.sample_rate(x, rev_rate)

        # Final denoising step
        x = projector(x)
        t = timesteps[-1].expand(B * K)
        sigma = noise(t)[0]
        z_gen = make_z_for_grpo(B * K, L, d_model, device)
        ix = model.front(x, sigma)
        for _ in range(max_act_steps):
            z_gen, log_score, q, _ = model.step(z_gen, ix, memories=memories_expanded)
            if (q.squeeze(-1) > 0).all():
                break
        score = log_score.exp()
        stag_score = graph.staggered_score(score, sigma)
        probs = stag_score * graph.transp_transition(x.long(), sigma)
        probs = probs[..., :-1]
        x = sample_categorical(probs)

    x = projector(x)
    generated = x  # (B*K, L)

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
    # Before any optimization, compute score_entropy for the current model
    # at pre-sampled noise levels. These serve as the DGPO reference.
    sampling_eps = 1e-3
    ref_cache = []  # [epoch][k] = {sigma, dsigma, perturbed, candidate, ref_loss}

    with torch.no_grad():
        for ep in range(epochs):
            ep_data = []
            for k in range(K):
                idx = torch.arange(B, device=device) * K + k
                candidate_k = generated[idx.cpu()].to(device)

                t = ((1 - sampling_eps) * torch.rand(B, device=device) + sampling_eps)
                sigma, dsigma = noise(t)
                perturbed = graph.sample_transition(candidate_k, sigma[:, None])
                perturbed = torch.where(visible, prompt_batch, perturbed)

                z_k = tuple(zi[:B].detach() for zi in z) if z[0].shape[0] >= B else z
                z_k, log_score, _ = model(z_k, perturbed, sigma, memories=memories)
                ref_loss_pp = graph.score_entropy(log_score, sigma[:, None], perturbed, candidate_k)
                ref_loss_ps = (dsigma[:, None] * ref_loss_pp).sum(dim=-1)  # (B,)

                ep_data.append({
                    'sigma': sigma,
                    'dsigma': dsigma,
                    'perturbed': perturbed.clone(),
                    'candidate': candidate_k,
                    'ref_loss': ref_loss_ps,
                })
            ref_cache.append(ep_data)

    # --- 5. DGPO optimization with Q_head ---
    model.train()
    total_loss = 0.0

    for ep in range(epochs):
        optimizer.zero_grad()
        n_tokens = 0

        if fused:
            # Fused mode: single backward over all K (faster, more VRAM)
            losses_k = []
            q_loss_total = torch.tensor(0.0, device=device)
            aux_total = torch.tensor(0.0, device=device)
            for k in range(K):
                c = ref_cache[ep][k]
                z_k = tuple(zi[:B].detach() for zi in z) if z[0].shape[0] >= B else z
                ix = model.front(c['perturbed'], c['sigma'], memories=memories)
                z_k, log_score, q, aux_loss = model.step(z_k, ix)
                loss_pp = graph.score_entropy(log_score, c['sigma'][:, None], c['perturbed'], c['candidate'])
                loss_ps = (c['dsigma'][:, None] * loss_pp).sum(dim=-1)
                losses_k.append(loss_ps)
                aux_total = aux_total + aux_loss

                # Q_head BCE: predict whether candidate is good (reward > 0.5)
                idx = torch.arange(B, device=device) * K + k
                q_target = (rewards[idx] > 0.5).float()
                q_loss_total = q_loss_total + F.binary_cross_entropy_with_logits(
                    q.squeeze(-1), q_target)

            group_terms = torch.zeros(B, device=device)
            for k in range(K):
                idx = torch.arange(B, device=device) * K + k
                delta = losses_k[k].detach() - ref_cache[ep][k]['ref_loss']
                group_terms += advantages[idx] * beta_dgpo * delta / K
            group_weights = torch.sigmoid(group_terms)

            weighted_sum = torch.tensor(0.0, device=device)
            for k in range(K):
                idx = torch.arange(B, device=device) * K + k
                weighted_sum = weighted_sum + (group_weights.detach() * advantages[idx] * losses_k[k]).sum()
                n_masked = (~visible).sum().item()
                n_tokens += max(n_masked, 1)

            (weighted_sum + q_loss_total + aux_total).backward()
            total_loss += weighted_sum.item()

        else:
            # Sequential mode: per-k backward (less VRAM, extra no-grad pass)
            cur_losses_detached = []
            with torch.no_grad():
                for k in range(K):
                    c = ref_cache[ep][k]
                    z_k = tuple(zi[:B].detach() for zi in z) if z[0].shape[0] >= B else z
                    z_k, log_score, _ = model(z_k, c['perturbed'], c['sigma'], memories=memories)
                    loss_pp = graph.score_entropy(log_score, c['sigma'][:, None], c['perturbed'], c['candidate'])
                    loss_ps = (c['dsigma'][:, None] * loss_pp).sum(dim=-1)
                    cur_losses_detached.append(loss_ps)

            group_terms = torch.zeros(B, device=device)
            for k in range(K):
                idx = torch.arange(B, device=device) * K + k
                delta = cur_losses_detached[k] - ref_cache[ep][k]['ref_loss']
                group_terms += advantages[idx] * beta_dgpo * delta / K
            group_weights = torch.sigmoid(group_terms)

            for k in range(K):
                c = ref_cache[ep][k]
                idx = torch.arange(B, device=device) * K + k
                adv_k = advantages[idx]

                z_k = tuple(zi[:B].detach() for zi in z) if z[0].shape[0] >= B else z
                ix = model.front(c['perturbed'], c['sigma'], memories=memories)
                z_k, log_score, q, aux_loss = model.step(z_k, ix)
                loss_pp = graph.score_entropy(log_score, c['sigma'][:, None], c['perturbed'], c['candidate'])
                loss_ps = (c['dsigma'][:, None] * loss_pp).sum(dim=-1)

                # Q_head BCE: predict whether candidate is good (reward > 0.5)
                q_target = (rewards[idx] > 0.5).float()
                q_loss = F.binary_cross_entropy_with_logits(q.squeeze(-1), q_target)

                weighted_loss = (group_weights.detach() * adv_k * loss_ps).sum()
                n_masked = (~visible).sum().item()
                n_tokens += max(n_masked, 1)

                (weighted_loss + q_loss + aux_loss).backward()
                total_loss += weighted_loss.item()

        if n_tokens > 0:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.div_(n_tokens)

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        z_out = tuple(zi.detach() for zi in z_k)

    metrics = {
        'mean_reward': rewards.mean().item(),
        'max_reward': rewards.max().item(),
        'min_reward': rewards.min().item(),
        'std_reward': rewards.std().item(),
        'frac_correct': (rewards > 0.5).float().mean().item(),
    }

    return total_loss / max(n_tokens, 1), z_out, metrics


def make_z_for_grpo(batch_size, seq_len, d_model, device):
    """Create fresh HRM state for GRPO sampling."""
    z = torch.zeros(batch_size, seq_len, d_model, device=device)
    return z, z.clone()
