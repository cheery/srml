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


# def grpo_step(
#     denoiser,
#     optimizer,
#     schedule,
#     prompt_batch,        # (B, L) int tensor -- prompts with masks
#     clean_batch,         # (B, L) int tensor -- ground truth tokens
#     reward_fn,           # callable(generated_tokens) -> float
#     device,
#     verbose=True,
#     memory=None,         # G-Mem state or None
#     answer_mask=None,    # (B, L) bool -- True at answer positions
#     K=4,                 # candidates per prompt
#     sampling_steps=50,   # diffusion steps for generation
#     grad_clip=1.0,
#     epochs=5,
#     beta_dgpo=100.0,      # DGPO sigmoid temperature (ref uses 100)
#     clip_range=0.05,      # PPO-style ratio clipping (0 = disabled)
#     t_min=1e-4,
# ):
#     memory = None
#     """DGPO-style GRPO update for masked discrete diffusion.
# 
#     1. Generate K candidates per prompt via reverse diffusion
#     2. Score each with reward_fn
#     3. Compute group-relative z-score advantages
#     4. Cache reference CE loss (model before optimization)
#     5. Optimize with DGPO loss: sigma(group_drift) * advantage * CE_loss
#     """
#     B, L = prompt_batch.shape
#     n_vocab = denoiser.cfg.vocab_size
#     mask_id = n_vocab  # MASK token
# 
#     prompt_batch = prompt_batch.to(device)
#     clean_batch = clean_batch.to(device)
# 
#     # Visible positions: not masked in prompt
#     visible = (prompt_batch != mask_id)
# 
#     denoiser.eval()
# 
#     # --- 1. Generate K candidates per prompt via diffusion ---
#     prompt_expanded = prompt_batch.repeat_interleave(K, dim=0)   # (B*K, L)
#     visible_expanded = prompt_expanded != mask_id
# 
#     memory_expanded = None
#     if memory is not None:
#         memory_expanded = memory.repeat_interleave(K, dim=0)
# 
#     sampler = Sampler(schedule, mask_id, n_vocab)
#     xt, stepper = sampler(B * K, L, device, sampling_steps)
#     z = None
# 
#     ref_logits = None
# 
#     with torch.no_grad():
#         for s in stepper:
#             # Clamp visible (query) positions each step
#             xt = torch.where(visible_expanded, prompt_expanded, xt)
#             z, logits, memory_expanded, _ = denoiser(z, xt, s.t, memory_expanded)
#             if ref_logits is None:
#                 ref_logits = logits
#             x0 = s.propose_x0(xt, logits)
#             xt = s.reverse_step(xt, x0)
# 
#     # Final projection: ensure query positions are correct
#     generated = torch.where(visible_expanded, prompt_expanded, xt)
# 
#     # --- 2. Score candidates with reward_fn ---
#     rewards = torch.zeros(B * K, device=device)
#     for i in range(B * K):
#         rewards[i] = reward_fn(generated[i]) * 2.0 - 1.0
# 
#     if verbose:
#         def _tok2str(t):
#             return bytes(t.clamp(0, 255).cpu().tolist()).decode("utf-8", errors="replace").rstrip()
#         print("    GRPO candidates:")
#         for b in range(min(B, 4)):
#             prompt_str = _tok2str(prompt_batch[b])
#             clean_str = _tok2str(clean_batch[b])
#             print(f"      prompt: {repr(prompt_str):30s}  target: {repr(clean_str)}")
#             for k in range(K):
#                 idx = b * K + k
#                 gen_str = _tok2str(generated[idx])
#                 r = rewards[idx].item()
#                 print(f"        k={k}: {repr(gen_str):40s} reward={r:.3f}")
#         print()
# 
#     # --- 3. Group-relative advantages (z-score per group) ---
#     rewards_grouped = rewards.view(B, K)
#     mean_r = rewards_grouped.mean(dim=1, keepdim=True)
#     std_r = rewards_grouped.std(dim=1, keepdim=True)
#     advantages = (rewards_grouped - mean_r) / (std_r + 1e-4)
#     advantages = advantages.view(B * K)
# 
#     # --- 4. GRPO step (single update over all K candidates) ---
#     denoiser.train()
# 
#     loss_fn = MDLMLoss(schedule, mask_id, t_min=t_min)
# 
#     # Prepare all candidates and their reference log-probs (from frozen reference)
#     # We need the reference log-probs for each candidate to compute KL.
#     # Since we don't have a separate reference model, we use the current model's
#     # logits from the generation phase (they were computed under no_grad, so they are frozen).
#     # We'll detach them explicitly.
# 
#     # ref_logits is of shape (B*K, L, vocab) and has no grad already.
#     # Compute reference log-probs per token for all candidates at once.
#     with torch.no_grad():
#         ref_log_probs = ref_logits.log_softmax(dim=-1)  # (B*K, L, vocab)
#         # Gather the log prob of the actual token at each position
#         # generated_tokens shape: (B*K, L)
#         ref_token_logps = torch.gather(
#             ref_log_probs, dim=-1, index=generated.unsqueeze(-1)
#         ).squeeze(-1)  # (B*K, L)
# 
#     # Reshape advantages to match (B*K, 1) for broadcasting
#     advantages = advantages.unsqueeze(1)  # (B*K, 1)
# 
#     # We'll collect losses for each candidate group
#     total_loss = 0.0
# 
#     # For numerical stability, we'll compute the per-token loss once for all K
#     # First, compute current model's log-probs for all candidates.
#     # We need to perturb each candidate separately because the mask positions differ.
#     # But we can loop over K and accumulate a single loss.
# 
#     optimizer.zero_grad()
# 
#     for k in range(K):
#         candidate_k = generated[k::K]          # (B, L)
#         adv_k = advantages[k::K]               # (B, 1)
# 
#         # Perturb the candidate (mask random positions according to schedule)
#         xt_k, t_k, is_masked_k = loss_fn.perturb(candidate_k, answer_mask=answer_mask)
#         # Keep visible/prompt positions fixed
#         xt_k = torch.where(visible, prompt_batch, xt_k)
# 
#         # Forward pass through current model
#         _, logits_k, _, _ = denoiser(None, xt_k, t_k, memory=None)
#         curr_log_probs = logits_k.log_softmax(dim=-1)   # (B, L, vocab)
#         curr_token_logps = torch.gather(
#             curr_log_probs, dim=-1, index=candidate_k.unsqueeze(-1)
#         ).squeeze(-1)  # (B, L)
# 
#         # Reference token log-probs for this candidate (already frozen)
#         ref_token_logps_k = ref_token_logps[k::K]  # (B, L)
# 
#         # Compute per-token KL divergence (forward KL: E_ref[log(ref/curr)])
#         # Using the formula: KL(ref || curr) = ref_logp - curr_logp
#         # But we want a penalty that discourages large drifts.
#         # The DGPO paper uses a sigmoid-weighted advantage * CE, but here we simplify.
#         # We'll use a standard KL penalty with coefficient beta.
#         per_token_kl = ref_token_logps_k - curr_token_logps   # (B, L)
# 
#         # Policy gradient loss: - advantage * log_prob(action)
#         # Only apply on masked (generated) positions.
#         per_token_pg_loss = -adv_k * curr_token_logps   # (B, L)
# 
#         # Combined loss: PG + beta * KL
#         beta_kl = 0.04  # tune as needed
#         per_token_loss = per_token_pg_loss + beta_kl * per_token_kl
# 
#         # Mask out non-generated positions (only compute loss where we actually generated)
#         completion_mask = is_masked_k.float()  # (B, L)
#         # Also mask out padding or answer_mask if needed
#         if answer_mask is not None:
#             completion_mask = completion_mask * answer_mask.float()
# 
#         # Average over tokens per sample, then mean over batch
#         sample_loss = (per_token_loss * completion_mask).sum(dim=1) / (completion_mask.sum(dim=1) + 1e-8)
#         loss = sample_loss.mean()
# 
#         # Accumulate loss (we'll backward once after the loop)
#         total_loss = total_loss + loss
# 
#     # Single backward and step
#     total_loss.backward()
#     torch.nn.utils.clip_grad_norm_(denoiser.parameters(), grad_clip)
#     optimizer.step()
# 
#     metrics = {
#         'mean_reward': rewards.mean().item(),
#         'max_reward': rewards.max().item(),
#         'min_reward': rewards.min().item(),
#         'std_reward': rewards.std().item(),
#         'frac_correct': (rewards > 0.5).float().mean().item(),
#     }
# 
#     memory_out = None
# 
#     return total_loss.item() / K, memory_out, metrics

# 
# 
# 
# 
# 
#     # --- 4. GRPO step
#     denoiser.train()
#     total_loss = 0.0
#     n_steps = 0
# 
#     loss_fn = MDLMLoss(schedule, mask_id, t_min=t_min)
# 
#     losses = 0.0
#     for k in range(K):
#         optimizer.zero_grad()
# 
#         candidate_k = generated[k::K].to(device)  # (B, L)
#         refs = ref_logits[k::K].to(device)
# 
#         xt_k, t_k, is_masked_k = loss_fn.perturb(candidate_k, answer_mask=answer_mask)
#         _, logits_k, _, _ = denoiser(None, xt_k, t_k, memory=None)
# 
#         per_token_logps = []
#         for logits_row, input_ids_row in zip(logits_k, candidate_k):
#             log_probs = logits_row.log_softmax(dim=-1)
#             token_log_prob = torch.gather(log_probs,
#                                           dim=1,
#                                           index=input_ids_row.unsqueeze(1)).squeeze(1)
#             per_token_logps.append(token_log_prob)
#         per_token_logps = torch.stack(per_token_logps)
# 
#         with torch.no_grad():
#             ref_per_token_logps = []
#             for logits_row, input_ids_row in zip(refs, candidate_k):
#                 log_probs = logits_row.log_softmax(dim=-1)
#                 token_log_prob = torch.gather(log_probs,
#                                               dim=1,
#                                               index=input_ids_row.unsqueeze(1)).squeeze(1)
#                 ref_per_token_logps.append(token_log_prob)
#             ref_per_token_logps = torch.stack(ref_per_token_logps)
# 
#         per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
#         completion_mask = is_masked_k.int()
# 
# #    if 'gen_logps' in batch:
# #        ratio = torch.exp(per_token_logps - batch['gen_logps'].to(engine.device))
# #        clipped_ratio = torch.clamp(ratio, 1-clip_param, 1+clip_param)
# #        per_token_loss = torch.min(ratio * advantages, clipped_ratio * advantages)
# #    else:
#         per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages[k::K].unsqueeze(1)
# 
#         beta = 0.04
#         per_token_loss = -(per_token_loss - beta * per_token_kl)
# 
#         loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
#         total_loss += loss.item()
#         n_steps += 1
# 
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 0.01)
#         optimizer.step()

#    # --- 4. Pre-cache reference losses and perturbations ---
#    loss_fn = MDLMLoss(schedule, mask_id, t_min=t_min)
#    # advantages reshaped to (B, K) for per-prompt, per-candidate access
#    adv_bk = advantages.view(B, K)
#    ref_cache = []
#
#    with torch.no_grad():
#        for ep in range(epochs):
#            ep_data = []
#            for k in range(K):
#                candidate_k = generated[k::K].to(device)  # (B, L)
#
#                xt_k, t_k, is_masked_k = loss_fn.perturb(
#                    candidate_k, answer_mask=answer_mask)
#
#                # Keep visible/query positions clean
#                xt_k = torch.where(visible, prompt_batch, xt_k)
#
#                _, logits_k, _, _ = ref_denoiser(None, xt_k, t_k, memory)
#                ref_loss_k = loss_fn.per_sample(
#                    logits_k, candidate_k, is_masked_k)  # (B,)
#
#                ep_data.append({
#                    'xt': xt_k.clone(),
#                    't': t_k.clone(),
#                    'is_masked': is_masked_k.clone(),
#                    'candidate': candidate_k,
#                    'ref_loss': ref_loss_k.clone(),       # (B,) per-sample
#                })
#            ref_cache.append(ep_data)
#
#    # --- 5. DGPO optimization ---
#    denoiser.train()
#    total_loss = 0.0
#    n_steps = 0
#
#    for ep in range(epochs):
#        optimizer.zero_grad()
#
#        # Forward all K candidates, get per-sample losses (B,) each
#        losses_k = []
#        for k in range(K):
#            c = ref_cache[ep][k]
#            _, logits_k, _, _ = denoiser(None, c['xt'], c['t'], memory)
#            loss_k = loss_fn.per_sample(
#                logits_k, c['candidate'], c['is_masked'])  # (B,)
#            losses_k.append(loss_k)
#
#        # DGPO loss (Eq. 17): -log σ(-β * Σ_k A_k * (L_θ - L_ref) / K)
#        # Gradient flows through (L_θ - L_ref); the -log σ provides
#        # implicit KL regularization against the reference model.
#        inner = torch.zeros(B, device=device)
#        for k in range(K):
#            ref_loss_k = ref_cache[ep][k]['ref_loss']       # (B,) constant
#            delta_k = losses_k[k] - ref_loss_k              # (B,) has grad
#            inner = inner + adv_bk[:, k] * delta_k / K
#
#        # PPO-style clipping: detach per-sample when drift is too large
##        if clip_range > 0:
##            with torch.no_grad():
##                should_clip = inner.abs() > clip_range
##            inner = torch.where(should_clip, inner.detach(), inner)
#
#        loss = -torch.log(torch.sigmoid(-beta_dgpo * inner) + 1e-8).sum()
#
#        loss.backward()
#        total_loss += loss.item()
#        n_steps += 1
#
#        torch.nn.utils.clip_grad_norm_(denoiser.parameters(), grad_clip)
#        optimizer.step()
#
#    # Detach memory for return
#    #memory_out = memory.detach() if memory is not None else None
    memory_out = None

    metrics = {
        'mean_reward': rewards.mean().item(),
        'max_reward': rewards.max().item(),
        'min_reward': rewards.min().item(),
        'std_reward': rewards.std().item(),
        'frac_correct': (rewards > 0.5).float().mean().item(),
    }

    return total_loss / max(n_steps, 1), memory_out, metrics

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
    K=12,
    sampling_steps=50,
    grad_clip=1.0,
    epochs=3,
    clip_epsilon=0.2,
    kl_coef=0.01,
    t_min=1e-4,
    lambda_temp=5.0,
    train_steps=5,       # number of trajectory steps to train on
    T=1.0,
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
        memory = denoiser.pioneer(prompt_batch, memory=memory)
        ref_memory = ref_denoiser.pioneer(prompt_batch, memory=memory)

    prompt_expanded = prompt_batch.repeat_interleave(K, dim=0)
    visible_expanded = prompt_expanded != mask_id
    memory_expanded = memory.repeat_interleave(K, dim=0)
    ref_memory_expanded = ref_memory.repeat_interleave(K, dim=0)

    sampler = Sampler(schedule, mask_id, denoiser.cfg.vocab_size)
    xt, stepper = sampler(B * K, L, device, sampling_steps)
    xt = torch.where(visible_expanded, prompt_expanded, xt)

    trajectory = []

    with torch.no_grad():
        for s in stepper:
            is_masked = (xt == mask_id)

            logits = denoiser(xt, s.t, memory_expanded)
            x0 = s.propose_x0(xt, logits/T)

            old_logp = F.log_softmax(logits/T, dim=-1)
            old_logp = torch.gather(old_logp, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)

            ref_logits = ref_denoiser(xt, s.t, ref_memory_expanded)
            ref_logp = F.log_softmax(ref_logits/T, dim=-1)
            ref_logp = torch.gather(ref_logp, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)

            trajectory.append({
                "xt": xt.clone(),
                "x0": x0.clone(),
                "t":  s.t.clone(),
                "old_logp": old_logp.clone(),
                "ref_logp": ref_logp.clone(),
                "is_masked": is_masked.clone(),
                "memory": memory_expanded,
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
    debug()

    # --- 3. RKL policy gradient ---
    full_loss = 0.0
    for epoch in range(epochs):
        total_loss = torch.tensor(0.0, device=device)

        for step_data in reversed(trajectory):
            optimizer.zero_grad()
            xt_s       = step_data["xt"]
            x0_s       = step_data["x0"]
            t_s        = step_data["t"]
            old_logp_s = step_data["old_logp"]
            ref_logp_s = step_data["ref_logp"]
            is_masked  = step_data["is_masked"]

            if not is_masked.any():
                continue

            logits_s = denoiser(xt_s, t_s, memory=step_data["memory"])
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

            # Don't divide by number of steps — keep gradient strong
            step_loss.backward()
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), grad_clip)
            optimizer.step()
        full_loss += total_loss.item()

    metrics = {
        'mean_reward': rewards.mean().item(),
        'frac_correct': (rewards > -0.5).float().mean().item(),
    }
    return full_loss / max(len(trajectory), 1), memory, metrics
