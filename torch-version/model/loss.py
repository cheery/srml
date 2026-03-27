import torch
import torch.nn.functional as F


def loss_function(model, graph, noise):
    """Returns a loss function for SEDD with HRM.

    The returned function computes the score entropy loss over a batch,
    threading z (HRM state) through as auxiliary output.

    Args:
        model:  nn.Module with signature forward(z, x, sigma) -> (z, log_score)
        graph:  Graph instance
        noise:  Noise instance
    Returns:
        sedd_hrm_loss(z, batch, t=None, perturbed_batch=None) -> (loss, z)
    """
    def sedd_hrm_loss(z, batch, t=None, perturbed_batch=None, memories=None):
        sampling_eps = 1e-3
        device = batch.device
        MASK_TOKEN = graph.dim - 1

        if t is None:
            t = ((1 - sampling_eps)
                 * torch.rand(batch.shape[0], device=device)
                 + sampling_eps)

        sigma, dsigma = noise(t)

        if perturbed_batch is None:
            perturbed_batch = graph.sample_transition(batch, sigma[:, None])
        else:
            # Pre-masked batch (arithmetic, sudoku): apply diffusion noise
            # consistent with sigma, but protect visible (non-mask) positions
            visible = (perturbed_batch != MASK_TOKEN)
            noised = graph.sample_transition(batch, sigma[:, None])
            perturbed_batch = torch.where(visible, perturbed_batch, noised)

        z, log_score, aux_loss = model(z, perturbed_batch, sigma, memories=memories)
        loss = graph.score_entropy(log_score, sigma[:, None], perturbed_batch, batch)
        loss = (dsigma[:, None] * loss).sum(dim=-1)

        return loss.mean() + aux_loss, tuple(zi.detach() for zi in z)

    return sedd_hrm_loss


def deep_supervision_step(model, optimizer, graph, noise, z, batch,
                          n_supervision=4, perturbed_batch=None, memories=None,
                          grad_clip=0.1, ema=None):
    """TRM-style deep supervision for SEDD.

    Calls model.front() + model.step() n_supervision times, with an
    optimizer update after each step. Q_head learns to predict whether
    the output is correct (halting signal for adaptive computation).
    EMA shadow weights are updated after each optimizer step for stability.

    Matches TRM pseudocode:
        for step in range(N_supervision):
            x = input_embedding(x_input)
            (y, z), y_hat, q_hat = deep_recursion(x, y, z)
            loss = cross_entropy(y_hat, y_true) + BCE(q_hat, y_hat == y_true)
            loss.backward(); opt.step(); opt.zero_grad()
            if q_hat > 0: break

    Args:
        model: SRLM with front()/step() API
        optimizer: optimizer to step
        graph: Graph instance
        noise: Noise instance
        z: HRM state tuple
        batch: (B, L) clean token IDs
        n_supervision: number of deep supervision steps
        perturbed_batch: optional pre-masked batch
        memories: optional memory tensors
        grad_clip: gradient clipping norm
        ema: optional EMA instance — updated after each optimizer step
    Returns:
        avg_loss: average SEDD loss across supervision steps
        z: final HRM state (detached)
    """
    sampling_eps = 1e-3
    device = batch.device
    MASK_TOKEN = graph.dim - 1

    # Sample noise once — same input across all supervision steps.
    # Deep supervision is iterative refinement on the same problem,
    # matching TRM and the front()+step() pattern used everywhere else.
    t = (1 - sampling_eps) * torch.rand(batch.shape[0], device=device) + sampling_eps
    sigma, dsigma = noise(t)

    if perturbed_batch is None:
        perturbed_batch = graph.sample_transition(batch, sigma[:, None])
    else:
        visible = (perturbed_batch != MASK_TOKEN)
        noised = graph.sample_transition(batch, sigma[:, None])
        perturbed_batch = torch.where(visible, perturbed_batch, noised)

    total_loss = 0.0
    n_steps = 0

    for step_i in range(n_supervision):
        optimizer.zero_grad()

        # Recompute front() each step — model weights change after optimizer.step()
        ix = model.front(perturbed_batch, sigma, memories=memories)
        z, log_score, q, aux_loss = model.step(z, ix)

        # SEDD loss
        sedd_loss = graph.score_entropy(log_score, sigma[:, None], perturbed_batch, batch)
        sedd_loss = (dsigma[:, None] * sedd_loss).sum(dim=-1).mean()

        # Q_head BCE loss: predict whether output matches ground truth
        with torch.no_grad():
            preds = log_score.argmax(dim=-1)                   # (B, L)
            accuracy = (preds == batch).float().mean(dim=-1)   # (B,)
            q_target = (accuracy > 0.5).float()
        q_loss = F.binary_cross_entropy_with_logits(q.squeeze(-1), q_target)

        loss = sedd_loss + q_loss + aux_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # EMA: smooth shadow weights toward current weights
        if ema is not None:
            ema.update(model)

        total_loss += sedd_loss.item()
        n_steps += 1

        # Early stopping: Q > 0 means "output is good enough, stop recursing"
        with torch.no_grad():
            if n_supervision > 1 and (q.squeeze(-1) > 0).all():
                break

    return total_loss / n_steps, tuple(zi.detach() for zi in z)
