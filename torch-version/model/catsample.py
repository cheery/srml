import torch

def gumbel_softmax(categorical_probs, hard=False, eps=1e-9):
    logits = categorical_probs.clamp(min=1e-9).log()
    return F.gumbel_softmax(logits, hard=hard)

def sample_categorical(categorical_probs: torch.Tensor) -> torch.Tensor:
    """Sample indices from a categorical distribution via Gumbel-argmax."""
#    eps = 1e-10
#
#    # 1. Convert probabilities to log-probabilities safely
#    logits = torch.log(categorical_probs.clamp(min=eps))
#
#    # 2. Generate Gumbel noise
#    u = torch.rand_like(categorical_probs).clamp(min=eps)
#    gumbel_noise = -torch.log(-torch.log(u))
#
#    # 3. Add noise to LOGITS, not raw probabilities
#    return torch.argmax(logits + gumbel_noise, dim=-1)

    gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
    return (categorical_probs / gumbel_norm).argmax(dim=-1)
