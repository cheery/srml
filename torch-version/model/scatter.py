import torch

def scatter(
    indices: torch.Tensor,  # (B, L) int
    x:       torch.Tensor,  # (B, L, vocab) float
    sigma:   torch.Tensor,  # (B,) float
) -> torch.Tensor:

    esigm1_log = torch.where(sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1).log().to(x.dtype)[:, None, None]
    x = x - esigm1_log - torch.log(
        torch.tensor(x.shape[-1] - 1, dtype=x.dtype, device=x.device)
    )
    x = torch.scatter(x, -1, indices[..., None], torch.zeros_like(x[..., :1]))

#    esigm1 = torch.where(sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1.0)
#    esigm1 = esigm1.clamp(min=1e-6)
#    esigm1_log = esigm1.log().to(x.dtype)[:, None, None]
#    x = x - esigm1_log - torch.log(
#        torch.tensor(x.shape[-1] - 1, dtype=x.dtype, device=x.device)
#    )
#    x = x.clone()
#    B, L = indices.shape
#    b_idx = torch.arange(B, device=x.device)[:, None].expand(B, L)
#    l_idx = torch.arange(L, device=x.device)[None, :].expand(B, L)
#    x[b_idx, l_idx, indices] = 0.0
    return x
