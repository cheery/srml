import abc
import torch
import torch.nn.functional as F
from .catsample import sample_categorical


def unsqueeze_as(x: torch.Tensor, y: torch.Tensor, back: bool = True) -> torch.Tensor:
    """Reshape x to broadcast against y by appending or prepending unit dims."""
    if back:
        return x.reshape(*x.shape, *((1,) * (y.ndim - x.ndim)))
    else:
        return x.reshape(*((1,) * (y.ndim - x.ndim)), *x.shape)


class Graph(abc.ABC):
    @property
    def dim(self) -> int:
        pass

    @property
    def absorb(self) -> bool:
        pass

    @abc.abstractmethod
    def rate(self, i: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def transp_rate(self, i: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def transition(self, i: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        pass

    def sample_transition(self, i: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        transition_vector = self.transition(i, sigma)
        return sample_categorical(transition_vector)

    def reverse_rate(self, i: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
        normalized_rate = self.transp_rate(i) * score
        # Zero diagonal, set to -row_sum
        # Use gather/scatter for correct multi-dim indexing
        diag_idx = i.unsqueeze(-1)  # (..., 1)
        diag_vals = torch.gather(normalized_rate, -1, diag_idx)
        normalized_rate.scatter_(-1, diag_idx, torch.zeros_like(diag_vals))
        row_sum = normalized_rate.sum(dim=-1, keepdim=True)
        normalized_rate.scatter_(-1, diag_idx, -row_sum)
        return normalized_rate

    def sample_rate(self, i: torch.Tensor, rate: torch.Tensor) -> torch.Tensor:
        one_hot_i = F.one_hot(i, self.dim).float()
        return sample_categorical(one_hot_i + rate)

    @abc.abstractmethod
    def staggered_score(self, score: torch.Tensor, dsigma: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def sample_limit(self, *batch_dims, device=None) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def score_entropy(self, score: torch.Tensor, sigma: torch.Tensor,
                      x: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        pass


class AbsorbingGraph(Graph):
    """Absorbing-state discrete diffusion. Last token index is the mask state."""

    def __init__(self, vocab_size: int):
        self._dim = vocab_size + 1

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def absorb(self) -> bool:
        return True

    def rate(self, i: torch.Tensor) -> torch.Tensor:
        mask = torch.full_like(i, self.dim - 1)
        return (F.one_hot(mask, self.dim).float()
                - F.one_hot(i, self.dim).float())

    def transp_rate(self, i: torch.Tensor) -> torch.Tensor:
        edge = -F.one_hot(i, self.dim).float()
        is_mask = (i == self.dim - 1)
        edge[is_mask] = edge[is_mask] + 1.0
        return edge

    def transition(self, i: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def transp_transition(self, i: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        sigma = unsqueeze_as(sigma, i.unsqueeze(-1))
        edge = torch.exp(-sigma) * F.one_hot(i, self.dim).float()
        stay_mask = torch.where(
            i == self.dim - 1,
            1 - torch.exp(-sigma[..., 0]),
            torch.zeros_like(sigma[..., 0]),
        )
        edge = edge + stay_mask.unsqueeze(-1)
        return edge

    def sample_transition(self, i: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        move_chance = 1 - torch.exp(-sigma)
        move = torch.rand(*i.shape, device=i.device) < move_chance
        return torch.where(move, torch.full_like(i, self.dim - 1), i)

    def staggered_score(self, score: torch.Tensor, dsigma: torch.Tensor) -> torch.Tensor:
        extra_const = (1 - torch.exp(dsigma))[:, None] * score.sum(dim=-1)
        score = score * torch.exp(dsigma)[:, None, None]
        score[..., -1] = score[..., -1] + extra_const
        return score

    def sample_limit(self, *batch_dims, device=None) -> torch.Tensor:
        return torch.full(batch_dims, self.dim - 1, dtype=torch.long, device=device)

    def score_entropy(self, score: torch.Tensor, sigma: torch.Tensor,
                      x: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        rel_ind = (x == self.dim - 1)

        esigm1 = torch.where(
            sigma < 0.5,
            torch.expm1(sigma),
            sigma.exp() - 1,
        )

        ratio = 1.0 / torch.where(rel_ind, esigm1, torch.ones_like(esigm1))
        neg_term = ratio * score.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
        pos_term = score[..., :-1].exp().sum(dim=-1)
        const = ratio * (torch.log(ratio.clamp(min=1e-9)) - 1)

        entropy = torch.where(rel_ind, pos_term - neg_term + const,
                              torch.zeros_like(pos_term))
        return entropy

    def score_entropy(self, score, sigma, x, x0):
        # This is where the sample has been absorbed / perturbed
        # [[ True, False,  True,  True,  True, False, False,  True]]
        rel_ind = x == self.dim - 1
        # print("rel_ind", rel_ind.shape, rel_ind)
        
        esigm1 = torch.where(
            sigma < 0.5,
            torch.expm1(sigma),
            torch.exp(sigma) - 1
        )
        # print("esigm1", esigm1.shape, esigm1)

        ratio = 1 / esigm1.expand_as(x)[rel_ind]
        # print("ratio", ratio.shape, ratio)
        
        other_ind = x0[rel_ind]
        # print("other_ind", other_ind.shape, other_ind)

        # negative_term
        neg_term = ratio * torch.gather(score[rel_ind], -1, other_ind[..., None]).squeeze(-1)
        # print("neg_term", neg_term.shape, neg_term)

        #positive term
        pos_term = score[rel_ind][:, :-1].exp().sum(dim=-1)
        # print("pos_term", pos_term.shape, pos_term)

        # constant term
        const = ratio * (ratio.log() - 1)
        # print("const", const.shape, const)

        entropy = torch.zeros(*x.shape, device=x.device)
        entropy[rel_ind] += pos_term - neg_term + const
        return entropy
