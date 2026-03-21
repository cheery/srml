import abc
import jax
import jax.numpy as jnp
import haiku as hk
from .catsample import sample_categorical

def unsqueeze_as(x, y, back=True):
    """Reshape x to broadcast against y by appending or prepending unit dims."""
    if back:
        return x.reshape(*x.shape, *((1,) * (y.ndim - x.ndim)))
    else:
        return x.reshape(*((1,) * (y.ndim - x.ndim)), *x.shape)

class Graph(abc.ABC):
    """Abstract base for discrete diffusion graphs.

    Note: Graph classes are pure (no learned parameters), so they are plain
    Python classes rather than hk.Modules. Instantiate them outside of
    hk.transform and pass them in as arguments.
    """

    @property
    def dim(self):
        pass

    @property
    def absorb(self):
        """Whether index (dim - 1) is an absorbing/mask state."""
        pass

    @abc.abstractmethod
    def rate(self, i):
        """i-th column of rate matrix Q. i: (...,) int."""
        pass

    @abc.abstractmethod
    def transp_rate(self, i):
        """i-th row of rate matrix Q."""
        pass

    @abc.abstractmethod
    def transition(self, i, sigma):
        """i-th column of transition matrix e^{sigma Q}."""
        pass

    def sample_transition(self, key, i, sigma):
        transition_vector = self.transition(i, sigma)
        return sample_categorical(key, transition_vector, method="hard")

    def reverse_rate(self, i, score):
        """Reverse rate: score * transp_rate, with diagonal corrected to sum to zero."""
        normalized_rate = self.transp_rate(i) * score
        # Zero the diagonal entry, then set it to -row_sum
        diag_idx = i[..., None]
        normalized_rate = normalized_rate.at[..., :].set(normalized_rate)
        # Scatter zeros at diagonal positions
        normalized_rate = normalized_rate.at[
            tuple(jnp.indices(i.shape)) + (i,)
        ].set(0.0)
        row_sum = normalized_rate.sum(axis=-1, keepdims=True)
        normalized_rate = normalized_rate.at[
            tuple(jnp.indices(i.shape)) + (i,)
        ].set(-row_sum[..., 0])
        return normalized_rate

    def sample_rate(self, key, i, rate):
        one_hot_i = jax.nn.one_hot(i, self.dim, dtype=rate.dtype)
        return sample_categorical(key, one_hot_i + rate)

    @abc.abstractmethod
    def staggered_score(self, score, dsigma):
        """Approximates p_{sigma-dsigma}(z) / p_{sigma}(x)."""
        pass

    @abc.abstractmethod
    def sample_limit(self, key, *batch_dims):
        """Sample from the limiting (fully noised) distribution."""
        pass

    @abc.abstractmethod
    def score_entropy(self, score, sigma, x, x0):
        """Score entropy loss."""
        pass


class AbsorbingGraph(Graph):
    """Absorbing-state discrete diffusion graph.

    Tokens are corrupted by being replaced with a single mask token at
    index (vocab_size), i.e. index (dim - 1).
    """

    def __init__(self, vocab_size: int):
        self._dim = vocab_size + 1  # last index is the absorbing/mask state

    @property
    def dim(self):
        return self._dim

    @property
    def absorb(self):
        return True

    def rate(self, i):
        """Rate column: transitions i -> mask, removes self-loop."""
        mask = (self.dim - 1) * jnp.ones_like(i)
        return (
            jax.nn.one_hot(mask, self.dim, dtype=jnp.float32)
            - jax.nn.one_hot(i, self.dim, dtype=jnp.float32)
        )

    def transp_rate(self, i):
        """Rate row: -1 at i, +1 at mask position if i is mask."""
        edge = -jax.nn.one_hot(i, self.dim, dtype=jnp.float32)
        is_mask = (i == self.dim - 1)
        edge = jnp.where(is_mask[..., None], edge + 1.0, edge)
        return edge

    def transition(self, i, sigma):
        # Not needed for absorbing graph (sample_transition is overridden)
        raise NotImplementedError

    def transp_transition(self, i, sigma):
        sigma = unsqueeze_as(sigma, i[..., None])
        edge = jnp.exp(-sigma) * jax.nn.one_hot(i, self.dim, dtype=jnp.float32)
        stay_mask = jnp.where(
            i == self.dim - 1,
            1 - jnp.exp(-sigma[..., 0]),
            0.0
        )
        edge = edge + stay_mask[..., None]
        return edge

    def sample_transition(self, key, i, sigma):
        """Each token moves to mask independently with prob (1 - exp(-sigma))."""
        move_chance = 1 - jnp.exp(-sigma)
        move_indices = jax.random.uniform(key, i.shape) < move_chance
        return jnp.where(move_indices, self.dim - 1, i)

    def staggered_score(self, score, dsigma):
        """Approximates ratio p_{sigma-dsigma} / p_{sigma}."""
        extra_const = (1 - jnp.exp(dsigma))[:,None] * score.sum(axis=-1)
        score = score * jnp.exp(dsigma)[:, None, None]
        score = score.at[..., -1].add(extra_const)
        return score

    def sample_limit(self, key, *batch_dims):
        """Fully absorbed: all tokens are the mask token."""
        return (self.dim - 1) * jnp.ones(batch_dims, dtype=jnp.int32)

    def score_entropy(self, score, sigma, x, x0):
        """Score entropy loss for training.

        Only positions that have been absorbed (x == mask) contribute.
        """
        rel_ind = x == self.dim - 1

        esigm1 = jnp.where(
            sigma < 0.5,
            jnp.expm1(sigma),
            jnp.exp(sigma) - 1,
        )

        ratio = 1.0 / jnp.where(rel_ind, esigm1, 1.0)  # avoid div-by-zero elsewhere
        other_ind = x0  # original token indices

        # Gather score at the original token position
        neg_term = ratio * jnp.take_along_axis(score, other_ind[..., None], axis=-1)[..., 0]

        # Sum of exp(score) over non-mask vocab
        pos_term = jnp.exp(score[..., :-1]).sum(axis=-1)

        const = ratio * (jnp.log(ratio + 1e-9) - 1)

        entropy = jnp.where(rel_ind, pos_term - neg_term + const, 0.0)
        return entropy
