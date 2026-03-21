import jax
import jax.numpy as jnp

def gumbel_softmax(key, categorical_probs, hard=False, eps=1e-9):
    """Gumbel-softmax relaxation of a categorical distribution.

    Args:
        key: JAX PRNG key
        categorical_probs: (..., vocab) float32 probability vectors
        hard: if True, returns one-hot (straight-through estimator)
        eps: numerical stability floor
    Returns:
        (..., vocab) float32
    """
    logits = jnp.log(jnp.clip(categorical_probs, a_min=eps))
    gumbel_noise = -jnp.log(-jnp.log(jax.random.uniform(key, logits.shape) + eps) + eps)
    y = jax.nn.softmax(logits + gumbel_noise)
    if hard:
        y_hard = jax.nn.one_hot(jnp.argmax(y, axis=-1), y.shape[-1])
        # Straight-through: forward is one-hot, backward is soft
        y = jax.lax.stop_gradient(y_hard - y) + y
    return y


def sample_categorical(key, categorical_probs, method="hard"):
    """Sample indices from a categorical distribution.

    Args:
        key: JAX PRNG key
        categorical_probs: (..., vocab) float32 probability vectors
        method: only "hard" (Gumbel-argmax) is supported
    Returns:
        (...,) int32 sampled indices
    """
    if method == "hard":
        gumbel_noise = -jnp.log(
            -jnp.log(jax.random.uniform(key, categorical_probs.shape) + 1e-10) + 1e-10
        )
        return jnp.argmax(categorical_probs / jnp.exp(-gumbel_noise), axis=-1)
    else:
        raise ValueError(f"Method '{method}' for sampling categorical variables is not valid.")
