"""
Sampler for discrete diffusion models using JAX.

Key differences from the PyTorch version:
- All random operations take an explicit JAX PRNG key.
- No mutable state: keys are threaded through via jax.random.split.
- score_fn signature: score_fn(params, x, sigma) -> logits, managed outside haiku.
- graph and noise are plain Python objects (no nn.Module needed).
"""

import abc
import jax
import jax.numpy as jnp
from .catsample import sample_categorical

class EulerPredictor:
    """Predictor (reverse diffusion step)."""

    def __init__(self, graph, noise):
        self.graph = graph
        self.noise = noise

    def update_fn(self, key, score_fn, params, arg, x, t, step_size):
        """One predictor step.

        Args:
            key: JAX PRNG key
            score_fn: fn(params, x, sigma) -> score logits
            params: model parameters (haiku FlatMap or pytree)
            x: current token sequence (..., L) int32
            t: current timestep scalar or (..., 1) float32
            step_size: dt float
        Returns:
            x: updated token sequence, same shape
        """
        sigma, dsigma = self.noise(t)
        key, score_key = jax.random.split(key)
        arg, log_score = score_fn(params, score_key, arg, x, sigma)
        return arg, self.predict(key, x, log_score, dsigma)

    def predict(self, key, x, log_score, dsigma, step_size):
        score = jnp.exp(log_score)
        rev_rate = step_size * dsigma[..., None, None] * self.graph.reverse_rate(x, score)
        x = self.graph.sample_rate(key, x, rev_rate)
        return x

    def _tree_flatten(self):
        # first group hashables, second group not hashable.
        return (self.graph, self.noise,), ()

    @classmethod
    def _tree_unflatten(cls, aux, children):
      return cls(*children)

jax.tree_util.register_pytree_node(EulerPredictor,
                                   EulerPredictor._tree_flatten,
                                   EulerPredictor._tree_unflatten)

class Denoiser:
    """Final denoising step at the end of sampling."""

    def __init__(self, graph, noise):
        self.graph = graph
        self.noise = noise

    def update_fn(self, key, score_fn, params, arg, x, t):
        sigma = self.noise(t)[0]
        key, score_key = jax.random.split(key)
        arg, log_score = score_fn(params, score_key, arg, x, sigma)
        return arg, self.denoise(key, x, log_score, sigma)

    def denoise(self, key, x, log_score, sigma):
        score = jnp.exp(log_score)
        stag_score = self.graph.staggered_score(score, sigma)
        probs = stag_score * self.graph.transp_transition(x, sigma)
        if self.graph.absorb:
            probs = probs[..., :-1]
        return sample_categorical(key, probs)

    def _tree_flatten(self):
        # first group hashables, second group not hashable.
        return (self.graph, self.noise,), ()

    @classmethod
    def _tree_unflatten(cls, aux, children):
      return cls(*children)

jax.tree_util.register_pytree_node(Denoiser,
                                   Denoiser._tree_flatten,
                                   Denoiser._tree_unflatten)

class Sampler:
    """Runs the full reverse diffusion sampling loop."""
    def __init__(self, graph, noise):
        self.graph = graph
        self.noise = noise
        self.predictor = EulerPredictor(graph, noise)
        self.denoiser = Denoiser(graph, noise)

    def begin(self, key, batch_size, batch_len, steps=10, eps=1e-5):
        x = self.graph.sample_limit2(key, batch_size, batch_len)
        timesteps = jnp.linspace(1.0, eps, steps + 1)
        dt = (1 - eps) / steps
        return x, timesteps, dt, steps

    def sample2(self, score_fn, projector, batch_size, batch_len):
        def _impl_(key, params, z, q, steps=10, eps=1e-5):
            x, timesteps, dt, steps = self.begin(key, batch_size, batch_len, steps, eps)
            for i in range(steps):
                t = timesteps[i] * jnp.ones((x.shape[0],))
                x = projector(x, q)
                sigma, dsigma = self.noise(t)
                key, score_key = jax.random.split(key)
                z, log_score = score_fn(params, score_key, z, x, sigma)
                key, predict_key = jax.random.split(key)
                x = self.predictor.predict(predict_key, x, log_score, dsigma, dt)

            x = projector(x, q)
            t = timesteps[-1] * jnp.ones((x.shape[0],))
            sigma, _ = self.noise(t)
            key, score_key = jax.random.split(key)
            z, log_score = score_fn(params, score_key, z, x, sigma)
            return z, self.denoiser.denoise(key, x, log_score, sigma)
        return _impl_

    def sample(
        self,
        key,
        tokenizer,
        score_fn,
        params,
        arg,
        batch_size=1,
        batch_len=32,
        steps=1024,
        eps=1e-5,
        denoise=True,
        projector=lambda x: x,
        show_intermediate=False,
    ):
        """Run the full sampling loop.

        Args:
            key: JAX PRNG key
            tokenizer: callable(x: int32 array) -> list[str]
            score_fn: fn(params, x, sigma) -> score logits  (B, L, vocab)
            params: model parameters pytree
            graph: Graph instance
            noise: Noise instance (hk.transform applied externally if needed)
            batch_size: number of sequences to generate
            steps: number of Euler steps
            eps: small t offset so we don't reach t=0 exactly
            denoise: whether to apply final denoising step
            projector: optional fn to project x between steps (e.g. for constrained decoding)
            show_intermediate: print decoded text at each step
        Returns:
            list of decoded strings
        """
        try:
            key, limit_key = jax.random.split(key)
            x = graph.sample_limit2(limit_key, batch_size, bacth_len)
            timesteps = jnp.linspace(1.0, eps, steps + 1)
            dt = (1 - eps) / steps

            #print(f"Sampling with {steps} steps")
            for i in range(steps):
                t = timesteps[i] * jnp.ones((x.shape[0],))
                x = projector(x)

                key, step_key = jax.random.split(key)
                arg, x = self.predictor.update_fn(step_key, score_fn, params, arg, x, t, dt)

                if show_intermediate:
                    print(f"{i} @ {timesteps[i].item():.5f}:")
                    sentences = [tokenizer(i) for i in x]
                    print(repr(sentences))

        except KeyboardInterrupt:
            pass
        x = projector(x)
        t = timesteps[-1] * jnp.ones((x.shape[0],))
        key, denoise_key = jax.random.split(key)
        arg, x = self.denoiser.update_fn(denoise_key, score_fn, params, arg, x, t)
        if show_intermediate:
            sentences = [tokenizer(i) for i in x]
            print("Denoised:")
            print(repr(sentences))

        return arg, [tokenizer(i) for i in x]

    def _tree_flatten(self):
        # first group hashables, second group not hashable.
        return (self.graph, self.noise,), ()

    @classmethod
    def _tree_unflatten(cls, aux, children):
      return cls(*children)

jax.tree_util.register_pytree_node(Sampler,
                                   Sampler._tree_flatten,
                                   Sampler._tree_unflatten)
