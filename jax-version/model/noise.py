import abc
import jax.numpy as jnp

class Noise(abc.ABC):
    """Abstract noise schedule.

    forward(t) returns (total_noise, rate_noise) — mirrors the PyTorch version
    but as a plain method call since hk.Module uses __call__.
    """

    def __call__(self, t):
        return self.total_noise(t), self.rate_noise(t)

    @abc.abstractmethod
    def rate_noise(self, t):
        """Rate of change of noise: g(t)."""
        pass

    @abc.abstractmethod
    def total_noise(self, t):
        """Total noise: integral of g from 0 to t."""
        pass


class LogLinearNoise(Noise):
    """Log-linear noise schedule for absorbing diffusion.

    total_noise(t) = -log(1 - (1 - eps) * t)
    rate_noise(t)  = (1 - eps) / (1 - (1 - eps) * t)

    At t=0: total=0, move_chance=0.
    At t=1: total -> inf, move_chance -> ~1.

    Note: the PyTorch version had a dummy nn.Parameter('empty') to ensure the
    module appeared in the parameter dict. In Haiku this is unnecessary — the
    module is stateless and needs no parameters.
    """

    def __init__(self, eps: float = 1e-3):
        self.eps = eps

    def rate_noise(self, t):
        return (1 - self.eps) / (1 - (1 - self.eps) * t)

    def total_noise(self, t):
        return -jnp.log1p(-(1 - self.eps) * t)
