import abc
import torch


class Noise(abc.ABC):
    """Abstract noise schedule. forward(t) returns (total_noise, rate_noise)."""

    def __call__(self, t: torch.Tensor):
        return self.total_noise(t), self.rate_noise(t)

    @abc.abstractmethod
    def rate_noise(self, t: torch.Tensor) -> torch.Tensor:
        """Rate of change of noise: g(t)."""

    @abc.abstractmethod
    def total_noise(self, t: torch.Tensor) -> torch.Tensor:
        """Total noise: integral of g from 0 to t."""


class LogLinearNoise(Noise):
    """Log-linear noise schedule for absorbing diffusion.

    total_noise(t) = -log(1 - (1 - eps) * t)
    rate_noise(t)  = (1 - eps) / (1 - (1 - eps) * t)
    """

    def __init__(self, eps: float = 1e-3):
        self.eps = eps

    def rate_noise(self, t: torch.Tensor) -> torch.Tensor:
        return (1 - self.eps) / (1 - (1 - self.eps) * t)

    def total_noise(self, t: torch.Tensor) -> torch.Tensor:
        return -torch.log1p(-(1 - self.eps) * t)
