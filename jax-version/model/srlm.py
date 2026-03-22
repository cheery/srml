from dataclasses import dataclass
import jax
import jax.numpy as jnp
import haiku as hk
import math
from .s5 import S5Dual

@dataclass
class SRLMConfig:
    vocab_size : int
    d_model : int
    d_state : int
    n_priors : int
    n_posteriors : int
    d_frequency_embedding : int = 256
    N : int = 2
    T : int = 4

class SRLM(hk.Module):
    def __init__(self, cfg, name="srlm"):
        super().__init__(name=name)
        self.cfg = cfg

        self.input = InputLayer(cfg, name="input")
        self.prior = S5Stack(cfg, cfg.n_priors, use_attention=False, name="prior")
        self.main = HRM(cfg, name="main")
        self.posterior = S5Stack(cfg, cfg.n_posteriors, use_attention=False, name="posterior")
        self.norm = AdaLN(cfg.d_model)
        self.output = OutputLayer(cfg, name="output")

    def __call__(self, z, x, sigma, is_training=False):
        cfg = self.cfg
        q, t = self.input(x, sigma)
        y = self.prior(q, t)
        z, y = self.main(z, y, t)
        y = self.posterior(y, t) + q
        y = self.norm(y, t)
        return z, self.output(x, y, sigma)

class S5Stack(hk.Module):
    def __init__(self, cfg, n_layers=1, use_attention=False, name=None):
        super().__init__(name=name)
        self.n_layers = n_layers
        self.layers = [S5Layer(cfg, name=f"layer_{i}")
                       for i in range(n_layers)]

    def __call__(self, y, t, is_training=False):
        for layer in self.layers:
            if is_training:
                y = hk.dropout(hk.next_rng_key(), 0.1, layer(y, t)) + y
            else:
                y = layer(y, t) + y
        return y

class S5Layer(hk.Module):
    def __init__(self, cfg, name=None):
        super().__init__(name=name)
        self.norm1 = AdaLN(cfg.d_model)
        self.s5d = S5Dual(cfg.d_model, cfg.d_state, name="s5d")
        self.norm2 = AdaLN(cfg.d_model)
        self.ff = FeedForward(cfg.d_model)

    def __call__(self, y, t):
        @hk.remat
        def _fwd(y):
            y_norm = self.norm1(y, t)
            s = jax.vmap(self.s5d)(y_norm)
            s_norm = self.norm2(s, t)
            return self.ff(s_norm)
        return _fwd(y)
        #y_norm = self.norm1(y, t)
        #y = jax.vmap(self.s5d)(y_norm)
        #y_norm = self.norm2(y, t)
        #y = self.ff(y_norm)
        #return y

class HRM(hk.Module):
    def __init__(self, cfg, name=None):
        super().__init__(name=name)
        self.N = cfg.N
        self.T = cfg.T
        self.fast = FastLayer(cfg, name=f"fast")
        self.slow = SlowLayer(cfg, name=f"slow")

    def __call__(self, z, x, t):
        zH, zL = z
        zL = jax.lax.stop_gradient(zL)
        zH = jax.lax.stop_gradient(zH)
        for i in range(self.N * self.T - 1):
            zL = self.fast(zH, zL, x, t)
            if (i + 1) % self.T == 0:
                zH = self.slow(zH, zL, t)
            zL = jax.lax.stop_gradient(zL)
            zH = jax.lax.stop_gradient(zH)

        zL = self.fast(zH, zL, x, t)
        zH = self.slow(zH, zL, t)
        return (zH, zL), zH

class FastLayer(hk.Module):
    def __init__(self, cfg, name=None):
        super().__init__(name=name)
        self.normL = AdaLN(cfg.d_model)
        self.normH = AdaLN(cfg.d_model)
        self.inj = hk.Linear(cfg.d_model, name=f"inj")
        self.s5d = S5Dual(cfg.d_model, cfg.d_state // 4, name=f"s5d")
        #self.proj = hk.Linear(cfg.d_model)

    def __call__(self, zH, zL, x, t):
        zH = self.normH(zH, t)
        zL = self.normL(zL, t)
        x = jnp.concatenate([zH, zL, x], axis=-1)
        x = self.inj(x)
        x = jax.vmap(self.s5d)(x)
        #x = self.proj(x)
        return x

class SlowLayer(hk.Module):
    def __init__(self, cfg, name=None):
        super().__init__(name=name)
        self.normL = AdaLN(cfg.d_model)
        self.normH = AdaLN(cfg.d_model)
        self.s5d = S5Dual(cfg.d_model*2, cfg.d_state, name="s5d")
        self.proj = hk.Linear(cfg.d_model, name=f"proj")

    def __call__(self, zH, zL, t):
        zH = self.normH(zH, t)
        zL = self.normL(zL, t)
        x = jnp.concatenate([zH, zL], axis=-1)
        x = jax.vmap(self.s5d)(x)
        x = self.proj(x)
        return x

class InputLayer(hk.Module):
    def __init__(self, cfg, name=None):
        super().__init__(name=name)
        self.input_emb = hk.Embed(cfg.vocab_size, cfg.d_model, name=f"input_emb")
        self.timestep_emb = TimestepEmbedder(cfg.d_model, cfg.d_frequency_embedding, name=f"timestep_emb")

    def __call__(self, x, sigma, is_training=False):
        y = self.input_emb(x)
        if is_training:
            y = hk.dropout(hk.next_rng_key(), 0.1, y)
        sigma_emb = jax.nn.silu(self.timestep_emb(sigma))
        return y, sigma_emb

class OutputLayer(hk.Module):
    def __init__(self, cfg, name=None):
        super().__init__(name=name)
        self.ff = FeedForward(cfg.d_model, name=f"ff")
        self.proj = hk.Linear(cfg.vocab_size, with_bias=False, name=f"proj")

    def __call__(self, x, y, sigma, is_training=False):
        y = self.ff(y)
        if is_training:
            y = hk.dropout(hk.next_rng_key(), 0.1, y)
        y = self.proj(y)
        return scatter(x, y, sigma)

class FeedForward(hk.Module):
    def __init__(self, dim, name=None):
        super().__init__(name=name)
        self.dim = dim
        self.pre = hk.Linear(4 * dim, name=f"pre")
        self.pos = hk.Linear(dim, w_init=jnp.zeros, b_init=jnp.zeros, name=f"pos")

    def __call__(self, x):
        x = self.pre(x)
        x = jax.nn.gelu(x) # or relu
        x = self.pos(x)
        return x

class AdaLN(hk.Module):
    def __init__(self, d_model, name=None):
        super().__init__(name=name)
        self.norm = hk.LayerNorm(axis=-1, create_scale=False, create_offset=False)
        self.proj = hk.Linear(2 * d_model)

    def __call__(self, x, cond):
        # cond: (B, d_cond) — e.g. your sigma_emb (or t)
        scale, shift = jnp.split(self.proj(cond), 2, axis=-1)
        return self.norm(x) * (1 + scale[:, None, :]) + shift[:, None, :]

class TimestepEmbedder(hk.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256, name=None):
        super().__init__(name=name)
        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size

    def __call__(self, t):
        t_freq = timestep_embedding(t, self.frequency_embedding_size)
        x = hk.Linear(self.hidden_size, name="pre")(t_freq)
        x = jax.nn.silu(x)
        x = hk.Linear(self.hidden_size, name="pos")(x)
        return x

def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = jnp.exp(
        -math.log(max_period) * jnp.arange(half, dtype=jnp.float32) / half
    )
    args = t[:, None].astype(jnp.float32) * freqs[None]
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    if dim % 2:
        embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
    return embedding

def mk_z(b, l, d_model):
    zH = jnp.zeros((b, l, d_model), dtype=jnp.float32)
    zL = jnp.zeros((b, l, d_model), dtype=jnp.float32)
    return zH, zL

def scatter(indices, x, sigma):
    eps = 1e-3
    esigm1 = jnp.where(sigma < 0.5, jnp.expm1(sigma), jnp.exp(sigma) - 1)
    esigm1 = jnp.clip(esigm1, a_min=1e-6, a_max=None)
    esigm1_log = jnp.log(esigm1).astype(x.dtype)[:, None, None]
    x = x - esigm1_log - jnp.log(x.shape[-1] - 1)
    
    B, L = indices.shape
    x = x.at[jnp.arange(B)[:, None], jnp.arange(L)[None, :], indices].set(0.0)
    return x
