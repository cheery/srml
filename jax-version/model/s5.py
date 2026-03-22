import jax
import jax.numpy as jnp
from jax.lax import associative_scan
import haiku as hk

class S5Dual(hk.Module):
    def __init__(self, d_model, d_state, name=None):
        super().__init__(name=name)
        self.fwd = S5(d_model, d_state, "fwd")
        self.bwd = S5(d_model, d_state, "bwd")

    # Note that I'd like torch version to use shape (B,L,D)
    def __call__(self, x):
        x_flip = jnp.flip(x, axis=0)
        return self.fwd(x) + jnp.flip(self.bwd(x_flip), axis=0)

class S5(hk.Module):
    """Single S5 layer: discretized diagonal SSM + GELU activation.

    Args:
        d_model: input/output feature dimension H
        d_state: SSM state dimension P (number of complex poles)
        name: haiku module name
    """

    def __init__(self, d_model: int, d_state: int, name: str = None):
        super().__init__(name=name)
        self.H = d_model
        self.P = d_state

    def __call__(self, input_sequence: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            input_sequence: (L, H) float32
        Returns:
            output: (L, H) float32
        """
        H, P = self.H, self.P

        # --- Lambda: stable complex eigenvalues, Re(Lambda) < 0 always ---
        # Parameterize as -exp(log_real) + i*imag to enforce stability.
        log_real = hk.get_parameter(
            "log_real", shape=(P,), dtype=jnp.float32,
            init=hk.initializers.RandomNormal(stddev=0.5)
        )
        imag = hk.get_parameter(
            "imag", shape=(P,), dtype=jnp.float32,
            init=hk.initializers.RandomNormal(stddev=1.0)
        )
        Lambda = (-jnp.exp(log_real) + 1j * imag).astype(jnp.complex64)

        # --- B_tilde: (P, H) complex input matrix ---
        B_real = hk.get_parameter(
            "B_real", shape=(P, H), dtype=jnp.float32,
            init=hk.initializers.TruncatedNormal(stddev=1.0 / jnp.sqrt(H))
        )
        B_imag = hk.get_parameter(
            "B_imag", shape=(P, H), dtype=jnp.float32,
            init=hk.initializers.TruncatedNormal(stddev=1.0 / jnp.sqrt(H))
        )
        B_tilde = (B_real + 1j * B_imag).astype(jnp.complex64)

        # --- C_tilde: (H, P) complex output matrix ---
        C_real = hk.get_parameter(
            "C_real", shape=(H, P), dtype=jnp.float32,
            init=hk.initializers.TruncatedNormal(stddev=1.0 / jnp.sqrt(P))
        )
        C_imag = hk.get_parameter(
            "C_imag", shape=(H, P), dtype=jnp.float32,
            init=hk.initializers.TruncatedNormal(stddev=1.0 / jnp.sqrt(P))
        )
        C_tilde = (C_real + 1j * C_imag).astype(jnp.complex64)

        # --- D: skip connection, log_Delta: learned step size ---
        D = hk.get_parameter(
            "D", shape=(H,), dtype=jnp.float32,
            init=hk.initializers.Constant(1.0)
        )
        log_Delta = hk.get_parameter(
            "log_Delta", shape=(P,), dtype=jnp.float32,
            init=hk.initializers.Constant(0.0)  # Delta = exp(0) = 1
        )

        Lambda_bar, B_bar = discretize(Lambda, B_tilde, jnp.exp(log_Delta))
        preactivations = apply_ssm(Lambda_bar, B_bar, C_tilde, D, input_sequence)
        return jax.nn.gelu(preactivations)

def discretize(Lambda, B_tilde, Delta):
    """ Discretize a diagonalized, continuous-time linear SSM
    Args:
    Lambda (complex64): diagonal state matrix (P,)
    B_tilde (complex64): input matrix (P, H)
    Delta (float32): discretization step sizes (P,)
    Returns:
    discretized Lambda_bar (complex64), B_bar (complex64) (P,), (P,H)"""
    Identity = jnp.ones(Lambda.shape[0], dtype=jnp.complex64)
    Lambda_bar = jnp.exp(Lambda * Delta)
    Lambda_safe = jnp.where(jnp.abs(Lambda) < 1e-6, 
                            jnp.ones_like(Lambda) * 1e-6, 
                            Lambda)
    B_bar = (1 / Lambda_safe * (Lambda_bar - Identity))[..., None] * B_tilde
    return Lambda_bar, B_bar

#def discretize(Lambda, B_tilde, Delta):
#    Identity = jnp.ones(Lambda.shape[0], dtype=jnp.complex64)
#    Lambda_bar = jnp.exp(Lambda * Delta)
#    jax.debug.print("Lambda_bar mag min/max: {a} {b}",
#                    a=jnp.abs(Lambda_bar).min(),
#                    b=jnp.abs(Lambda_bar).max())
#    B_bar = (1 / Lambda * (Lambda_bar - Identity))[..., None] * B_tilde
#    return Lambda_bar, B_bar

from jax.lax import associative_scan

def binary_operator(element_i, element_j):
    """ Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
    Args:
    element_i: tuple containing A_i and Bu_i at position i (P,), (P,)
    element_j: tuple containing A_j and Bu_j at position j (P,), (P,)
    Returns:
    new element ( A_out, Bu_out ) """
    A_i, Bu_i = element_i
    A_j, Bu_j = element_j
    return A_j * A_i, A_j * Bu_i + Bu_j

def apply_ssm(Lambda_bar, B_bar, C_tilde, D, input_sequence):
    """ Compute the LxH output of discretized SSM given an LxH input.
    Args:
    Lambda_bar (complex64): discretized diagonal state matrix (P,)
    B_bar (complex64): discretized input matrix (P, H)
    C_tilde (complex64): output matrix (H, P)
    D (float32): feedthrough matrix (H,)
    input_sequence (float32): input sequence of features (L, H)
    Returns:
    ys (float32): the SSM outputs (S5 layer preactivations) (L, H) """
    L = input_sequence.shape[0]
    #Lambda_elements = jnp.repeat(Lambda_bar[None, ...], L, axis=0)
    Lambda_elements = jnp.broadcast_to(Lambda_bar[None, ...], (L,) + Lambda_bar.shape)
    #Bu_elements = jax.vmap(lambda u: B_bar @ u)(input_sequence)
    Bu_elements = jax.vmap(lambda u: B_bar @ u.astype(jnp.complex64))(input_sequence)
    elements = (Lambda_elements, Bu_elements)
    _, xs = associative_scan(binary_operator, elements)
    ys = jax.vmap(lambda x, u: (C_tilde @ x + D * u).real)(xs, input_sequence)
    return ys
apply_ssm = jax.checkpoint(apply_ssm)
