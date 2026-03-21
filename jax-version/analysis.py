"""
s5hrm_analysis.py — Analysis tool for S5HRM checkpoints.

Usage:
    python s5hrm_analysis.py <init_checkpoint> <trained_checkpoint>

Runs the following analyses:
    1. Parameter count and memory per module
    2. Parameter drift (how much each module moved from init)
    3. Gradient norms on a sample batch
    4. Activation statistics (mean/std per layer)
    5. HRM slow pathway contribution (zH ablation)
    6. Loss ablation (freeze each top-level module, measure loss impact)
    7. SSM eigenvalue distribution (Lambda_bar magnitudes)
"""

# This needs to be fixed.

import sys
import os
import math
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax
from pathlib import Path
from jax.lax import associative_scan
from orbax import checkpoint as ocp
from dataclasses import dataclass

# ── Import everything from s5hrm so we don't duplicate definitions ────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s5hrm import (
    Config, S5HRM, S5Stack, S5Layer, HRM, FastLayer, SlowLayer,
    InputLayer, OutputLayer, FeedForward, S5, TimestepEmbedder,
    discretize, apply_ssm, binary_operator, scatter,
    timestep_embedding, loss_function, from_text,
    config,
    training_data
)
from hk_models.graph import AbsorbingGraph
from hk_models.noise import LogLinearNoise

# ── Helpers ───────────────────────────────────────────────────────────────────

def separator(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)

def flat_module_leaves(tree, prefix=""):
    """Recursively yield (path_str, array) for every leaf in a nested dict."""
    if isinstance(tree, dict):
        for k, v in tree.items():
            yield from flat_module_leaves(v, prefix + "/" + k if prefix else k)
    else:
        yield prefix, tree

def top_level_key(path):
    """Return the first path component (top-level module name)."""
    return path.split("/")[0]

def module_key(path):
    """Return the first two path components as module identifier."""
    parts = path.split("/")
    return "/".join(parts[:2]) if len(parts) > 1 else parts[0]

# ── 1. Parameter count and memory ─────────────────────────────────────────────

def analyse_parameters(params):
    separator("1. Parameter count and memory per module")
    module_counts = {}
    module_bytes = {}
    for path, arr in flat_module_leaves(params):
        key = module_key(path)
        module_counts[key] = module_counts.get(key, 0) + arr.size
        module_bytes[key] = module_bytes.get(key, 0) + arr.size * arr.dtype.itemsize

    total_params = sum(module_counts.values())
    total_mb = sum(module_bytes.values()) / 1024**2

    print(f"{'Module':<55} {'Params':>12} {'MB':>8}")
    print("-" * 77)
    for k in sorted(module_counts):
        print(f"  {k:<53} {module_counts[k]:>12,} {module_bytes[k]/1024**2:>8.2f}")
    print("-" * 77)
    print(f"  {'TOTAL':<53} {total_params:>12,} {total_mb:>8.2f}")

# ── 2. Parameter drift ────────────────────────────────────────────────────────

def analyse_drift(params_init, params_trained):
    separator("2. Parameter drift from init (mean |trained - init|)")
    module_drift = {}
    module_rel = {}
    for (path, arr_trained), (_, arr_init) in zip(
        flat_module_leaves(params_trained),
        flat_module_leaves(params_init)
    ):
        key = module_key(path)
        abs_drift = float(jnp.mean(jnp.abs(arr_trained - arr_init)))
        rel_drift = abs_drift / (float(jnp.mean(jnp.abs(arr_init))) + 1e-8)
        if key not in module_drift:
            module_drift[key] = []
            module_rel[key] = []
        module_drift[key].append(abs_drift)
        module_rel[key].append(rel_drift)

    mean_drift = {k: np.mean(v) for k, v in module_drift.items()}
    mean_rel   = {k: np.mean(v) for k, v in module_rel.items()}

    print(f"{'Module':<55} {'Abs drift':>10} {'Rel drift':>10}")
    print("-" * 77)
    for k in sorted(mean_drift, key=lambda x: -mean_drift[x]):
        print(f"  {k:<53} {mean_drift[k]:>10.5f} {mean_rel[k]:>10.3f}x")

# ── 3. Gradient norms ─────────────────────────────────────────────────────────

def analyse_gradients(params, loss_fn, key, z, batch):
    separator("3. Gradient norms per module")
    (loss, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, key, z, batch)
    print(f"  Loss: {float(loss):.4f}")
    print()

    module_norms = {}
    for path, arr in flat_module_leaves(grads):
        key_m = module_key(path)
        if key_m not in module_norms:
            module_norms[key_m] = 0.0
        module_norms[key_m] += float(jnp.sum(arr ** 2))

    module_norms = {k: math.sqrt(v) for k, v in module_norms.items()}

    print(f"{'Module':<55} {'Grad norm':>10}")
    print("-" * 67)
    for k in sorted(module_norms, key=lambda x: -module_norms[x]):
        print(f"  {k:<53} {module_norms[k]:>10.5f}")

    total_norm = math.sqrt(sum(v**2 for v in module_norms.values()))
    print("-" * 67)
    print(f"  {'TOTAL GRAD NORM':<53} {total_norm:>10.5f}")
    return grads

# ── 4. Activation statistics ──────────────────────────────────────────────────

activation_log = {}

def make_instrumented_model(config):
    """Build a model variant that logs activation stats."""
    global activation_log
    activation_log = {}

    class InstrumentedS5(S5):
        def __call__(self, input_sequence):
            out = super().__call__(input_sequence)
            name = self.name
            def _log(mean, std, max_val):
                activation_log[name] = {
                    "mean": float(mean),
                    "std":  float(std),
                    "max":  float(max_val),
                }
            jax.debug.callback(
                _log,
                jnp.mean(out),
                jnp.std(out),
                jnp.max(jnp.abs(out)),
            )
            return out

    class InstrumentedS5Layer(S5Layer):
        def __init__(self, cfg, name=None):
            hk.Module.__init__(self, name=name)
            self.pre  = hk.Linear(2 * cfg.d_model, name=f"{name}_pre")
            self.norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name=f"{name}_norm")
            self.fwd  = InstrumentedS5(cfg.d_model, cfg.d_state, name=f"{name}_fwd")
            self.bwd  = InstrumentedS5(cfg.d_model, cfg.d_state, name=f"{name}_bwd")

    class InstrumentedS5Stack(S5Stack):
        def __init__(self, cfg, n_layers=1, name=None):
            hk.Module.__init__(self, name=name)
            self.n_layers = n_layers
            self.layers = [InstrumentedS5Layer(cfg, name=f"{name}_{i}") for i in range(n_layers)]

    class InstrumentedS5HRM(S5HRM):
        def __init__(self, cfg, name="s5hrm"):
            hk.Module.__init__(self, name=name)
            self.cfg       = cfg
            self.input     = InputLayer(cfg, name="input")
            self.prior     = InstrumentedS5Stack(cfg, cfg.n_layers, name="prior")
            self.main      = HRM(cfg, name="main")
            self.posterior = InstrumentedS5Stack(cfg, cfg.n_layers, name="posterior")
            self.output    = OutputLayer(cfg, name="output")

    def model_spec(z, x, sigma):
        return InstrumentedS5HRM(config)(z, x, sigma)

    return hk.transform(model_spec)

def analyse_activations(params, config, key, z, batch, sigma):
    separator("4. Activation statistics (S5 layer outputs)")
    global activation_log
    model = make_instrumented_model(config)
    # forward only — no grad needed
    _ = model.apply(params, key, z, batch, sigma)

    print(f"  {'Layer':<50} {'mean':>8} {'std':>8} {'max|x|':>8}")
    print("-" * 78)
    for name, stats in sorted(activation_log.items()):
        flag = "  *** DEAD" if stats["std"] < 1e-3 else ""
        print(f"  {name:<50} {stats['mean']:>8.4f} {stats['std']:>8.4f} {stats['max']:>8.4f}{flag}")

# ── 5. HRM slow pathway ablation ─────────────────────────────────────────────

def analyse_hrm_contribution(params, config, key, z, batch, sigma):
    separator("5. HRM slow pathway (zH) contribution")

    results = {}

    def model_with_zeroed_zh(z, x, sigma):
        class ZeroedHRM(HRM):
            def __call__(self, z, x):
                zH, zL = z
                zL = jax.lax.stop_gradient(zL)
                zH = jax.lax.stop_gradient(zH)
                for i in range(self.N * self.T - 1):
                    zL = self.fast(zL, jnp.zeros_like(zH), x)
                    if (i + 1) % self.T == 0:
                        zH = self.slow(zH, zL)
                    zL = jax.lax.stop_gradient(zL)
                    zH = jax.lax.stop_gradient(zH)
                zL = self.fast(zL, jnp.zeros_like(zH), x)
                zH = self.slow(zH, zL)
                return (zL, zH), zH

        class AblatedModel(S5HRM):
            def __init__(self, cfg, name="s5hrm"):
                hk.Module.__init__(self, name=name)
                self.cfg       = cfg
                self.input     = InputLayer(cfg, name="input")
                self.prior     = S5Stack(cfg, cfg.n_layers, name="prior")
                self.main      = ZeroedHRM(cfg, name="main")
                self.posterior = S5Stack(cfg, cfg.n_layers, name="posterior")
                self.output    = OutputLayer(cfg, name="output")

        return AblatedModel(config)(z, x, sigma)

    def normal_model(z, x, sigma):
        return S5HRM(config)(z, x, sigma)

    model_normal  = hk.transform(normal_model)
    model_ablated = hk.transform(model_with_zeroed_zh)

    _, out_normal  = model_normal.apply(params, key, z, batch, sigma)
    _, out_ablated = model_ablated.apply(params, key, z, batch, sigma)

    diff = float(jnp.mean(jnp.abs(out_normal - out_ablated)))
    rel  = diff / (float(jnp.mean(jnp.abs(out_normal))) + 1e-8)
    print(f"  Mean |output_normal - output_zeroed_zH|: {diff:.6f}")
    print(f"  Relative to output magnitude:            {rel:.4f}")
    if rel < 0.01:
        print("  *** zH has almost NO effect — slow pathway hasn't learned to contribute.")
    elif rel < 0.1:
        print("  zH has modest effect — slow pathway is beginning to contribute.")
    else:
        print("  zH has significant effect — slow pathway is active.")

# ── 6. Loss ablation ──────────────────────────────────────────────────────────

def analyse_loss_ablation(init_params, params, loss_fn, key, z, batch):
    separator("6. Loss ablation (freeze each top-level module)")

    baseline_loss, _ = loss_fn(params, key, z, batch)
    print(f"  Baseline loss: {float(baseline_loss):.4f}")
    print()

    top_level_modules = list(params.keys())

    print(f"  {'Frozen module':<40} {'Loss':>10} {'Delta':>10} {'Impact':>8}")
    print("-" * 72)
    for module in sorted(top_level_modules):
        def freeze_module(p, name=module):
            return {
                k: init_params[k] if k == name else v
                for k, v in p.items()
            }
        frozen_params = freeze_module(params)
        frozen_loss, _ = loss_fn(frozen_params, key, z, batch)
        delta = float(frozen_loss) - float(baseline_loss)
        impact = "higher (module helps)" if delta > 0 else \
                 "same (module idle)"    if delta == 0 else \
                 "lower (module hurts?)"
        print(f"  {module:<40} {float(frozen_loss):>10.4f} {delta:>+10.4f}  {impact}")

# ── 7. SSM eigenvalue distribution ───────────────────────────────────────────

def analyse_eigenvalues(params):
    separator("7. SSM eigenvalue distribution (Lambda_bar magnitudes)")

    ssm_params = {
        path: arr
        for path, arr in flat_module_leaves(params)
        if path.endswith("log_real") or path.endswith("imag") or path.endswith("log_Delta")
    }

    # Group by SSM instance
    ssm_instances = {}
    for path, arr in ssm_params.items():
        parts = path.split("/")
        param_name = parts[-1]
        instance = "/".join(parts[:-1])
        if instance not in ssm_instances:
            ssm_instances[instance] = {}
        ssm_instances[instance][param_name] = arr

    print(f"  {'SSM instance':<55} {'|λ| min':>8} {'|λ| mean':>8} {'|λ| max':>8} {'dead%':>7}")
    print("-" * 90)
    for instance, p in sorted(ssm_instances.items()):
        if "log_real" not in p or "imag" not in p or "log_Delta" not in p:
            continue
        Lambda = (-jnp.exp(p["log_real"]) + 1j * p["imag"]).astype(jnp.complex64)
        Delta  = jnp.exp(p["log_Delta"])
        Lambda_bar = jnp.exp(Lambda * Delta)
        mags = jnp.abs(Lambda_bar)
        dead_pct = 100.0 * float(jnp.mean(mags < 0.01))
        print(f"  {instance:<55} {float(mags.min()):>8.4f} {float(mags.mean()):>8.4f} {float(mags.max()):>8.4f} {dead_pct:>6.1f}%")
        if float(mags.max()) > 0.999:
            print(f"  *** WARNING: some eigenvalues near 1.0 — possible slow forgetting or instability")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 3:
        print("Usage: python s5hrm_analysis.py <init_checkpoint> <trained_checkpoint>")
        sys.exit(1)

    base = Path.cwd()
    init_ckpt    = base / sys.argv[1]
    trained_ckpt = base / sys.argv[2]

    VOCAB_SIZE  = 256
    TOTAL_VOCAB = 257
    B           = 8
    SEQ_LEN     = 64

    graph = AbsorbingGraph(VOCAB_SIZE)
    noise = LogLinearNoise()

    rng = hk.PRNGSequence(42)

    def model_spec(z, x, sigma):
        return S5HRM(config)(z, x, sigma)

    model = hk.transform(model_spec)

    def mk_z(b, l):
        return (
            jnp.zeros((b, l, config.d_model), dtype=jnp.float32),
            jnp.zeros((b, l, config.d_model), dtype=jnp.float32),
        )

    # Initialise with dummy data to get param structure
    print("Initialising model structure...")
    x_dummy    = jax.random.randint(next(rng), (B, SEQ_LEN), 0, VOCAB_SIZE)
    sigma_dummy = jnp.abs(jax.random.normal(next(rng), (B,))) + 0.1
    z_dummy    = mk_z(B, SEQ_LEN)
    params_dummy = model.init(rng=next(rng), x=x_dummy, z=z_dummy, sigma=sigma_dummy)

    # Load checkpoints
    checkpointer = ocp.StandardCheckpointer()
    print(f"Loading init checkpoint:    {init_ckpt}")
    params_init    = checkpointer.restore(init_ckpt, params_dummy)
    print(f"Loading trained checkpoint: {trained_ckpt}")
    params_trained = checkpointer.restore(trained_ckpt, params_dummy)

    # Sample batch for gradient/loss analyses
    key = next(rng)
    x_batch = training_data()(next(rng), SEQ_LEN, B)
    #x_batch = jax.random.randint(next(rng), (B, SEQ_LEN), 0, VOCAB_SIZE)
    sigma_b = jnp.abs(jax.random.normal(next(rng), (B,))) + 0.1
    z_batch = mk_z(B, SEQ_LEN)

    loss_fn = loss_function(model, graph, noise)

    # Wrap loss_fn to match (params, key, z, batch) signature expected below
    def simple_loss(p, k, z, x):
        return loss_fn(p, k, z, x)

    # ── Run analyses ──────────────────────────────────────────────────────────
    #analyse_parameters(params_trained)
    #analyse_drift(params_init, params_trained)
    #analyse_gradients(params_trained, simple_loss, key, z_batch, x_batch)
    #analyse_activations(params_trained, config, key, z_batch, x_batch, sigma_b)
    #analyse_hrm_contribution(params_trained, config, key, z_batch, x_batch, sigma_b)
    analyse_loss_ablation(params_init, params_trained, simple_loss, key, z_batch, x_batch)
    analyse_eigenvalues(params_trained)

    print()
    print("Analysis complete.")

if __name__ == "__main__":
    main()
