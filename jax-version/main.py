"""
S5 (Simplified Structured State Space) layer as a dm-haiku module.
HRM added in.

Dependencies:
    pip install dm-haiku optax jax jaxlib jmp pyarrow orbax-checkpoint

"""
from dataclasses import dataclass
import argparse
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["JAX_LOG_COMPILES"] = "0"
import jax
import jax.numpy as jnp
import jax.numpy as np
import jmp
import haiku as hk
import optax
import sys
import math
from jax.lax import associative_scan
from orbax import checkpoint as ocp
from pathlib import Path
from model import SRLMConfig, SRLM, AbsorbingGraph, LogLinearNoise, Sampler, mk_z, loss_function
from typing import Any

VOCAB_SIZE = 256
TOTAL_VOCAB = 257

@dataclass
class Specification:
    CONFIG : SRLMConfig
    STEP_REPORT_EVERY: int = 10
    SAVE_EVERY : int = 10000
    BATCH : int = 32
    SEQ_LEN : int = 32
    N_STEPS : int = 500
    SUPERVISION : int = 5

specifications = {
        "1024x1024": Specification(
            CONFIG=SRLMConfig(vocab_size = TOTAL_VOCAB,
                              d_model = 1024, d_state = 1024,
                              n_priors = 4, n_posteriors = 3),
            SEQ_LEN=128,
            N_STEPS=500,
            SAVE_EVERY=500),
        "512x64": Specification(
            CONFIG=SRLMConfig(vocab_size = TOTAL_VOCAB,
                              d_model = 512, d_state = 64,
                              n_priors=3, n_posteriors=2))
}


def setup(args):
    cwd = Path.cwd()
    print("Current path:", cwd)

    import warnings
    warnings.filterwarnings("error", category=np.ComplexWarning)
    
    #jax.config.update("jax_enable_custom_prng", True)
    #jax.config.update("jax_debug_nans", True)
    #jax.config.update("jax_platform_name", "cpu")
    
    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 5)

    # Policy for haiku — cast all compute to bfloat16
    policy = jmp.Policy(
        compute_dtype=jnp.bfloat16,
        param_dtype=jnp.float32,  # keep params in float32
        output_dtype=jnp.bfloat16
    )
    hk.mixed_precision.set_policy(SRLM, policy)

    spec = specifications[args.spec]

    graph = AbsorbingGraph(VOCAB_SIZE)
    noise = LogLinearNoise()
    sampler = Sampler()
    rng = hk.PRNGSequence(7)

    def model_spec(z, x, sigma):
        assert len(x.shape) == 2, x.shape
        assert len(sigma.shape) == 1, sigma.shape
        hrm = SRLM(spec.CONFIG)
        return hrm(z, x, sigma)
    model = hk.transform(model_spec)

    print("initializing model...")
    z_init = mk_z(1, 1, spec.CONFIG.d_model)
    x_all = jax.random.randint(next(rng), (1,1), 0, 256)
    sigma = jax.random.normal(next(rng), shape=(1,))
    params = model.init(rng=next(rng), x=x_all, z=z_init, sigma=sigma)
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Parameters: {param_count:,}")
    print(f"Parameter memory: {param_memory_mb(params):.1f} MB")

    z_init = mk_z(spec.BATCH, spec.SEQ_LEN, spec.CONFIG.d_model)
    return Setup(
            cwd,
            spec,
            graph,
            noise,
            sampler,
            rng,
            model,
            params,
            z_init)

@dataclass
class Setup:
    cwd : Any
    spec : Any
    graph : Any
    noise : Any
    sampler : Any
    rng : Any
    model : Any
    params : Any
    z_init : Any

def prepare_for_exam(args, s):
    ckdir = Path(args.checkpoint).absolute()
    ckdir.mkdir(parents=True, exist_ok=True)
    checkpointer = ocp.StandardCheckpointer()
    params = s.params
    if os.path.exists(ckdir):
        (epoch, step), item = max(list(subitems(ckdir)), key=lambda x: x[0], default=((0,0),None))
        if item is not None:
            params = checkpointer.restore(item, params)
            print(f"epoch {epoch} at step {step}")
    return params

def prepare_for_train(args, s):
    ckdir = Path(args.checkpoint).absolute()
    ckdir.mkdir(parents=True, exist_ok=True)

    checkpointer = ocp.StandardCheckpointer()
    def restore_checkpoint(params):
        if os.path.exists(ckdir):
            (epoch, step), item = max(list(subitems(ckdir)), key=lambda x: x[0], default=((0,0),None))
            if item is not None:
                params = checkpointer.restore(item, params)
            return epoch, step, params
        else:
            checkpointer.save(ckdir / "00000.0", params)
            return 0, 0, params

    epoch, step, params = restore_checkpoint(s.params)
    def save_checkpoint(params, epoch, step):
        checkpointer.save(ckdir, str(epoch).zfill(5) + "." + str(step))

    print("Setting up optimizer, loss function, training step")
    lr_scheduler = optax.schedules.cosine_decay_schedule(1e-4, 500, 0.01)
    optimizer = optax.chain(
        optax.clip_by_global_norm(0.1),
        optax.zero_nans(),
        optax.adamw(lr_scheduler, weight_decay=0.01),
    )
    opt_state = optimizer.init(params)
    loss_fn = loss_function(s.model, s.graph, s.noise)
    @jax.jit
    def train_step_single(key, params, opt_state, x, z):
        (loss, z), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, key, z, x)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, z

    return Trainer(
            ckdir,
            epoch,
            step,
            checkpointer,
            params,
            optimizer,
            opt_state,
            loss_fn,
            train_step_single,
            save_checkpoint)

@dataclass
class Trainer:
    ckdir : Any
    epoch : int
    step : int
    checkpointer : Any
    params : Any
    optimizer : Any
    opt_state : Any
    loss_fn : Any
    train_step_single : Any
    save : Any

def subitems(ckdir):
    for subitem in ckdir.iterdir():
        if subitem.name == "progress.json":
            continue
        if "_" in subitem.name:
            epoch_s, step_s = subitem.name.split("_")
            yield (int(epoch_s), int(step_s)), subitem

def as_text(p):
    return p.astype(np.uint8).tobytes().decode("utf-8", errors="replace")

def from_text(text):
    data = text.encode("utf-8")
    return np.frombuffer(bytearray(data), dtype=np.uint8).astype(np.int16)

def load_wikipedia_finnish(cwd, batch_size, seq_len):
    from wiki_data import WikiDataLoader
    loader = WikiDataLoader(
        parquet_files=[
            cwd / "../../data/train-00000-of-00002.parquet",
            cwd / "../../data/train-00001-of-00002.parquet",
        ],
        batch_size=batch_size,
        seq_len=seq_len,
        seed=42,
    )
    return loader

def load_kalevala(cwd):
    with open(cwd / "../../data/kalevala.plain.txt", "r", encoding="utf-8") as fd:
        text = fd.read().replace("\n", " ")
    raw = np.frombuffer(bytearray(text.encode("utf-8")), dtype=np.uint8).astype(np.int32)
    N   = raw.shape[0]
    print(f"Data: {N} tavua")
    def sample_batch(key, seq_len, batch):
        starts = jax.random.randint(key, (batch,), 0, raw.shape[0] - seq_len)
        starts = np.array(starts)  # bring to CPU
        result = np.stack([raw[s:s+seq_len] for s in starts])
        return jax.device_put(result)
    return sample_batch

parser = argparse.ArgumentParser(
                    prog='srlm',
                    description='Evaluates and trains SRLM -models',
                    epilog='It is a mess.')
parser.set_defaults(run=None)
parser.add_argument("-c", "--checkpoint", help="checkpoint directory")
parser.add_argument("-s", "--spec",
                    default="512x64",
                    help="specification of the model (512x64, 1024x1024)")
subparsers = parser.add_subparsers(help='subcommand help')

def train(args):
    s = setup(args)
    t = prepare_for_train(args, s)
    sample_batch = load_kalevala(s.cwd)
    print(f"Training S5 HRM haiku model | d_model={s.spec.CONFIG.d_model}, d_state={s.spec.CONFIG.d_state}")
    print("-" * 55)
    for epoch in range(t.epoch, 5000):
        total_loss = 0
        for step in range(t.step, s.spec.N_STEPS):
            batch = sample_batch(next(s.rng), s.spec.SEQ_LEN, s.spec.BATCH)
            session_loss = supervision_train(s, t, batch)
            if t.step % s.spec.STEP_REPORT_EVERY == 0:
                print(f"  step {step:4d} | loss {session_loss/s.spec.SUPERVISION:.4f}")
            t.step += 1
            total_loss += session_loss
        print(f"epoch {epoch+1}, loss {total_loss / s.spec.N_STEPS / s.spec.SUPERVISION}")
        t.step = 0
        t.save(t.params, epoch+1, 0)
    print("Done.")

parser_train = subparsers.add_parser('train', help='train SRLM from ../../data/kalevala.plaintext.txt')
parser_train.set_defaults(run=train)

def wikitrain(args):
    s = setup(args)
    t = prepare_for_train(args, s)
    loader = load_wikipedia_finnish(s.cwd, s.spec.BATCH, s.spec.SEQ_LEN)
    print(f"Training S5 HRM haiku model | d_model={s.spec.CONFIG.d_model}, d_state={s.spec.CONFIG.d_state}")
    print(f"-" * 55)
    resuming = loader.load_state(t.ckdir / f"progress.json")
    for epoch in range(t.epoch, 3):
        total_loss = 0
        if not resuming:
            loader.shuffle(epoch)
        resuming = False
        while not loader.epoch_done():
            batch = loader.next_batch() # (B, seq_len) int32, or None if epoch done
            if batch is not None:
                session_loss = supervision_train(s, t, batch)
                k = loader.steps_this_epoch
                if k % s.spec.STEP_REPORT_EVERY == 0:
                    print(f"{k} | session loss:", session_loss / s.spec.SUPERVISION)
                if k % s.spec.SAVE_EVERY == 0:
                    print(f"-" * 55)
                    t.save(t.params, t.epoch+1, loader.steps_this_epoch)
                    loader.save_state(t.ckdir / f"progress.json")
        print(f"last | session loss:", session_loss / s.spec.SUPERVISION)
        print(f"epoch {epoch} done, {loader.steps_this_epoch} steps")
        t.save(t.params, t.epoch+1, 0)
        loader.save_state(t.ckdir / f"progress.json")

def supervision_train(s, t, batch):
    z = s.z_init
    session_loss = 0
    for _ in range(s.spec.SUPERVISION):
        t.params, t.opt_state, loss, z = t.train_step_single(next(s.rng), t.params, t.opt_state, batch, z)
        session_loss += loss
    if np.isnan(session_loss):
        print(f"Training has failed")
        sys.exit(0)
    return session_loss

parser_wikitrain = subparsers.add_parser('wikitrain', help='train from finnish wikipedia')
parser_wikitrain.set_defaults(run=wikitrain)

def wikidry(args):
    i = 0
    s = setup(args)
    t = prepare_for_train(args, s)
    loader = load_wikipedia_finnish(s.cwd, s.spec.BATCH, s.spec.SEQ_LEN)
    loader.shuffle(0)           # shuffle article order for this epoch
    while not loader.epoch_done():
        batch = loader.next_batch() # (B, seq_len) int32, or None if epoch done
        if i % 10000 == 0:
            print(f"at {i}")
        i += 1
    print("total batches: ", i)

parser_wikidry = subparsers.add_parser('wikidry', help='dry run finnish wikipedia')
parser_wikidry.set_defaults(run=wikidry)

def evaluate(args):
    s = setup(args)
    params = prepare_for_exam(args, s)
    while True:
        query = from_text(input("> "))
        def projector(x):
            return x.at[:,0:min(64, len(query))].set(query[:min(64,len(query))])
        _, x = s.sampler.sample(next(s.rng), as_text,
                           s.model.apply,
                           params, mk_z(1,64, s.spec.CONFIG.d_model),
                           s.graph,
                           s.noise,
                           batch_size=1,
                           batch_len=64,
                           steps=10,
                           projector = projector)
        print(repr(x))

parser_eval = subparsers.add_parser('eval', help='evaluate on model')
parser_eval.set_defaults(run=evaluate)

def param_memory_mb(params):
    leaves = jax.tree_util.tree_leaves(params)
    total_bytes = sum(x.size * x.dtype.itemsize for x in leaves)
    return total_bytes / 1024**2

if __name__=="__main__":
    args = parser.parse_args()
    if args.run is None:
        print("no command given")
    else:
        args.run(args)
