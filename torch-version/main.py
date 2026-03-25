"""
main.py — Training and inference entry point for SRLM in PyTorch.

Usage:
    python main.py train <checkpoint> [--kalevala STEPS] [--wikipedia STEPS]
                                      [--save-every N] [--report-every N]
                                      [--supervision K] [--batch-size B]
                                      [--seq-len L] [--lr LR]
    python main.py eval <checkpoint>  [--steps N] [--seq-len L]
"""

import argparse
import json
import math
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from model import SRLM, SRLMConfig, make_z, AbsorbingGraph, LogLinearNoise, Sampler, loss_function, MemoryBank
from wiki_data import WikiDataLoader

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VOCAB_SIZE  = 256
TOTAL_VOCAB = 257

PARQUET_FILES = [
    "../../data/train-00000-of-00002.parquet",
    "../../data/train-00001-of-00002.parquet",
]

DEFAULT_CONFIG = SRLMConfig(
    vocab_size     = TOTAL_VOCAB,
    context_length = 256,
    d_model        = 256,
    n_priors       = 3,
    n_posteriors   = 2,
    n_heads        = 8,
)

MEDIUM_CONFIG = SRLMConfig(
    vocab_size     = TOTAL_VOCAB,
    context_length = 256,
    d_model        = 1152,
    n_priors       = 3,
    n_posteriors   = 2,
    n_heads        = 16,
    dropout        = 0.2,
)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def as_text(p: torch.Tensor) -> str:
    return p.cpu().to(torch.uint8).numpy().tobytes().decode("utf-8", errors="replace")

def from_text(text: str) -> torch.Tensor:
    data = text.encode("utf-8")
    if len(data) == 0:
        return None
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).to(torch.int32)

def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def param_memory_mb(model: nn.Module) -> float:
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2

def sample_batch_random(raw: torch.Tensor, seq_len: int, batch: int) -> torch.Tensor:
    N = raw.shape[0]
    starts = torch.randint(0, N - seq_len, (batch,))
    return torch.stack([raw[s:s+seq_len] for s in starts])

# ---------------------------------------------------------------------------
# Checkpoint I/O (new format per spec)
# ---------------------------------------------------------------------------

def save_checkpoint(ckpt_dir: Path, model: nn.Module, config: SRLMConfig,
                    training_log: str, wiki_state: dict | None = None):
    """Save checkpoint directory with all spec files."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "parameters.pt")
    with open(ckpt_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)
    with open(ckpt_dir / "training.txt", "w") as f:
        f.write(training_log)
    if wiki_state is not None:
        with open(ckpt_dir / "wiki.json", "w") as f:
            json.dump(wiki_state, f)
    print(f"  Saved: {ckpt_dir}")

def _strip_orig_mod(state_dict: dict) -> dict:
    """Strip _orig_mod. prefix left by torch.compile when saving state."""
    prefix = "_orig_mod."
    if any(k.startswith(prefix) for k in state_dict):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict

def load_checkpoint(model: nn.Module, ckpt_dir: Path) -> bool:
    """Load model parameters from checkpoint directory."""
    params = ckpt_dir / "parameters.pt"
    if not params.exists():
        # Also try legacy format
        legacy = ckpt_dir / "checkpoint.pt"
        if legacy.exists():
            state = torch.load(legacy, map_location="cpu")
            model.load_state_dict(_strip_orig_mod(state["model"]))
            print(f"  Loaded (legacy): {legacy}")
            return True
        return False
    model.load_state_dict(_strip_orig_mod(torch.load(params, map_location="cpu")))
    print(f"  Loaded: {params}")
    return True

def load_config(ckpt_dir: Path) -> SRLMConfig | None:
    cfg_path = ckpt_dir / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return SRLMConfig(**json.load(f))
    return None

# ---------------------------------------------------------------------------
# Training programs
# ---------------------------------------------------------------------------

class KalevalaProgram:
    """Trains on Kalevala text. Runs for `max_steps` steps (None = indefinite)."""

    def __init__(self, device, seq_len, batch_size, max_steps=None):
        self.device = device
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.step_count = 0
        cwd = Path.cwd()
        with open(cwd / "../../data/kalevala.plain.txt", "r", encoding="utf-8") as f:
            text = f.read().replace("\n", " ")
        self.raw = torch.frombuffer(bytearray(text.encode("utf-8")),
                                    dtype=torch.uint8).to(torch.int32).to(device)
        print(f"Kalevala: {self.raw.shape[0]:,} bytes")

    def done(self):
        if self.max_steps is None:
            return False
        return self.step_count >= self.max_steps

    def next_batch(self):
        """Return (batch, None), or None if done."""
        if self.done():
            return None
        self.step_count += 1
        return sample_batch_random(self.raw, self.seq_len, self.batch_size), None

    def description(self):
        if self.max_steps is None:
            return f"kalevala(indefinite, done={self.step_count})"
        return f"kalevala({self.max_steps}, done={self.step_count})"


class WikipediaProgram:
    """Trains on Wikipedia. Runs for `max_steps` steps or one epoch."""

    def __init__(self, seq_len, batch_size, max_steps=None, seed=42):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.step_count = 0
        self.loader = WikiDataLoader(
            parquet_files=PARQUET_FILES,
            seq_len=seq_len,
            batch_size=batch_size,
            seed=seed,
        )
        self.loader.shuffle(0)

    def done(self):
        if self.max_steps is not None:
            return self.step_count >= self.max_steps
        return self.loader.epoch_done()

    def next_batch(self):
        """Return (batch, None), or None if done/exhausted."""
        if self.done():
            return None
        batch = self.loader.next_batch()
        if batch is None:
            return None
        self.step_count += 1
        return torch.from_numpy(batch), None

    def wiki_state(self):
        return self.loader.state_dict()

    def description(self):
        if self.max_steps is not None:
            return f"wikipedia({self.max_steps}, done={self.step_count})"
        return f"wikipedia(epoch, done={self.step_count})"

MASK_TOKEN = VOCAB_SIZE  # = 256, the absorbing state

class ArithmeticProgram:
    """Trains on N+M=Y facts. Problem shown unmasked; rest of buffer masked.

    This teaches the model to predict the masked suffix given the arithmetic
    fact as a visible prefix. Loss is computed only on masked positions.
    """

    def __init__(self, seq_len, batch_size, max_steps=None, max_operand=99):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.max_operand = max_operand
        self.step_count = 0

    def done(self):
        if self.max_steps is None:
            return False
        return self.step_count >= self.max_steps

    def next_batch(self):
        """Return (batch, perturbed_batch), or None if done.

        batch:           clean tokens — problem prefix + spaces for the rest
        perturbed_batch: problem prefix shown, rest filled with MASK_TOKEN
        """
        if self.done():
            return None

        batch_clean     = torch.full((self.batch_size, self.seq_len), ord(' '), dtype=torch.int32)
        batch_perturbed = torch.full((self.batch_size, self.seq_len), MASK_TOKEN, dtype=torch.int32)

        for i in range(self.batch_size):
            n = torch.randint(0, self.max_operand + 1, ()).item()
            m = torch.randint(0, self.max_operand + 1, ()).item()
            problem = f"{n}+{m}={n+m}"
            tokens  = torch.frombuffer(bytearray(problem.encode()), dtype=torch.uint8).to(torch.int32)
            L = min(len(tokens), self.seq_len)
            batch_clean[i, :L]     = tokens[:L]
            batch_perturbed[i, :L] = tokens[:L]

        self.step_count += 1
        return batch_clean, batch_perturbed

    def description(self):
        if self.max_steps is None:
            return f"arithmetic(indefinite, done={self.step_count})"
        return f"arithmetic({self.max_steps}, done={self.step_count})"


class QAProgram:
    """Trains on question-answer pairs from a JSONL file.

    Each entry is formatted as "{question} {answer}". The question + separator
    space are shown unmasked; the answer is masked. Loss is computed only on
    masked (answer) positions.
    """

    def __init__(self, path, seq_len, batch_size, max_steps=None):
        import json
        self.seq_len    = seq_len
        self.batch_size = batch_size
        self.max_steps  = max_steps
        self.step_count = 0
        self.pairs = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.pairs.append((obj["question"], obj["answer"]))
        print(f"QA: {len(self.pairs)} pairs loaded from {path}")

    def done(self):
        if self.max_steps is None:
            return False
        return self.step_count >= self.max_steps

    def next_batch(self):
        """Return (batch, perturbed_batch), or None if done.

        batch:           clean tokens — full question + space + answer + spaces
        perturbed_batch: question + space shown, answer positions masked
        """
        if self.done():
            return None

        batch_clean     = torch.full((self.batch_size, self.seq_len), ord(' '), dtype=torch.int32)
        batch_perturbed = torch.full((self.batch_size, self.seq_len), MASK_TOKEN, dtype=torch.int32)

        indices = torch.randint(0, len(self.pairs), (self.batch_size,))
        for i, idx in enumerate(indices.tolist()):
            question, answer = self.pairs[idx]
            prefix = question + " "
            full   = prefix + answer
            prefix_tok = torch.frombuffer(bytearray(prefix.encode()), dtype=torch.uint8).to(torch.int32)
            full_tok   = torch.frombuffer(bytearray(full.encode()),   dtype=torch.uint8).to(torch.int32)
            Lp = min(len(prefix_tok), self.seq_len)
            Lf = min(len(full_tok),   self.seq_len)
            batch_clean[i, :Lf]     = full_tok[:Lf]
            batch_perturbed[i, :Lp] = prefix_tok[:Lp]   # answer stays masked

        self.step_count += 1
        return batch_clean, batch_perturbed

    def description(self):
        if self.max_steps is None:
            return f"qa(indefinite, done={self.step_count})"
        return f"qa({self.max_steps}, done={self.step_count})"

# ---------------------------------------------------------------------------
# Score function wrapper for sampler
# ---------------------------------------------------------------------------

def make_score_fn(model, device, memories=None):
    @torch.no_grad()
    def score_fn(z, x, sigma):
        x = x.to(device)
        sigma = sigma.to(device)
        z = tuple(zi.to(device) for zi in z)
        z, log_score, _aux = model(z, x, sigma, memories=memories)
        return z, log_score
    return score_fn

# ---------------------------------------------------------------------------
# Train step factory
# ---------------------------------------------------------------------------

def make_train_step(model, optimizer, loss_fn, device):
    def train_step(z, batch, perturbed_batch=None, memories=None):
        batch = batch.to(device)
        if perturbed_batch is not None:
            perturbed_batch = perturbed_batch.to(device)
        optimizer.zero_grad()
        loss, z = loss_fn(z, batch, perturbed_batch=perturbed_batch, memories=memories)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        return loss.item(), z
    return train_step

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_train(args):
    cwd    = Path.cwd()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Resolve config: checkpoint > preset flag > default
    ckpt_path = cwd / args.checkpoint
    base = load_config(ckpt_path)
    if base is None:
        base = MEDIUM_CONFIG if args.medium else DEFAULT_CONFIG
    config = SRLMConfig(
        vocab_size       = base.vocab_size,
        context_length   = args.seq_len or base.context_length,
        d_model          = args.d_model or base.d_model,
        n_priors         = args.n_priors if args.n_priors is not None else base.n_priors,
        n_posteriors     = args.n_posteriors if args.n_posteriors is not None else base.n_posteriors,
        n_heads          = args.n_heads or base.n_heads,
        dropout          = args.dropout if args.dropout is not None else base.dropout,
        N                = args.N if args.N is not None else base.N,
        T                = args.T if args.T is not None else base.T,
    )

    seq_len     = config.context_length
    batch_size  = args.batch_size
    supervision = args.supervision
    save_every  = args.save_every
    report_every = args.report_every

    # Graph / noise / sampler
    graph   = AbsorbingGraph(VOCAB_SIZE)
    noise   = LogLinearNoise()

    # Model
    print("Initialising model...")
    model = SRLM(config).to(device)
    # Load existing checkpoint
    load_checkpoint(model, ckpt_path)
    model = torch.compile(model)
    print(f"Parameters:       {param_count(model):,}")
    print(f"Parameter memory: {param_memory_mb(model):.1f} MB")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    loss_fn   = loss_function(model, graph, noise)

    train_step = make_train_step(model, optimizer, loss_fn, device)

    # Build training programs
    programs = []
    if args.kalevala is not None:
        max_steps = None if args.kalevala == 0 else args.kalevala
        programs.append(KalevalaProgram(device, seq_len, batch_size, max_steps))
    if args.wikipedia is not None:
        max_steps = None if args.wikipedia == 0 else args.wikipedia
        programs.append(WikipediaProgram(seq_len, batch_size, max_steps))
    if args.arithmetic is not None:
        max_steps = None if args.arithmetic == 0 else args.arithmetic
        programs.append(ArithmeticProgram(seq_len, batch_size, max_steps))
    if args.qa is not None:
        max_steps = None if args.qa == 0 else args.qa
        programs.append(QAProgram(args.qa_file, seq_len, batch_size, max_steps))

    if not programs:
        print("No training programs specified. Use --kalevala and/or --wikipedia.")
        sys.exit(1)

    # Training log
    training_log_lines = []
    start_time = datetime.now()
    training_log_lines.append(f"started: {start_time.isoformat()}")
    for p in programs:
        training_log_lines.append(f"program: {p.description()}")

    # Create checkpoint dir and loss.txt
    ckpt_path.mkdir(parents=True, exist_ok=True)
    loss_fd = open(ckpt_path / "loss.txt", "w")

    # Scheduler: linear warmup then cosine decay
    warmup_steps = args.warmup_steps
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(save_every, 1000), eta_min=1e-6
    )
    if warmup_steps > 0:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-2, end_factor=1.0, total_iters=warmup_steps
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
    else:
        scheduler = cosine_scheduler

    # Memory bank (Anki-style rolling recall)
    use_memory = args.memory_size > 0
    if use_memory:
        memory_bank = MemoryBank()
        print(f"Memory bank: size={args.memory_size}, k={args.memory_k}, refresh every {args.memory_refresh}")

    print(f"Training | programs: {[p.description() for p in programs]}")
    print("-" * 55)

    global_step = 0
    running_loss = 0.0
    program_idx = 0

    try:
        while True:
            # Check if all programs are done
            if all(p.done() for p in programs):
                break

            # Round-robin: pick next non-done program
            attempts = 0
            while programs[program_idx].done():
                program_idx = (program_idx + 1) % len(programs)
                attempts += 1
                if attempts > len(programs):
                    break
            if attempts > len(programs):
                break

            program = programs[program_idx]

            # Get batch from program
            result = program.next_batch()
            if result is None:
                program_idx = (program_idx + 1) % len(programs)
                continue
            batch, perturbed_batch = result

            # Store this batch in the memory bank
            if use_memory:
                model.eval()
                memory_bank.encode(model, [batch], device)
                model.train()
                # Drop oldest if over capacity
                while len(memory_bank) > args.memory_size:
                    memory_bank.tokens.pop(0)
                    memory_bank.memories.pop(0)
                    memory_bank.summaries.pop(0)

            # Anki: randomly replay a past batch, retrieve relevant memories
            if use_memory and len(memory_bank) >= args.memory_k + 1:
                idx = torch.randint(0, len(memory_bank), ()).item()
                batch = memory_bank.tokens[idx].to(device)
                perturbed_batch = None
                # Route to find relevant memories (target's own memory will score high)
                with torch.no_grad():
                    query = model.input.input_emb(batch.clamp(0, config.vocab_size - 1))
                memories = memory_bank.retrieve(query, args.memory_k)
            else:
                memories = None

            # Run supervision steps
            z = make_z(batch_size, seq_len, config.d_model, device=device)
            session_loss = 0.0
            for _ in range(supervision):
                loss, z = train_step(z, batch, perturbed_batch, memories=memories)
                session_loss += loss

            avg_session = session_loss / supervision
            running_loss += avg_session
            global_step += 1

            loss_fd.write(f"{global_step} {avg_session}\n")
            loss_fd.flush()

            if math.isnan(avg_session):
                print("Training failed — NaN loss")
                sys.exit(1)

            if global_step % report_every == 0:
                avg = running_loss / report_every
                mem_info = f" | mem {len(memory_bank)}" if use_memory else ""
                print(f"  step {global_step:6d} | loss {avg:.4f} | lr {scheduler.get_last_lr()[0]:.2e}{mem_info}")
                running_loss = 0.0

            # Periodic memory refresh — re-encode with improved model
            if use_memory and global_step % args.memory_refresh == 0 and len(memory_bank) > 0:
                model.eval()
                memory_bank.refresh(model, device)
                model.train()

            if global_step % save_every == 0:
                wiki_state = None
                for p in programs:
                    if isinstance(p, WikipediaProgram):
                        wiki_state = p.wiki_state()
                elapsed = datetime.now() - start_time
                log = "\n".join(training_log_lines)
                log += f"\nsaved at step {global_step}, elapsed {elapsed}"
                save_checkpoint(ckpt_path, model, config, log, wiki_state)

            scheduler.step()
            program_idx = (program_idx + 1) % len(programs)

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        loss_fd.close()
        # Final save
        elapsed = datetime.now() - start_time
        training_log_lines.append(f"finished: {datetime.now().isoformat()}")
        training_log_lines.append(f"elapsed: {elapsed}")
        training_log_lines.append(f"total steps: {global_step}")
        for p in programs:
            training_log_lines.append(f"final: {p.description()}")
        log = "\n".join(training_log_lines)

        wiki_state = None
        for p in programs:
            if isinstance(p, WikipediaProgram):
                wiki_state = p.wiki_state()

        save_checkpoint(ckpt_path, model, config, log, wiki_state)
        print(f"Done. {global_step} steps in {elapsed}.")


def cmd_eval(args):
    cwd    = Path.cwd()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path = cwd / args.checkpoint
    config = load_config(ckpt_path)
    if config is None:
        print(f"No config.json found in {ckpt_path}, using defaults.")
        config = DEFAULT_CONFIG

    seq_len = args.seq_len or config.context_length
    steps   = args.steps

    graph   = AbsorbingGraph(VOCAB_SIZE)
    noise   = LogLinearNoise()
    sampler = Sampler()

    print("Initialising model...")
    model = SRLM(config).to(device)
    if not load_checkpoint(model, ckpt_path):
        print(f"No checkpoint found at {ckpt_path}")
        sys.exit(1)
    model = torch.compile(model)
    model.eval()

    score_fn = make_score_fn(model, device)

    while True:
        query = from_text(input("> "))
        def projector(x, q=query):
            if q is None:
                return x
            q = q.to(x.device)
            x[:, :len(q)] = q
            return x
        z = make_z(1, seq_len, config.d_model, device=device)
        z, outputs = sampler.sample(
            score_fn, z, graph, noise,
            tokenizer=as_text,
            batch_size=1,
            batch_len=seq_len,
            steps=steps,
            projector=projector,
            device=device,
        )
        print(repr(outputs))

# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SRLM training and evaluation")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- train ---
    p_train = sub.add_parser("train", help="Train the model")
    p_train.add_argument("checkpoint", help="Checkpoint directory name")
    p_train.add_argument("--kalevala", type=int, nargs="?", const=0, default=None,
                         help="Enable Kalevala training (0 or omit count = indefinite)")
    p_train.add_argument("--wikipedia", type=int, nargs="?", const=0, default=None,
                         help="Enable Wikipedia training (0 or omit count = one epoch)")
    p_train.add_argument("--arithmetic", type=int, nargs="?", const=0, default=None,
                         help="Enable arithmetic N+M=Y training (0 or omit count = indefinite)")
    p_train.add_argument("--qa", type=int, nargs="?", const=0, default=None,
                         help="Enable QA training (0 or omit count = indefinite)")
    p_train.add_argument("--qa-file", type=str, default="../../data/finnish_qa_663.jsonl",
                         help="Path to JSONL file with question/answer fields")
    # Model config presets and overrides
    p_train.add_argument("--medium", action="store_true",
                         help="Use medium config (~200M params, dropout=0.2)")
    p_train.add_argument("--d-model", type=int, default=None, dest="d_model",
                         help="Override d_model")
    p_train.add_argument("--n-priors", type=int, default=None, dest="n_priors",
                         help="Override number of prior layers")
    p_train.add_argument("--n-posteriors", type=int, default=None, dest="n_posteriors",
                         help="Override number of posterior layers")
    p_train.add_argument("--n-heads", type=int, default=None, dest="n_heads",
                         help="Override number of attention heads")
    p_train.add_argument("--dropout", type=float, default=None,
                         help="Override dropout rate")
    p_train.add_argument("--N", type=int, default=None,
                         help="Override HRM N (outer iterations)")
    p_train.add_argument("--T", type=int, default=None,
                         help="Override HRM T (inner iterations per slow step)")
    p_train.add_argument("--save-every", type=int, default=1000,
                         help="Save checkpoint every N global steps (default: 1000)")
    p_train.add_argument("--report-every", type=int, default=10,
                         help="Report loss every N steps (default: 10)")
    p_train.add_argument("--supervision", type=int, default=5,
                         help="Supervision steps per training step (default: 5)")
    p_train.add_argument("--batch-size", type=int, default=32,
                         help="Batch size (default: 32)")
    p_train.add_argument("--seq-len", type=int, default=None,
                         help="Sequence length (default: from config)")
    p_train.add_argument("--lr", type=float, default=1e-4,
                         help="Learning rate (default: 1e-4)")
    p_train.add_argument("--warmup-steps", type=int, default=100, dest="warmup_steps",
                         help="Linear LR warmup steps (default: 100)")
    p_train.add_argument("--memory-size", type=int, default=0, dest="memory_size",
                         help="Rolling memory bank size (0 = disabled, default: 0)")
    p_train.add_argument("--memory-k", type=int, default=2, dest="memory_k",
                         help="Number of memories to retrieve per step (default: 2)")
    p_train.add_argument("--memory-refresh", type=int, default=100, dest="memory_refresh",
                         help="Re-encode memory bank every N steps (default: 100)")

    # --- eval ---
    p_eval = sub.add_parser("eval", help="Interactive evaluation")
    p_eval.add_argument("checkpoint", help="Checkpoint directory name")
    p_eval.add_argument("--steps", type=int, default=10,
                        help="Sampling steps (default: 10)")
    p_eval.add_argument("--seq-len", type=int, default=None,
                        help="Sequence length (default: from config)")

    args = parser.parse_args()
    if args.command == "train":
        cmd_train(args)
    elif args.command == "eval":
        cmd_eval(args)

if __name__ == "__main__":
    main()
