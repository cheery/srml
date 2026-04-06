"""
main.py — Training and inference for SRLM v2.

SRLM = MDLM denoiser + G-Mem + gated PonderBlock.

Usage:
    python main.py train <checkpoint> [--kalevala STEPS] [--wikipedia STEPS]
                                      [--sudoku STEPS] [--arithmetic STEPS]
                                      [--qa STEPS --qa-file PATH]
                                      [--memory-size N] [--memory-alternate N]
    python main.py eval <checkpoint>  [--steps N]
"""

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model.model import (
    SRLMConfig, GMemConfig, PonderConfig,
    SRLMDenoiser, SRLMEnergyModel,
    mdlm_loss, nce_loss, sample, PonderTrainer,
)
from model.gmem import MemoryLoss
from model.edlm import LogLinearSchedule, Sampler
from model.ema import EMA
from model.grpo import grpo_step, arithmetic_reward, sudoku_reward
from wiki_data import WikiDataLoader

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VOCAB_SIZE = 256
MASK_TOKEN = VOCAB_SIZE  # 256 = absorbing state

PARQUET_FILES = [
    "../../data/train-00000-of-00002.parquet",
    "../../data/train-00001-of-00002.parquet",
]

DEFAULT_CONFIG = SRLMConfig(
    gmem   = GMemConfig(memory_dim=384, num_slots=64),
    ponder = PonderConfig(N_H=2, N_L=4),
    hidden_dim = 384,
    num_heads = 12,
    front_layers = 3,
    back_layers = 3,

)

MEDIUM_CONFIG = SRLMConfig(
    gmem              = GMemConfig(memory_dim=384, num_slots=1024),
    ponder            = PonderConfig(N_H=2, N_L=4),
    hidden_dim        = 384,
    num_heads         = 12,
    front_layers      = 3,
    back_layers       = 3,
    max_context_length = 256,
)

LARGE_CONFIG = SRLMConfig(
    gmem              = GMemConfig(memory_dim=512, num_slots=2048),
    ponder            = PonderConfig(N_H=2, N_L=4),
    hidden_dim        = 768,
    num_heads         = 12,
    mlp_ratio         = 4,
    front_layers      = 4,
    back_layers       = 4,
    max_context_length = 256,
    dropout           = 0.1,
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
# Checkpoint I/O
# ---------------------------------------------------------------------------

def save_checkpoint(ckpt_dir: Path, model: nn.Module, config: SRLMConfig,
                    training_log: str, wiki_state: dict | None = None,
                    ema=None):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "parameters.pt")
    if ema is not None:
        torch.save(ema.state_dict(), ckpt_dir / "ema.pt")
    with open(ckpt_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)
    with open(ckpt_dir / "training.txt", "w") as f:
        f.write(training_log)
    if wiki_state is not None:
        with open(ckpt_dir / "wiki.json", "w") as f:
            json.dump(wiki_state, f)
    print(f"  Saved: {ckpt_dir}")


def load_checkpoint(model: nn.Module, ckpt_dir: Path) -> bool:
    params = ckpt_dir / "parameters.pt"
    if not params.exists():
        return False
    state = torch.load(params, map_location="cpu")
    prefix = "_orig_mod."
    if any(k.startswith(prefix) for k in state):
        state = {k[len(prefix):]: v for k, v in state.items()}
    model.load_state_dict(state)
    print(f"  Loaded: {params}")
    return True


def load_config(ckpt_dir: Path) -> SRLMConfig | None:
    cfg_path = ckpt_dir / "config.json"
    if not cfg_path.exists():
        return None
    with open(cfg_path) as f:
        d = json.load(f)
    gmem = GMemConfig(**d.pop("gmem"))
    ponder = PonderConfig(**d.pop("ponder"))
    return SRLMConfig(gmem=gmem, ponder=ponder, **d)


# ---------------------------------------------------------------------------
# Memory Replay Buffer (Anki-style, backed by G-Mem)
# ---------------------------------------------------------------------------

class MemoryReplayBuffer:
    """
    Stores recent (tokens, G-Mem state) pairs for memory-jogging replay.

    The idea: after the model processes a batch, we snapshot the G-Mem
    state. Later, we replay that batch with the stored memory context.
    The model should reconstruct the old content better because memory
    carries information from when it first saw it.

    This teaches the model to actually use G-Mem for cross-segment
    coherence rather than ignoring it.
    """

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.entries = []  # list of (tokens_cpu, memory_cpu, answer_mask_cpu_or_None)

    def __len__(self):
        return len(self.entries)

    def store(self, tokens, memory, answer_mask=None):
        """Store a training batch + its G-Mem snapshot."""
        entry = (
            tokens.detach().cpu(),
            memory.detach().cpu(),
            answer_mask.detach().cpu() if answer_mask is not None else None,
        )
        self.entries.append(entry)
        while len(self.entries) > self.max_size:
            self.entries.pop(0)

    def sample(self, device):
        """Randomly retrieve a stored batch and its memory context."""
        idx = torch.randint(0, len(self.entries), ()).item()
        tokens, memory, answer_mask = self.entries[idx]
        return (
            tokens.to(device),
            memory.to(device),
            answer_mask.to(device) if answer_mask is not None else None,
        )

    @torch.no_grad()
    def refresh(self, denoiser, device):
        """Re-encode all stored data with current model weights.

        Keeps stored memories aligned with the evolving model so
        old snapshots don't become stale / incompatible.
        """
        for i, (tokens, _, answer_mask) in enumerate(self.entries):
            x0 = tokens.to(device)
            t_zero = torch.zeros(x0.shape[0], device=device)
            h, c, p_emb = denoiser.input(x0, t_zero)
            for layer in denoiser.front_layers:
                h = layer(h, c, p_emb)
            memory = denoiser.init_memory(x0.shape[0], device)
            _, memory, _ = denoiser.latent_memory(h, memory)
            self.entries[i] = (tokens, memory.detach().cpu(), answer_mask)


# ---------------------------------------------------------------------------
# Training programs
# ---------------------------------------------------------------------------

class KalevalaProgram:
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
        return self.max_steps is not None and self.step_count >= self.max_steps

    def next_batch(self):
        if self.done():
            return None
        self.step_count += 1
        return sample_batch_random(self.raw, self.seq_len, self.batch_size), None

    def description(self):
        return f"kalevala({self.max_steps or 'inf'}, done={self.step_count})"


class WikipediaProgram:
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
        return f"wikipedia({self.max_steps or 'epoch'}, done={self.step_count})"


class ArithmeticProgram:
    """N+M=Y facts. Question shown, answer masked.

    Returns 3-tuple: (full_tokens, answer_mask, prompt_tokens)
      - full_tokens: "N+M=Y" (denoiser target)
      - answer_mask: True at answer digit positions
      - prompt_tokens: "N+M=" only (ponder input)
    """

    def __init__(self, seq_len, batch_size, max_steps=None, max_operand=99):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.max_operand = max_operand
        self.step_count = 0

    def done(self):
        return self.max_steps is not None and self.step_count >= self.max_steps

    def next_batch(self):
        if self.done():
            return None
        batch = torch.full((self.batch_size, self.seq_len), ord(' '), dtype=torch.int32)
        prompt_batch = torch.full((self.batch_size, self.seq_len), ord(' '), dtype=torch.int32)
        answer_mask = torch.zeros(self.batch_size, self.seq_len, dtype=torch.bool)
        for i in range(self.batch_size):
            n = torch.randint(0, self.max_operand + 1, ()).item()
            m = torch.randint(0, self.max_operand + 1, ()).item()
            #n,m = 1,20
            prompt = f"{n}+{m}="
            answer = f"{n+m}"
            full = prompt + answer
            full_tok = torch.frombuffer(bytearray(full.encode()), dtype=torch.uint8).to(torch.int32)
            prompt_tok = torch.frombuffer(bytearray(prompt.encode()), dtype=torch.uint8).to(torch.int32)
            L = min(len(full_tok), self.seq_len)
            Lp = min(len(prompt_tok), self.seq_len)
            batch[i, :L] = full_tok[:L]
            prompt_batch[i, :Lp] = prompt_tok[:Lp]
            answer_mask[i, Lp:self.seq_len] = True
        self.step_count += 1
        return batch, answer_mask

    def description(self):
        return f"arithmetic({self.max_steps or 'inf'}, done={self.step_count})"


class SudokuProgram:
    """Sudoku puzzles. Clue positions shown, blank positions masked.

    Format: 9 rows of 9 digits separated by newlines (89 bytes).
    answer_mask is True where puzzle has '0' (blanks to predict).

    Returns 3-tuple: (solution_tokens, answer_mask, puzzle_tokens)
      - solution_tokens: full solution (denoiser target)
      - answer_mask: True at blank positions
      - puzzle_tokens: puzzle with '_' for blanks (ponder input)
    """

    def __init__(self, seq_len, batch_size, max_steps=None,
                 parquet_path="../../data/valid_0.parquet"):
        import pandas as pd
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.step_count = 0
        df = pd.read_parquet(parquet_path)
        self.puzzles = df["puzzle"].values
        self.solutions = df["solution"].values
        print(f"Sudoku: {len(self.puzzles)} puzzles loaded")

    def _format_grid(self, digits_81):
        rows = [digits_81[i:i+9] for i in range(0, 81, 9)]
        return "\n".join(rows)

    def _format_puzzle(self, digits_81):
        """Format puzzle with '_' replacing '0' blanks."""
        return self._format_grid(digits_81.replace('0', '_'))

    def done(self):
        return self.max_steps is not None and self.step_count >= self.max_steps

    def next_batch(self):
        if self.done():
            return None

        batch = torch.full((self.batch_size, self.seq_len), ord(' '), dtype=torch.int32)
        puzzle_batch = torch.full((self.batch_size, self.seq_len), ord(' '), dtype=torch.int32)
        answer_mask = torch.zeros(self.batch_size, self.seq_len, dtype=torch.bool)

        for i in range(self.batch_size):
            idx = torch.randint(0, len(self.puzzles), ()).item()
            solution_grid = self._format_grid(self.solutions[idx])
            puzzle_grid   = self._format_puzzle(self.puzzles[idx])

            sol_tokens = torch.frombuffer(
                bytearray(solution_grid.encode()), dtype=torch.uint8).to(torch.int32)
            puz_tokens = torch.frombuffer(
                bytearray(puzzle_grid.encode()), dtype=torch.uint8).to(torch.int32)

            L = min(len(sol_tokens), self.seq_len)
            batch[i, :L] = sol_tokens[:L]
            puzzle_batch[i, :L] = puz_tokens[:L]
            # Mask where puzzle has '_' (was '0')
            for j in range(L):
                if puz_tokens[j] == ord('_'):
                    answer_mask[i, j] = True

        self.step_count += 1
        return batch, answer_mask, puzzle_batch

    def description(self):
        return f"sudoku({self.max_steps or 'inf'}, done={self.step_count})"


class QAProgram:
    """Question-answer pairs from JSONL.

    Returns (answer_tokens, answer_mask, question_tokens):
      - answer_tokens: full sequence with answer filled in (denoiser target)
      - answer_mask: True at answer positions
      - question_tokens: question only, padded (ponder input)
    """

    def __init__(self, path, seq_len, batch_size, max_steps=None):
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
        return self.max_steps is not None and self.step_count >= self.max_steps

    def next_batch(self):
        if self.done():
            return None

        batch = torch.full((self.batch_size, self.seq_len), ord(' '), dtype=torch.int32)
        question_batch = torch.full((self.batch_size, self.seq_len), ord(' '), dtype=torch.int32)
        answer_mask = torch.zeros(self.batch_size, self.seq_len, dtype=torch.bool)

        indices = torch.randint(0, len(self.pairs), (self.batch_size,))
        for i, idx in enumerate(indices.tolist()):
            question, answer = self.pairs[idx]
            prefix = question + " "
            full   = prefix + answer
            prefix_tok = torch.frombuffer(
                bytearray(prefix.encode()), dtype=torch.uint8).to(torch.int32)
            full_tok = torch.frombuffer(
                bytearray(full.encode()), dtype=torch.uint8).to(torch.int32)
            Lf = min(len(full_tok), self.seq_len)
            Lp = min(len(prefix_tok), self.seq_len)
            batch[i, :Lf] = full_tok[:Lf]
            question_batch[i, :Lp] = prefix_tok[:Lp]
            answer_mask[i, Lp:Lf] = True

        self.step_count += 1
        return batch, answer_mask, question_batch

    def description(self):
        return f"qa({self.max_steps or 'inf'}, done={self.step_count})"


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_train(args):
    cwd    = Path.cwd()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Config: checkpoint > preset > default
    ckpt_path = cwd / args.checkpoint
    config = load_config(ckpt_path)
    if config is None:
        if args.large:
            config = LARGE_CONFIG
        elif args.medium:
            config = MEDIUM_CONFIG
        else:
            config = DEFAULT_CONFIG

    # CLI overrides
    if args.seq_len:
        config.max_context_length = args.seq_len
    if args.hidden_dim:
        config.hidden_dim = args.hidden_dim
        config.gmem.memory_dim = args.hidden_dim
    if args.num_heads:
        config.num_heads = args.num_heads
    if args.num_slots:
        config.gmem.num_slots = args.num_slots

    seq_len    = config.max_context_length
    batch_size = args.batch_size
    schedule   = LogLinearSchedule(eps=1e-3)

    # Model
    print("Initialising model...")
    denoiser = SRLMDenoiser(config).to(device)
    load_checkpoint(denoiser, ckpt_path)

    ref_denoiser = SRLMDenoiser(config).to(device)
    load_checkpoint(ref_denoiser, ckpt_path)
    ref_denoiser.eval()
    for p in ref_denoiser.parameters():
        p.requires_grad_(False)
    print(f"Denoiser params:  {param_count(denoiser):,}")
    print(f"Denoiser memory:  {param_memory_mb(denoiser):.1f} MB")

    # Optimizer covers both denoiser and ponder
    optimizer = optim.AdamW(denoiser.parameters(), lr=args.lr, weight_decay=0.01)
    grpo_optimizer = optim.AdamW(denoiser.parameters(), lr=1e-6)#, weight_decay=0.01)
    ema = EMA(denoiser, mu=0.999)
    mem_loss_fn = MemoryLoss()

    ema_path = ckpt_path / "ema.pt"
    if ema_path.exists():
        ema.load_state_dict(torch.load(ema_path, map_location=device, weights_only=True))
        print(f"Loaded EMA from {ema_path}")

    # Ponder trainer
    ponder_trainer = PonderTrainer(
        denoiser=denoiser,
        schedule=schedule,
        N_super=args.supervision,
    )

    # Memory replay buffer (Anki-style)
    use_memory = args.memory_size > 0
    memory_alternate = args.memory_alternate
    replay_buffer = MemoryReplayBuffer(max_size=args.memory_size) if use_memory else None
    if use_memory:
        if memory_alternate > 0:
            print(f"Memory replay: size={args.memory_size}, "
                  f"alternate every {memory_alternate}, "
                  f"refresh every {args.memory_refresh}")
        else:
            print(f"Memory replay: size={args.memory_size}, "
                  f"refresh every {args.memory_refresh}")

    # LR schedule
    warmup_steps = args.warmup_steps
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.save_every, 1000), eta_min=1e-6
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

    # Training programs
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
    if args.sudoku is not None:
        max_steps = None if args.sudoku == 0 else args.sudoku
        programs.append(SudokuProgram(seq_len, batch_size, max_steps))
    if args.qa is not None:
        max_steps = None if args.qa == 0 else args.qa
        programs.append(QAProgram(args.qa_file, seq_len, batch_size, max_steps))
    if not programs:
        print("No training programs specified. Use --kalevala and/or --wikipedia.")
        sys.exit(1)

    # Training log
    training_log_lines = []
    prev_log = ckpt_path / "training.txt"
    if prev_log.exists():
        training_log_lines.append(prev_log.read_text().rstrip())
        training_log_lines.append("")
    start_time = datetime.now()
    training_log_lines.append(f"started: {start_time.isoformat()}")
    for p in programs:
        training_log_lines.append(f"program: {p.description()}")

    ckpt_path.mkdir(parents=True, exist_ok=True)
    loss_fd = open(ckpt_path / "loss.txt", "w")

    print(f"Training | programs: {[p.description() for p in programs]}")
    print("-" * 55)

    global_step = 0
    running_loss = 0.0
    program_idx = 0
    ema_loss = None
    memory = denoiser.init_memory(batch_size, device)

    try:
        while True:
            if all(p.done() for p in programs):
                break

            # Pick next active program
            attempts = 0
            while programs[program_idx].done():
                program_idx = (program_idx + 1) % len(programs)
                attempts += 1
                if attempts > len(programs):
                    break
            if attempts > len(programs):
                break

            program = programs[program_idx]
            result = program.next_batch()
            if result is None:
                program_idx = (program_idx + 1) % len(programs)
                continue

            # Unpack: QAProgram returns 3-tuple, others return 2-tuple
            ponder_x0 = None
            if len(result) == 3:
                batch, answer_mask, ponder_x0 = result
            else:
                batch, answer_mask = result

            x0 = batch.to(device)
            if answer_mask is not None:
                answer_mask = answer_mask.to(device)
            if ponder_x0 is not None:
                ponder_x0 = ponder_x0.to(device)

            is_reasoning = isinstance(program, (SudokuProgram, ArithmeticProgram, QAProgram))

            # Determine if this is a study or practice step
            if use_memory and memory_alternate > 0:
                memory_active = ((global_step // memory_alternate) % 2) == 0
            else:
                memory_active = False  # no alternation = no replay

            ponder_info = ""

            # --- GRPO reinforcement (always) ---
            if args.grpo_every == 1:
                if isinstance(program, ArithmeticProgram):
                    reward_fn = arithmetic_reward
                else:
                    reward_fn = sudoku_reward
                # Build prompt: mask answer positions
                prompt = x0.clone()
                if answer_mask is not None:
                    prompt[answer_mask] = MASK_TOKEN
                grpo_loss, memory, grpo_metrics = grpo_step(
                        denoiser,
                        ref_denoiser,
                        grpo_optimizer, schedule,
                        prompt_batch=prompt,
                        clean_batch=x0,
                        reward_fn=reward_fn,
                        device=device,
                        memory=memory)
                        
                #grpo_loss, memory, grpo_metrics = grpo_step(
                #    denoiser, grpo_optimizer, schedule,
                #    prompt_batch=prompt,
                #    clean_batch=x0,
                #    reward_fn=reward_fn,
                #    device=device,
                #    memory=memory,
                #    answer_mask=answer_mask,
                #    K=args.grpo_k,
                #    sampling_steps=args.grpo_steps,
                #    beta_dgpo=args.grpo_beta,
                #    #clip_range=args.grpo_clip,
                #    verbose=(global_step % args.report_every == 0),
                #)
                ema.update(denoiser)
                avg_loss = grpo_loss
                ponder_info += (f" | grpo={grpo_loss:.4f}"
                                f" r={grpo_metrics['mean_reward']:.3f}")

            elif is_reasoning:
                # --- Reasoning tasks: always use ponder deep supervision ---
                denoiser.train()
                optimizer.zero_grad()
                ponder_losses, memory = ponder_trainer.train_step(
                    x0, memory=memory, answer_mask=answer_mask,
                )
                # --- Gradient flow diagnostic ---
                if global_step % args.report_every == 0:
                    groups = {
                        "ponder.block": denoiser.ponder.block,
                        "ponder.q_head": denoiser.ponder.q_head,
                        "front_layers": denoiser.front_layers,
                        "back_layers": denoiser.back_layers,
                        "latent_memory": denoiser.latent_memory,
                        "latent_memory_in": denoiser.latent_memory_in,
                        "out_proj": denoiser.out_proj,
                    }
                    grad_parts = []
                    for name, mod in groups.items():
                        grads = [p.grad for p in mod.parameters() if p.grad is not None]
                        if grads:
                            norm = torch.cat([g.flatten() for g in grads]).norm().item()
                        else:
                            norm = 0.0
                        grad_parts.append(f"{name}={norm:.4f}")
                    print(f"    grad norms: {' | '.join(grad_parts)}")

                torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                ema.update(denoiser)
                avg_loss = ponder_losses['LM']
                ponder_info = (f" | ponder LM={ponder_losses['LM']:.4f}"
                               f" BCE={ponder_losses['BCE']:.4f}")

                # Store in replay buffer
                if replay_buffer is not None and memory is not None:
                    replay_buffer.store(x0, memory, answer_mask)

                # --- GRPO reinforcement (interleaved) ---
                if (args.grpo_every > 0
                        and global_step % args.grpo_every == 0
                        and global_step > 0):
                    if isinstance(program, ArithmeticProgram):
                        reward_fn = arithmetic_reward
                    else:
                        reward_fn = sudoku_reward
                    # Build prompt: mask answer positions
                    prompt = x0.clone()
                    if answer_mask is not None:
                        prompt[answer_mask] = MASK_TOKEN
                    grpo_loss, memory, grpo_metrics = grpo_step(
                        denoiser,
                        ref_denoiser,
                        grpo_optimizer, schedule,
                        prompt_batch=prompt,
                        clean_batch=x0,
                        reward_fn=reward_fn,
                        device=device,
                        memory=memory,
                        answer_mask=answer_mask,
                        K=args.grpo_k,
                        sampling_steps=args.grpo_steps,
                        beta_dgpo=args.grpo_beta,
                        #clip_range=args.grpo_clip,
                        verbose=(global_step % args.report_every == 0),
                    )
                    ema.update(denoiser)
                    ponder_info += (f" | grpo={grpo_loss:.4f}"
                                    f" r={grpo_metrics['mean_reward']:.3f}")

            # --- Study phase: train on fresh data, store in replay buffer ---
            elif not memory_active or replay_buffer is None or len(replay_buffer) == 0:
                denoiser.train()
                optimizer.zero_grad()

                loss, memory, importance_scores = mdlm_loss(
                    denoiser, x0, schedule, memory,
                    answer_mask=answer_mask,
                )
                loss += mem_loss_fn(importance_scores)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                ema.update(denoiser)

                if memory is not None:
                    memory = memory.detach()

                # Store in replay buffer
                if replay_buffer is not None and memory is not None:
                    replay_buffer.store(x0, memory, answer_mask)

                avg_loss = loss.item()

                # Ponder step (interleaved for non-reasoning tasks)
                if args.ponder_every > 0 and global_step % args.ponder_every == 0:
                    denoiser.train()
                    optimizer.zero_grad()
                    ponder_losses, memory = ponder_trainer.train_step(
                        x0, memory=memory, answer_mask=answer_mask,
                    )
                    torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
                    optimizer.step()
                    ponder_info = f" | ponder LM={ponder_losses['LM']:.4f}"
                    scheduler.step()

            # --- Practice phase: replay old data with stored memory context ---
            else:
                replay_x0, replay_mem, replay_mask = replay_buffer.sample(device)
                denoiser.train()
                optimizer.zero_grad()

                loss, memory, importance_scores = mdlm_loss(
                    denoiser, replay_x0, schedule, replay_mem,
                    answer_mask=replay_mask,
                )
                loss += mem_loss_fn(importance_scores)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                ema.update(denoiser)

                if memory is not None:
                    memory = memory.detach()

                avg_loss = loss.item()

            running_loss += avg_loss
            global_step += 1

            loss_fd.write(f"{global_step} {avg_loss}\n")
            loss_fd.flush()

            if math.isnan(avg_loss):
                print("Training failed -- NaN loss")
                sys.exit(1)

            # Periodic memory refresh
            if (replay_buffer is not None
                    and global_step % args.memory_refresh == 0
                    and len(replay_buffer) > 0):
                denoiser.eval()
                replay_buffer.refresh(denoiser, device)
                denoiser.train()

            if global_step % args.report_every == 0:
                avg = running_loss / args.report_every
                ema_loss = avg if ema_loss is None else 0.99 * ema_loss + 0.01 * avg
                prog_name = program.description().split("(")[0]
                lr = scheduler.get_last_lr()[0]
                phase = ""
                if use_memory:
                    if memory_active and len(replay_buffer) > 0:
                        phase = f" | practice ({len(replay_buffer)})"
                    else:
                        phase = f" | study ({len(replay_buffer) if replay_buffer else 0})"
                print(f"  step {global_step:6d} | loss {avg:.4f} | ema {ema_loss:.4f}"
                      f" | {prog_name} | lr {lr:.2e}{phase}{ponder_info}")
                running_loss = 0.0

            if global_step % args.save_every == 0:
                wiki_state = None
                for p in programs:
                    if isinstance(p, WikipediaProgram):
                        wiki_state = p.wiki_state()
                log = "\n".join(training_log_lines)
                log += f"\nsaved at step {global_step}"
                save_checkpoint(ckpt_path, denoiser, config, log, wiki_state, ema=ema)
                # Update reference model for GRPO (paper Algorithm 1, line 3)
                ref_denoiser.load_state_dict(denoiser.state_dict())
                ref_denoiser.eval()

            if global_step % args.sample_every == 0:
                denoiser.eval()
                ema.apply(denoiser)
                s, _ = sample(denoiser, schedule, batch_size=2, seq_len=seq_len,
                              num_steps=64, device=device)
                for i in range(min(2, s.shape[0])):
                    print(f"    sample {i+1}: {repr(as_text(s[i][:80]))}")
                ema.restore(denoiser)

            program_idx = (program_idx + 1) % len(programs)

    except KeyboardInterrupt:
        print("\nInterrupted.")

    loss_fd.close()

    # Final save
    elapsed = datetime.now() - start_time
    training_log_lines.append(f"finished: {datetime.now().isoformat()}")
    training_log_lines.append(f"total steps: {global_step}, elapsed: {elapsed}")
    log = "\n".join(training_log_lines)

    wiki_state = None
    for p in programs:
        if isinstance(p, WikipediaProgram):
            wiki_state = p.wiki_state()
    save_checkpoint(ckpt_path, denoiser, config, log, wiki_state, ema=ema)
    print(f"Done. {global_step} steps in {elapsed}.")


def cmd_eval(args):
    cwd    = Path.cwd()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path = cwd / args.checkpoint
    config = load_config(ckpt_path)
    if config is None:
        print(f"No config.json in {ckpt_path}, using defaults.")
        config = DEFAULT_CONFIG

    seq_len = args.seq_len or config.max_context_length
    schedule = LogLinearSchedule(eps=1e-3)

    print("Initialising model...")
    denoiser = SRLMDenoiser(config).to(device)
    if not load_checkpoint(denoiser, ckpt_path):
        print(f"No checkpoint found at {ckpt_path}")
        sys.exit(1)

    if args.ema:
        ema_path = ckpt_path / "ema.pt"
        if ema_path.exists():
            ema = EMA(denoiser, mu=0.999)
            ema.load_state_dict(torch.load(ema_path, map_location=device, weights_only=True))
            ema.apply(denoiser)
            print("Using EMA weights")
        else:
            print("Warning: --ema requested but no ema.pt found")

    denoiser.eval()
    print(f"Denoiser params: {param_count(denoiser):,}")

    # Session-persistent memory
    memory = denoiser.init_memory(1, device)
    n_ponder = args.n_ponder
    puzzles = None
    solutions = None

    print(f"Memory persists across the session ({config.gmem.num_slots} slots).")
    print(f"Commands: !reset  !sudoku  !ponder N  (empty line = free generate)")
    print()

    while True:
        try:
            query_text = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if query_text.strip() == "!reset":
            memory = denoiser.init_memory(1, device)
            print("Memory reset.")
            continue

        if query_text.strip().startswith("!ponder"):
            parts = query_text.strip().split()
            if len(parts) >= 2:
                n_ponder = int(parts[1])
            print(f"Ponder iterations: {n_ponder}")
            continue

        if query_text.strip() == "!sudoku":
            if puzzles is None:
                import pandas as pd
                parquet_path = Path.cwd() / "../../data/valid_0.parquet"
                df = pd.read_parquet(parquet_path)
                puzzles   = df["puzzle"].values
                solutions = df["solution"].values
                print(f"Loaded {len(puzzles)} puzzles")

            def format_grid(digits_81):
                return "\n".join(digits_81[i:i+9] for i in range(0, 81, 9))

            idx = torch.randint(0, len(puzzles), ()).item()
            puzzle = puzzles[idx]
            solution = solutions[idx]
            missing = puzzle.count('0')

            puzzle_grid   = format_grid(puzzle)
            solution_grid = format_grid(solution)

            print(f"\nPuzzle #{idx} ({missing} blanks):")
            print(puzzle_grid.replace('0', '.'))
            print()

            # Ponder reads the puzzle into memory
            puz_tokens = torch.frombuffer(
                bytearray(puzzle_grid.encode()), dtype=torch.uint8).to(torch.int32)
            sol_tokens = torch.frombuffer(
                bytearray(solution_grid.encode()), dtype=torch.uint8).to(torch.int32)

            ponder_input = torch.full((1, seq_len), ord(' '), dtype=torch.int32, device=device)
            p_len = min(len(puz_tokens), seq_len)
            ponder_input[0, :p_len] = puz_tokens[:p_len].to(device)

            #with torch.no_grad():
            #    h_p, p_emb, _ = ponder_model.get_front(ponder_input, memory)
            #    z_H, z_L = ponder_model.init_states(h_p)
            #    for _ in range(n_ponder):
            #        z_H, z_L, _ = ponder_model.ponder(h_p, p_emb, z_H, z_L)
            #    memory = ponder_model(z_H, memory)

            # Denoiser generates with clue clamping
            clue_mask = (puz_tokens != ord('0'))
            clue_values = sol_tokens.clone()

            sampler_obj = Sampler(schedule, MASK_TOKEN, VOCAB_SIZE)
            xt, stepper = sampler_obj(1, seq_len, device, args.steps)
            q_len = min(len(clue_values), seq_len)
            cm = clue_mask[:q_len].to(device)
            cv = clue_values[:q_len].to(device)
            xt[0, :q_len] = torch.where(cm, cv, xt[0, :q_len])
            memory = denoiser.pioneer(xt, None, memory)
            with torch.no_grad():
                for step in stepper:
                    logits = denoiser(xt, step.t, memory)
                    x0 = step.propose_x0(xt, logits)
                    xt = step.reverse_step(xt, x0)
                    xt[0, :q_len] = torch.where(cm, cv, xt[0, :q_len])

            result_text = as_text(xt[0])[:89]
            result_digits = result_text.replace("\n", "")

            correct = 0
            total_blanks = 0
            for p_ch, s_ch, r_ch in zip(puzzle, solution, result_digits):
                if p_ch == '0':
                    total_blanks += 1
                    if r_ch == s_ch:
                        correct += 1

            print(f"Model output:")
            print(result_text)
            print(f"\nScore: {correct}/{total_blanks} blanks correct "
                  f"({100*correct/max(total_blanks,1):.0f}%)")
            if correct < total_blanks:
                print(f"\nSolution:")
                print(solution_grid)
            print()
            continue

        query = from_text(query_text)

        if query is not None and len(query) > 0:
            # Step 1: Ponder reads user input into session memory
            #ponder_input = torch.full((1, seq_len), ord(' '), dtype=torch.int32, device=device)
            #ponder_input[0, :q_len] = query[:q_len].to(device)

            #with torch.no_grad():
            #    h_p, p_emb, _ = ponder_model.get_front(ponder_input, memory)
            #    z_H, z_L = ponder_model.init_states(h_p)
            #    for _ in range(n_ponder):
            #        z_H, z_L, _ = ponder_model.ponder(h_p, p_emb, z_H, z_L)
            #    memory = ponder_model(z_H, memory)

            # Step 2: Denoiser generates from all-masked using enriched memory

            sampler_obj = Sampler(schedule, MASK_TOKEN, VOCAB_SIZE)
            xt, stepper = sampler_obj(1, seq_len, device, args.steps)
            q_len = min(len(query), seq_len)
            xt[0, :q_len] = query[:q_len].to(device)
            memory = denoiser.pioneer(xt, None, memory)
            with torch.no_grad():
                for step in stepper:
                    logits = denoiser(xt, step.t, memory)
                    x0 = step.propose_x0(xt, logits)
                    xt = step.reverse_step(xt, x0)
            print(as_text(xt[0]))
        else:
            # Empty input: free generate from current memory state
            sampler_obj = Sampler(schedule, MASK_TOKEN, VOCAB_SIZE)
            xt, stepper = sampler_obj(1, seq_len, device, args.steps)
            memory = denoiser.pioneer(xt, None, memory)
            with torch.no_grad():
                for step in stepper:
                    logits = denoiser(xt, step.t, memory)
                    x0 = step.propose_x0(xt, logits)
                    xt = step.reverse_step(xt, x0)

            print(as_text(xt[0]))


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SRLM v2")
    parser.add_argument("--tf32", action="store_true",
                        help="Enable TF32 matmul precision")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- train ---
    p_train = sub.add_parser("train")
    p_train.add_argument("checkpoint")
    # Data programs
    p_train.add_argument("--kalevala", type=int, nargs="?", const=0, default=None)
    p_train.add_argument("--wikipedia", type=int, nargs="?", const=0, default=None)
    p_train.add_argument("--arithmetic", type=int, nargs="?", const=0, default=None)
    p_train.add_argument("--sudoku", type=int, nargs="?", const=0, default=None)
    p_train.add_argument("--qa", type=int, nargs="?", const=0, default=None)
    p_train.add_argument("--qa-file", type=str, default="../../data/finnish_qa_663.jsonl",
                         dest="qa_file")
    # Model config
    p_train.add_argument("--medium", action="store_true")
    p_train.add_argument("--large", action="store_true")
    p_train.add_argument("--hidden-dim", type=int, default=None, dest="hidden_dim")
    p_train.add_argument("--num-heads", type=int, default=None, dest="num_heads")
    p_train.add_argument("--num-slots", type=int, default=None, dest="num_slots")
    p_train.add_argument("--seq-len", type=int, default=None, dest="seq_len")
    # Training
    p_train.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    p_train.add_argument("--lr", type=float, default=3e-4)
    p_train.add_argument("--warmup-steps", type=int, default=100, dest="warmup_steps")
    p_train.add_argument("--save-every", type=int, default=1000, dest="save_every")
    p_train.add_argument("--report-every", type=int, default=10, dest="report_every")
    p_train.add_argument("--sample-every", type=int, default=500, dest="sample_every")
    # Ponder
    p_train.add_argument("--supervision", type=int, default=8,
                         help="Ponder supervision segments (default: 8)")
    p_train.add_argument("--ponder-every", type=int, default=0, dest="ponder_every",
                         help="Interleave ponder training every N steps (0=off)")
    # Memory replay (Anki-style)
    p_train.add_argument("--memory-size", type=int, default=0, dest="memory_size",
                         help="Replay buffer size (0=off, default: 0)")
    p_train.add_argument("--memory-alternate", type=int, default=50, dest="memory_alternate",
                         help="Alternate study/practice every N steps (default: 50)")
    p_train.add_argument("--memory-refresh", type=int, default=200, dest="memory_refresh",
                         help="Re-encode replay buffer every N steps (default: 200)")
    # GRPO (for reasoning tasks)
    p_train.add_argument("--grpo-every", type=int, default=0, dest="grpo_every",
                         help="GRPO reinforcement every N steps on reasoning tasks (0=off)")
    p_train.add_argument("--grpo-k", type=int, default=4, dest="grpo_k",
                         help="GRPO candidates per prompt (default: 4)")
    p_train.add_argument("--grpo-steps", type=int, default=32, dest="grpo_steps",
                         help="GRPO sampling steps (default: 32)")
    p_train.add_argument("--grpo-clip", type=float, default=0.05, dest="grpo_clip",
                         help="GRPO PPO-style clip range (default: 0.05, 0=disabled)")
    p_train.add_argument("--grpo-beta", type=float, default=100.0, dest="grpo_beta",
                         help="DGPO sigmoid temperature (default: 100)")
    # Program interleaving
    p_train.add_argument("--interleave", type=int, default=0,
                         help="Switch programs every N steps (0=sequential)")

    # --- eval ---
    p_eval = sub.add_parser("eval")
    p_eval.add_argument("checkpoint")
    p_eval.add_argument("--steps", type=int, default=64)
    p_eval.add_argument("--seq-len", type=int, default=None, dest="seq_len")
    p_eval.add_argument("--ema", action="store_true")
    p_eval.add_argument("--n-ponder", type=int, default=3, dest="n_ponder",
                         help="Ponder iterations per input (default: 3)")

    args = parser.parse_args()
    if args.tf32:
        torch.set_float32_matmul_precision('high')
        print("TF32 enabled")
    if args.command == "train":
        cmd_train(args)
    elif args.command == "eval":
        cmd_eval(args)


if __name__ == "__main__":
    main()
