"""
main.py — Training and inference entry point for SRLM in PyTorch.

Usage:
    python main.py <checkpoint_name> train
    python main.py <checkpoint_name> wikitrain
    python main.py <checkpoint_name> eval
    python main.py <checkpoint_name> wikidry
"""

import os
import sys
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from model import SRLM, SRLMConfig, make_z, AbsorbingGraph, LogLinearNoise, Sampler, loss_function
from wiki_data import WikiDataLoader

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VOCAB_SIZE  = 256
TOTAL_VOCAB = 257

B                  = 32
SEQ_LEN            = 32
N_STEPS            = 500
STEP_REPORT_EVERY  = 10
SUPERVISION        = 5

config = SRLMConfig(
    vocab_size   = TOTAL_VOCAB,
    context_length = SEQ_LEN,
    d_model      = 768,
    d_state      = 128,
    n_priors     = 3,
    n_posteriors = 2,
    n_heads      = 12
)

PARQUET_FILES = [
    "../../data/train-00000-of-00002.parquet",
    "../../data/train-00001-of-00002.parquet",
]

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def as_text(p: torch.Tensor) -> str:
    return p.cpu().to(torch.uint8).numpy().tobytes().decode("utf-8", errors="replace")

def from_text(text: str) -> torch.Tensor:
    data = text.encode("utf-8")
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).to(torch.int32)

def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def param_memory_mb(model: nn.Module) -> float:
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2

def sample_batch_random(raw: torch.Tensor, seq_len: int, batch: int) -> torch.Tensor:
    N = raw.shape[0]
    starts = torch.randint(0, N - seq_len, (batch,))
    return torch.stack([raw[s:s+seq_len] for s in starts])

def save_checkpoint(model: nn.Module, optimizer, path: Path):
    path.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, path / "checkpoint.pt")
    print(f"  Saved: {path}")

def load_checkpoint(model: nn.Module, optimizer, path: Path) -> bool:
    ckpt = path / "checkpoint.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state["model"])
        if optimizer is not None and "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        print(f"  Loaded: {ckpt}")
        return True
    return False

def training_data_kalevala(device):
    with open(cwd / "../../data/kalevala.plain.txt", "r", encoding="utf-8") as f:
        text = f.read().replace("\n", " ")
    raw = torch.frombuffer(bytearray(text.encode("utf-8")),
                           dtype=torch.uint8).to(torch.int32).to(device)
    print(f"Data: {raw.shape[0]:,} tavua")
    return raw

# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def make_train_step(model, optimizer, loss_fn, device):
    def train_step(z, batch):
        batch = batch.to(device)
        optimizer.zero_grad()
        loss, z = loss_fn(z, batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        return loss.item(), z
    return train_step

# ---------------------------------------------------------------------------
# Score function wrapper for sampler
# ---------------------------------------------------------------------------

def make_score_fn(model, device):
    @torch.no_grad()
    def score_fn(z, x, sigma):
        x = x.to(device)
        sigma = sigma.to(device)
        z = tuple(zi.to(device) for zi in z)
        z, log_score = model(z, x, sigma)
        return z, log_score
    return score_fn

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python main.py <checkpoint_name> <train|wikitrain|eval|wikidry>")
        sys.exit(1)

    checkpoint_name = sys.argv[1]
    command         = sys.argv[2]
    cwd             = Path.cwd()
    device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Setup
    graph   = AbsorbingGraph(VOCAB_SIZE)
    noise   = LogLinearNoise()
    sampler = Sampler()

    print("Initialising model...")
    model = SRLM(config).to(device)
    model = torch.compile(model)
    print(f"Parameters:      {param_count(model):,}")
    print(f"Parameter memory: {param_memory_mb(model):.1f} MB")


    LR        = 1e-4
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    loss_fn   = loss_function(model, graph, noise)

    # Try to load existing checkpoint
    ckpt_path = cwd / checkpoint_name
    if not load_checkpoint(model, optimizer, ckpt_path):
        save_checkpoint(model, optimizer, cwd / f"{checkpoint_name}_0")

    # lr scheduler — cosine decay
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-6)

    # -----------------------------------------------------------------------
    if command == "eval":
        model.eval()
        score_fn = make_score_fn(model, device)
        while True:
            query = from_text(input("> "))
            def projector(x):
                q = query.to(x.device)
                x[:, :len(q)] = q
                return x
            z = make_z(1, SEQ_LEN, config.d_model, device=device)
            z, outputs = sampler.sample(
                score_fn, z, graph, noise,
                tokenizer=as_text,
                batch_size=1,
                batch_len=SEQ_LEN,
                steps=10,
                projector=projector,
                device=device,
            )
            print(repr(outputs))

    # -----------------------------------------------------------------------
    elif command == "wikidry":
        loader = WikiDataLoader(
            parquet_files=[cwd / name for name in PARQUET_FILES],
            seq_len=SEQ_LEN,
            batch_size=B,
            seed=42,
        )
        loader.shuffle(0)
        i = 0
        while not loader.epoch_done():
            batch = loader.next_batch()
            if i % 10000 == 0:
                print(f"  step {i}: {batch.shape}")
            i += 1
        print(f"Total batches: {i}")

    # -----------------------------------------------------------------------
    elif command == "wikitrain":
        train_step = make_train_step(model, optimizer, loss_fn, device)
        loader = WikiDataLoader(
            parquet_files=PARQUET_FILES,
            seq_len=SEQ_LEN,
            batch_size=B,
            seed=42,
        )
        n_epochs    = 3
        loader_path = cwd / "wiki_loader.json"
        resuming    = loader.load_state(str(loader_path))
        start_epoch = loader._epoch if resuming else 0

        print("-" * 55)
        for epoch in range(start_epoch, n_epochs):
            if not resuming:
                loader.shuffle(epoch)
            resuming = False

            total_loss = 0.0
            while not loader.epoch_done():
                batch = loader.next_batch()
                if batch is None:
                    continue
                batch = torch.from_numpy(batch).to(device)

                z = make_z(B, SEQ_LEN, config.d_model, device=device)
                session_loss = 0.0
                for _ in range(SUPERVISION):
                    loss, z = train_step(z, batch)
                    session_loss += loss
                    total_loss   += loss

                k = loader.steps_this_epoch
                if k % STEP_REPORT_EVERY == 0:
                    print(f"  step {k:6d} | loss {session_loss/SUPERVISION:.4f} | lr {scheduler.get_last_lr()[0]:.2e}")
                if k % 10000 == 0 and k > 0:
                    save_checkpoint(model, optimizer, cwd / f"{checkpoint_name}_{epoch+1}.{k}")
                    loader.save_state(str(loader_path))

                scheduler.step()

            steps = loader.steps_this_epoch
            print(f"epoch {epoch+1} done | {steps} steps | avg loss {total_loss/max(steps,1):.4f}")
            save_checkpoint(model, optimizer, cwd / f"{checkpoint_name}_{epoch+1}")
            loader.save_state(str(loader_path))

    # -----------------------------------------------------------------------
    elif command == "train":
        raw        = training_data_kalevala(device)
        train_step = make_train_step(model, optimizer, loss_fn, device)
        score_fn   = make_score_fn(model, device)

        print(f"Training SRLM | d_model={config.d_model}, d_state={config.d_state}")
        print("-" * 55)

        for epoch in range(5000):
            # Sample inference
            z = make_z(1, SEQ_LEN, config.d_model, device=device)
            prefix = from_text("Vaka vanha").to(device)
            def projector(x):
                x[:, :len(prefix)] = prefix
                return x
            _, outputs = sampler.sample(
                score_fn, z, graph, noise,
                tokenizer=as_text,
                batch_size=1,
                batch_len=SEQ_LEN,
                steps=10,
                projector=projector,
                device=device,
            )
            print(repr(outputs[0]))

            total_loss = 0.0
            for step in range(N_STEPS):
                batch = sample_batch_random(raw, SEQ_LEN, B)
                z = make_z(B, SEQ_LEN, config.d_model, device=device)
                session_loss = 0.0
                for _ in range(SUPERVISION):
                    loss, z = train_step(z, batch)
                    session_loss += loss
                    total_loss   += loss

                if step % STEP_REPORT_EVERY == 0:
                    print(f"  step {step:4d} | loss {session_loss/SUPERVISION:.4f}")
                if math.isnan(loss):
                    print("Training failed — NaN loss")
                    sys.exit(1)

                scheduler.step()

            avg = total_loss / (N_STEPS * SUPERVISION)
            print(f"epoch {epoch+1}, avg loss {avg:.4f}")
            save_checkpoint(model, optimizer, cwd / f"{checkpoint_name}_{epoch+1}")

        print("Done.")
