"""
wiki_data.py — Systematic Wikipedia data loader for S5HRM training.

Instead of random sampling, this streams through the entire Wikipedia
corpus sequentially in order, epoch by epoch. Each epoch covers every
article exactly once, in shuffled article order (but sequential bytes
within each article).

Usage:
    from wiki_data import WikiDataLoader

    loader = WikiDataLoader(
        parquet_files=[
            "data/train-00000-of-00002.parquet",
            "data/train-00001-of-00002.parquet",
        ],
        seq_len=256,
        batch_size=32,
        seed=42,
    )

    for epoch in range(n_epochs):
        loader.shuffle(epoch)           # shuffle article order for this epoch
        while not loader.epoch_done():
            batch = loader.next_batch() # (B, seq_len) int32, or None if epoch done
            if batch is not None:
                train_step(batch)
        print(f"epoch {epoch} done, {loader.steps_this_epoch} steps")

    loader.save_state(f"/home/cheery/ai/2026/{checkpoint_name}_loader.json")
    loader.load_state(f"/home/cheery/ai/2026/{checkpoint_name}_loader.json")

Design:
    - Articles are concatenated with a separator byte (0x00) between them
    - Each epoch shuffles the article order so the model sees different
      article boundaries and context combinations
    - Within each article, bytes are consumed sequentially in non-overlapping
      windows of seq_len
    - Short trailing chunks at the end of each article (< seq_len bytes) are
      either padded or discarded (configurable)
    - Batches are formed by packing B independent article streams in parallel,
      similar to how GPT training packs sequences
    - State is serializable so training can resume mid-epoch after a crash
"""

import numpy as np
import os
import json
from typing import List, Optional

SEPARATOR = 0  # byte used between articles


class WikiDataLoader:
    def __init__(
        self,
        parquet_files: List[str],
        seq_len: int,
        batch_size: int,
        seed: int = 42,
        pad_short: bool = False,   # if True, pad short trailing chunks; else discard
        text_col: str = "text",
    ):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.seed = seed
        self.pad_short = pad_short

        print("Loading Wikipedia articles...")
        self.articles = self._load_articles(parquet_files, text_col)
        print(f"  {len(self.articles)} articles loaded")

        # Convert each article to bytes once, store as list of np arrays
        print("Encoding articles to bytes...")
        self.article_bytes = []
        total_bytes = 0
        for text in self.articles:
            b = np.frombuffer(
                bytearray(text.encode("utf-8")),
                dtype=np.uint8
            ).astype(np.int32)
            self.article_bytes.append(b)
            total_bytes += len(b)
        print(f"  {total_bytes:,} bytes total ({total_bytes/1024**2:.1f} MB)")
        self.total_bytes = total_bytes

        # Estimate steps per epoch
        chunks_per_article = [
            max(0, len(b) // seq_len)
            for b in self.article_bytes
        ]
        self.estimated_steps = sum(chunks_per_article) // batch_size
        print(f"  ~{self.estimated_steps:,} steps per epoch at "
              f"batch={batch_size}, seq_len={seq_len}")

        # State
        self._epoch = -1
        self._order = None          # shuffled article indices for current epoch
        self._article_idx = 0       # which article we're currently reading
        self._byte_pos = 0          # position within current article
        self._steps_this_epoch = 0
        self._buffer = []           # list of in-progress sequences, one per batch slot

    def _load_articles(self, parquet_files: List[str], text_col: str) -> List[str]:
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError("pyarrow required: pip install pyarrow")

        articles = []
        for path in parquet_files:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Parquet file not found: {path}")
            table = pq.read_table(path, columns=[text_col])
            for text in table.column(text_col).to_pylist():
                if text and len(text.strip()) > 0:
                    articles.append(text.replace("\n", " "))
        return articles

    def shuffle(self, epoch: int):
        """Shuffle article order for a new epoch. Call at the start of each epoch."""
        self._epoch = epoch
        rng = np.random.default_rng(self.seed + epoch)
        self._order = rng.permutation(len(self.articles))
        self._article_idx = 0
        self._byte_pos = 0
        self._steps_this_epoch = 0
        self._buffer = [[] for _ in range(self.batch_size)]
        print(f"  Epoch {epoch}: shuffled {len(self.articles)} articles")

    def epoch_done(self) -> bool:
        """Returns True when all articles have been consumed this epoch."""
        return self._article_idx >= len(self._order)

    @property
    def steps_this_epoch(self) -> int:
        return self._steps_this_epoch

    def _current_article(self) -> Optional[np.ndarray]:
        """Return current article bytes, or None if epoch is done."""
        if self._article_idx >= len(self._order):
            return None
        return self.article_bytes[self._order[self._article_idx]]

    def _advance_article(self):
        """Move to the next article."""
        self._article_idx += 1
        self._byte_pos = 0

    def _fill_slot(self, slot: int) -> bool:
        """
        Fill buffer slot with seq_len bytes from the current stream position.
        Returns True if successful, False if epoch is done.
        """
        seq = []
        needed = self.seq_len

        while needed > 0:
            article = self._current_article()
            if article is None:
                # Epoch done — pad with zeros if we have something, else fail
                if seq and self.pad_short:
                    seq.extend([0] * needed)
                    self._buffer[slot] = seq
                    return True
                return False

            remaining = article[self._byte_pos:]
            available = len(remaining)

            if available <= needed:
                # Take everything from this article and move on
                seq.extend(remaining.tolist())
                needed -= available
                self._advance_article()
                # Add separator byte between articles
                if needed > 0:
                    seq.append(SEPARATOR)
                    needed -= 1
            else:
                # Take what we need from this article
                seq.extend(remaining[:needed].tolist())
                self._byte_pos += needed
                needed = 0

        self._buffer[slot] = seq
        return True

    def next_batch(self) -> Optional[np.ndarray]:
        """
        Return next batch of shape (batch_size, seq_len) as int32 numpy array.
        Returns None if epoch is done.
        """
        if self._order is None:
            raise RuntimeError("Call shuffle(epoch) before next_batch()")

        if self.epoch_done():
            return None

        rows = []
        for slot in range(self.batch_size):
            ok = self._fill_slot(slot)
            if not ok:
                # Epoch ended mid-batch — return partial batch or None
                if len(rows) == 0:
                    return None
                # Pad remaining slots with zeros
                while len(rows) < self.batch_size:
                    rows.append([0] * self.seq_len)
                break
            rows.append(self._buffer[slot])

        if not rows:
            return None

        batch = np.array(rows, dtype=np.int32)
        self._steps_this_epoch += 1
        return batch

    def state_dict(self) -> dict:
        """Serialize loader state for checkpoint resumption."""
        return {
            "epoch": self._epoch,
            "article_idx": self._article_idx,
            "byte_pos": self._byte_pos,
            "steps_this_epoch": self._steps_this_epoch,
            "order": self._order.tolist() if self._order is not None else None,
        }

    def load_state_dict(self, state: dict):
        """Restore loader state from checkpoint."""
        self._epoch = state["epoch"]
        self._article_idx = state["article_idx"]
        self._byte_pos = state["byte_pos"]
        self._steps_this_epoch = state["steps_this_epoch"]
        self._order = np.array(state["order"]) if state["order"] is not None else None
        self._buffer = [[] for _ in range(self.batch_size)]

    def save_state(self, path: str):
        with open(path, "w") as f:
            json.dump(self.state_dict(), f)

    def load_state(self, path: str):
        if os.path.exists(path):
            with open(path) as f:
                self.load_state_dict(json.load(f))
            print(f"  Resumed from step {self._steps_this_epoch} "
                  f"of epoch {self._epoch}")
            return True
        return False


class InterleavedWikiLoader:
    """
    Alternative loader that maintains B independent article streams in parallel,
    advancing each stream independently. This is closer to how GPT-style training
    works and gives better gradient diversity within a batch.

    Each batch slot reads from a different article position, so the B sequences
    in a batch come from B different parts of the corpus simultaneously.
    """

    def __init__(
        self,
        parquet_files: List[str],
        seq_len: int,
        batch_size: int,
        seed: int = 42,
        text_col: str = "text",
    ):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.seed = seed

        print("Loading Wikipedia articles (interleaved mode)...")
        # Reuse WikiDataLoader for loading
        self._base = WikiDataLoader(
            parquet_files, seq_len, batch_size=1, seed=seed, text_col=text_col
        )

        # Concatenate everything into one big flat array with separators
        print("Building flat corpus...")
        parts = []
        for b in self._base.article_bytes:
            parts.append(b)
            parts.append(np.array([SEPARATOR], dtype=np.int32))
        self.corpus = np.concatenate(parts)
        print(f"  Flat corpus: {len(self.corpus):,} bytes")

        self._epoch = -1
        self._positions = None   # (batch_size,) current position per slot
        self._steps = 0

    def shuffle(self, epoch: int):
        """Initialize B independent stream positions spread across the corpus."""
        self._epoch = epoch
        rng = np.random.default_rng(self.seed + epoch)
        N = len(self.corpus)
        # Spread starting positions evenly, then add small random jitter
        base_positions = np.linspace(0, N - self.seq_len, self.batch_size).astype(np.int64)
        jitter = rng.integers(0, max(1, N // (self.batch_size * 4)), size=self.batch_size)
        self._positions = np.clip(base_positions + jitter, 0, N - self.seq_len)
        self._steps = 0
        total_steps = (len(self.corpus) // self.seq_len) // self.batch_size
        print(f"  Epoch {epoch}: {self.batch_size} streams, "
              f"~{total_steps:,} steps to cover corpus")

    def epoch_done(self) -> bool:
        if self._positions is None:
            return True
        N = len(self.corpus)
        # Done when all streams have wrapped around past their starting point
        # Simple heuristic: done after corpus_size / (batch_size * seq_len) steps
        return self._steps >= (len(self.corpus) // (self.batch_size * self.seq_len))

    @property
    def steps_this_epoch(self) -> int:
        return self._steps

    def next_batch(self) -> Optional[np.ndarray]:
        if self.epoch_done():
            return None

        N = len(self.corpus)
        rows = []
        for slot in range(self.batch_size):
            pos = int(self._positions[slot])
            end = pos + self.seq_len
            if end <= N:
                rows.append(self.corpus[pos:end])
            else:
                # Wrap around
                tail = self.corpus[pos:]
                head = self.corpus[:end - N]
                rows.append(np.concatenate([tail, head]))
            self._positions[slot] = (pos + self.seq_len) % N

        self._steps += 1
        return np.stack(rows, axis=0).astype(np.int32)


if __name__ == "__main__":
    # Quick smoke test with a tiny synthetic corpus
    import tempfile
    import pyarrow as pa
    import pyarrow.parquet as pq

    print("=== Smoke test: WikiDataLoader ===")
    # Create a tiny fake parquet
    texts = [
        "Tämä on ensimmäinen artikkeli. Se kertoo asioista.",
        "Toinen artikkeli käsittelee eri aiheita kuin ensimmäinen.",
        "Kolmas artikkeli on hieman pidempi ja sisältää enemmän tekstiä tässä.",
        "Neljäs artikkeli." * 10,
    ]
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        tmp_path = f.name
    table = pa.table({"text": texts})
    pq.write_table(table, tmp_path)

    loader = WikiDataLoader(
        parquet_files=[tmp_path],
        seq_len=16,
        batch_size=2,
        seed=0,
    )
    loader.shuffle(0)
    steps = 0
    while not loader.epoch_done():
        batch = loader.next_batch()
        if batch is not None:
            steps += 1
    print(f"  WikiDataLoader: {steps} steps in epoch 0")

    loader.shuffle(1)
    steps = 0
    while not loader.epoch_done():
        batch = loader.next_batch()
        if batch is not None:
            steps += 1
    print(f"  WikiDataLoader: {steps} steps in epoch 1")

    print("\n=== Smoke test: InterleavedWikiLoader ===")
    loader2 = InterleavedWikiLoader(
        parquet_files=[tmp_path],
        seq_len=16,
        batch_size=2,
        seed=0,
    )
    loader2.shuffle(0)
    steps = 0
    while not loader2.epoch_done():
        batch = loader2.next_batch()
        if batch is not None:
            steps += 1
    print(f"  InterleavedWikiLoader: {steps} steps in epoch 0")

    os.unlink(tmp_path)
    print("\nAll tests passed.")
