"""
MemoryBank — holds pre-encoded document memories on CPU, transfers to GPU on demand.

Usage:
    bank = MemoryBank()
    bank.encode(model, tokens_list, device)   # encode & store on CPU
    memories = bank.retrieve(query, k)         # route & load top-k to GPU
    bank.refresh(model, device)                # re-encode with improved model
"""

import torch
import torch.nn.functional as F


class MemoryBank:
    """Stores encoded document memories on CPU. Routes and retrieves on demand."""

    def __init__(self):
        self.tokens = []        # list of (B, L_doc) int tensors — source documents
        self.memories = []      # list of (B, L, D) tensors on CPU — encoded
        self.summaries = []     # list of (B, D) tensors on CPU — for fast routing

    def clear(self):
        self.tokens.clear()
        self.memories.clear()
        self.summaries.clear()

    def __len__(self):
        return len(self.memories)

    @torch.no_grad()
    def encode(self, model, tokens_list: list[torch.Tensor], device=None):
        """Encode a list of documents and store on CPU.

        Args:
            model: SRLM model (will use encode_document)
            tokens_list: list of (B, L_doc) int tensors
            device: device to run encoding on (then move to CPU)
        """
        for tokens in tokens_list:
            self.tokens.append(tokens.cpu())
            if device is not None:
                tokens = tokens.to(device)
            mem = model.encode_document(tokens)
            mem_cpu = mem.cpu()
            self.memories.append(mem_cpu)
            self.summaries.append(mem_cpu.mean(dim=1))

    @torch.no_grad()
    def refresh(self, model, device=None):
        """Re-encode all documents with the current model weights.

        Call periodically during training so memories benefit from
        improved prior layers.
        """
        new_memories = []
        new_summaries = []
        for tokens in self.tokens:
            if device is not None:
                tokens = tokens.to(device)
            mem = model.encode_document(tokens)
            mem_cpu = mem.cpu()
            new_memories.append(mem_cpu)
            new_summaries.append(mem_cpu.mean(dim=1))
        self.memories = new_memories
        self.summaries = new_summaries

    @torch.no_grad()
    def retrieve(self, query: torch.Tensor, k: int) -> list[torch.Tensor]:
        """Score all memories against query, return top-k on query's device.

        Args:
            query: (B, L, D) — current hidden state to route against
            k: number of memories to retrieve
        Returns:
            list of k tensors, each (B, L, D), on query's device
        """
        if len(self.memories) == 0 or k == 0:
            return []

        device = query.device
        k = min(k, len(self.memories))

        q = F.normalize(query.mean(dim=1), dim=-1)  # (B, D)

        scores = []
        for summary in self.summaries:
            s = F.normalize(summary.to(device), dim=-1)
            sim = (q * s).sum(dim=-1).mean()
            scores.append(sim)

        scores = torch.stack(scores)
        topk_idx = scores.argsort(descending=True)[:k]

        return [self.memories[i].to(device) for i in topk_idx.tolist()]

    def state_dict(self) -> dict:
        """Serialize for checkpointing."""
        return {
            'tokens': self.tokens,
            'memories': self.memories,
            'summaries': self.summaries,
        }

    def load_state_dict(self, state: dict):
        self.tokens = state['tokens']
        self.memories = state['memories']
        self.summaries = state['summaries']
