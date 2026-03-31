"""
G-MemLLM: Gated Latent Memory Augmentation for Long-Context Reasoning
in Large Language Models (Xu, 2025)

A memory-augmented architecture that integrates a frozen LLM backbone with a
trainable Latent Memory Bank. Uses GRU-style gated update logic to selectively
update, preserve, or overwrite latent memory slots.

Usage:
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    gmem = GMemLLM(model, num_slots=1024, memory_dim=256)
    # Only gmem's memory parameters are trained; the LLM stays frozen.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryEncoder(nn.Module):
    """Projects LLM hidden states into lower-dimensional memory space."""

    def __init__(self, hidden_dim: int, memory_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, memory_dim)
        self.norm = nn.LayerNorm(memory_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (batch, seq_len, hidden_dim)
        return self.norm(self.proj(hidden_states))


class MemoryDecoder(nn.Module):
    """Projects memory representations back to LLM hidden dimension."""

    def __init__(self, memory_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(memory_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, memory_states: torch.Tensor) -> torch.Tensor:
        # memory_states: (batch, seq_len, memory_dim)
        return self.norm(self.proj(memory_states))


class CrossAttention(nn.Module):
    """Cross-attention where memory slots are Queries, encoded states are Keys/Values."""

    def __init__(self, memory_dim: int, num_heads: int = 4):
        super().__init__()
        assert memory_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = memory_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(memory_dim, memory_dim)
        self.k_proj = nn.Linear(memory_dim, memory_dim)
        self.v_proj = nn.Linear(memory_dim, memory_dim)
        self.out_proj = nn.Linear(memory_dim, memory_dim)

    def forward(
        self,
        queries: torch.Tensor,   # memory slots: (batch, num_slots, memory_dim)
        keys: torch.Tensor,      # encoded hidden: (batch, seq_len, memory_dim)
        values: torch.Tensor,    # encoded hidden: (batch, seq_len, memory_dim)
    ) -> torch.Tensor:
        B, S, D = queries.shape
        _, T, _ = keys.shape

        q = self.q_proj(queries).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(keys).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(values).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)


class GatedUpdate(nn.Module):
    """GRU-style gated update for memory consolidation.

    M_new = (1 - g) * M_old + g * M_attended
    where g = sigmoid(W_z [M_old; M_attended] + b_z)
    Gate bias initialized to -3 so updates are conservative initially.
    """

    def __init__(self, memory_dim: int):
        super().__init__()
        self.gate = nn.Linear(memory_dim * 2, memory_dim)
        nn.init.constant_(self.gate.bias, -3.0)

    def forward(
        self,
        m_old: torch.Tensor,      # (batch, num_slots, memory_dim)
        m_attended: torch.Tensor,  # (batch, num_slots, memory_dim)
    ) -> torch.Tensor:
        g = torch.sigmoid(self.gate(torch.cat([m_old, m_attended], dim=-1)))
        return (1 - g) * m_old + g * m_attended


class RelevanceRetriever(nn.Module):
    """Computes importance scores for each memory slot given the current context.

    Returns weighted memory representation and the importance scores (for loss).
    """

    def __init__(self, memory_dim: int):
        super().__init__()
        self.query_proj = nn.Linear(memory_dim, memory_dim)
        self.key_proj = nn.Linear(memory_dim, memory_dim)

    def forward(
        self,
        encoded_hidden: torch.Tensor,  # (batch, seq_len, memory_dim)
        memory_slots: torch.Tensor,    # (batch, num_slots, memory_dim)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Compute context query as mean-pooled encoded hidden states
        context = encoded_hidden.mean(dim=1, keepdim=True)  # (batch, 1, memory_dim)
        q = self.query_proj(context)   # (batch, 1, memory_dim)
        k = self.key_proj(memory_slots)  # (batch, num_slots, memory_dim)

        # Importance scores per slot
        scores = (q @ k.transpose(-2, -1)).squeeze(1) / math.sqrt(k.size(-1))
        # scores: (batch, num_slots)

        weights = F.softmax(scores, dim=-1)  # (batch, num_slots)

        # Weighted sum of memory slots
        retrieved = torch.bmm(weights.unsqueeze(1), memory_slots)  # (batch, 1, memory_dim)

        return retrieved, scores


class GatedEnhancement(nn.Module):
    """Gated injection of retrieved memory into LLM hidden states.

    Combines original hidden states with decoded memory via a learned gate.
    Gate bias initialized to -3 so g ≈ 0.05 initially (near-identity).
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        nn.init.constant_(self.gate.bias, -3.0)

    def forward(
        self,
        hidden_states: torch.Tensor,     # (batch, seq_len, hidden_dim)
        decoded_memory: torch.Tensor,     # (batch, 1, hidden_dim) or (batch, seq_len, hidden_dim)
    ) -> torch.Tensor:
        # Broadcast decoded_memory across sequence length if needed
        if decoded_memory.size(1) == 1:
            decoded_memory = decoded_memory.expand_as(hidden_states)

        g = torch.sigmoid(self.gate(torch.cat([hidden_states, decoded_memory], dim=-1)))
        return hidden_states + g * decoded_memory


class LatentMemoryBank(nn.Module):
    """The full memory module: encoder, decoder, cross-attention, gated update,
    relevance retriever, and gated enhancement."""

    def __init__(
        self,
        hidden_dim: int,
        memory_dim: int = 256,
        num_slots: int = 1024,
        num_heads: int = 4,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.memory_dim = memory_dim

        # Learnable memory slots
        self.memory_slots = nn.Parameter(torch.randn(1, num_slots, memory_dim) * 0.02)

        # Sub-modules
        self.encoder = MemoryEncoder(hidden_dim, memory_dim)
        self.decoder = MemoryDecoder(memory_dim, hidden_dim)
        self.cross_attn = CrossAttention(memory_dim, num_heads)
        self.gated_update = GatedUpdate(memory_dim)
        self.retriever = RelevanceRetriever(memory_dim)
        self.enhancement = GatedEnhancement(hidden_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,   # (batch, seq_len, hidden_dim)
        memory: torch.Tensor | None = None,  # (batch, num_slots, memory_dim) or None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            enhanced_hidden: hidden states augmented with memory (batch, seq_len, hidden_dim)
            updated_memory: consolidated memory for next step (batch, num_slots, memory_dim)
            importance_scores: raw scores for loss computation (batch, num_slots)
        """
        B = hidden_states.size(0)

        # Initialize memory from learned slots if not provided
        if memory is None:
            memory = self.memory_slots.expand(B, -1, -1)

        # Encode hidden states into memory space
        encoded = self.encoder(hidden_states)  # (B, T, D_m)

        # --- Retrieval ---
        retrieved, importance_scores = self.retriever(encoded, memory)
        # retrieved: (B, 1, D_m), importance_scores: (B, S)

        # Decode and inject into hidden states
        decoded = self.decoder(retrieved)  # (B, 1, D_h)
        enhanced_hidden = self.enhancement(hidden_states, decoded)

        # --- Consolidation ---
        # Cross-attention: memory attends to new encoded states
        m_attended = self.cross_attn(memory, encoded, encoded)
        # GRU gated update
        updated_memory = self.gated_update(memory, m_attended)

        return enhanced_hidden, updated_memory, importance_scores


class GMemLLM(nn.Module):
    """G-MemLLM wrapper around a frozen HuggingFace causal LM.

    Freezes the base model and trains only the LatentMemoryBank.

    Args:
        base_model: A HuggingFace CausalLM (e.g. GPT2LMHeadModel, LlamaForCausalLM)
        num_slots: Number of memory slots (default 1024)
        memory_dim: Dimension of memory space (default 256)
        num_heads: Number of attention heads in cross-attention
        lambda_sparsity: Weight for sparsity loss
        lambda_entropy: Weight for entropy loss
    """

    def __init__(
        self,
        base_model: nn.Module,
        num_slots: int = 1024,
        memory_dim: int = 256,
        num_heads: int = 4,
        lambda_sparsity: float = 0.01,
        lambda_entropy: float = 0.01,
    ):
        super().__init__()
        self.base_model = base_model
        self.lambda_sparsity = lambda_sparsity
        self.lambda_entropy = lambda_entropy

        # Freeze the base model
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Detect hidden dimension from model config
        config = base_model.config
        if hasattr(config, "hidden_size"):
            hidden_dim = config.hidden_size
        elif hasattr(config, "n_embd"):
            hidden_dim = config.n_embd
        else:
            raise ValueError("Cannot detect hidden_dim from model config")

        self.memory_bank = LatentMemoryBank(
            hidden_dim=hidden_dim,
            memory_dim=memory_dim,
            num_slots=num_slots,
            num_heads=num_heads,
        )

    def get_hidden_states(self, input_ids, attention_mask=None):
        """Extract hidden states from the frozen LLM (before the LM head)."""
        with torch.no_grad():
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        # Use the last hidden state (before LM head)
        return outputs.hidden_states[-1]

    def get_lm_logits(self, hidden_states):
        """Pass hidden states through the frozen LM head.

        No torch.no_grad() here — the LM head weights are already frozen
        (requires_grad=False), but we need gradients to flow back through
        the enhanced hidden states to train the memory bank.
        """
        if hasattr(self.base_model, "lm_head"):
            return self.base_model.lm_head(hidden_states)
        elif hasattr(self.base_model, "embed_out"):
            return self.base_model.embed_out(hidden_states)
        else:
            raise ValueError("Cannot find LM head in base model")

    def compute_memory_loss(self, importance_scores):
        """Compute sparsity and entropy regularization losses."""
        # Sparsity: L1 on importance scores
        l_sparsity = importance_scores.abs().mean()

        # Entropy: encourage diverse slot usage
        p = F.softmax(importance_scores, dim=-1)
        l_entropy = (p * p.log()).sum(dim=-1).mean()  # negative entropy

        return self.lambda_sparsity * l_sparsity + self.lambda_entropy * l_entropy

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        memory: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            labels: (batch, seq_len) target token ids for CLM loss
            memory: optional memory state from previous segment

        Returns:
            dict with keys: logits, loss (if labels), memory, importance_scores
        """
        # Step 1: Extraction - get hidden states from frozen LLM
        hidden_states = self.get_hidden_states(input_ids, attention_mask)

        # Step 2-3: Retrieval + Injection + Consolidation via memory bank
        enhanced_hidden, updated_memory, importance_scores = self.memory_bank(
            hidden_states, memory
        )

        # Step 4: Get logits from enhanced hidden states through frozen LM head
        logits = self.get_lm_logits(enhanced_hidden)

        result = {
            "logits": logits,
            "memory": updated_memory,
            "importance_scores": importance_scores,
        }

        # Compute loss if labels provided
        if labels is not None:
            # Primary CLM loss: shift logits and labels for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            l_clm = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            # Memory regularization losses
            l_mem = self.compute_memory_loss(importance_scores)

            result["loss"] = l_clm + l_mem
            result["loss_clm"] = l_clm
            result["loss_memory"] = l_mem

        return result

    def process_segments(
        self,
        input_ids_list: list[torch.Tensor],
        attention_mask_list: list[torch.Tensor] | None = None,
        labels_list: list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Process a sequence of segments, passing memory between them.

        This is the main training/inference loop for long documents split
        into segments that exceed the base model's context window.
        """
        memory = None
        total_loss = 0.0
        all_logits = []
        num_segments = len(input_ids_list)

        for i, input_ids in enumerate(input_ids_list):
            attn_mask = attention_mask_list[i] if attention_mask_list else None
            labels = labels_list[i] if labels_list else None

            result = self.forward(
                input_ids=input_ids,
                attention_mask=attn_mask,
                labels=labels,
                memory=memory,
            )

            all_logits.append(result["logits"])
            memory = result["memory"]

            if "loss" in result:
                total_loss = total_loss + result["loss"]

        output = {
            "logits": all_logits,
            "memory": memory,
        }
        if labels_list is not None:
            output["loss"] = total_loss / num_segments

        return output


# --- Convenience: standalone test ---
if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading GPT-2...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    base_model = AutoModelForCausalLM.from_pretrained("gpt2")

    print("Creating G-MemLLM...")
    model = GMemLLM(base_model, num_slots=64, memory_dim=128)

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # Test forward pass
    text = "The capital of France is Paris. Paris is known for the Eiffel Tower."
    inputs = tokenizer(text, return_tensors="pt")
    result = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        labels=inputs["input_ids"],
    )
    print(f"Loss: {result['loss'].item():.4f}")
    print(f"  CLM: {result['loss_clm'].item():.4f}")
    print(f"  Mem: {result['loss_memory'].item():.4f}")
    print(f"Logits shape: {result['logits'].shape}")
    print(f"Memory shape: {result['memory'].shape}")

    # Test multi-segment processing
    segments = [
        "Marie Curie was born in Warsaw, Poland.",
        "She moved to Paris to study at the Sorbonne.",
        "She won the Nobel Prize in Physics in 1903.",
    ]
    ids_list = [tokenizer(s, return_tensors="pt")["input_ids"] for s in segments]
    labels_list = [ids.clone() for ids in ids_list]
    result = model.process_segments(ids_list, labels_list=labels_list)
    print(f"\nMulti-segment loss: {result['loss'].item():.4f}")
    print(f"Final memory shape: {result['memory'].shape}")
    print("Done!")
