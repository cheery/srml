import torch
import torch.nn as nn
import torch.nn.functional as F

class BlockDivider:
    def __init__(self, blocks, partial_block, index=1, block_size=1):
        self.blocks = blocks
        self.partial_block = partial_block
        self.index = index
        self.block_size = block_size

    @classmethod
    def top(cls, h, index=1, block_size=1):
        return cls([h], torch.zeros_like(h), index, block_size)

    def div(self, force=False):
        if force or self.index % self.block_size == 0:
            blocks = self.blocks + [self.partial_block]
            partial_block = torch.zeros_like(self.partial_block)
            return BlockDivider(blocks, partial_block, self.index+1, self.block_size)
        else:
            return BlockDivider(self.blocks, self.partial_block, self.index+1, self.block_size)

    def whole(self):
        return torch.stack(self.blocks + [self.partial_block], dim=0)  # (N+1, B, T, D)

    def __call__(self, h):
        return BlockDivider(self.blocks, self.partial_block + h, self.index, self.block_size)

class BlockAttnResOp(nn.Module):
    """Block Attention Residual operator (from arXiv:2603.15031).

    Given completed block representations and a partial block sum,
    computes softmax attention over them using a learned pseudo-query.

    - Query is a learned parameter (not input-dependent)
    - Keys are RMSNorm'd values
    - Zero-init query → uniform attention at init → reduces to averaging
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Linear(d_model, 1, bias=False)
        nn.init.zeros_(self.query.weight)
        self.key_norm = nn.RMSNorm(d_model)

    def forward(self, bd: BlockDivider) -> torch.Tensor:
        """
        Args:
            blocks: list of [B, T, D] — completed block representations
            partial_block: [B, T, D] — current intra-block partial sum
        Returns:
            h: [B, T, D] — aggregated input for this layer
        """
        V = bd.whole()
        K = self.key_norm(V)
        logits = torch.einsum('d, n b t d -> n b t', self.query.weight.squeeze(0), K)
        weights = logits.softmax(dim=0)
        h = torch.einsum('n b t, n b t d -> b t d', weights, V)
        return h
