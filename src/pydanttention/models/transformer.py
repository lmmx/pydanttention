from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from .ops.main import AttentionBlock, TransformerBlock

__all__ = ["GPT"]


class GPT(BaseModel, arbitrary_types_allowed=True):
    inputs: list[int] | np.ndarray
    wte: np.ndarray
    wpe: np.ndarray
    blocks: list[AttentionBlock]
    """
    [n_seq] -> [n_seq, n_vocab]
    """

    def generate(self):
        # token + positional embeddings: [n_seq] -> [n_seq, n_embd]
        x = self.wte[self.inputs] + self.wpe[range(len(self.inputs))]
        # forward pass through n_layer transformer blocks
        for block in self.blocks:
            # [n_seq, n_embd] -> [n_seq, n_embd]
            x = TransformerBlock(x=x, attn=block.attn).process()
        # projection to vocab: [n_seq, n_embd] -> [n_seq, n_vocab]
        return x @ self.wte.T
