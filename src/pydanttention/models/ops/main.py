from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from ..attention.block import AttentionBlock, AttentionConfig
from ..attention.main import Attention
from .base import Operation
from .simple import Linear

__all__ = ["CausalSelfAttention", "TransformerBlock"]


class CausalSelfAttention(AttentionConfig, Operation):
    def self_attend(self) -> np.ndarray:
        """[n_seq, n_embd] -> [n_seq, n_embd]"""
        # qkv projections: [n_seq, n_embd] -> [n_seq, 3*n_embd]
        x = Linear(x=self.x, w=self.c_attn.w, b=self.c_attn.b).project()
        # split into qkv: [n_seq, 3*n_embd] -> 3 of [n_seq, n_embd]
        q, k, v = np.split(x, 3, axis=-1)
        # causal mask to hide future inputs from being attended to
        causal_mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10  # [n_seq, n_seq]
        # perform causal self attention: [n_seq, n_embd] -> [n_seq, n_embd]
        x = Attention(q=q, k=k, v=v, mask=causal_mask).attend()
        # out projection: [n_seq, n_embd] @ [n_embd, n_embd] = [n_seq, n_embd]
        x = Linear(x=x, w=self.c_proj.w, b=self.c_proj.b).project()
        return x


class TransformerBlock(AttentionBlock, Operation):
    def process(self) -> np.ndarray:
        """[n_seq, n_embd] -> [n_seq, n_embd]"""
        a = CausalSelfAttention(x=self.x, **self.attn.model_dump())
        # NOTE: removed ffn
        return self.x + a.self_attend()


class GPT(BaseModel, arbitrary_types_allowed=True):
    inputs: np.ndarray
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
