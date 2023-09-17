from __future__ import annotations

import numpy as np
from pydantic import BaseModel

__all__ = [
    "softmax",
    "linear",
    "attention",
    "causal_self_attention",
    "transformer_block",
    "GPT",
]


class Operation(BaseModel, arbitrary_types_allowed=True):
    x: np.ndarray


class Softmax(Operation):
    def normalise(self) -> np.ndarray:
        exp_x = np.exp(self.x - np.max(self.x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class Linear(Operation, arbitrary_types_allowed=True):
    w: np.ndarray
    b: list[int]  # | np.ndarray

    def project(self) -> np.ndarray:
        """[m, in], [in, out], [out] -> [m, out]"""
        return self.x @ self.w + self.b


class Attention(BaseModel, arbitrary_types_allowed=True):
    q: np.ndarray
    k: np.ndarray
    v: np.ndarray
    mask: np.ndarray

    def attend(self) -> np.ndarray:
        """
        In the attention mechanism, "scores" are the raw similarity metrics computed
        between queries and keys, essentially measuring how much 'focus' each element in
        the query should have on each element in the key. The scores are computed using
        a dot product between the query and key, then smoothed, normalised, and scaled.

        Scaling by the square root of the query dimension (smoothing factor) prevents
        the dot-product between large vectors from growing too large.

        [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v]
        """
        similarity_scores = self.q @ self.k.T  # [n_q, n_k]
        query_dimension = self.q.shape[-1]  # d_k
        smoothing_factor = np.sqrt(query_dimension)
        scaled_similarity = similarity_scores / smoothing_factor  # [n_q, n_k]
        attention_logits = scaled_similarity + self.mask  # [n_q, n_k]
        attention_weights = Softmax(x=attention_logits).normalise()  # [n_q, n_k]
        attention_output = attention_weights @ self.v  # [n_q, d_v]
        return attention_output


class CausalSelfAttention(Operation, arbitrary_types_allowed=True):
    c_attn: dict  # np.ndarray # TODO make a model type
    c_proj: dict  # np.ndarray # TODO make a model type

    def self_attend(self) -> np.ndarray:
        """[n_seq, n_embd] -> [n_seq, n_embd]"""
        # qkv projections [n_seq, n_embd] -> [n_seq, 3*n_embd]
        x = Linear.model_validate(dict(x=self.x, **self.c_attn)).project()
        # split into qkv
        q, k, v = np.split(x, 3, axis=-1)  # [n_seq, 3*n_embd] -> 3 of [n_seq, n_embd]
        # causal mask to hide future inputs from being attended to
        causal_mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10  # [n_seq, n_seq]
        # perform causal self attention: [n_seq, n_embd] -> [n_seq, n_embd]
        x = Attention(q=q, k=k, v=v, mask=causal_mask).attend()
        # out projection [n_seq, n_embd] @ [n_embd, n_embd] = [n_seq, n_embd]
        x = Linear.model_validate(dict(x=x, **self.c_proj)).project()
        return x


class TransformerBlock(Operation, arbitrary_types_allowed=True):
    attn: dict  # np.ndarray # TODO make this another model type

    def process(self) -> np.ndarray:
        """[n_seq, n_embd] -> [n_seq, n_embd]"""
        # NOTE: removed ffn
        return (
            self.x
            + CausalSelfAttention.model_validate(
                dict(x=self.x, **self.attn)
            ).self_attend()
        )


class GPT(BaseModel, arbitrary_types_allowed=True):
    inputs: np.ndarray
    wte: np.ndarray
    wpe: np.ndarray
    blocks: list
    """
    [n_seq] -> [n_seq, n_vocab]
    """

    def transform(self):
        # token + positional embeddings: [n_seq] -> [n_seq, n_embd]
        x = self.wte[self.inputs] + self.wpe[range(len(self.inputs))]
        # forward pass through n_layer transformer blocks
        for block in self.blocks:
            # [n_seq, n_embd] -> [n_seq, n_embd]
            x = TransformerBlock.model_validate(dict(x=x, **block)).process()
        # projection to vocab
        return x @ self.wte.T  # [n_seq, n_embd] -> [n_seq, n_vocab]
