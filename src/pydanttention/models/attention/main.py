from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from ..ops.simple import Softmax

__all__ = ["Attention"]


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
