from __future__ import annotations

import numpy as np
from pydantic import BaseModel

__all__ = ["AttentionWeights", "AttentionConfig", "AttentionBlock"]


class AttentionWeights(BaseModel, arbitrary_types_allowed=True):
    w: np.ndarray
    b: list[float]


class AttentionConfig(BaseModel):
    c_attn: AttentionWeights  # qkv queries
    c_proj: AttentionWeights  # projection


class AttentionBlock(BaseModel):
    attn: AttentionConfig
