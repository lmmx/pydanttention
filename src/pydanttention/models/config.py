from typing import ClassVar

import numpy as np
from pydantic import BaseModel, Field

from .attention.block import AttentionBlock, AttentionConfig, AttentionWeights

__all__ = ["Config"]


class DefaultParameters:
    """
    Purely used within the field defaults in the `Config` class.
    """

    N_EMBED: ClassVar[int] = 8
    Lg: ClassVar[int] = 1024  # "Large"


# Assign the Lg default parameter into the global scope so the Config default value can
# use it (accessing a full class attribute path would disrupt the nice formatting)
Lg = DefaultParameters.Lg


class ModelFieldDefaults(BaseModel):
    """
    Only `N_CTX` is used at runtime (as the tokenization tail size) so must be set on
    the `Config` data model, and thus by extension on the `Transformer`.
    """

    n_ctx: int = Field(5, exclude=True)


class Config(ModelFieldDefaults, arbitrary_types_allowed=True):
    """
    One-hot encoded token (WTE) and position embeddings (WPE), and attention blocks.

    EMBEDDING USAGE
     P = Position embeddings (one-hot)
     T = Token embeddings (one-hot, first is `a`, second is `b`)
     V = Prediction scratch space

    [P, P, P, P, P, T, T, V]
    """

    wte: np.ndarray = np.array(
        # one-hot token embeddings
        [
            [0, 0, 0, 0, 0, 1, 0, 0],  # token `a` (id 0)
            [0, 0, 0, 0, 0, 0, 1, 0],  # token `b` (id 1)
        ],
    )
    wpe: np.ndarray = np.array(
        # one-hot position embeddings
        [
            [1, 0, 0, 0, 0, 0, 0, 0],  # position 0
            [0, 1, 0, 0, 0, 0, 0, 0],  # position 1
            [0, 0, 1, 0, 0, 0, 0, 0],  # position 2
            [0, 0, 0, 1, 0, 0, 0, 0],  # position 3
            [0, 0, 0, 0, 1, 0, 0, 0],  # position 4
        ],
    )
    blocks: list[AttentionBlock] = [
        AttentionBlock(
            attn=AttentionConfig(
                c_attn=AttentionWeights(
                    # generates qkv matrix
                    b=np.zeros(DefaultParameters.N_EMBED * 3),
                    w=np.array(
                        # this is where the magic happens
                        # fmt: off
                        [
                          [
                              Lg, 0., 0., 0., 0., 0., 0., 0.,  # q
                              1., 0., 0., 0., 0., 0., 0., 0.,  # k
                              0., 0., 0., 0., 0., 0., 0., 0.,  # v
                          ],
                          [
                              Lg, Lg, 0., 0., 0., 0., 0., 0.,  # q
                              0., 1., 0., 0., 0., 0., 0., 0.,  # k
                              0., 0., 0., 0., 0., 0., 0., 0.,  # v
                          ],
                          [
                              0., Lg, Lg, 0., 0., 0., 0., 0.,  # q
                              0., 0., 1., 0., 0., 0., 0., 0.,  # k
                              0., 0., 0., 0., 0., 0., 0., 0.,  # v
                          ],
                          [
                              0., 0., Lg, Lg, 0., 0., 0., 0.,  # q
                              0., 0., 0., 1., 0., 0., 0., 0.,  # k
                              0., 0., 0., 0., 0., 0., 0., 0.,  # v
                          ],
                          [
                              0., 0., 0., Lg, Lg, 0., 0., 0.,  # q
                              0., 0., 0., 0., 1., 0., 0., 0.,  # k
                              0., 0., 0., 0., 0., 0., 0., 0.,  # v
                          ],
                          [
                              0., 0., 0., 0., 0., 0., 0., 0.,  # q
                              0., 0., 0., 0., 0., 0., 0., 0.,  # k
                              0., 0., 0., 0., 0., 0., 0., 1.,  # v
                          ],
                          [
                              0., 0., 0., 0., 0., 0., 0., 0.,  # q
                              0., 0., 0., 0., 0., 0., 0., 0.,  # k
                              0., 0., 0., 0., 0., 0., 0., -1,  # v
                          ],
                          [
                              0., 0., 0., 0., 0., 0., 0., 0.,  # q
                              0., 0., 0., 0., 0., 0., 0., 0.,  # k
                              0., 0., 0., 0., 0., 0., 0., 0.,  # v
                          ],
                        ],
                        # fmt: on
                    ),
                ),
                c_proj=AttentionWeights(
                    # weights to project attn result back to embedding space
                    b=[0, 0, 0, 0, 0, Lg, 0, 0],
                    w=np.array(
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, -Lg, Lg, 0],
                        ],
                    ),
                ),
            ),
        ),
    ]
