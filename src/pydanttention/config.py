import numpy as np
from pydantic import BaseModel

__all__ = ["Config"]


class AttentionWeights(BaseModel, arbitrary_types_allowed=True):
    w: np.ndarray
    b: list[float]


class AttentionConfig(BaseModel):
    c_attn: AttentionWeights  # qkv queries
    c_proj: AttentionWeights  # projection


class AttentionBlock(BaseModel):
    attn: AttentionConfig


class Config(BaseModel, arbitrary_types_allowed=True):
    """
    EMBEDDING USAGE
     P = Position embeddings (one-hot)
     T = Token embeddings (one-hot, first is `a`, second is `b`)
     V = Prediction scratch space

    [P, P, P, P, P, T, T, V]
    """

    N_CTX: int = 5
    N_VOCAB: int = 2
    N_EMBED: int = 8
    Lg: int = 1024  # Large
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
                    b=np.zeros(N_EMBED * 3),
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
