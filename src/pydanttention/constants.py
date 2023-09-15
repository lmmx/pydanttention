import numpy as np

__all__ = ["CHARS", "N_CTX", "N_VOCAB", "N_EMBED", "Lg", "MODEL"]

CHARS = ["a", "b"]

N_CTX = 5
N_VOCAB = 2
N_EMBED = 8

Lg = 1024  # Large

MODEL = {
    # EMBEDDING USAGE
    #  P = Position embeddings (one-hot)
    #  T = Token embeddings (one-hot, first is `a`, second is `b`)
    #  V = Prediction scratch space
    #
    #       [P, P, P, P, P, T, T, V]
    "wte": np.array(
        # one-hot token embeddings
        [
            [0, 0, 0, 0, 0, 1, 0, 0],  # token `a` (id 0)
            [0, 0, 0, 0, 0, 0, 1, 0],  # token `b` (id 1)
        ],
    ),
    "wpe": np.array(
        # one-hot position embeddings
        [
            [1, 0, 0, 0, 0, 0, 0, 0],  # position 0
            [0, 1, 0, 0, 0, 0, 0, 0],  # position 1
            [0, 0, 1, 0, 0, 0, 0, 0],  # position 2
            [0, 0, 0, 1, 0, 0, 0, 0],  # position 3
            [0, 0, 0, 0, 1, 0, 0, 0],  # position 4
        ],
    ),
    "blocks": [
        {
            "attn": {
                "c_attn": {  # generates qkv matrix
                    "b": np.zeros(N_EMBED * 3),
                    "w": np.array(
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
                },
                "c_proj": {  # weights to project attn result back to embedding space
                    "b": [0, 0, 0, 0, 0, Lg, 0, 0],
                    "w": np.array(
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
                },
            },
        },
    ],
}
