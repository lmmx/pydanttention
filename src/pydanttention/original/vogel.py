"""
Manual transformer from https://vgel.me/posts/handmade-transformer/#Completed_code
Model ops from https://github.com/jaymody/picoGPT/blob/main/gpt2.py (MIT license)
"""

import numpy as np


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# [m, in], [in, out], [out] -> [m, out]
def linear(x, w, b):
    return x @ w + b


# [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v]
def attention(q, k, v, mask):
    return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v


# [n_seq, n_embd] -> [n_seq, n_embd]
def causal_self_attention(x, c_attn, c_proj):
    # qkv projections
    x = linear(x, **c_attn)  # [n_seq, n_embd] -> [n_seq, 3*n_embd]

    # split into qkv
    q, k, v = np.split(x, 3, axis=-1)  # [n_seq, 3*n_embd] -> 3 of [n_seq, n_embd]

    # causal mask to hide future inputs from being attended to
    causal_mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10  # [n_seq, n_seq]

    # perform causal self attention
    x = attention(q, k, v, causal_mask)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # out projection
    x = linear(x, **c_proj)  # [n_seq, n_embd] @ [n_embd, n_embd] = [n_seq, n_embd]

    return x


# [n_seq, n_embd] -> [n_seq, n_embd]
def transformer_block(x, attn):
    x = x + causal_self_attention(x, **attn)
    # NOTE: removed ffn
    return x


# [n_seq] -> [n_seq, n_vocab]
def gpt(inputs, wte, wpe, blocks):
    # token + positional embeddings
    x = wte[inputs] + wpe[range(len(inputs))]  # [n_seq] -> [n_seq, n_embd]

    # forward pass through n_layer transformer blocks
    for block in blocks:
        x = transformer_block(x, **block)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # projection to vocab
    return x @ wte.T  # [n_seq, n_embd] -> [n_seq, n_vocab]


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

CHARS = ["a", "b"]


def tokenize(s):
    return [CHARS.index(c) for c in s]


def untok(tok):
    return CHARS[tok]


def predict(s):
    tokens = tokenize(s)[-5:]
    logits = gpt(np.array(tokens), **MODEL)
    probs = softmax(logits)

    for i, tok in enumerate(tokens):
        pred = np.argmax(probs[i])
        print(
            f"{untok(tok)} ({tok}): next={untok(pred)} ({pred}) probs={probs[i]} logits={logits[i]}",
        )

    return np.argmax(probs[-1])


def complete(s, max_new_tokens=10):
    tokens = tokenize(s)
    while len(tokens) < len(s) + max_new_tokens:
        logits = gpt(np.array(tokens[-5:]), **MODEL)
        probs = softmax(logits)
        pred = np.argmax(probs[-1])
        tokens.append(pred)
    return s + " :: " + "".join(untok(t) for t in tokens[len(s) :])


class VogelManualTransformer:
    def __init__(self, report=False):
        self.test = "aab" * 10
        self.total, self.correct = 0, 0
        for i in range(2, len(self.test) - 1):
            ctx = self.test[:i]
            expected = self.test[i]
            self.total += 1
            if untok(predict(ctx)) == expected:
                self.correct += 1
        if report:
            print(
                f"ACCURACY: {self.correct / self.total * 100}% ({self.correct} / {self.total})",
            )


if __name__ == "__main__":
    VogelManualTransformer(report=True)
