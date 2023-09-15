from __future__ import annotations

import numpy as np

from .constants import CHARS, MODEL
from .sources import gpt, softmax, tokenize

__all__ = ["untok", "predict"]


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
