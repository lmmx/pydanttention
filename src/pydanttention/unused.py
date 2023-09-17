"""
I didn't keep this in the loop during development, but can pick it up once finished.
"""
import numpy as np

from .constants import MODEL
from .sources import gpt, softmax, tokenize, untok


def complete(s, max_new_tokens=10):
    tokens = tokenize(s)
    while len(tokens) < len(s) + max_new_tokens:
        logits = gpt(np.array(tokens[-5:]), **MODEL)
        probs = softmax(logits)
        pred = np.argmax(probs[-1])
        tokens.append(pred)
    return s + " :: " + "".join(untok(t) for t in tokens[len(s) :])
