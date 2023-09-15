from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from .constants import CHARS, MODEL
from .sources import gpt, softmax, tokenize

__all__ = ["ManualTransformer"]


class ManualTransformer(BaseModel):
    test: str = "aab" * 10
    total: int = 0
    correct: int = 0
    report: bool = False

    def __init__(self):
        super().__init__()
        self.run()

    def run(self) -> None:
        for i in range(2, len(self.test) - 1):
            ctx = self.test[:i]
            expected = self.test[i]
            self.total += 1
            if self.untok(self.predict(ctx)) == expected:
                self.correct += 1
        if self.report:
            print(
                f"ACCURACY: {self.correct / self.total * 100}% ({self.correct} / {self.total})",
            )

    @staticmethod
    def untok(tok):
        return CHARS[tok]

    def predict(self, s, report=True):
        tokens = tokenize(s)[-5:]
        logits = gpt(np.array(tokens), **MODEL)
        probs = softmax(logits)
        for i, tok in enumerate(tokens):
            pred = np.argmax(probs[i])
            if report:
                print(
                    f"{self.untok(tok)} ({tok}): next={self.untok(pred)} ({pred}) probs={probs[i]} logits={logits[i]}",
                )
        return np.argmax(probs[-1])
