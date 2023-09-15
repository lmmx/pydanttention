from __future__ import annotations

from functools import cached_property

import numpy as np
from pydantic import BaseModel, Field, computed_field

from .constants import CHARS, MODEL
from .sources import gpt, softmax, tokenize

__all__ = ["ManualTransformer"]


class Token(BaseModel):
    tok: int
    vocab: list[str] = Field(repr=False)

    def decode(self) -> str:
        return self.vocab[self.tok]

    @computed_field
    @cached_property
    def char(self) -> str:
        return self.decode()


class ManualTransformer(BaseModel):
    test: str = "aab" * 10
    vocab: list[str] = list("ab")
    total: int = 0
    correct: int = 0
    report: bool = False

    def run(self) -> None:
        for i in range(2, len(self.test) - 1):
            ctx = self.test[:i]
            expected = self.test[i]
            self.total += 1
            if self.untok(self.predict(ctx)) == expected:
                self.correct += 1
        if self.report:
            pct = self.correct / self.total * 100
            print(f"ACCURACY: {pct}% ({self.correct} / {self.total})")

    def untok(self, tok):
        return self.vocab[tok]

    def predict(self, s, report=True):
        tokens = tokenize(s)[-5:]
        logits = gpt(np.array(tokens), **MODEL)
        probs = softmax(logits)
        if report:
            for i, current_token in enumerate(tokens):
                prob_values = probs[i]
                predicted_next_token = np.argmax(prob_values)
                pred_str = self.untok(predicted_next_token)
                token_str = self.untok(current_token)
                logit_values = logits[i]
                token_repr = f"{token_str} ({current_token})"
                pred_repr = f"next={pred_str} ({predicted_next_token})"
                probs_repr = f"probs={prob_values}"
                logits_repr = f"logits={logit_values}"
                print(f"{token_repr}: {pred_repr} {probs_repr} {logits_repr}")
        return np.argmax(probs[-1])
