from __future__ import annotations

from functools import cached_property

import numpy as np
from pydantic import BaseModel, Field, computed_field

from .config import Config
from .sources import gpt, softmax, tokenize

__all__ = ["ManualTransformer"]


class Token(BaseModel):
    idx: int
    vocab: list[str] = Field(repr=False)

    def decode(self) -> str:
        return self.vocab[self.idx]

    @computed_field
    @cached_property
    def char(self) -> str:
        return self.decode()

    def __str__(self) -> str:
        return f"{self.char} ({self.idx})"


class ManualTransformer(BaseModel):
    test: str = "aab" * 10
    vocab: list[str] = list("ab")
    total: int = 0
    correct: int = 0
    report: bool = False
    logs: list[str] = []

    def run(self) -> None:
        for i in range(2, len(self.test) - 1):
            ctx = self.test[:i]
            expected = self.test[i]
            self.total += 1
            if self.untok(self.predict(ctx)) == expected:
                self.correct += 1
        self.send_report()

    def send_report(self) -> None:
        if self.report:
            pct = self.correct / self.total * 100
            print(f"ACCURACY: {pct}% ({self.correct} / {self.total})")
        return

    def untok(self, tok_idx: int) -> str:
        return self.vocab[tok_idx]

    def tokenize(self, value: str):
        return tokenize(value, vocab=self.vocab)

    def log(self, log_entry: str) -> None:
        self.logs.append(log_entry)
        return

    def predict(self, s, report=True):
        tokens = self.tokenize(s)[-5:]
        model_kwargs = Config().model_dump(include=["inputs", "wte", "wpe", "blocks"])
        logits = gpt(np.array(tokens), **model_kwargs)
        probs = softmax(logits)
        for i, current_idx in enumerate(tokens):
            token_probs = probs[i]
            next_idx = np.argmax(token_probs)
            current = Token(idx=current_idx, vocab=self.vocab)
            pred = Token(idx=next_idx, vocab=self.vocab)
            raw_logits = logits[i]
            self.log(f"{current}: next={pred} probs={token_probs} logits={raw_logits}")
            if report:
                print(self.logs[-1])
        return np.argmax(probs[-1])
