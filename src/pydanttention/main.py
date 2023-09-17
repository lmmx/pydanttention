from __future__ import annotations

from functools import cached_property

import numpy as np
from pydantic import BaseModel, Field, computed_field

from .config import Config
from .sources import GPT, Softmax

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
    config: Config = Config()

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
        if self.report:
            print(self.logs[-1])
        return

    def make_token(self, idx: int) -> Token:
        return Token(idx=idx, vocab=self.vocab)

    def gpt(self, tokens: list[int]):
        gpt = GPT(inputs=np.array(tokens), **self.config.model_dump())
        logits = gpt.transform()
        return logits

    def tokenize(self, string: str) -> list[int]:
        ctx_tail = -(self.config.N_CTX)
        tail = string[ctx_tail:]
        return [self.vocab.index(char) for char in tail]

    def predict(self, string: s) -> int:
        tokens = self.tokenize(string)
        logits = self.gpt(tokens)
        probs = Softmax(x=logits).normalise()
        for i, (current_idx, token_probs, raw_logits) in enumerate(
            zip(tokens, probs, logits)
        ):
            next_idx = np.argmax(token_probs)
            current, pred = map(self.make_token, (current_idx, next_idx))
            self.log(f"{current}: next={pred} probs={token_probs} logits={raw_logits}")
        most_probable_next_token_idx = np.argmax(probs[-1])
        return most_probable_next_token_idx
