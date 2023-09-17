from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from .models.config import Config
from .models.ops.main import GPT
from .models.ops.simple import Softmax
from .models.tokens import Token

__all__ = ["ManualTransformer"]


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
        self.emit_report()

    def emit_report(self) -> None:
        if self.report:
            pct = self.correct / self.total * 100
            print(f"ACCURACY: {pct}% ({self.correct} / {self.total})")
        return

    def untok(self, tok_idx: int) -> str:
        return self.vocab[tok_idx]

    def log(self, log_entry: str) -> None:
        self.logs.append(log_entry)
        if self.report:
            print(self.logs[-1])
        return

    def make_token(self, idx: int) -> Token:
        return Token(idx=idx, vocab=self.vocab)

    def generate(self, tokens: list[int]):
        model_config = self.config.model_dump()
        transformer_model = GPT(inputs=np.array(tokens), **model_config)
        logits = transformer_model.generate()
        return logits

    def normalise(self, x):
        return Softmax(x=x).normalise()

    def tokenize(self, string: str) -> list[int]:
        ctx_tail = -(self.config.N_CTX)
        tail = string[ctx_tail:]
        return [self.vocab.index(char) for char in tail]

    def predict(self, string: str) -> int:
        tokens = self.tokenize(string)
        logits = self.generate(tokens)
        probs = self.normalise(logits)
        for i, (current_idx, token_probs, raw_logits) in enumerate(
            zip(tokens, probs, logits),
        ):
            next_idx = np.argmax(token_probs)
            current, pred = map(self.make_token, (current_idx, next_idx))
            self.log(f"{current}: next={pred} probs={token_probs} logits={raw_logits}")
        most_probable_next_token_idx = np.argmax(probs[-1])
        return most_probable_next_token_idx
