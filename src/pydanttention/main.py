from __future__ import annotations

from .prediction import predict, untok

__all__ = ["ManualTransformer"]


class ManualTransformer:
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
