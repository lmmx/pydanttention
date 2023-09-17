from __future__ import annotations

import numpy as np

from .base import Operation

__all__ = ["Softmax", "Linear"]


class Softmax(Operation):
    def normalise(self) -> np.ndarray:
        exp_x = np.exp(self.x - np.max(self.x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class Linear(Operation, arbitrary_types_allowed=True):
    w: np.ndarray
    b: list[int]  # | np.ndarray

    def project(self) -> np.ndarray:
        """[m, in], [in, out], [out] -> [m, out]"""
        return self.x @ self.w + self.b
