from __future__ import annotations

import numpy as np
from pydantic import BaseModel

__all__ = ["Operation"]


class Operation(BaseModel, arbitrary_types_allowed=True):
    x: np.ndarray
