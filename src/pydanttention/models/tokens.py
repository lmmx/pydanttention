from __future__ import annotations

from functools import cached_property

from pydantic import BaseModel, Field, computed_field

__all__ = ["Token"]


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
