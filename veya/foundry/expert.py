"""Veya Foundry expert interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from .contracts import VeyaRequest, VeyaResponse


class VeyaExpert(ABC):
    """A normalized decision expert participating in the Veya Foundry."""

    @property
    @abstractmethod
    def expert_id(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def predict(self, request: VeyaRequest) -> VeyaResponse:
        raise NotImplementedError

    def predict_many(self, requests: Sequence[VeyaRequest]) -> list[VeyaResponse]:
        return [self.predict(request) for request in requests]

    def close(self) -> None:
        """Release resources. Implementations may override."""

    def __enter__(self) -> "VeyaExpert":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
