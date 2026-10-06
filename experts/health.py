"""Small circuit breaker. Three failures open the provider; success closes it."""

from __future__ import annotations


class Circuit:
    def __init__(self, limit: int = 3):
        self.limit = limit
        self.failures = 0
        self.open = False

    def allow(self) -> bool:
        return not self.open

    def record(self, success: bool) -> None:
        if success:
            self.failures = 0
            self.open = False
            return
        self.failures += 1
        if self.failures >= self.limit:
            self.open = True
