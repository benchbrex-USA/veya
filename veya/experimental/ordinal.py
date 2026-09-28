"""Ordinal score head. Experimental: Agent.predict still uses a categorical softmax.

Cumulative thresholds are a softplus chain, so they stay ordered. Class probabilities
are the adjacent differences of those cumulative probabilities and sum to one.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalHead(nn.Module):
    def __init__(self, dim: int, n_levels: int):
        super().__init__()
        if n_levels < 2:
            raise ValueError("ordinal score needs at least 2 levels")
        self.n_levels = n_levels
        self.proj = nn.Linear(dim, 1)
        self.gaps = nn.Parameter(torch.zeros(n_levels - 1))

    def thresholds(self) -> torch.Tensor:
        return torch.cumsum(F.softplus(self.gaps) + 1e-4, dim=0)

    def cumulative_probability(self, hidden: torch.Tensor) -> torch.Tensor:
        """P(Y > k) for k = 0 .. n_levels-2. Shape [B, n_levels - 1], nonincreasing in k."""
        return torch.sigmoid(self.proj(hidden) - self.thresholds())

    def distribution(self, hidden: torch.Tensor) -> torch.Tensor:
        """Class probabilities, shape [B, n_levels], each row sums to 1."""
        greater = self.cumulative_probability(hidden)
        ones = torch.ones(hidden.size(0), 1, device=hidden.device, dtype=greater.dtype)
        zeros = torch.zeros(hidden.size(0), 1, device=hidden.device, dtype=greater.dtype)
        upper = torch.cat([ones, greater], dim=1)
        lower = torch.cat([greater, zeros], dim=1)
        return (upper - lower).clamp(min=0.0)
