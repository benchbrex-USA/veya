"""State-once candidate scorer. Experimental: not wired into Agent.predict.

Each candidate is scored from the state vector plus that candidate alone, so the
score of candidate i does not depend on where it sits in the list. Candidates can
be cached and concatenated later. This is not a trained production head.
"""
import torch
import torch.nn as nn


class CandidateSetHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.state_proj = nn.Linear(dim, dim)
        self.candidate_proj = nn.Linear(dim, dim)
        self.score = nn.Linear(dim, 1)

    def forward(self, state: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        """state [B, D], candidates [B, N, D] -> scores [B, N]."""
        if state.dim() != 2 or candidates.dim() != 3:
            raise ValueError("state must be [B, D] and candidates [B, N, D]")
        if state.size(0) != candidates.size(0) or state.size(-1) != candidates.size(-1):
            raise ValueError("state and candidate batch/dimension do not match")
        mixed = torch.tanh(self.state_proj(state).unsqueeze(1) + self.candidate_proj(candidates))
        return self.score(mixed).squeeze(-1)
