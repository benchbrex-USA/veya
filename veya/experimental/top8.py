"""Small top-8 interaction scorer. Experimental.

The encoder stays frozen. One state vector and eight candidate vectors attend
to each other, then each candidate gets a logit. This is not the production
reranker unless a validation comparison promotes it.
"""
import torch
import torch.nn as nn


class Top8Interaction(nn.Module):
    def __init__(self, dim=1024, width=256):
        super().__init__()
        self.query_proj = nn.Linear(dim, width)
        self.cand_proj = nn.Linear(dim, width)
        layer = nn.TransformerEncoderLayer(width, 4, 4 * width, 0.1, batch_first=True, norm_first=True)
        self.block = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.score = nn.Linear(width, 1)

    def forward(self, query, candidates):
        """query [B, D], candidates [B, 8, D] -> logits [B, 8]."""
        tokens = torch.cat([self.query_proj(query).unsqueeze(1), self.cand_proj(candidates)], dim=1)
        hidden = self.block(tokens)
        return self.score(hidden[:, 1:, :]).squeeze(-1)
