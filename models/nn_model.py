"""
Tiny MLP model definition for tabular market features.
"""

from __future__ import annotations

import torch
from torch import nn


class TinyMLP(nn.Module):
    """Configurable tiny MLP for tabular features."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dims: tuple[int, int] = (32, 16),
        dropout: tuple[float, float] = (0.2, 0.1),
    ) -> None:
        super().__init__()
        h1, h2 = hidden_dims
        d1, d2 = dropout
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Dropout(d1),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(d2),
            nn.Linear(h2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return logits.squeeze(-1)
