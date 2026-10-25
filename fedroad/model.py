"""Neural-network models used by FedRoad."""

from __future__ import annotations

import torch
from torch import nn


class SensorLevelGRU(nn.Module):
    """Forecast one sensor's future speeds from its recent history.

    Each time step contains two features: normalized speed and an observation
    indicator. The output contains predictions for 15, 30, and 60 minutes ahead.
    """

    def __init__(self, hidden_size: int = 32, num_horizons: int = 3) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=2,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, num_horizons)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return forecasts for ``x`` with shape ``(batch, time, 2)``."""
        sequence, _ = self.gru(x)
        return self.output(sequence[:, -1, :])
