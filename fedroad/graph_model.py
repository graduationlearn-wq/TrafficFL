"""Directed spatio-temporal graph GRU for traffic-speed forecasting.

The temporal encoder processes a speed/mask sequence per sensor. Two
message-passing layers use forward and backward random-walk transitions so the
model can retain directionality in the METR-LA sensor graph.

This is a DCRNN-inspired spatial encoder, not a full reproduction of DCRNN's
diffusion-convolutional recurrent cell.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def normalize_directed_adjacencies(
    adjacency: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build row-normalized forward and reverse random-walk matrices.

    The supplied adjacency is kept directed; it is not averaged with its
    transpose. Rows with zero outgoing weight remain zero. A self-information
    path is learned inside each graph-convolution layer.

    Returns:
        (forward_transition, backward_transition), each shaped (N, N).
    """
    matrix = torch.as_tensor(adjacency, dtype=torch.float32)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("adjacency must be a square 2D matrix")
    if matrix.shape[0] == 0:
        raise ValueError("adjacency must contain at least one node")
    if not torch.isfinite(matrix).all():
        raise ValueError("adjacency must contain only finite values")
    if torch.any(matrix < 0):
        raise ValueError("adjacency values must be non-negative")

    def row_normalize(values: torch.Tensor) -> torch.Tensor:
        degree = values.sum(dim=1, keepdim=True)
        safe_degree = degree.clamp_min(torch.finfo(values.dtype).eps)
        normalized = values / safe_degree
        return torch.where(degree > 0, normalized, torch.zeros_like(values))

    forward_transition = row_normalize(matrix)
    backward_transition = row_normalize(matrix.T)
    return forward_transition, backward_transition


class DiffusionGraphConvolution(nn.Module):
    """Graph message passing over directed forward/backward random walks."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        diffusion_steps: int = 2,
    ) -> None:
        super().__init__()
        if in_features < 1 or out_features < 1 or diffusion_steps < 1:
            raise ValueError("feature sizes and diffusion_steps must be positive")
        self.diffusion_steps = diffusion_steps
        self.self_linear = nn.Linear(in_features, out_features, bias=False)
        self.forward_linears = nn.ModuleList(
            nn.Linear(in_features, out_features, bias=False)
            for _ in range(diffusion_steps)
        )
        self.backward_linears = nn.ModuleList(
            nn.Linear(in_features, out_features, bias=False)
            for _ in range(diffusion_steps)
        )
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(
        self,
        node_features: torch.Tensor,
        forward_transition: torch.Tensor,
        backward_transition: torch.Tensor,
    ) -> torch.Tensor:
        if node_features.ndim != 3:
            raise ValueError("node_features must have shape (batch, nodes, features)")
        node_count = node_features.shape[1]
        expected = (node_count, node_count)
        if forward_transition.shape != expected or backward_transition.shape != expected:
            raise ValueError("transition matrix dimensions must match input node count")
        if (forward_transition.device != node_features.device
                or backward_transition.device != node_features.device):
            raise ValueError("node features and transition matrices must share a device")

        output = self.self_linear(node_features)
        forward_state = node_features
        backward_state = node_features
        for step in range(self.diffusion_steps):
            forward_state = torch.einsum(
                "ij,bjf->bif", forward_transition, forward_state
            )
            backward_state = torch.einsum(
                "ij,bjf->bif", backward_transition, backward_state
            )
            output = output + self.forward_linears[step](forward_state)
            output = output + self.backward_linears[step](backward_state)
        return output + self.bias


class SpatioTemporalGraphGRU(nn.Module):
    """Encode each sensor's temporal history, then diffuse across the road graph.

    Input x shape: (batch, nodes, time_steps, 2), with speed and observation
    mask as the two features. The two transition matrices are returned by
    :func:`normalize_directed_adjacencies`. Output shape is (batch, nodes, 3).
    """

    def __init__(
        self,
        hidden_size: int = 32,
        num_horizons: int = 3,
        dropout: float = 0.1,
        diffusion_steps: int = 2,
    ) -> None:
        super().__init__()
        if hidden_size < 1 or num_horizons < 1:
            raise ValueError("hidden_size and num_horizons must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.diffusion_steps = diffusion_steps
        self.temporal_gru = nn.GRU(
            input_size=2,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.graph_conv1 = DiffusionGraphConvolution(
            hidden_size, hidden_size, diffusion_steps
        )
        self.graph_conv2 = DiffusionGraphConvolution(
            hidden_size, hidden_size, diffusion_steps
        )
        self.dropout = nn.Dropout(dropout)
        self.forecast_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_horizons),
        )

    def forward(
        self,
        x: torch.Tensor,
        forward_transition: torch.Tensor,
        backward_transition: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("x must have shape (batch, nodes, time, features)")
        batch_size, node_count, time_steps, feature_count = x.shape
        if feature_count != 2:
            raise ValueError("each time step must contain speed and observation mask")
        expected = (node_count, node_count)
        if forward_transition.shape != expected or backward_transition.shape != expected:
            raise ValueError("transition matrix dimensions must match the input node count")
        if (forward_transition.device != x.device
                or backward_transition.device != x.device):
            raise ValueError("x and transition matrices must be on the same device")

        sequences = x.reshape(batch_size * node_count, time_steps, feature_count)
        _, hidden = self.temporal_gru(sequences)
        temporal_features = hidden[-1].reshape(batch_size, node_count, -1)

        spatial_features = F.relu(
            self.graph_conv1(
                temporal_features, forward_transition, backward_transition
            )
        )
        spatial_features = self.dropout(spatial_features)
        spatial_features = F.relu(
            self.graph_conv2(
                spatial_features, forward_transition, backward_transition
            )
        )

        fused = torch.cat((temporal_features, spatial_features), dim=-1)
        return self.forecast_head(fused)
