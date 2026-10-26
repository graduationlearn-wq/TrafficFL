"""Tests for direction-preserving spatio-temporal graph GRU."""

import numpy as np
import pytest
import torch

from fedroad.graph_data import align_adjacency_to_sensor_order
from fedroad.graph_model import (
    SpatioTemporalGraphGRU,
    normalize_directed_adjacencies,
)


def test_directed_normalization_preserves_forward_and_reverse_walks():
    adjacency = torch.tensor(
        [[0.0, 2.0, 0.0], [0.0, 0.0, 1.0], [3.0, 0.0, 0.0]]
    )
    forward, backward = normalize_directed_adjacencies(adjacency)
    assert forward.shape == (3, 3)
    assert backward.shape == (3, 3)
    assert torch.isfinite(forward).all()
    assert torch.isfinite(backward).all()
    assert torch.allclose(forward.sum(dim=1), torch.ones(3))
    assert torch.allclose(backward.sum(dim=1), torch.ones(3))
    assert not torch.allclose(forward, backward)
    assert forward[0, 1].item() == pytest.approx(1.0)
    assert backward[1, 0].item() == pytest.approx(1.0)


def test_directed_normalization_handles_isolated_node():
    adjacency = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    forward, backward = normalize_directed_adjacencies(adjacency)
    assert torch.equal(forward[1], torch.zeros(3))
    assert torch.equal(forward[2], torch.zeros(3))
    assert torch.isfinite(backward).all()


def test_graph_gru_output_and_gradients_with_directed_graph():
    torch.manual_seed(7)
    nodes = 5
    adjacency = torch.triu(torch.ones(nodes, nodes), diagonal=1)
    forward, backward = normalize_directed_adjacencies(adjacency)
    model = SpatioTemporalGraphGRU(hidden_size=8)
    x = torch.randn(2, nodes, 12, 2)

    output = model(x, forward, backward)
    assert output.shape == (2, nodes, 3)
    assert torch.isfinite(output).all()

    output.square().mean().backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_graph_model_rejects_node_mismatch():
    model = SpatioTemporalGraphGRU(hidden_size=8)
    x = torch.randn(2, 4, 12, 2)
    forward, backward = normalize_directed_adjacencies(torch.eye(3))
    with pytest.raises(ValueError, match="node count"):
        model(x, forward, backward)


def test_graph_adjacency_is_reordered_to_csv_sensor_order():
    graph_sensor_ids = ["s2", "s1", "s3"]
    adjacency = np.array(
        [[2, 12, 23], [12, 1, 13], [23, 13, 3]], dtype=np.float32
    )
    aligned = align_adjacency_to_sensor_order(
        ["s1", "s2", "s3"], graph_sensor_ids, adjacency
    )
    expected = np.array(
        [[1, 12, 13], [12, 2, 23], [13, 23, 3]], dtype=np.float32
    )
    np.testing.assert_array_equal(aligned, expected)


def test_graph_adjacency_rejects_missing_sensor():
    with pytest.raises(ValueError, match="missing from the graph"):
        align_adjacency_to_sensor_order(
            ["s1", "not-present"], ["s1"], np.array([[1.0]])
        )
