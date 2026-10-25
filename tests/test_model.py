import numpy as np
import torch

from fedroad.data import TrafficWindowDataset
from fedroad.model import SensorLevelGRU


def test_sensor_gru_output_shape():
    model = SensorLevelGRU(hidden_size=16)
    x = torch.randn(5, 12, 2)
    y = model(x)
    assert y.shape == (5, 3)


def test_traffic_window_shapes_and_horizons():
    values = np.arange(40 * 4, dtype=np.float32).reshape(40, 4)
    mask = np.ones_like(values, dtype=np.float32)
    dataset = TrafficWindowDataset(
        values,
        mask,
        sensor_indices=np.array([0, 2]),
        max_windows=None,
    )
    x, y, target_mask = dataset[0]
    assert x.shape == (2, 12, 2)
    assert y.shape == (2, 3)
    assert target_mask.shape == (2, 3)
    assert len(dataset) == 40 - 12 - 12 + 1


def test_mask_is_preserved_for_missing_target():
    values = np.ones((40, 2), dtype=np.float32)
    mask = np.ones_like(values)
    mask[14, 0] = 0.0
    dataset = TrafficWindowDataset(values, mask, max_windows=None)
    _, _, target_mask = dataset[0]
    # First target is at row 14 (12-step input ending at row 11 + 3 steps).
    assert target_mask[0, 0].item() == 0.0
