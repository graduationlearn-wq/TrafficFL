"""Small deterministic tests for federated sensor partitioning."""
import numpy as np
import pytest

from fedroad.federated_data import partition_sensor_indices, _window_limit


def test_sensor_partitions_are_deterministic_disjoint_and_complete():
    first = partition_sensor_indices(207, 4)
    second = partition_sensor_indices(207, 4)
    assert [part.tolist() for part in first] == [part.tolist() for part in second]
    joined = np.concatenate(first)
    assert len(joined) == 207
    assert len(np.unique(joined)) == 207
    assert sorted(len(part) for part in first) == [51, 52, 52, 52]


def test_partition_config_validation():
    with pytest.raises(ValueError):
        partition_sensor_indices(10, 1)
    with pytest.raises(ValueError):
        partition_sensor_indices(3, 4)


def test_zero_window_limit_means_all_windows():
    assert _window_limit(0) is None
    assert _window_limit(None) is None
    assert _window_limit(1200) == 1200
