"""Client-local METR-LA loading and preprocessing for Flower simulation.

Each simulated client selects its assigned sensor columns before calculating
its own per-sensor training statistics. The original CSV is never modified.
"""
from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from fedroad.data import (
    DEFAULT_DATA_PATH,
    HORIZONS,
    INPUT_STEPS,
    TrafficWindowDataset,
    download_metr_la,
)

PARTITION_SEED = 42


def partition_sensor_indices(num_sensors: int, num_partitions: int, seed: int = PARTITION_SEED) -> list[np.ndarray]:
    """Deterministically partition sensor indices without overlap."""
    if num_sensors < 1:
        raise ValueError("num_sensors must be positive")
    if num_partitions < 2:
        raise ValueError("num_partitions must be at least 2")
    if num_partitions > num_sensors:
        raise ValueError("num_partitions cannot exceed num_sensors")
    shuffled = np.random.default_rng(seed).permutation(num_sensors)
    return [part.astype(np.int64) for part in np.array_split(shuffled, num_partitions)]


def _window_limit(value: int | None) -> int | None:
    if value is None or int(value) <= 0:
        return None
    return int(value)


@lru_cache(maxsize=16)
def _load_local_arrays(
    partition_id: int,
    num_partitions: int,
    data_path_string: str,
) -> tuple[np.ndarray, ...]:
    """Load/preprocess only the columns assigned to one simulated client."""
    requested_path = Path(data_path_string).expanduser()
    env_path = os.environ.get("FEDROAD_DATA")
    if env_path:
        data_path = Path(env_path).expanduser()
    elif requested_path.exists():
        data_path = requested_path
    else:
        # Flower may execute the packaged app from an isolated run directory,
        # so also look for the project-local dataset relative to the working dir.
        cwd_path = Path.cwd() / "data" / "raw" / "METR-LA.csv"
        module_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "METR-LA.csv"
        if cwd_path.exists():
            data_path = cwd_path
        elif module_path.exists():
            data_path = module_path
        else:
            data_path = cwd_path
    # The downloader validates the checksum if the file already exists and
    # downloads it only when it does not.
    download_metr_la(data_path)

    header = list(pd.read_csv(data_path, nrows=0).columns)
    if len(header) < 3:
        raise ValueError(f"Unexpected METR-LA columns: {len(header)}")

    sensor_columns = header[1:]
    partitions = partition_sensor_indices(len(sensor_columns), num_partitions)
    if not 0 <= partition_id < num_partitions:
        raise ValueError("partition_id must be between 0 and num_partitions - 1")

    global_indices = partitions[partition_id]
    local_sensor_columns = [sensor_columns[i] for i in global_indices]
    # Read only this client's sensor columns (plus timestamp), not other sensors.
    frame = pd.read_csv(data_path, usecols=[header[0], *local_sensor_columns])
    frame = frame[[header[0], *local_sensor_columns]]
    raw = frame[local_sensor_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)

    if raw.shape[0] < 100:
        raise ValueError(f"Dataset unexpectedly small: {raw.shape[0]} rows")

    observed = np.isfinite(raw) & (raw > 0)
    raw = raw.astype(np.float32)

    n_rows = len(raw)
    train_end = int(n_rows * 0.70)
    val_end = int(n_rows * 0.80)

    training_observed = np.where(observed[:train_end], raw[:train_end], np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        medians = np.nanmedian(training_observed, axis=0)
        means = np.nanmean(training_observed, axis=0)
        stds = np.nanstd(training_observed, axis=0)

    means = np.where(np.isfinite(means), means, 0.0).astype(np.float32)
    medians = np.where(np.isfinite(medians), medians, means).astype(np.float32)
    stds = np.where(np.isfinite(stds) & (stds > 1e-6), stds, 1.0).astype(np.float32)

    imputed = np.where(observed, raw, medians[None, :])
    scaled = ((imputed - means[None, :]) / stds[None, :]).astype(np.float32)
    observed = observed.astype(np.float32)

    return (
        scaled[:train_end], observed[:train_end],
        scaled[train_end:val_end], observed[train_end:val_end],
        scaled[val_end:], observed[val_end:],
        means, stds, global_indices, np.asarray(local_sensor_columns, dtype=str),
        np.asarray([100.0 * (1.0 - observed.mean())], dtype=np.float32),
    )


def _valid_target_count(mask: np.ndarray, dataset: TrafficWindowDataset) -> int:
    """Count observed targets used in all sampled windows and horizons."""
    starts = dataset.starts
    offsets = np.asarray(HORIZONS, dtype=np.int64)
    rows = starts[:, None] + INPUT_STEPS - 1 + offsets[None, :]
    return int(mask[rows].sum())


def load_client_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int = 16,
    max_train_windows: int | None = 1200,
    max_eval_windows: int | None = 1200,
    data_path: str | Path = DEFAULT_DATA_PATH,
) -> dict[str, Any]:
    """Return loaders and train-only scaler stats for one sensor partition.

    Setting max_*_windows to 0 or None uses every valid window. Limited window
    runs are for smoke tests only; use all windows for reportable experiments.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    arrays = _load_local_arrays(
        int(partition_id), int(num_partitions), str(Path(data_path).resolve())
    )
    x_train, m_train, x_val, m_val, x_test, m_test, mean, std, global_ids, names, missing = arrays

    train_ds = TrafficWindowDataset(
        x_train, m_train, max_windows=_window_limit(max_train_windows)
    )
    val_ds = TrafficWindowDataset(
        x_val, m_val, max_windows=_window_limit(max_eval_windows)
    )
    test_ds = TrafficWindowDataset(
        x_test, m_test, max_windows=_window_limit(max_eval_windows)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "mean": mean,
        "std": std,
        "global_sensor_indices": global_ids,
        "sensor_names": names.tolist(),
        "num_sensors": len(global_ids),
        "unavailable_percent": float(missing[0]),
        "train_examples": _valid_target_count(m_train, train_ds),
        "val_examples": _valid_target_count(m_val, val_ds),
        "test_examples": _valid_target_count(m_test, test_ds),
        "train_windows": len(train_ds),
        "val_windows": len(val_ds),
        "test_windows": len(test_ds),
    }
