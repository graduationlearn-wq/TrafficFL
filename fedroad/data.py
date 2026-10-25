"""Download and preprocess the real METR-LA traffic-speed dataset."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import requests
from torch.utils.data import Dataset
import torch

DATA_URL = "https://zenodo.org/records/5724362/files/METR-LA.csv?download=1"
EXPECTED_MD5 = "7c463fcbd28ecc6c9cd93e1ca120aaaa"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "METR-LA.csv"
INPUT_STEPS = 12
HORIZONS = (3, 6, 12)  # five-minute samples: 15, 30 and 60 minutes


@dataclass
class PreparedTrafficData:
    """Chronological normalized splits plus training-only scaler statistics."""

    x_train: np.ndarray
    m_train: np.ndarray
    x_val: np.ndarray
    m_val: np.ndarray
    x_test: np.ndarray
    m_test: np.ndarray
    sensor_ids: list[str]
    mean: np.ndarray
    std: np.ndarray
    zero_percent: float
    timestamps: pd.DatetimeIndex


class WindowBatch(NamedTuple):
    x: torch.Tensor
    y: torch.Tensor
    mask: torch.Tensor


def file_md5(path: Path) -> str:
    """Compute a file's MD5 checksum in chunks."""
    digest = hashlib.md5()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_metr_la(path: Path = DEFAULT_DATA_PATH, timeout: int = 180) -> Path:
    """Download METR-LA from Zenodo and verify its published checksum."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        actual = file_md5(path)
        if actual == EXPECTED_MD5:
            print(f"Dataset already exists and checksum is valid: {path}")
            return path
        raise ValueError(
            f"Existing dataset checksum mismatch for {path}. "
            "Move it aside and download it again; it has not been overwritten."
        )

    temporary_path = path.with_suffix(path.suffix + ".part")
    digest = hashlib.md5()
    try:
        with requests.get(DATA_URL, stream=True, timeout=(20, timeout)) as response:
            response.raise_for_status()
            with temporary_path.open("wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file.write(chunk)
                        digest.update(chunk)

        actual = digest.hexdigest()
        if actual != EXPECTED_MD5:
            raise ValueError(
                "Downloaded METR-LA checksum mismatch. "
                f"Expected {EXPECTED_MD5}, received {actual}."
            )
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    print(f"Downloaded and verified METR-LA: {path}")
    print(f"MD5: {actual}")
    return path


def _normalize_split(
    raw_values: np.ndarray,
    observed: np.ndarray,
    median: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Impute unavailable INPUT readings with train medians, then scale."""
    imputed = np.where(observed, raw_values, median[None, :])
    return ((imputed - mean[None, :]) / std[None, :]).astype(np.float32)


def load_and_preprocess(path: Path = DEFAULT_DATA_PATH) -> PreparedTrafficData:
    """Load METR-LA, preserve observation masks, and split time chronologically.

    The first CSV column is expected to contain timestamps. Non-positive or
    non-finite sensor readings are treated as unavailable, consistent with the
    documented preprocessing assumption used in this project. The original CSV
    is never changed. Statistics are fitted on observed TRAIN readings only.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Traffic dataset not found at {path}. Run: "
            "python -m scripts.download_data"
        )

    frame = pd.read_csv(path)
    if frame.shape[1] < 2:
        raise ValueError("Expected a timestamp column and at least one sensor column.")

    timestamps = pd.to_datetime(frame.iloc[:, 0], errors="coerce")
    if timestamps.notna().mean() < 0.95:
        raise ValueError("The first CSV column does not appear to contain timestamps.")

    sensor_ids = [str(column) for column in frame.columns[1:]]
    raw = frame.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(
        dtype=np.float64
    )
    observed = np.isfinite(raw) & (raw > 0)
    raw = raw.astype(np.float32)
    n_rows = raw.shape[0]
    if n_rows < 100:
        raise ValueError(f"Dataset unexpectedly small: {n_rows} rows.")

    train_end = int(n_rows * 0.70)
    val_end = int(n_rows * 0.80)
    raw_train = raw[:train_end]
    mask_train = observed[:train_end]

    # Fit preprocessing statistics on observed training readings only.
    masked_train = np.where(mask_train, raw_train, np.nan).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nanmean(masked_train, axis=0)
        median = np.nanmedian(masked_train, axis=0)
        std = np.nanstd(masked_train, axis=0)

    mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
    median = np.where(np.isfinite(median), median, mean).astype(np.float32)
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0).astype(np.float32)

    def make_part(start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
        part = raw[start:end]
        part_mask = observed[start:end]
        scaled = _normalize_split(part, part_mask, median, mean, std)
        return scaled, part_mask.astype(np.float32)

    x_train, m_train = make_part(0, train_end)
    x_val, m_val = make_part(train_end, val_end)
    x_test, m_test = make_part(val_end, n_rows)

    return PreparedTrafficData(
        x_train=x_train,
        m_train=m_train,
        x_val=x_val,
        m_val=m_val,
        x_test=x_test,
        m_test=m_test,
        sensor_ids=sensor_ids,
        mean=mean,
        std=std,
        zero_percent=float(100.0 * (1.0 - observed.mean())),
        timestamps=pd.DatetimeIndex(timestamps),
    )


class TrafficWindowDataset(Dataset[WindowBatch]):
    """Return a time window for each selected sensor, with masked targets.

    An item has shapes x=(sensors, 12, 2), y=(sensors, 3),
    mask=(sensors, 3). ``max_windows`` can be used for a quick CPU experiment;
    set it to None to use every valid window.
    """

    def __init__(
        self,
        values: np.ndarray,
        mask: np.ndarray,
        sensor_indices: np.ndarray | None = None,
        input_steps: int = INPUT_STEPS,
        horizons: tuple[int, ...] = HORIZONS,
        max_windows: int | None = 1200,
    ) -> None:
        if values.ndim != 2 or mask.shape != values.shape:
            raise ValueError("values and mask must be matching 2D arrays.")
        self.values = np.asarray(values, dtype=np.float32)
        self.mask = np.asarray(mask, dtype=np.float32)
        self.sensor_indices = (
            np.arange(values.shape[1], dtype=np.int64)
            if sensor_indices is None
            else np.asarray(sensor_indices, dtype=np.int64)
        )
        if self.sensor_indices.size == 0:
            raise ValueError("At least one sensor must be selected.")
        if np.any(self.sensor_indices < 0) or np.any(
            self.sensor_indices >= values.shape[1]
        ):
            raise ValueError("sensor_indices contain an out-of-range index.")

        self.input_steps = input_steps
        self.horizons = tuple(horizons)
        if not self.horizons or min(self.horizons) < 1:
            raise ValueError("horizons must contain positive future offsets.")
        count = len(values) - input_steps - max(self.horizons) + 1
        if count <= 0:
            raise ValueError("Not enough rows to create forecasting windows.")

        if max_windows is None or max_windows <= 0 or max_windows >= count:
            self.starts = np.arange(count, dtype=np.int64)
        else:
            self.starts = np.linspace(0, count - 1, max_windows, dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self.starts))

    def __getitem__(self, index: int) -> WindowBatch:
        start = int(self.starts[index])
        sensors = self.sensor_indices
        end = start + self.input_steps

        speed_history = self.values[start:end][:, sensors].T
        observed_history = self.mask[start:end][:, sensors].T
        x = np.stack((speed_history, observed_history), axis=-1)

        target_rows = [end - 1 + horizon for horizon in self.horizons]
        y = self.values[target_rows][:, sensors].T
        target_mask = self.mask[target_rows][:, sensors].T

        return WindowBatch(
            torch.from_numpy(x.copy()),
            torch.from_numpy(y.copy()),
            torch.from_numpy(target_mask.copy()),
        )
