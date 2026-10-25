"""Training and evaluation utilities for FedRoad GRU models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from fedroad.data import HORIZONS
from fedroad.model import SensorLevelGRU


def _flatten_batch(x, y, mask, device):
    """Flatten (batch, sensors, ...) into independent sensor sequences."""
    batch_size, sensor_count, steps, features = x.shape
    x = x.to(device).reshape(batch_size * sensor_count, steps, features)
    y = y.to(device).reshape(batch_size * sensor_count, -1)
    mask = mask.to(device).reshape(batch_size * sensor_count, -1)
    return x, y, mask


def train_one_epoch(
    model: SensorLevelGRU,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch using masked MAE in normalized units."""
    model.train()
    total_error = 0.0
    total_valid = 0.0

    for x, y, mask in loader:
        x, y, mask = _flatten_batch(x, y, mask, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x)
        error = (prediction - y).abs() * mask
        valid_count = mask.sum().clamp_min(1.0)
        loss = error.sum() / valid_count
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_error += error.detach().sum().item()
        total_valid += mask.sum().item()

    return total_error / max(total_valid, 1.0)


def normalized_validation_mae(
    model: SensorLevelGRU,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Calculate masked MAE in normalized units."""
    model.eval()
    total_error = 0.0
    total_valid = 0.0
    with torch.no_grad():
        for x, y, mask in loader:
            x, y, mask = _flatten_batch(x, y, mask, device)
            pred = model(x)
            total_error += ((pred - y).abs() * mask).sum().item()
            total_valid += mask.sum().item()
    return total_error / max(total_valid, 1.0)


def evaluate(
    model: SensorLevelGRU | None,
    loader: DataLoader,
    sensor_indices: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    model_name: str,
    persistence: bool = False,
) -> list[dict[str, float | int | str]]:
    """Compute masked MAE/RMSE in the original speed units."""
    if not persistence and model is None:
        raise ValueError("A model is required unless persistence=True.")
    if model is not None:
        model.eval()

    errors = [[] for _ in HORIZONS]
    valid_counts = [0 for _ in HORIZONS]
    means = torch.as_tensor(mean[sensor_indices], dtype=torch.float32)
    stds = torch.as_tensor(std[sensor_indices], dtype=torch.float32)

    with torch.no_grad():
        for x, y, mask in loader:
            batch_size, sensor_count, steps, _ = x.shape
            if persistence:
                speeds = x[..., 0]
                availability = x[..., 1]
                reverse_mask = torch.flip(availability, dims=[2])
                offset = reverse_mask.argmax(dim=2)
                last_index = steps - 1 - offset
                latest = speeds.gather(2, last_index.unsqueeze(-1)).squeeze(-1)
                pred = latest.unsqueeze(-1).expand(-1, -1, len(HORIZONS))
            else:
                flat_x = x.to(device).reshape(-1, steps, 2)
                flat_pred = model(flat_x)
                pred = flat_pred.reshape(batch_size, sensor_count, len(HORIZONS)).cpu()

            mu = means.view(1, sensor_count, 1)
            sigma = stds.view(1, sensor_count, 1)
            pred_raw = pred * sigma + mu
            truth_raw = y * sigma + mu

            for horizon_index in range(len(HORIZONS)):
                valid = mask[:, :, horizon_index] > 0.5
                error = (pred_raw[:, :, horizon_index] - truth_raw[:, :, horizon_index])
                selected = error[valid].numpy()
                if selected.size:
                    errors[horizon_index].append(selected)
                    valid_counts[horizon_index] += int(selected.size)

    rows = []
    for i, offset in enumerate(HORIZONS):
        all_errors = np.concatenate(errors[i]) if errors[i] else np.array([], dtype=float)
        rows.append({
            "Model": model_name,
            "Horizon (minutes)": int(offset * 5),
            "MAE": float(np.abs(all_errors).mean()) if all_errors.size else float("nan"),
            "RMSE": float(np.sqrt(np.mean(all_errors ** 2))) if all_errors.size else float("nan"),
            "Valid targets": valid_counts[i],
        })
    return rows


def metrics_frame(rows: list[dict[str, float | int | str]]) -> pd.DataFrame:
    """Return a tidy metrics table."""
    return pd.DataFrame(rows, columns=[
        "Model", "Horizon (minutes)", "MAE", "RMSE", "Valid targets"
    ])
