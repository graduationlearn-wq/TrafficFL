"""Train and evaluate the experimental spatio-temporal graph GRU on METR-LA.

By default this is a smoke-sized run. Use --max-train-windows 0 and
--max-eval-windows 0 for a full benchmark (prefer a GPU-backed runtime).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from fedroad.data import (
    DEFAULT_DATA_PATH,
    HORIZONS,
    TrafficWindowDataset,
    download_metr_la,
    load_and_preprocess,
)
from fedroad.graph_data import DEFAULT_GRAPH_PATH, load_metr_la_adjacency
from fedroad.graph_model import SpatioTemporalGraphGRU, normalize_directed_adjacencies

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-train-windows", type=int, default=1200,
                        help="0 uses all windows; small positive values are for smoke tests.")
    parser.add_argument("--max-eval-windows", type=int, default=1200,
                        help="0 uses all windows; small positive values are for smoke tests.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--graph-path", type=Path, default=DEFAULT_GRAPH_PATH)
    return parser.parse_args()


def _window_limit(value: int) -> int | None:
    return None if value <= 0 else value


def _make_loader(
    values: np.ndarray,
    mask: np.ndarray,
    limit: int | None,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TrafficWindowDataset(
        values,
        mask,
        sensor_indices=np.arange(values.shape[1], dtype=np.int64),
        max_windows=limit,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _train_epoch(
    model: SpatioTemporalGraphGRU,
    loader: DataLoader,
    forward_adjacency: torch.Tensor,
    backward_adjacency: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_error = 0.0
    total_valid = 0.0
    for x, y, mask in loader:
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x, forward_adjacency, backward_adjacency)
        error = (prediction - y).abs() * mask
        loss = error.sum() / mask.sum().clamp_min(1.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_error += error.detach().sum().item()
        total_valid += mask.sum().item()
    return total_error / max(total_valid, 1.0)


def _validation_mae(
    model: SpatioTemporalGraphGRU,
    loader: DataLoader,
    forward_adjacency: torch.Tensor,
    backward_adjacency: torch.Tensor,
    device: torch.device,
) -> float:
    model.eval()
    total_error = 0.0
    total_valid = 0.0
    with torch.no_grad():
        for x, y, mask in loader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            prediction = model(x, forward_adjacency, backward_adjacency)
            total_error += ((prediction - y).abs() * mask).sum().item()
            total_valid += mask.sum().item()
    return total_error / max(total_valid, 1.0)


def _evaluate(
    model: SpatioTemporalGraphGRU | None,
    loader: DataLoader,
    forward_adjacency: torch.Tensor,
    backward_adjacency: torch.Tensor,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    model_name: str,
    persistence: bool = False,
) -> list[dict[str, float | int | str]]:
    if model is None and not persistence:
        raise ValueError("model is required unless persistence=True")
    if model is not None:
        model.eval()

    errors_by_horizon: list[list[np.ndarray]] = [[] for _ in HORIZONS]
    valid_counts = [0 for _ in HORIZONS]
    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=device).view(1, -1, 1)
    std_t = torch.as_tensor(std, dtype=torch.float32, device=device).view(1, -1, 1)

    with torch.no_grad():
        for x, y, mask in loader:
            x_device = x.to(device)
            if persistence:
                speeds = x_device[..., 0]
                availability = x_device[..., 1]
                reverse_mask = torch.flip(availability, dims=[2])
                offset = reverse_mask.argmax(dim=2)
                last_index = x_device.shape[2] - 1 - offset
                latest = speeds.gather(2, last_index.unsqueeze(-1)).squeeze(-1)
                prediction = latest.unsqueeze(-1).expand(-1, -1, len(HORIZONS))
            else:
                prediction = model(x_device, forward_adjacency, backward_adjacency)

            target = y.to(device)
            target_mask = mask.to(device)
            pred_raw = prediction * std_t + mean_t
            target_raw = target * std_t + mean_t
            for horizon_index in range(len(HORIZONS)):
                valid = target_mask[:, :, horizon_index] > 0.5
                error = pred_raw[:, :, horizon_index] - target_raw[:, :, horizon_index]
                chosen = error[valid].detach().cpu().numpy()
                if chosen.size:
                    errors_by_horizon[horizon_index].append(chosen)
                    valid_counts[horizon_index] += int(chosen.size)

    rows: list[dict[str, float | int | str]] = []
    for horizon_index, offset in enumerate(HORIZONS):
        errors = (
            np.concatenate(errors_by_horizon[horizon_index])
            if errors_by_horizon[horizon_index]
            else np.array([], dtype=np.float32)
        )
        rows.append({
            "Model": model_name,
            "Horizon (minutes)": int(offset * 5),
            "MAE": float(np.abs(errors).mean()) if errors.size else float("nan"),
            "RMSE": float(np.sqrt(np.mean(errors ** 2))) if errors.size else float("nan"),
            "Valid targets": valid_counts[horizon_index],
        })
    return rows


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.patience < 1 or args.lr_patience < 1:
        raise ValueError("epochs and patience values must be positive")
    if args.batch_size < 1 or args.hidden_size < 1 or args.learning_rate <= 0:
        raise ValueError("batch-size, hidden-size and learning-rate must be positive")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.data_path.exists():
        download_metr_la(args.data_path)
    data = load_and_preprocess(args.data_path)
    raw_adjacency = load_metr_la_adjacency(data.sensor_ids, args.graph_path)
    forward_adjacency, backward_adjacency = normalize_directed_adjacencies(raw_adjacency)
    forward_adjacency = forward_adjacency.to(device)
    backward_adjacency = backward_adjacency.to(device)
    graph_asymmetry = torch.max(torch.abs(raw_adjacency - raw_adjacency.T)).item()

    max_train = _window_limit(args.max_train_windows)
    max_eval = _window_limit(args.max_eval_windows)
    train_loader = _make_loader(
        data.x_train, data.m_train, max_train, args.batch_size, True
    )
    val_loader = _make_loader(
        data.x_val, data.m_val, max_eval, args.batch_size, False
    )
    test_loader = _make_loader(
        data.x_test, data.m_test, max_eval, args.batch_size, False
    )

    model = SpatioTemporalGraphGRU(hidden_size=args.hidden_size).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.lr_patience, min_lr=1e-5
    )

    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale_epochs = 0
    epochs_run = 0

    print("FedRoad experimental spatio-temporal graph GRU")
    print(f"Device: {device}")
    print(f"Rows/sensors: {len(data.timestamps):,} / {len(data.sensor_ids)}")
    print(f"Unavailable readings: {data.zero_percent:.2f}%")
    print(
        f"Windows: train={len(train_loader.dataset):,}, "
        f"val={len(val_loader.dataset):,}, test={len(test_loader.dataset):,}"
    )
    print(
        f"Directed graph: {tuple(raw_adjacency.shape)}; "
        f"maximum adjacency asymmetry={graph_asymmetry:.6f}; max epochs={args.epochs}"
    )
    print("Spatial operator: separate forward/reverse random walks; no symmetrization.")
    if max_train is not None or max_eval is not None:
        print("Window limits active: this is a smoke/development run, not a full benchmark.")

    for epoch in range(1, args.epochs + 1):
        epochs_run = epoch
        train_mae = _train_epoch(model, train_loader, forward_adjacency, backward_adjacency, optimizer, device)
        val_mae = _validation_mae(model, val_loader, forward_adjacency, backward_adjacency, device)
        scheduler.step(val_mae)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | train masked MAE: {train_mae:.4f} | "
            f"validation masked MAE: {val_mae:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )
        if val_mae < best_val - 1e-6:
            best_val = val_mae
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ARTIFACT_DIR / "graph_spatiotemporal_gru.pt"
    torch.save({
        "model_state_dict": best_state,
        "sensor_ids": data.sensor_ids,
        "mean": data.mean,
        "std": data.std,
        "horizons": [int(h * 5) for h in HORIZONS],
        "hidden_size": args.hidden_size,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_validation_masked_mae_normalized": best_val,
    }, checkpoint_path)

    rows = _evaluate(
        None, test_loader, forward_adjacency, backward_adjacency, data.mean, data.std, device,
        "Persistence", persistence=True,
    )
    rows += _evaluate(
        model, test_loader, forward_adjacency, backward_adjacency, data.mean, data.std, device,
        "SpatioTemporalGraphGRU",
    )
    metrics = pd.DataFrame(
        rows, columns=["Model", "Horizon (minutes)", "MAE", "RMSE", "Valid targets"]
    )
    metrics_path = ARTIFACT_DIR / "graph_model_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    metadata_path = ARTIFACT_DIR / "graph_model_metadata.json"
    metadata_path.write_text(json.dumps({
        "dataset": "METR-LA",
        "graph_md5": "55996dfce3e39cc981e6e42e3f427da8",
        "model": "SpatioTemporalGraphGRU",
        "graph_operator": "directed_forward_backward_random_walk",
        "diffusion_steps": model.diffusion_steps,
        "adjacency_symmetrized": False,
        "original_graph_max_asymmetry": graph_asymmetry,
        "training_mode": "centralized experimental candidate",
        "rows": len(data.timestamps),
        "sensors": len(data.sensor_ids),
        "unavailable_readings_percent": data.zero_percent,
        "train_windows_used": len(train_loader.dataset),
        "validation_windows_used": len(val_loader.dataset),
        "test_windows_used": len(test_loader.dataset),
        "max_epochs": args.epochs,
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "best_validation_masked_mae_normalized": best_val,
        "batch_size": args.batch_size,
        "hidden_size": args.hidden_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "device": str(device),
        "training_windows_limited": max_train is not None,
        "evaluation_windows_limited": max_eval is not None,
        "note": "Directed graph model is a centralized experimental candidate and is not yet integrated into the federated Flower apps.",
    }, indent=2), encoding="utf-8")

    print("\nHeld-out test metrics (original speed units):")
    print(metrics.round(3).to_string(index=False))
    print(f"Best validation epoch: {best_epoch}/{epochs_run}")
    print(f"Saved model: {checkpoint_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
