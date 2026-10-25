"""Train and evaluate a reproducible centralized GRU benchmark on METR-LA.

The default configuration uses all available windows, early stopping on the
validation split, and saves the best validation checkpoint. Use window limits
only for quick development smoke tests; do not report those as full benchmarks.
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
    TrafficWindowDataset,
    download_metr_la,
    load_and_preprocess,
)
from fedroad.engine import (
    evaluate,
    metrics_frame,
    normalized_validation_mae,
    train_one_epoch,
)
from fedroad.model import SensorLevelGRU

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Maximum epochs; early stopping selects the best validation checkpoint.",
    )
    parser.add_argument("--patience", type=int, default=7,
                        help="Stop after this many epochs without validation improvement.")
    parser.add_argument("--lr-patience", type=int, default=2,
                        help="Reduce learning rate after this many stagnant validation epochs.")
    parser.add_argument("--batch-windows", type=int, default=16)
    parser.add_argument(
        "--max-train-windows", type=int, default=0,
        help="Maximum training windows; 0 uses every available training window.",
    )
    parser.add_argument(
        "--max-eval-windows", type=int, default=0,
        help="Maximum validation/test windows; 0 uses every available window.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    return parser.parse_args()


def _window_limit(value: int) -> int | None:
    return None if value <= 0 else value


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_windows < 1:
        raise ValueError("epochs and batch-windows must be positive.")
    if args.patience < 1 or args.lr_patience < 1:
        raise ValueError("patience and lr-patience must be positive.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.data_path.exists():
        download_metr_la(args.data_path)
    data = load_and_preprocess(args.data_path)
    sensor_indices = np.arange(len(data.sensor_ids), dtype=np.int64)
    max_train = _window_limit(args.max_train_windows)
    max_eval = _window_limit(args.max_eval_windows)

    train_ds = TrafficWindowDataset(
        data.x_train, data.m_train, sensor_indices, max_windows=max_train
    )
    val_ds = TrafficWindowDataset(
        data.x_val, data.m_val, sensor_indices, max_windows=max_eval
    )
    test_ds = TrafficWindowDataset(
        data.x_test, data.m_test, sensor_indices, max_windows=max_eval
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_windows, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_windows, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_windows, shuffle=False, num_workers=0
    )

    model = SensorLevelGRU().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5,
        patience=args.lr_patience, min_lr=1e-5
    )

    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    epochs_without_improvement = 0
    epochs_run = 0

    print("FedRoad centralized sensor-level GRU baseline")
    print(f"Device: {device}")
    print(f"Rows/sensors: {len(data.timestamps):,} / {len(data.sensor_ids)}")
    print(f"Unavailable readings: {data.zero_percent:.2f}%")
    print(
        f"Window counts: train={len(train_ds):,}, "
        f"val={len(val_ds):,}, test={len(test_ds):,}"
    )
    print(f"Maximum epochs: {args.epochs}; early-stopping patience: {args.patience}")

    for epoch in range(1, args.epochs + 1):
        epochs_run = epoch
        train_mae = train_one_epoch(model, train_loader, optimizer, device)
        val_mae = normalized_validation_mae(model, val_loader, device)
        scheduler.step(val_mae)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train masked MAE (normalized): {train_mae:.4f} | "
            f"Val masked MAE (normalized): {val_mae:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        if val_mae < best_val - 1e-6:
            best_val = val_mae
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(
                f"Early stopping at epoch {epoch}; "
                f"best validation epoch was {best_epoch}."
            )
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    model_path = ARTIFACT_DIR / "centralized_sensor_gru.pt"
    torch.save({
        "model_state_dict": best_state,
        "sensor_ids": data.sensor_ids,
        "mean": data.mean,
        "std": data.std,
        "horizons": [15, 30, 60],
        "hidden_size": 32,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_validation_masked_mae_normalized": best_val,
    }, model_path)

    rows = evaluate(
        None, test_loader, sensor_indices, data.mean, data.std, device,
        "Persistence", persistence=True,
    )
    rows += evaluate(
        model, test_loader, sensor_indices, data.mean, data.std, device,
        "Centralized shared GRU",
    )
    result_df = metrics_frame(rows)
    metrics_path = ARTIFACT_DIR / "baseline_metrics.csv"
    result_df.to_csv(metrics_path, index=False)

    metadata_path = ARTIFACT_DIR / "baseline_metadata.json"
    metadata_path.write_text(json.dumps({
        "dataset": "METR-LA",
        "dataset_md5": "7c463fcbd28ecc6c9cd93e1ca120aaaa",
        "rows": len(data.timestamps),
        "sensors": len(data.sensor_ids),
        "unavailable_readings_percent": data.zero_percent,
        "train_windows_used": len(train_ds),
        "validation_windows_used": len(val_ds),
        "test_windows_used": len(test_ds),
        "max_epochs": args.epochs,
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "early_stopping_patience": args.patience,
        "lr_patience": args.lr_patience,
        "initial_learning_rate": args.learning_rate,
        "batch_windows": args.batch_windows,
        "seed": args.seed,
        "device": str(device),
        "best_validation_masked_mae_normalized": best_val,
        "training_windows_limited": max_train is not None,
        "evaluation_windows_limited": max_eval is not None,
    }, indent=2), encoding="utf-8")

    print("\nTest metrics (original speed units):")
    print(result_df.round(3).to_string(index=False))
    print(f"\nBest validation epoch: {best_epoch}/{epochs_run}")
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
