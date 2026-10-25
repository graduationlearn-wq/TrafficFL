"""Evaluate the saved FedAvg model and persistence baseline on held-out data."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fedroad.data import DEFAULT_DATA_PATH, HORIZONS
from fedroad.federated_data import load_client_data
from fedroad.model import SensorLevelGRU

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "artifacts" / "fedavg_sensor_gru.pt")
    parser.add_argument("--num-clients", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-eval-windows", type=int, default=0,
                        help="0 uses every held-out window (recommended for final metrics).")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts" / "federated_metrics.csv")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"FedAvg checkpoint not found: {args.model}. Run Flower first.")
    if args.num_clients < 2:
        raise ValueError("num-clients must be at least 2")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SensorLevelGRU().to(device)
    state = torch.load(args.model, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    model_sums = {key: {"abs": np.zeros(len(HORIZONS)), "sq": np.zeros(len(HORIZONS)),
                        "count": np.zeros(len(HORIZONS), dtype=np.int64)}
                  for key in ("Persistence", "FedAvg GRU")}

    for client_id in range(args.num_clients):
        data = load_client_data(
            partition_id=client_id,
            num_partitions=args.num_clients,
            batch_size=args.batch_size,
            max_train_windows=1200,
            max_eval_windows=args.max_eval_windows,
            data_path=args.data_path,
        )
        mean = torch.as_tensor(data["mean"], dtype=torch.float32).view(1, -1, 1)
        std = torch.as_tensor(data["std"], dtype=torch.float32).view(1, -1, 1)

        with torch.no_grad():
            for x, y, mask in data["test_loader"]:
                batch_size, sensor_count, steps, features = x.shape
                flat_x = x.to(device).reshape(batch_size * sensor_count, steps, features)
                pred_fed = model(flat_x).reshape(batch_size, sensor_count, len(HORIZONS)).cpu()

                # Persistence uses the latest observed value in each input sequence.
                speed = x[..., 0]
                observed = x[..., 1]
                reverse_mask = torch.flip(observed, dims=[2])
                offset = reverse_mask.argmax(dim=2)
                last_index = steps - 1 - offset
                latest = speed.gather(2, last_index.unsqueeze(-1)).squeeze(-1)
                pred_persistence = latest.unsqueeze(-1).expand(-1, -1, len(HORIZONS))

                true_raw = y * std + mean
                mask_bool = mask > 0.5

                for name, prediction in (("Persistence", pred_persistence), ("FedAvg GRU", pred_fed)):
                    pred_raw = prediction * std + mean
                    error = pred_raw - true_raw
                    for h in range(len(HORIZONS)):
                        valid = mask_bool[:, :, h]
                        selected = error[:, :, h][valid]
                        model_sums[name]["abs"][h] += selected.abs().sum().item()
                        model_sums[name]["sq"][h] += selected.square().sum().item()
                        model_sums[name]["count"][h] += selected.numel()
        print(f"Evaluated client {client_id}: {data['num_sensors']} sensors")

    rows = []
    for name, sums in model_sums.items():
        for index, horizon in enumerate(HORIZONS):
            count = max(int(sums["count"][index]), 1)
            rows.append({
                "Model": name,
                "Horizon (minutes)": int(horizon * 5),
                "MAE": float(sums["abs"][index] / count),
                "RMSE": float(np.sqrt(sums["sq"][index] / count)),
                "Valid targets": int(sums["count"][index]),
            })

    results = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print("\nHeld-out test metrics (original speed units):")
    print(results.round(3).to_string(index=False))
    print(f"\nSaved metrics: {args.output}")
    print("Use these test results only after model/configuration choices are finalized.")


if __name__ == "__main__":
    main()
