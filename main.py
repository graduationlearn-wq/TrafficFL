"""TrafficFL: simple multi-target traffic forecasting with GRU + simulated FedAvg."""

from __future__ import annotations

import argparse
import hashlib
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw"
ARTIFACT_DIR = ROOT / "artifacts"
RESULT_DIR = ROOT / "results"
DATA_FILE = DATA_DIR / "PEMS04.npz"
PART_FILE = DATA_DIR / "PEMS04.npz.part"
CHECKPOINT = ARTIFACT_DIR / "traffic_gru_fedavg.pt"

DATA_URL = "https://raw.githubusercontent.com/guoshnBJTU/ASTGNN/main/data/PEMS04/PEMS04.npz"
DATA_SHA256 = "95a3c9b720fffdb85f0330d09bfab41b0b3cad0ca86c0d7d5f3accacb4ac999a"
EXPECTED_SHAPE = (16992, 307, 3)
FEATURES = ["flow", "occupancy", "speed"]  # PEMS04 channel order
INPUT_STEPS = 12
HORIZONS = [3, 6, 12]  # 15, 30 and 60 minutes at 5-minute intervals
TARGET_OFFSETS = [14, 17, 23]  # relative to the start of each 12-step input
START_TIME = "2018-01-01 00:00:00"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TrafficGRU(nn.Module):
    """Small GRU that predicts flow, occupancy and speed at three horizons."""

    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.gru = nn.GRU(input_size=3, hidden_size=hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        return self.output(hidden[-1]).reshape(-1, 3, 3)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset() -> Path:
    """Download with retry + resume. A partial file is kept if a connection drops."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if DATA_FILE.exists():
        if sha256_file(DATA_FILE) == DATA_SHA256:
            print(f"Dataset already verified: {DATA_FILE}")
            return DATA_FILE
        print("Existing dataset checksum is incorrect; downloading a clean copy.")
        DATA_FILE.unlink()

    for attempt in range(1, 7):
        downloaded = PART_FILE.stat().st_size if PART_FILE.exists() else 0
        headers = {"User-Agent": "TrafficFL-student-project/1.0"}
        if downloaded:
            headers["Range"] = f"bytes={downloaded}-"

        try:
            print(f"Dataset download attempt {attempt}/6 (resume at {downloaded // (1024 * 1024)} MB).")
            with requests.get(DATA_URL, headers=headers, stream=True, timeout=(20, 90)) as response:
                if response.status_code == 416:
                    if PART_FILE.exists() and sha256_file(PART_FILE) == DATA_SHA256:
                        os.replace(PART_FILE, DATA_FILE)
                        print("Download already complete; checksum verified.")
                        return DATA_FILE
                    PART_FILE.unlink(missing_ok=True)
                    raise requests.HTTPError("Server rejected the resume request; partial file cleared for a clean retry.")

                response.raise_for_status()
                if downloaded and response.status_code == 206:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {downloaded}-"):
                        PART_FILE.unlink(missing_ok=True)
                        raise requests.RequestException("Unexpected Content-Range; partial file cleared for a clean retry.")
                    mode = "ab"
                else:
                    # The server may ignore Range. In that case, safely restart the partial download.
                    downloaded = 0
                    mode = "wb"

                content_length = response.headers.get("Content-Length")
                expected_total = None
                content_range = response.headers.get("Content-Range", "")
                if "/" in content_range and content_range.rsplit("/", 1)[1].isdigit():
                    expected_total = int(content_range.rsplit("/", 1)[1])
                elif content_length and content_length.isdigit():
                    expected_total = downloaded + int(content_length)

                last_reported = downloaded // (5 * 1024 * 1024)
                with PART_FILE.open(mode) as file:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        file.write(chunk)
                        current = file.tell()
                        report = current // (5 * 1024 * 1024)
                        if report > last_reported:
                            print(f"  Downloaded {current // (1024 * 1024)} MB")
                            last_reported = report

            actual_size = PART_FILE.stat().st_size
            if expected_total is not None and actual_size != expected_total:
                raise requests.ConnectionError(
                    f"Download ended early ({actual_size} of {expected_total} bytes). Partial file retained."
                )

            actual_hash = sha256_file(PART_FILE)
            if actual_hash != DATA_SHA256:
                PART_FILE.unlink(missing_ok=True)
                raise ValueError("Dataset checksum mismatch. The incomplete/corrupt file was removed.")

            os.replace(PART_FILE, DATA_FILE)
            print("Dataset download completed and SHA-256 checksum verified.")
            return DATA_FILE

        except (requests.RequestException, OSError, ValueError) as error:
            print(f"  Download attempt failed: {error}")
            if attempt < 6:
                wait = min(2 ** (attempt - 1), 20)
                print(f"  Retrying in {wait} seconds. Any valid partial download will be resumed.")
                time.sleep(wait)

    raise RuntimeError(
        "Dataset download failed after six attempts. Run `python main.py inspect` again; "
        "the downloader will try to resume data/raw/PEMS04.npz.part."
    )


def load_dataset() -> np.ndarray:
    path = download_dataset()
    with np.load(path, allow_pickle=False) as archive:
        if "data" not in archive:
            raise ValueError("PEMS04 archive does not contain the expected 'data' array.")
        values = np.array(archive["data"], dtype=np.float32, copy=True)

    if values.shape != EXPECTED_SHAPE:
        raise ValueError(f"Unexpected PEMS04 shape {values.shape}; expected {EXPECTED_SHAPE}.")
    return values


def inspect_dataset() -> None:
    values = load_dataset()
    print("\nPEMS04 DATASET CHECK")
    print(f"Shape: {values.shape[0]:,} time steps x {values.shape[1]} sensors x {values.shape[2]} features")
    print("Sampling: every 5 minutes")
    print("Period: 1 January 2018 to 28 February 2018")
    print("Channels: 0=flow (vehicles/5 min), 1=occupancy (fraction 0-1), 2=speed (mph)")
    print("Occupancy is stored as a fraction; multiply by 100 to display it as a percentage.")
    for index, name in enumerate(FEATURES):
        channel = values[:, :, index]
        valid = channel[np.isfinite(channel)]
        if name == "occupancy":
            print(
                f"{name:10s} | valid={len(valid):,} | min={valid.min():.3f} "
                f"| mean={valid.mean():.3f} ({valid.mean() * 100:.2f}%) "
                f"| max={valid.max():.3f} ({valid.max() * 100:.2f}%)"
            )
        else:
            print(
                f"{name:10s} | valid={len(valid):,} | min={valid.min():.3f} "
                f"| mean={valid.mean():.3f} | max={valid.max():.3f}"
            )
    print("\nDataset is readable and ready for training.")


def prepare_data(raw: np.ndarray):
    """Chronological 70/10/20 split; fit scalers using training rows only."""
    time_steps = raw.shape[0]
    train_end = int(time_steps * 0.70)
    val_end = int(time_steps * 0.80)

    observed = np.isfinite(raw) & (raw >= 0)
    train = raw[:train_end]
    train_mask = observed[:train_end]

    means = np.zeros(3, dtype=np.float32)
    medians = np.zeros(3, dtype=np.float32)
    stds = np.ones(3, dtype=np.float32)
    for feature in range(3):
        valid = train[:, :, feature][train_mask[:, :, feature]]
        if valid.size == 0:
            raise ValueError(f"No valid training values found for {FEATURES[feature]}.")
        means[feature] = float(valid.mean())
        medians[feature] = float(np.median(valid))
        scale = float(valid.std())
        stds[feature] = scale if scale > 1e-6 else 1.0

    filled = raw.copy()
    for feature in range(3):
        filled[:, :, feature][~observed[:, :, feature]] = medians[feature]

    normalized = (filled - means.reshape(1, 1, 3)) / stds.reshape(1, 1, 3)
    normalized = normalized.astype(np.float32)

    # Sensor-specific 85th-percentile speed is the reference for a transparent congestion heuristic.
    global_free_speed = float(np.percentile(train[:, :, 2][train_mask[:, :, 2]], 85))
    free_flow_speed = np.full(raw.shape[1], global_free_speed, dtype=np.float32)
    for sensor in range(raw.shape[1]):
        valid_speed = train[:, sensor, 2][train_mask[:, sensor, 2]]
        if valid_speed.size >= 20:
            free_flow_speed[sensor] = float(np.percentile(valid_speed, 85))

    return normalized, observed, means, medians, stds, free_flow_speed, train_end, val_end


def make_windows(
    normalized: np.ndarray,
    observed: np.ndarray,
    sensor: int,
    start: int,
    end: int,
    max_windows: int | None = None,
    seed: int = 42,
):
    """Create 12-step inputs and three multi-feature targets for one sensor."""
    starts = np.arange(start, end - 23, dtype=np.int64)
    if starts.size == 0:
        return None

    if max_windows is not None and starts.size > max_windows:
        # Evenly spaced windows retain coverage across the complete training period.
        chosen = np.linspace(0, starts.size - 1, max_windows, dtype=np.int64)
        starts = starts[chosen]

    x = np.stack([normalized[i : i + INPUT_STEPS, sensor, :] for i in starts])
    target_indices = starts[:, None] + np.asarray(TARGET_OFFSETS, dtype=np.int64)[None, :]
    y = normalized[target_indices, sensor, :]
    target_mask = observed[target_indices, sensor, :]
    return x.astype(np.float32), y.astype(np.float32), target_mask.astype(np.float32)


def masked_mae(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (torch.abs(prediction - target) * mask).sum() / mask.sum().clamp_min(1.0)


def evaluate(
    model: nn.Module,
    normalized: np.ndarray,
    observed: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    start: int,
    end: int,
    device: torch.device,
):
    """Evaluate every valid window in the requested split."""
    model.eval()
    abs_sum = np.zeros((3, 3), dtype=np.float64)
    sq_sum = np.zeros((3, 3), dtype=np.float64)
    base_abs_sum = np.zeros((3, 3), dtype=np.float64)
    base_sq_sum = np.zeros((3, 3), dtype=np.float64)
    counts = np.zeros((3, 3), dtype=np.float64)
    normalized_abs = 0.0
    normalized_count = 0.0

    mean_tensor = means.reshape(1, 1, 3)
    std_tensor = stds.reshape(1, 1, 3)

    with torch.no_grad():
        for sensor in range(normalized.shape[1]):
            windows = make_windows(normalized, observed, sensor, start, end)
            if windows is None:
                continue
            x, y, mask = windows
            for offset in range(0, len(x), 512):
                xb = torch.from_numpy(x[offset : offset + 512]).to(device)
                yb = y[offset : offset + 512]
                mb = mask[offset : offset + 512]
                prediction = model(xb).cpu().numpy()
                normalized_abs += float((np.abs(prediction - yb) * mb).sum())
                normalized_count += float(mb.sum())

                prediction_real = prediction * std_tensor + mean_tensor
                # Enforce physically valid ranges consistently with the Streamlit app.
                prediction_real[:, :, 0] = np.maximum(prediction_real[:, :, 0], 0.0)
                prediction_real[:, :, 1] = np.clip(prediction_real[:, :, 1], 0.0, 1.0)
                prediction_real[:, :, 2] = np.maximum(prediction_real[:, :, 2], 0.0)
                target_real = yb * std_tensor + mean_tensor
                # Persistence repeats the latest input measurement for all horizons.
                baseline_norm = np.repeat(x[offset : offset + 512, -1:, :], 3, axis=1)
                baseline_real = baseline_norm * std_tensor + mean_tensor
                error = prediction_real - target_real
                base_error = baseline_real - target_real
                abs_sum += (np.abs(error) * mb).sum(axis=0)
                sq_sum += (np.square(error) * mb).sum(axis=0)
                base_abs_sum += (np.abs(base_error) * mb).sum(axis=0)
                base_sq_sum += (np.square(base_error) * mb).sum(axis=0)
                counts += mb.sum(axis=0)

    if normalized_count == 0:
        raise ValueError("No valid target observations were found in this split.")

    result = {
        "normalized_mae": normalized_abs / normalized_count,
        "mae": abs_sum / np.maximum(counts, 1.0),
        "rmse": np.sqrt(sq_sum / np.maximum(counts, 1.0)),
        "baseline_mae": base_abs_sum / np.maximum(counts, 1.0),
        "baseline_rmse": np.sqrt(base_sq_sum / np.maximum(counts, 1.0)),
        "counts": counts,
    }
    return result


def average_weights(client_states: list[dict], client_sizes: list[int]) -> dict:
    total = float(sum(client_sizes))
    averaged = {}
    for key in client_states[0]:
        averaged[key] = sum(
            client_states[i][key] * (client_sizes[i] / total) for i in range(len(client_states))
        )
    return averaged


def make_client_data(normalized, observed, sensors, max_windows, seed):
    x_parts, y_parts, mask_parts = [], [], []
    train_end = int(normalized.shape[0] * 0.70)
    for sensor in sensors:
        result = make_windows(normalized, observed, int(sensor), 0, train_end, max_windows, seed + int(sensor))
        if result is not None:
            x_parts.append(result[0])
            y_parts.append(result[1])
            mask_parts.append(result[2])
    if not x_parts:
        raise ValueError("A federated client has no training windows.")
    return np.concatenate(x_parts), np.concatenate(y_parts), np.concatenate(mask_parts)


def train_model(args) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"Device: {DEVICE}")
    raw = load_dataset()
    normalized, observed, means, medians, stds, free_speed, train_end, val_end = prepare_data(raw)
    print(f"Dataset: {raw.shape[0]:,} time steps x {raw.shape[1]} sensors x 3 features")
    print("Targets: flow, occupancy and speed at 15, 30 and 60 minutes")
    print(f"Maximum FedAvg rounds: {args.rounds}; local epochs per round: {args.local_epochs}")

    client_sensors = np.array_split(np.arange(raw.shape[1]), 4)
    client_datasets = []
    for index, sensors in enumerate(client_sensors, start=1):
        x, y, mask = make_client_data(normalized, observed, sensors, args.max_windows, args.seed)
        client_datasets.append(TensorDataset(torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask)))
        print(f"Client {index}: {len(sensors)} sensors, {len(x):,} training windows")

    global_model = TrafficGRU().to(DEVICE)
    print(f"GRU parameters: {sum(p.numel() for p in global_model.parameters()):,}")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    best_validation = float("inf")
    best_round = 0
    wait = 0
    history = []

    for round_number in range(1, args.rounds + 1):
        client_states, client_sizes, local_losses = [], [], []
        for client_index, dataset in enumerate(client_datasets):
            local_model = TrafficGRU().to(DEVICE)
            local_model.load_state_dict(global_model.state_dict())
            optimizer = torch.optim.Adam(local_model.parameters(), lr=args.learning_rate)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
            loss_total, batches = 0.0, 0
            local_model.train()
            for _ in range(args.local_epochs):
                for xb, yb, mb in loader:
                    xb, yb, mb = xb.to(DEVICE), yb.to(DEVICE), mb.to(DEVICE)
                    optimizer.zero_grad()
                    prediction = local_model(xb)
                    loss = masked_mae(prediction, yb, mb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), 1.0)
                    optimizer.step()
                    loss_total += float(loss.item())
                    batches += 1
            local_losses.append(loss_total / max(batches, 1))
            client_states.append({k: v.detach().cpu().clone() for k, v in local_model.state_dict().items()})
            client_sizes.append(len(dataset))
            del local_model, optimizer, loader

        global_model.load_state_dict(average_weights(client_states, client_sizes))
        validation = evaluate(global_model, normalized, observed, means, stds, train_end, val_end, DEVICE)
        val_score = float(validation["normalized_mae"])
        train_score = float(np.average(local_losses, weights=client_sizes))
        improved = val_score < best_validation - args.min_delta

        if improved:
            best_validation = val_score
            best_round = round_number
            wait = 0
            torch.save(
                {
                    "model_state_dict": global_model.state_dict(),
                    "means": torch.tensor(means),
                    "medians": torch.tensor(medians),
                    "stds": torch.tensor(stds),
                    "free_flow_speed": torch.tensor(free_speed),
                    "feature_names": FEATURES,
                    "sensor_count": raw.shape[1],
                    "input_steps": INPUT_STEPS,
                    "horizons": HORIZONS,
                    "best_round": best_round,
                    "best_validation_mae_normalized": best_validation,
                },
                CHECKPOINT,
            )
        else:
            wait += 1

        speed_mae = validation["mae"][:, 2].mean()
        flow_mae = validation["mae"][:, 0].mean()
        occupancy_mae_percentage_points = validation["mae"][:, 1].mean() * 100.0
        history.append(
            {
                "round": round_number,
                "train_mae_normalized": train_score,
                "validation_mae_normalized": val_score,
                "validation_flow_mae_veh_per_5min": flow_mae,
                "validation_occupancy_mae_percentage_points": occupancy_mae_percentage_points,
                "validation_speed_mae_mph": speed_mae,
            }
        )
        label = "BEST" if improved else f"no improvement {wait}/{args.patience}"
        print(
            f"Round {round_number:02d}/{args.rounds} | train MAE norm={train_score:.4f} "
            f"| validation MAE norm={val_score:.4f} | flow MAE={flow_mae:.3f} veh/5min "
            f"| occupancy MAE={occupancy_mae_percentage_points:.3f} percentage points "
            f"| speed MAE={speed_mae:.3f} mph | {label}"
        )
        pd.DataFrame(history).to_csv(RESULT_DIR / "training_history.csv", index=False)
        if wait >= args.patience:
            print("Early stopping: validation score did not improve.")
            break

    print(f"\nTraining complete. Best round: {best_round}")
    print(f"Best validation normalized MAE: {best_validation:.4f}")
    print(f"Best checkpoint: {CHECKPOINT}")
    print("Next: python main.py test")


def test_model() -> None:
    if not CHECKPOINT.exists():
        raise FileNotFoundError("Trained checkpoint not found. Run `python main.py train` first.")
    raw = load_dataset()
    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    if raw.shape[1] != int(checkpoint["sensor_count"]):
        raise ValueError("Dataset sensor count does not match the saved checkpoint.")
    means = checkpoint["means"].cpu().numpy()
    medians = checkpoint["medians"].cpu().numpy()
    stds = checkpoint["stds"].cpu().numpy()
    train_end = int(raw.shape[0] * 0.70)
    val_end = int(raw.shape[0] * 0.80)
    # Reuse only the training scaler saved with the model.
    filled = raw.copy()
    valid = np.isfinite(filled) & (filled >= 0)
    for feature in range(3):
        filled[:, :, feature][~valid[:, :, feature]] = medians[feature]
    normalized = ((filled - means.reshape(1, 1, 3)) / stds.reshape(1, 1, 3)).astype(np.float32)

    model = TrafficGRU().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    result = evaluate(model, normalized, valid, means, stds, val_end, raw.shape[0], DEVICE)

    rows = []
    horizons_minutes = [15, 30, 60]
    for horizon_index, minutes in enumerate(horizons_minutes):
        for feature_index, feature in enumerate(FEATURES):
            scale = 100.0 if feature == "occupancy" else 1.0
            mae = float(result["mae"][horizon_index, feature_index] * scale)
            rmse = float(result["rmse"][horizon_index, feature_index] * scale)
            base_mae = float(result["baseline_mae"][horizon_index, feature_index] * scale)
            base_rmse = float(result["baseline_rmse"][horizon_index, feature_index] * scale)
            unit = {
                "flow": "vehicles/5 min",
                "occupancy": "percentage points",
                "speed": "mph",
            }[feature]
            improvement = 100.0 * (base_mae - mae) / base_mae if base_mae > 0 else 0.0
            rows.append(
                {
                    "horizon_minutes": minutes,
                    "target": feature,
                    "unit": unit,
                    "GRU_MAE": mae,
                    "Persistence_MAE": base_mae,
                    "MAE_improvement_percent": improvement,
                    "GRU_RMSE": rmse,
                    "Persistence_RMSE": base_rmse,
                    "valid_targets": int(result["counts"][horizon_index, feature_index]),
                }
            )

    frame = pd.DataFrame(rows)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULT_DIR / "test_metrics.csv"
    frame.to_csv(output, index=False)
    print("\nHELD-OUT TEST RESULTS (PeMSD4)")
    print(frame.round(3).to_string(index=False))
    print(f"\nSaved metrics to: {output}")
    print("Note: congestion is derived from predicted speed; it is not a fourth learned output.")


def export_sensor(args) -> None:
    raw = load_dataset()
    if args.sensor < 0 or args.sensor >= raw.shape[1]:
        raise ValueError(f"Sensor must be between 0 and {raw.shape[1] - 1}.")
    timestamps = pd.date_range(START_TIME, periods=raw.shape[0], freq="5min")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "sensor_id": args.sensor,
            "flow_veh_5min": raw[:, args.sensor, 0],
            "occupancy_fraction": raw[:, args.sensor, 1],
            "speed_mph": raw[:, args.sensor, 2],
        }
    )
    output = ROOT / "data" / f"example_sensor_{args.sensor}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"Exported real PeMSD4 observations for sensor {args.sensor}: {output.relative_to(ROOT)}")
    print("Occupancy is exported as a fraction (0-1); the Streamlit app displays it as a percentage.")
    print("This file is actual dataset data, not synthetic data.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TrafficFL multi-target GRU + simulated FedAvg")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("inspect", help="Download/verify and inspect PeMSD4")

    train = subparsers.add_parser("train", help="Train the multi-target GRU with four simulated clients")
    train.add_argument("--rounds", type=int, default=20)
    train.add_argument("--local-epochs", type=int, default=1)
    train.add_argument("--max-windows", type=int, default=800, help="Evenly sampled training windows per sensor")
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--learning-rate", type=float, default=0.001)
    train.add_argument("--patience", type=int, default=4)
    train.add_argument("--min-delta", type=float, default=0.0001)
    train.add_argument("--seed", type=int, default=42)

    subparsers.add_parser("test", help="Evaluate the saved model on the chronological held-out test split")
    export = subparsers.add_parser("export", help="Export one sensor's real observations to a CSV for Streamlit")
    export.add_argument("--sensor", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "inspect":
        inspect_dataset()
    elif args.command == "train":
        if args.rounds < 1 or args.local_epochs < 1 or args.max_windows < 1:
            raise ValueError("Rounds, local epochs and maximum windows must be positive.")
        train_model(args)
    elif args.command == "test":
        test_model()
    elif args.command == "export":
        export_sensor(args)


if __name__ == "__main__":
    main()
