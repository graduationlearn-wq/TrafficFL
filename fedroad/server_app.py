"""Flower ServerApp that coordinates FedAvg and saves the global GRU."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from fedroad.model import SensorLevelGRU

app = ServerApp()
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"


@app.main()
def main(grid: Grid, context: Context) -> None:
    num_clients = int(context.run_config["num-clients"])
    num_rounds = int(context.run_config["num-server-rounds"])
    seed = int(context.run_config["seed"])
    torch.manual_seed(seed)

    model = SensorLevelGRU()
    strategy = FedAvg(
        fraction_train=float(context.run_config["fraction-train"]),
        fraction_evaluate=float(context.run_config["fraction-evaluate"]),
        min_train_nodes=num_clients,
        min_evaluate_nodes=num_clients,
        min_available_nodes=num_clients,
    )

    result = strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(model.state_dict()),
        train_config=ConfigRecord({
            "learning-rate": float(context.run_config["learning-rate"]),
        }),
        evaluate_config=ConfigRecord({}),
        num_rounds=num_rounds,
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    state_dict = result.arrays.to_torch_state_dict()
    model_path = ARTIFACT_DIR / "fedavg_sensor_gru.pt"
    torch.save({k: v.detach().cpu() for k, v in state_dict.items()}, model_path)

    metadata = {
        "method": "Flower FedAvg",
        "model": "SensorLevelGRU",
        "dataset": "METR-LA",
        "num_clients": num_clients,
        "num_server_rounds": num_rounds,
        "local_epochs": int(context.run_config["local-epochs"]),
        "batch_size": int(context.run_config["batch-size"]),
        "learning_rate": float(context.run_config["learning-rate"]),
        "max_train_windows_per_client": int(context.run_config["max-train-windows"]),
        "max_eval_windows_per_client": int(context.run_config["max-eval-windows"]),
        "seed": seed,
        "note": "A local Flower simulation; not a deployment-level privacy guarantee.",
    }
    metadata_path = ARTIFACT_DIR / "fedavg_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved federated model: {model_path}")
    print(f"Saved federated metadata: {metadata_path}")
