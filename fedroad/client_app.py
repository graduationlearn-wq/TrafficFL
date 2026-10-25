"""Flower ClientApp for local GRU training on a client's sensor partition."""
from __future__ import annotations

import numpy as np
import torch

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fedroad.data import DEFAULT_DATA_PATH
from fedroad.engine import evaluate, normalized_validation_mae, train_one_epoch
from fedroad.federated_data import load_client_data
from fedroad.model import SensorLevelGRU

app = ClientApp()


def _get_data(context: Context):
    return load_client_data(
        partition_id=int(context.node_config["partition-id"]),
        num_partitions=int(context.node_config["num-partitions"]),
        batch_size=int(context.run_config["batch-size"]),
        max_train_windows=int(context.run_config["max-train-windows"]),
        max_eval_windows=int(context.run_config["max-eval-windows"]),
        data_path=DEFAULT_DATA_PATH,
    )


def _seed_for_client(context: Context, partition_id: int, msg: Message) -> None:
    base_seed = int(context.run_config["seed"])
    round_id = int(msg.content["config"].get("server-round", 0))
    seed = base_seed + 1009 * (round_id + 1) + partition_id
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@app.train()
def train(msg: Message, context: Context) -> Message:
    """Train on local sensors and return the updated model plus sample weight."""
    partition_id = int(context.node_config["partition-id"])
    _seed_for_client(context, partition_id, msg)
    data = _get_data(context)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SensorLevelGRU().to(device)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(msg.content["config"]["learning-rate"]),
        weight_decay=1e-5,
    )

    epochs = int(context.run_config["local-epochs"])
    losses = [
        train_one_epoch(model, data["train_loader"], optimizer, device)
        for _ in range(epochs)
    ]

    arrays = ArrayRecord({k: v.detach().cpu() for k, v in model.state_dict().items()})
    metrics = MetricRecord({
        "train_mae_normalized": float(losses[-1]),
        "num-examples": max(1, int(data["train_examples"])),
        "num-sensors": int(data["num_sensors"]),
    })
    return Message(content=RecordDict({"arrays": arrays, "metrics": metrics}), reply_to=msg)


@app.evaluate()
def evaluate_global(msg: Message, context: Context) -> Message:
    """Evaluate the received global model on the client's validation period."""
    data = _get_data(context)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SensorLevelGRU().to(device)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    val_mae = normalized_validation_mae(model, data["val_loader"], device)
    rows = evaluate(
        model=model,
        loader=data["val_loader"],
        sensor_indices=np.arange(data["num_sensors"], dtype=np.int64),
        mean=data["mean"],
        std=data["std"],
        device=device,
        model_name="FedAvg validation",
    )
    valid_total = sum(int(row["Valid targets"]) for row in rows)
    pooled_mae = sum(
        float(row["MAE"]) * int(row["Valid targets"]) for row in rows
    ) / max(valid_total, 1)

    metrics = MetricRecord({
        "val_mae_normalized": float(val_mae),
        "val_mae_original_units": float(pooled_mae),
        "num-examples": max(1, int(valid_total)),
        "num-sensors": int(data["num_sensors"]),
    })
    return Message(content=RecordDict({"metrics": metrics}), reply_to=msg)
