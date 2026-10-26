"""Download and align the official METR-LA sensor adjacency matrix."""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np
import requests
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRAPH_PATH = PROJECT_ROOT / "data" / "graph" / "adj_mx_METR-LA.pkl"
GRAPH_URL = "https://zenodo.org/records/5724362/files/adj_mx_METR-LA.pkl?download=1"
EXPECTED_GRAPH_MD5 = "55996dfce3e39cc981e6e42e3f427da8"


def _file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_metr_la_graph(
    path: Path = DEFAULT_GRAPH_PATH, timeout: int = 180
) -> Path:
    """Download the official graph file and verify its published MD5 checksum.

    If a file already exists but has a different checksum, it is not overwritten.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        actual = _file_md5(path)
        if actual != EXPECTED_GRAPH_MD5:
            raise ValueError(
                f"Existing graph checksum mismatch at {path}: {actual}. "
                "Move the file aside before downloading a fresh copy."
            )
        print(f"METR-LA graph exists and checksum is valid: {path}")
        return path

    temporary = path.with_suffix(path.suffix + ".part")
    digest = hashlib.md5()
    try:
        with requests.get(GRAPH_URL, stream=True, timeout=(20, timeout)) as response:
            response.raise_for_status()
            with temporary.open("wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file.write(chunk)
                        digest.update(chunk)
        actual = digest.hexdigest()
        if actual != EXPECTED_GRAPH_MD5:
            raise ValueError(
                "Downloaded adjacency checksum mismatch: "
                f"expected {EXPECTED_GRAPH_MD5}, got {actual}."
            )
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    print(f"Downloaded and verified METR-LA graph: {path}")
    print(f"MD5: {actual}")
    return path


def _sensor_key(value: object) -> str:
    """Normalize IDs from CSV headers and the legacy adjacency pickle."""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def align_adjacency_to_sensor_order(
    sensor_ids: list[str] | tuple[str, ...],
    graph_sensor_ids: list[object] | tuple[object, ...] | np.ndarray,
    adjacency: np.ndarray,
) -> np.ndarray:
    """Reorder an adjacency matrix to match the traffic CSV's sensor columns."""
    csv_ids = [_sensor_key(value) for value in sensor_ids]
    graph_ids = [_sensor_key(value) for value in graph_sensor_ids]
    matrix = np.asarray(adjacency, dtype=np.float32)

    if len(set(csv_ids)) != len(csv_ids):
        raise ValueError("traffic sensor IDs contain duplicates")
    if len(set(graph_ids)) != len(graph_ids):
        raise ValueError("graph sensor IDs contain duplicates")
    if matrix.ndim != 2 or matrix.shape != (len(graph_ids), len(graph_ids)):
        raise ValueError("graph adjacency shape does not match graph sensor IDs")

    graph_index = {sensor_id: index for index, sensor_id in enumerate(graph_ids)}
    missing = [sensor_id for sensor_id in csv_ids if sensor_id not in graph_index]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"{len(missing)} traffic sensors are missing from the graph; "
            f"first missing IDs: {preview}"
        )

    order = [graph_index[sensor_id] for sensor_id in csv_ids]
    aligned = matrix[np.ix_(order, order)]
    if not np.isfinite(aligned).all() or np.any(aligned < 0):
        raise ValueError("adjacency must contain finite non-negative weights")
    return aligned.astype(np.float32, copy=False)


def load_metr_la_adjacency(
    sensor_ids: list[str] | tuple[str, ...],
    path: Path = DEFAULT_GRAPH_PATH,
) -> torch.Tensor:
    """Load the checksum-verified official adjacency and align sensor order.

    The upstream file is a legacy pickle containing (sensor_ids, id_to_index,
    adjacency). Pickle is only read after its complete file checksum is verified
    against the fixed checksum published for this official Zenodo asset.
    """
    path = download_metr_la_graph(path)
    with path.open("rb") as file:
        payload = pickle.load(file, encoding="latin1")
    if not isinstance(payload, (tuple, list)) or len(payload) != 3:
        raise ValueError("Unexpected METR-LA graph file format")

    graph_sensor_ids, _sensor_id_to_ind, adjacency = payload
    aligned = align_adjacency_to_sensor_order(
        sensor_ids, graph_sensor_ids, np.asarray(adjacency)
    )
    if aligned.shape != (len(sensor_ids), len(sensor_ids)):
        raise ValueError("Aligned graph size does not match the traffic sensor count")
    return torch.from_numpy(aligned.copy())
