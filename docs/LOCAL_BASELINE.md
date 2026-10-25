# FedRoad local baseline

This folder adds the first reproducible source-code stage to the existing TrafficFL repository.
It uses the real METR-LA benchmark. It does not generate synthetic traffic observations.

## Download and verify data

From the repository root, with the project virtual environment active (or by calling its Python executable directly):

```powershell
.\.venv\Scripts\python.exe -m scripts.download_data
```

The CSV is saved to `data/raw/METR-LA.csv`. The downloader verifies the MD5 checksum published on Zenodo (`7c463fcbd28ecc6c9cd93e1ca120aaaa`). The original dataset file is not modified.

## Run unit tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Train a quick centralized GRU baseline

```powershell
.\.venv\Scripts\python.exe -m fedroad.train --epochs 3
```

For the full available window sets, use:

```powershell
.\.venv\Scripts\python.exe -m fedroad.train --epochs 3 --max-train-windows 0 --max-eval-windows 0
```

The full option can take noticeably longer on CPU. The default uses evenly spaced windows for a quicker development run. Results from a subset should be labelled as a development benchmark, not directly compared with a full-test-set benchmark.

## Data handling

The first CSV column is parsed as timestamps. Non-positive or non-finite readings are marked unavailable for this experiment. Missing input readings are imputed with per-sensor medians calculated from observed training readings. Normalization statistics are also calculated using training data only. Target masks are retained, so unavailable target readings are excluded from training loss and evaluation metrics.

## Outputs

Generated model checkpoints and metrics are saved under `artifacts/`. The raw dataset should remain local and must not be committed to GitHub.

## Next engineering stage

Once this local data/model pipeline passes on the target computer, wrap the same model and training functions in Flower's `ClientApp` and `ServerApp`. The Colab FedAvg experiment is already a prototype, but local Flower execution still needs to be tested independently.
