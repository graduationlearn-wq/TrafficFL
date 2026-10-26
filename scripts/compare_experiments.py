
"""Compare centralized and federated FedRoad experiments."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "experiments"
DOCS = ROOT / "docs"

central = pd.read_csv(RESULTS / "centralized_metrics.csv")
federated = pd.read_csv(RESULTS / "federated_metrics.csv")

required = {
    "Model", "Horizon (minutes)", "MAE", "RMSE", "Valid targets"
}

for name, frame in [("centralized", central), ("federated", federated)]:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} metrics are missing columns: {missing}")

# Select the three models needed for comparison.
persistence = central[central["Model"] == "Persistence"].copy()
central_gru = central[
    central["Model"] == "Centralized shared GRU"
].copy()
fed_gru = federated[federated["Model"] == "FedAvg GRU"].copy()

for name, frame in [
    ("persistence", persistence),
    ("centralized GRU", central_gru),
    ("FedAvg GRU", fed_gru),
]:
    if len(frame) != 3:
        raise ValueError(
            f"Expected three horizon rows for {name}; found {len(frame)}."
        )

# Ensure the metrics use the same valid targets at every horizon.
for horizon in (15, 30, 60):
    counts = []
    for frame in (persistence, central_gru, fed_gru):
        row = frame[frame["Horizon (minutes)"] == horizon]
        if len(row) != 1:
            raise ValueError(f"Missing or duplicate {horizon}-minute row.")
        counts.append(int(row["Valid targets"].iloc[0]))

    if len(set(counts)) != 1:
        raise ValueError(
            f"Different valid-target counts at {horizon} minutes: {counts}"
        )

comparison = pd.concat(
    [persistence, central_gru, fed_gru],
    ignore_index=True
)

baseline_mae = persistence.set_index("Horizon (minutes)")["MAE"]
central_mae = central_gru.set_index("Horizon (minutes)")["MAE"]

comparison["MAE improvement vs persistence (%)"] = comparison.apply(
    lambda r: (
        100 * (
            baseline_mae[r["Horizon (minutes)"]] - r["MAE"]
        ) / baseline_mae[r["Horizon (minutes)"]]
    ),
    axis=1
)

comparison["MAE difference vs centralized (%)"] = comparison.apply(
    lambda r: (
        100 * (
            r["MAE"] - central_mae[r["Horizon (minutes)"]]
        ) / central_mae[r["Horizon (minutes)"]]
    ),
    axis=1
)

order = {
    "Persistence": 0,
    "Centralized shared GRU": 1,
    "FedAvg GRU": 2,
}
comparison["_order"] = comparison["Model"].map(order)
comparison = comparison.sort_values(
    ["Horizon (minutes)", "_order"]
).drop(columns="_order")

comparison.to_csv(
    RESULTS / "model_comparison.csv",
    index=False
)

# Generate a Markdown report from the actual metric files.
lines = [
    "# FedRoad Experiment Results",
    "",
    "## Test-set comparison",
    "",
    "MAE and RMSE are in the dataset's original speed units. "
    "Lower values indicate better forecasts.",
    "",
    "| Horizon | Model | MAE | RMSE | Valid targets | "
    "MAE improvement vs persistence | MAE difference vs centralized |",
    "|---:|---|---:|---:|---:|---:|---:|",
]

for _, row in comparison.iterrows():
    lines.append(
        f"| {int(row['Horizon (minutes)'])} min"
        f" | {row['Model']}"
        f" | {row['MAE']:.3f}"
        f" | {row['RMSE']:.3f}"
        f" | {int(row['Valid targets']):,}"
        f" | {row['MAE improvement vs persistence (%)']:.2f}%"
        f" | {row['MAE difference vs centralized (%)']:+.2f}% |"
    )

lines += [
    "",
    "## Interpretation and limitations",
    "",
    "- Persistence is the simple forecasting baseline.",
    "- The centralized GRU and FedAvg GRU use the same sensor-level "
    "model family and held-out test horizons.",
    "- The centralized run used validation-based early stopping; "
    "the federated run used 10 server rounds with one local epoch per round. "
    "Training budgets therefore differ.",
    "- The federated clients are simulated in one runtime. This experiment "
    "does not establish a deployment-level privacy guarantee.",
    "- Use validation data for model selection. Preserve the test results "
    "as a final evaluation rather than repeatedly tuning against them.",
    "",
    "## Source files",
    "",
    "- `centralized_metrics.csv`",
    "- `centralized_metadata.json`",
    "- `federated_metrics.csv`",
    "- `fedavg_metadata.json`",
    "",
]

DOCS.mkdir(parents=True, exist_ok=True)
report_path = DOCS / "experiment_results.md"
report_path.write_text("\n".join(lines), encoding="utf-8")

print("Comparison completed.")
print(comparison.round(3).to_string(index=False))
print(f"\nCSV saved to: {RESULTS / 'model_comparison.csv'}")
print(f"Report saved to: {report_path}")
