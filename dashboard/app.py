"""FedRoad experiment dashboard.

Run from the repository root with: streamlit run dashboard/app.py
The dashboard reads saved CSV/Markdown outputs; it does not train models.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "experiments"
COMPARISON_PATH = RESULTS_DIR / "model_comparison.csv"
ROUNDS_PATH = RESULTS_DIR / "fedavg_round_metrics.csv"
REPORT_PATH = ROOT / "docs" / "experiment_results.md"

st.set_page_config(
    page_title="FedRoad | Traffic Forecasting",
    page_icon="🚦",
    layout="wide",
)


def load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.warning(f"Could not read {path.relative_to(ROOT)}: {exc}")
        return None


st.title("FedRoad")
st.subheader("Federated deep learning for traffic-speed forecasting")
st.caption(
    "METR-LA benchmark · PyTorch GRU · Flower FedAvg · Four simulated clients · "
    "15, 30 and 60-minute forecast horizons"
)
st.info(
    "This dashboard visualizes saved experiment results. It does not start training. "
    "Federated clients were simulated in one runtime; this is not a deployment-level "
    "privacy guarantee."
)

comparison = load_csv(COMPARISON_PATH)
if comparison is None:
    st.error(
        "Missing results/experiments/model_comparison.csv. From the repository root, "
        "run: python scripts/compare_experiments.py"
    )
    st.stop()

required = {"Model", "Horizon (minutes)", "MAE", "RMSE", "Valid targets"}
missing = required - set(comparison.columns)
if missing:
    st.error(f"Comparison CSV is missing required columns: {sorted(missing)}")
    st.stop()

for col in ("Horizon (minutes)", "MAE", "RMSE", "Valid targets"):
    comparison[col] = pd.to_numeric(comparison[col], errors="coerce")
comparison = comparison.dropna(subset=["Model", "Horizon (minutes)", "MAE", "RMSE"])
if comparison.empty:
    st.error("The comparison CSV contains no valid rows.")
    st.stop()

model_names = comparison["Model"].drop_duplicates().tolist()
horizons = sorted(comparison["Horizon (minutes)"].astype(int).unique().tolist())

with st.sidebar:
    st.header("Explore results")
    metric = st.selectbox("Metric", ["MAE", "RMSE"])
    selected_horizons = st.multiselect("Forecast horizon (minutes)", horizons, default=horizons)
    selected_models = st.multiselect("Models", model_names, default=model_names)
    st.caption("Lower MAE/RMSE is better.")

filtered = comparison[
    comparison["Horizon (minutes)"].isin(selected_horizons)
    & comparison["Model"].isin(selected_models)
].copy()

fed_rows = comparison[comparison["Model"] == "FedAvg GRU"].copy()
central_rows = comparison[comparison["Model"] == "Centralized shared GRU"].copy()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Traffic sensors", "207")
c2.metric("Federated clients", "4")
c3.metric("FedAvg rounds", "10")
if not fed_rows.empty and not central_rows.empty:
    joined = fed_rows.merge(
        central_rows[["Horizon (minutes)", "MAE"]],
        on="Horizon (minutes)",
        suffixes=("_fed", "_central"),
    )
    mean_gap = ((joined["MAE_fed"] - joined["MAE_central"]) / joined["MAE_central"] * 100).mean()
    c4.metric("Mean FedAvg MAE gap vs central", f"{mean_gap:+.2f}%")
else:
    c4.metric("Mean FedAvg MAE gap vs central", "N/A")

st.divider()
st.header(f"Test-set {metric} by forecast horizon")
if filtered.empty:
    st.info("Select at least one model and forecast horizon in the sidebar.")
else:
    fig = px.bar(
        filtered,
        x="Horizon (minutes)",
        y=metric,
        color="Model",
        barmode="group",
        text_auto=".3f",
        labels={"Horizon (minutes)": "Forecast horizon (minutes)", metric: f"{metric} (original speed units)"},
    )
    fig.update_layout(legend_title_text="Model", xaxis_type="category")
    st.plotly_chart(fig, use_container_width=True)

st.header("Detailed test metrics")
show_cols = [c for c in [
    "Model", "Horizon (minutes)", "MAE", "RMSE", "Valid targets",
    "MAE improvement vs persistence (%)", "MAE difference vs centralized (%)"
] if c in filtered.columns]
st.dataframe(
    filtered[show_cols].sort_values(["Horizon (minutes)", "Model"]).round(3),
    use_container_width=True,
    hide_index=True,
)
st.download_button(
    "Download filtered metrics as CSV",
    data=filtered[show_cols].to_csv(index=False).encode("utf-8"),
    file_name="fedroad_filtered_metrics.csv",
    mime="text/csv",
)

st.header("Federated training curve")
rounds = load_csv(ROUNDS_PATH)
if rounds is not None and not rounds.empty:
    expected_round_cols = {"round", "train_mae_normalized", "val_mae_normalized"}
    if expected_round_cols.issubset(rounds.columns):
        curve = rounds.melt(
            id_vars="round",
            value_vars=["train_mae_normalized", "val_mae_normalized"],
            var_name="Series",
            value_name="Normalized MAE",
        )
        curve["Series"] = curve["Series"].map({
            "train_mae_normalized": "Train MAE",
            "val_mae_normalized": "Validation MAE",
        })
        fig_rounds = px.line(
            curve, x="round", y="Normalized MAE", color="Series", markers=True,
            title="FedAvg round metrics (normalized units)",
        )
        fig_rounds.update_layout(xaxis_title="Federated round", yaxis_title="Normalized MAE")
        st.plotly_chart(fig_rounds, use_container_width=True)
        st.caption("Round values were transcribed from the completed Flower console log.")
    else:
        st.warning(f"Round-metrics CSV needs columns: {sorted(expected_round_cols)}")
else:
    st.caption("Optional round-by-round file not found. Add results/experiments/fedavg_round_metrics.csv to display this chart.")

with st.expander("Methodology and limitations"):
    st.markdown(
        """
        - Dataset: METR-LA, 34,272 time steps at five-minute intervals, 207 sensors.
        - Input window: 12 readings (60 minutes); forecast horizons: 15, 30 and 60 minutes.
        - Missing/non-positive readings are masked as unavailable; metrics use observed targets only.
        - Centralized and federated experiments were run with different training budgets, so the comparison is informative but not a perfectly matched compute-budget study.
        - The four Flower clients are simulated in one runtime. Raw readings are loaded client-side in the simulation, but there is no secure aggregation, differential privacy, or separate-owner deployment in this prototype.
        - The directed graph GRU is an experimental extension. Its two-epoch / 128-window smoke-test metrics are not included in the final benchmark.
        """
    )

if REPORT_PATH.exists():
    with st.expander("Read the generated experiment report"):
        st.markdown(REPORT_PATH.read_text(encoding="utf-8"))

st.caption("FedRoad · Results are experimental and should be reproduced before external claims.")
