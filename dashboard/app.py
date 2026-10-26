
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import torch

from fedroad.data import (
    DEFAULT_DATA_PATH,
    HORIZONS,
    INPUT_STEPS,
    download_metr_la,
    load_and_preprocess,
)
from fedroad.model import SensorLevelGRU


# ---------------------------------------------------------
# Paths and application settings
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results" / "experiments"

COMPARISON_PATH = RESULTS / "model_comparison.csv"
ROUND_PATH = RESULTS / "fedavg_round_metrics.csv"
CENTRAL_MODEL_PATH = ARTIFACTS / "centralized_sensor_gru.pt"
FED_MODEL_PATH = ARTIFACTS / "fedavg_sensor_gru.pt"

st.set_page_config(
    page_title="FedRoad | Traffic Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------
# Styling: single-page dashboard
# ---------------------------------------------------------

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1450px;
        padding-top: 1.2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 1.7rem 1.8rem;
        border-radius: 22px;
        background:
            radial-gradient(circle at 85% 5%,
                rgba(45, 212, 191, 0.20), transparent 32%),
            linear-gradient(125deg, #0b1222, #142b42 65%, #123b42);
        border: 1px solid rgba(110, 231, 220, 0.22);
        color: #f1f5f9;
        margin-bottom: 1.2rem;
    }

    .hero h1 {
        color: #f8fafc;
        font-size: 2.5rem;
        margin: 0;
        letter-spacing: -1.2px;
    }

    .hero p {
        color: #c5d5e6;
        margin: 0.55rem 0 0 0;
        font-size: 1.03rem;
    }

    .eyebrow {
        color: #5eead4;
        font-weight: 700;
        font-size: 0.78rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    .section-intro {
        color: #64748b;
        margin-top: -0.5rem;
        margin-bottom: 1.1rem;
    }

    [data-testid="stMetric"] {
        padding: 0.9rem 1rem;
        border: 1px solid rgba(125, 145, 165, 0.25);
        border-radius: 14px;
        background: rgba(125, 145, 165, 0.06);
    }

    div[data-testid="stButton"] button {
        border-radius: 10px;
        font-weight: 600;
    }

    hr {
        margin: 1.6rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Data and model loading
# ---------------------------------------------------------

@st.cache_resource(show_spinner="Preparing the traffic dataset...")
def get_data():
    if not DEFAULT_DATA_PATH.exists():
        download_metr_la(DEFAULT_DATA_PATH)

    return load_and_preprocess(DEFAULT_DATA_PATH)


@st.cache_resource
def get_model(path_string: str, modified_time: float):
    """Load a locally generated FedRoad checkpoint."""
    checkpoint = torch.load(
        path_string,
        map_location="cpu",
        weights_only=False,
    )

    # Centralized checkpoints include metadata.
    # Flower checkpoints contain the state dictionary directly.
    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model = SensorLevelGRU(
        hidden_size=32,
        num_horizons=3,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.warning(f"Could not read {path.name}: {exc}")
        return None


# ---------------------------------------------------------
# Neural-network SVG visualization
# ---------------------------------------------------------

def build_network_html(animate: bool = True) -> str:
    """
    Draw an illustrative, animated view of the actual GRU:
    input sequence -> recurrent states -> three forecast outputs.

    Multiple GRU columns are time-unrolled copies of the same
    GRU cell, not independent learned hidden layers.
    """

    width, height = 1140, 450

    layers = [
        {
            "x": 85,
            "count": 7,
            "kind": "input",
            "title": "INPUT SEQUENCE",
            "subtitle": "12 × [speed, mask]",
        },
        {
            "x": 315,
            "count": 8,
            "kind": "hidden",
            "title": "GRU STATE",
            "subtitle": "Earlier step",
        },
        {
            "x": 535,
            "count": 8,
            "kind": "hidden",
            "title": "GRU STATE",
            "subtitle": "Middle step",
        },
        {
            "x": 755,
            "count": 8,
            "kind": "hidden",
            "title": "GRU STATE",
            "subtitle": "Final step",
        },
        {
            "x": 975,
            "count": 3,
            "kind": "output",
            "title": "FORECASTS",
            "subtitle": "Linear output",
        },
    ]

    positions = []

    for layer in layers:
        ys = np.linspace(132, 332, layer["count"]).tolist()
        positions.append([(layer["x"], float(y)) for y in ys])

    svg_parts = []

    # Layer-to-layer connections.
    connection_number = 0

    for layer_index in range(len(positions) - 1):
        left_nodes = positions[layer_index]
        right_nodes = positions[layer_index + 1]

        for i, (x1, y1) in enumerate(left_nodes):
            for j, (x2, y2) in enumerate(right_nodes):
                delay = (connection_number % 13) * 0.09

                svg_parts.append(
                    f'<line class="link" '
                    f'x1="{x1}" y1="{y1}" '
                    f'x2="{x2}" y2="{y2}" '
                    f'style="animation-delay:{delay:.2f}s" />'
                )
                connection_number += 1

    # Neuron nodes. The diagram shows representative neurons,
    # not all 32 hidden units.
    colors = {
        "input": "#38bdf8",
        "hidden": "#5eead4",
        "output": "#fbbf24",
    }

    for layer_index, layer in enumerate(layers):
        x = layer["x"]
        color = colors[layer["kind"]]

        for node_index, (_, y) in enumerate(positions[layer_index]):
            delay = (node_index % 8) * 0.16

            svg_parts.append(
                f'<circle class="neuron {layer["kind"]}" '
                f'cx="{x}" cy="{y}" r="9" '
                f'fill="{color}" '
                f'style="animation-delay:{delay:.2f}s" />'
            )

            svg_parts.append(
                f'<circle cx="{x}" cy="{y}" r="3" '
                f'fill="#ffffff" opacity="0.8" />'
            )

        svg_parts.append(
            f'<text x="{x}" y="62" text-anchor="middle" '
            f'class="layer-title">{layer["title"]}</text>'
        )
        svg_parts.append(
            f'<text x="{x}" y="84" text-anchor="middle" '
            f'class="layer-subtitle">{layer["subtitle"]}</text>'
        )

    # Output labels.
    for y, label in zip(
        [232.0, 282.0, 332.0],
        ["15 min", "30 min", "60 min"],
    ):
        svg_parts.append(
            f'<text x="1000" y="{y + 4}" '
            f'class="output-label">{label}</text>'
        )

    animation_css = (
        """
        .animated .link {
            stroke-dasharray: 4 9;
            animation: signal-flow 1.7s linear infinite;
        }
        .animated .neuron {
            animation: neuron-pulse 2.2s ease-in-out infinite;
            transform-box: fill-box;
            transform-origin: center;
        }
        @keyframes signal-flow {
            to { stroke-dashoffset: -26; }
        }
        @keyframes neuron-pulse {
            0%, 100% { opacity: 0.85; transform: scale(1); }
            50% { opacity: 1; transform: scale(1.14); }
        }
        """
        if animate
        else ""
    )

    network_class = "network animated" if animate else "network"

    svg_body = "".join(svg_parts)

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {{
            margin: 0;
            background: #0b1220;
            color: #e2e8f0;
            font-family: Arial, sans-serif;
            overflow-x: auto;
        }}
        .frame {{
            width: 100%;
            min-width: 760px;
            padding: 8px 0;
            box-sizing: border-box;
            background:
                radial-gradient(circle at 50% 35%,
                    rgba(45, 212, 191, 0.08), transparent 55%),
                #0b1220;
            border: 1px solid #263449;
            border-radius: 16px;
        }}
        svg {{
            display: block;
            width: 100%;
            height: auto;
        }}
        .link {{
            stroke: #4a9eaa;
            stroke-width: 1.1;
            opacity: 0.30;
            fill: none;
        }}
        .neuron {{
            stroke: rgba(255,255,255,0.8);
            stroke-width: 1.2;
        }}
        .layer-title {{
            fill: #e2e8f0;
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 1px;
        }}
        .layer-subtitle {{
            fill: #93a8bc;
            font-size: 11px;
        }}
        .output-label {{
            fill: #fbbf24;
            font-size: 12px;
            font-weight: 700;
        }}
        .footer {{
            color: #9fb1c4;
            text-align: center;
            padding: 0 15px 12px 15px;
            font-size: 11px;
        }}
        {animation_css}
    </style>
    </head>
    <body>
      <div class="frame">
        <svg class="{network_class}"
             viewBox="0 0 {width} {height}"
             role="img"
             aria-label="Animated GRU neural network showing input, recurrent states and forecasts">
          {svg_body}
          <text x="570" y="410" text-anchor="middle"
                fill="#5eead4" font-size="12">
            Same GRU cell and shared weights are reused across time
          </text>
        </svg>
        <div class="footer">
          Illustrative neural-network view · each displayed node represents
          a subset of the actual model units.
        </div>
      </div>
    </body>
    </html>
    """


# ---------------------------------------------------------
# Main page
# ---------------------------------------------------------

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Privacy-aware traffic intelligence</div>
      <h1>🚦 TrafficFL</h1>
      <p>
        Deep-learning traffic forecasting with GRU neural networks
        and federated learning.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "METR-LA · 207 sensors · 60-minute input history · "
    "15 / 30 / 60-minute forecasts"
)

with st.spinner("Loading verified traffic data..."):
    try:
        traffic = get_data()
    except Exception as exc:
        st.error(f"Unable to load METR-LA: {exc}")
        st.info(
            "Check the dataset path and run "
            "`python -m scripts.download_data` from the project root."
        )
        st.stop()


# Discover models available on this computer.
model_paths = {}

if CENTRAL_MODEL_PATH.exists():
    model_paths["Centralized GRU"] = CENTRAL_MODEL_PATH

if FED_MODEL_PATH.exists():
    model_paths["FedAvg GRU"] = FED_MODEL_PATH

predictor_options = ["Persistence"] + list(model_paths.keys())

default_predictor = (
    "FedAvg GRU"
    if "FedAvg GRU" in predictor_options
    else "Centralized GRU"
    if "Centralized GRU" in predictor_options
    else "Persistence"
)


# Metadata is informative only; don't assume checkpoints exist.
metadata_path = RESULTS / "fedavg_metadata.json"
fed_metadata = {}

if metadata_path.exists():
    try:
        fed_metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        fed_metadata = {}


k1, k2, k3, k4 = st.columns(4)

k1.metric("Traffic sensors", len(traffic.sensor_ids))
k2.metric("Input history", f"{INPUT_STEPS * 5} minutes")
k3.metric(
    "Federated clients",
    fed_metadata.get("num_clients", 4),
)
k4.metric(
    "FedAvg rounds",
    fed_metadata.get("num_server_rounds", 10),
)

st.divider()


# ---------------------------------------------------------
# Interactive prediction area
# ---------------------------------------------------------

st.markdown("### 🔮 Explore a traffic prediction")
st.markdown(
    '<div class="section-intro">'
    "Choose a sensor and a historical test window to see a forecast "
    "and compare it with the observed future."
    "</div>",
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns([1.1, 1.1, 1.4])

with c1:
    selected_model = st.selectbox(
        "Prediction model",
        predictor_options,
        index=predictor_options.index(default_predictor),
    )

with c2:
    selected_sensor = st.selectbox(
        "Traffic sensor",
        traffic.sensor_ids,
        index=0,
    )

sensor_index = traffic.sensor_ids.index(selected_sensor)

# x_test contains the chronologically held-out 20% of time rows.
maximum_start = max(
    0,
    len(traffic.x_test) - INPUT_STEPS - max(HORIZONS),
)

with c3:
    test_start = st.slider(
        "Historical test window",
        min_value=0,
        max_value=int(maximum_start),
        value=min(200, int(maximum_start)),
        step=1,
        help=(
            "Choose the starting position of a 60-minute input sequence "
            "from the held-out historical period."
        ),
    )

test_offset = len(traffic.timestamps) - len(traffic.x_test)

history_end = test_start + INPUT_STEPS

history_scaled = traffic.x_test[
    test_start:history_end, sensor_index
]
history_mask = traffic.m_test[
    test_start:history_end, sensor_index
]

# Convert normalized history back to original units.
mean_value = float(traffic.mean[sensor_index])
std_value = float(traffic.std[sensor_index])

history_raw = history_scaled * std_value + mean_value
history_plot = np.where(
    history_mask > 0.5,
    history_raw,
    np.nan,
)

history_times = traffic.timestamps[
    test_offset + test_start:
    test_offset + history_end
]

valid_history = np.flatnonzero(history_mask > 0.5)

if len(valid_history):
    latest_valid_index = int(valid_history[-1])
    latest_speed = float(history_raw[latest_valid_index])
else:
    latest_speed = float("nan")
    st.warning(
        "This input window has no observed readings for the selected sensor. "
        "Predictions can still be generated by a neural model, but persistence "
        "has no valid previous speed to repeat."
    )


# Three targets corresponding to the three configured forecast horizons.
future_indices = (
    test_start + INPUT_STEPS - 1 + np.asarray(HORIZONS)
)

future_times = traffic.timestamps[
    test_offset + future_indices
]

future_mask = traffic.m_test[
    future_indices, sensor_index
]

future_scaled = traffic.x_test[
    future_indices, sensor_index
]

future_actual = np.where(
    future_mask > 0.5,
    future_scaled * std_value + mean_value,
    np.nan,
)


# Run the selected model.
hidden_states = None
prediction_error = None

if selected_model == "Persistence":
    if len(valid_history):
        latest_normalized = float(
            history_scaled[latest_valid_index]
        )
        predictions_normalized = np.repeat(
            latest_normalized,
            len(HORIZONS),
        )
    else:
        predictions_normalized = np.full(
            len(HORIZONS), np.nan
        )

else:
    model_path = model_paths[selected_model]

    try:
        model = get_model(
            str(model_path.resolve()),
            model_path.stat().st_mtime,
        )

        input_array = np.stack(
            [history_scaled, history_mask],
            axis=-1,
        ).astype(np.float32)

        input_tensor = torch.from_numpy(
            input_array
        ).unsqueeze(0)

        with torch.no_grad():
            sequence, _ = model.gru(input_tensor)
            hidden_states = sequence[0].cpu().numpy()

            predicted_tensor = model(input_tensor)
            predictions_normalized = (
                predicted_tensor[0].cpu().numpy()
            )

    except Exception as exc:
        prediction_error = (
            f"Could not load {selected_model} checkpoint: {exc}"
        )
        predictions_normalized = np.full(
            len(HORIZONS), np.nan
        )

predictions_raw = (
    predictions_normalized * std_value + mean_value
)

st.caption(
    f"Selected sensor: {selected_sensor} · "
    f"Input period: {history_times[0]} to {history_times[-1]} · "
    "Forecasts are made from historical test data, not live traffic."
)

if prediction_error:
    st.error(prediction_error)


# Prediction summary cards.
forecast_columns = st.columns(3)

for i, minutes in enumerate([15, 30, 60]):
    value = predictions_raw[i]

    if np.isfinite(value):
        delta = None
        if np.isfinite(latest_speed):
            delta = f"{value - latest_speed:+.2f} vs latest observed"

        forecast_columns[i].metric(
            f"{minutes}-minute prediction",
            f"{value:.2f}",
            delta=delta,
        )
    else:
        forecast_columns[i].metric(
            f"{minutes}-minute prediction",
            "N/A",
        )


# Forecast chart: historical observations, forecast points and true future.
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=history_times,
        y=history_plot,
        mode="lines+markers",
        name="Observed input history",
        connectgaps=False,
        line=dict(color="#0891b2", width=2.7),
        marker=dict(size=5),
    )
)

forecast_base = (
    float(history_raw[-1])
    if history_mask[-1] > 0.5
    else None
)

fig.add_trace(
    go.Scatter(
        x=[history_times[-1], *list(future_times)],
        y=[forecast_base, *list(predictions_raw)],
        mode="lines+markers",
        name=f"{selected_model} forecast",
        line=dict(color="#0d9488", width=3, dash="dash"),
        marker=dict(size=10, symbol="diamond"),
        connectgaps=False,
    )
)

if np.isfinite(future_actual).any():
    fig.add_trace(
        go.Scatter(
            x=future_times,
            y=future_actual,
            mode="markers",
            name="Observed future (test reference)",
            marker=dict(
                color="#d97706",
                size=12,
                symbol="circle-open",
                line=dict(width=2),
            ),
        )
    )

fig.add_vline(
    x=history_times[-1],
    line_dash="dot",
    line_color="#94a3b8",
    annotation_text="Forecast origin",
)

fig.update_layout(
    title="Observed history → predicted traffic speed",
    xaxis_title="Timestamp",
    yaxis_title="Speed (dataset units)",
    height=430,
    template="plotly_white",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=15, r=15, t=75, b=15),
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "The model predicts specific future horizons. Markers show the "
    "prediction at each horizon; they are not a dense five-minute forecast."
)

st.divider()


# ---------------------------------------------------------
# Animated neural network and actual activations
# ---------------------------------------------------------

st.markdown("### 🧠 Inside the neural network")
st.markdown(
    '<div class="section-intro">'
    "See the GRU architecture and the hidden-state activations "
    "produced by your selected model for this input window."
    "</div>",
    unsafe_allow_html=True,
)

animate_signals = st.checkbox(
    "Animate signal flow",
    value=True,
)

components.html(
    build_network_html(animate=animate_signals),
    height=465,
    scrolling=False,
)

st.caption(
    "Architecture: 12 sequential steps × 2 input features → "
    "one GRU layer with 32 hidden units → linear head with 3 outputs. "
    "The repeated GRU columns above illustrate the same cell at different "
    "times; they are not separate learned layers."
)

if hidden_states is not None:
    heatmap = px.imshow(
        hidden_states.T,
        x=[f"t-{(INPUT_STEPS - 1 - i) * 5}m" for i in range(INPUT_STEPS)],
        y=[f"Unit {i + 1}" for i in range(hidden_states.shape[1])],
        aspect="auto",
        origin="lower",
        color_continuous_scale="RdBu_r",
        labels={
            "x": "Input time step",
            "y": "GRU hidden unit",
            "color": "Activation",
        },
        title="Actual GRU hidden-state activations",
    )

    heatmap.update_layout(
        height=420,
        template="plotly_white",
        margin=dict(l=15, r=15, t=60, b=15),
    )

    st.plotly_chart(heatmap, use_container_width=True)

    st.caption(
        "Each cell is an actual hidden-state activation from the selected "
        "checkpoint. It visualizes internal model values; it is not a causal "
        "explanation of why the forecast was made."
    )
else:
    st.info(
        "Select a trained GRU model to see its actual hidden-state activations. "
        "The neural-network drawing above is an architectural illustration."
    )

st.divider()


# ---------------------------------------------------------
# Model comparison and federated-learning history
# ---------------------------------------------------------

st.markdown("### 📊 How well do the models perform?")
st.markdown(
    '<div class="section-intro">'
    "Compare the saved held-out test results from persistence, centralized "
    "GRU, and federated GRU."
    "</div>",
    unsafe_allow_html=True,
)

comparison = load_csv(COMPARISON_PATH)

if comparison is not None and not comparison.empty:
    required_columns = {
        "Model", "Horizon (minutes)", "MAE", "RMSE", "Valid targets"
    }

    missing = required_columns - set(comparison.columns)

    if missing:
        st.warning(
            f"The comparison CSV is missing columns: {sorted(missing)}"
        )
    else:
        for column in [
            "Horizon (minutes)", "MAE", "RMSE", "Valid targets"
        ]:
            comparison[column] = pd.to_numeric(
                comparison[column], errors="coerce"
            )

        metric_choice = st.radio(
            "Metric",
            ["MAE", "RMSE"],
            horizontal=True,
        )

        fig_compare = px.bar(
            comparison,
            x="Horizon (minutes)",
            y=metric_choice,
            color="Model",
            barmode="group",
            text_auto=".3f",
            category_orders={
                "Horizon (minutes)": [15, 30, 60]
            },
            labels={
                "Horizon (minutes)": "Forecast horizon",
                metric_choice: f"{metric_choice} (dataset speed units)",
            },
            title=f"Test-set {metric_choice}: model comparison",
        )

        fig_compare.update_layout(
            template="plotly_white",
            height=420,
            xaxis_type="category",
            hovermode="x unified",
        )

        st.plotly_chart(fig_compare, use_container_width=True)

        columns_to_show = [
            c for c in [
                "Model",
                "Horizon (minutes)",
                "MAE",
                "RMSE",
                "Valid targets",
                "MAE improvement vs persistence (%)",
                "MAE difference vs centralized (%)",
            ]
            if c in comparison.columns
        ]

        st.dataframe(
            comparison[columns_to_show].round(3),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download model comparison",
            data=comparison.to_csv(index=False).encode("utf-8"),
            file_name="fedroad_model_comparison.csv",
            mime="text/csv",
        )
else:
    st.warning(
        "No experiment comparison file found. Run "
        "`python scripts/compare_experiments.py` from the project root."
    )


# Federated-round history, if available.
rounds = load_csv(ROUND_PATH)

if rounds is not None and not rounds.empty:
    needed = {
        "round",
        "train_mae_normalized",
        "val_mae_normalized",
    }

    if needed.issubset(rounds.columns):
        round_long = rounds.melt(
            id_vars="round",
            value_vars=[
                "train_mae_normalized",
                "val_mae_normalized",
            ],
            var_name="Series",
            value_name="Normalized MAE",
        )

        round_long["Series"] = round_long["Series"].map({
            "train_mae_normalized": "Train MAE",
            "val_mae_normalized": "Validation MAE",
        })

        fig_rounds = px.line(
            round_long,
            x="round",
            y="Normalized MAE",
            color="Series",
            markers=True,
            title="Federated learning progress",
        )

        fig_rounds.update_layout(
            xaxis_title="Federated server round",
            template="plotly_white",
            height=360,
        )

        st.plotly_chart(
            fig_rounds,
            use_container_width=True,
        )
    else:
        st.caption("Federated-round CSV exists but has an unexpected schema.")


# ---------------------------------------------------------
# Methodology and limitations - still on the same page
# ---------------------------------------------------------

with st.expander("How FedRoad works and what its limitations are"):
    st.markdown(
        """
        **Data and preprocessing**

        - Real METR-LA traffic-speed data with 207 sensors and five-minute readings.
        - The model takes the previous 12 readings, covering 60 minutes.
        - Unavailable/non-positive readings are masked. Predictions are evaluated
          only where the future target was observed.
        - Normalization statistics are fitted on training data, not the test set.

        **Neural network**

        - The main forecasting model is a PyTorch GRU neural network.
        - Each sensor's sequence has two features: normalized speed and an
          observation mask.
        - The GRU's 32-unit hidden representation feeds a linear layer that
          predicts three future horizons.

        **Federated learning**

        - Flower simulates four clients with separate sensor partitions.
        - FedAvg aggregates the local model parameters.
        - Raw observations are not sent to the coordinator by the training
          strategy, but the local simulation is not a production privacy guarantee.
        - This prototype does not implement secure aggregation or differential
          privacy.

        **Evaluation**

        - MAE and RMSE are reported in the dataset's original speed units.
        - The displayed observed-future points are from the historical held-out
          test period. The dashboard does not provide live traffic forecasts.
        - Centralized and federated runs used different training budgets, so
          their results should not be treated as a perfectly matched comparison.
        - The directed graph GRU is a separate experimental extension and should
          not be described as superior until a full benchmark is complete.
        """
    )

st.caption(
    "FedRoad · Forecasts are generated from historical METR-LA test windows. "
    "Model checkpoints must exist locally for neural-network predictions."
)
