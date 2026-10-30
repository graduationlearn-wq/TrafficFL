
from __future__ import annotations

import html
import json
from pathlib import Path
from textwrap import dedent
from typing import Any

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


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results" / "experiments"

CENTRAL_MODEL_PATH = ARTIFACTS / "centralized_sensor_gru.pt"
FED_MODEL_PATH = ARTIFACTS / "fedavg_sensor_gru.pt"

COMPARISON_PATH = RESULTS / "model_comparison.csv"
ROUND_PATH = RESULTS / "fedavg_round_metrics.csv"

CENTRAL_METADATA_PATH = RESULTS / "centralized_metadata.json"
FED_METADATA_PATH = RESULTS / "fedavg_metadata.json"

st.set_page_config(
    page_title="FedRoad | Traffic Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# 2. HTML RENDERING HELPER
# ============================================================

def render_html(container, markup: str):
    """
    Render HTML safely through Streamlit Markdown.

    dedent() removes indentation that could otherwise cause
    Streamlit to interpret HTML as a code block.
    """
    container.markdown(
        dedent(markup).strip(),
        unsafe_allow_html=True,
    )


# ============================================================
# 3. PROFESSIONAL STYLING
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1420;
    }

    .block-container {
        max-width: 1480px;
        padding-top: 1.4rem;
        padding-bottom: 3rem;
    }

    /* Header */

    .hero {
        padding: 1.7rem 1.9rem;
        border-radius: 13px;
        background: linear-gradient(120deg, #12243a, #19384d);
        border: 1px solid #29465e;
        margin-bottom: 1rem;
    }

    .hero .eyebrow {
        color: #79cfc5;
        font-weight: 700;
        font-size: 0.75rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    .hero h1 {
        color: #ffffff;
        font-size: 2.4rem;
        font-weight: 750;
        letter-spacing: -1px;
        margin: 0;
    }

    .hero p {
        color: #c5d4e2;
        margin-top: 0.5rem;
        margin-bottom: 0;
        font-size: 1.02rem;
    }

    .section-note {
        color: #a3b3c6;
        font-size: 0.94rem;
        margin-top: -0.45rem;
        margin-bottom: 1rem;
    }

    /* White metric cards with explicit dark typography */

    .metric-card {
        display: block;
        box-sizing: border-box;
        width: 100%;
        min-height: 112px;
        padding: 15px 17px;
        margin-bottom: 8px;

        background-color: #ffffff !important;
        color: #14243a !important;

        border: 1px solid #dce4ec;
        border-radius: 11px;

        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.10);
        transition: border-color 0.2s ease;
    }

    .metric-card:hover {
        border-color: #0f766e;
    }

    .metric-label {
        color: #536579 !important;
        font-size: 0.83rem;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #14243a !important;
        font-size: 1.75rem;
        font-weight: 750;
        line-height: 1.25;
    }

    .metric-description {
        color: #64748b !important;
        font-size: 0.77rem;
        margin-top: 5px;
    }

    .metric-positive {
        color: #087f5b !important;
    }

    .metric-negative {
        color: #b45309 !important;
    }

    /* Network diagram panel */

    .network-panel {
        width: 100%;
        background: #ffffff !important;
        border: 1px solid #dce4ec;
        border-radius: 12px;
        padding: 12px;
        overflow-x: auto;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.10);
        box-sizing: border-box;
    }

    /* Layout */

    hr {
        border-color: #344155;
        margin-top: 1.45rem;
        margin-bottom: 1.45rem;
    }

    div[data-testid="stDownloadButton"] button {
        border-radius: 8px;
        font-weight: 600;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #334155;
        border-radius: 9px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 4. METRIC CARD COMPONENT
# ============================================================

def metric_card(
    container,
    label: str,
    value: Any,
    description: str = "",
    description_type: str = "",
):
    """Create a white metric card with readable dark text."""

    safe_label = html.escape(str(label))
    safe_value = html.escape(str(value))
    safe_description = html.escape(str(description))

    description_class = ""

    if description_type == "positive":
        description_class = "metric-positive"
    elif description_type == "negative":
        description_class = "metric-negative"

    description_html = ""

    if safe_description:
        description_html = f"""
        <div class="metric-description {description_class}">
            {safe_description}
        </div>
        """

    render_html(
        container,
        f"""
        <div class="metric-card">
            <div class="metric-label">{safe_label}</div>
            <div class="metric-value">{safe_value}</div>
            {description_html}
        </div>
        """,
    )


# ============================================================
# 5. DATA AND MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner="Loading verified METR-LA dataset...")
def load_traffic_data():
    if not DEFAULT_DATA_PATH.exists():
        download_metr_la(DEFAULT_DATA_PATH)

    return load_and_preprocess(DEFAULT_DATA_PATH)


@st.cache_resource(show_spinner=False)
def load_gru_model(
    checkpoint_path: str,
    modified_time: float,
):
    """Load a locally generated centralized or federated GRU."""

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]

    elif (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):
        state_dict = checkpoint["state_dict"]

    else:
        state_dict = checkpoint

    model = SensorLevelGRU(
        hidden_size=32,
        num_horizons=3,
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}

    try:
        result = json.loads(
            path.read_text(encoding="utf-8")
        )
        return result if isinstance(result, dict) else {}

    except (OSError, json.JSONDecodeError):
        return {}


def read_csv(path: Path):
    if not path.exists():
        return None

    try:
        return pd.read_csv(path)

    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.warning(f"Unable to read {path.name}: {exc}")
        return None


# ============================================================
# 6. PLOTLY CHART STYLE
# ============================================================

def style_chart(fig: go.Figure, height: int = 390):

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(
            family="Arial, sans-serif",
            color="#28384a",
            size=12,
        ),
        height=height,
        margin=dict(
            l=25,
            r=20,
            t=70,
            b=30,
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0,
        ),
    )

    fig.update_xaxes(
        showgrid=False,
        linecolor="#cbd5e1",
        tickfont=dict(color="#475569"),
        title_font=dict(color="#334155"),
    )

    fig.update_yaxes(
        gridcolor="#e8edf3",
        zeroline=False,
        tickfont=dict(color="#475569"),
        title_font=dict(color="#334155"),
    )

    return fig


# ============================================================
# 7. GRU NEURAL NETWORK DIAGRAM
# ============================================================

def build_gru_diagram() -> str:
    """
    Professional static architecture diagram.

    Repeated GRU columns represent the same recurrent
    cell at different time steps, not independent layers.
    """

    width = 1180
    height = 455

    layers = [
        {
            "x": 95,
            "count": 6,
            "title": "INPUT",
            "subtitle": "12 × 2 features",
            "color": "#64748b",
        },
        {
            "x": 350,
            "count": 7,
            "title": "GRU STATE",
            "subtitle": "Earlier step",
            "color": "#0f766e",
        },
        {
            "x": 570,
            "count": 7,
            "title": "GRU STATE",
            "subtitle": "Intermediate step",
            "color": "#0f766e",
        },
        {
            "x": 790,
            "count": 7,
            "title": "GRU STATE",
            "subtitle": "Final time step",
            "color": "#0f766e",
        },
        {
            "x": 1050,
            "count": 3,
            "title": "OUTPUT",
            "subtitle": "Linear head",
            "color": "#b7791f",
        },
    ]

    positions = []

    for layer in layers:
        ys = np.linspace(140, 330, layer["count"])

        positions.append(
            [(layer["x"], float(y)) for y in ys]
        )

    pieces = []

    # Draw connections.
    for layer_index in range(len(positions) - 1):

        left = positions[layer_index]
        right = positions[layer_index + 1]

        for i, (x1, y1) in enumerate(left):

            for j, (x2, y2) in enumerate(right):

                if (
                    layer_index in (1, 2)
                    or (i + j) % 2 == 0
                ):

                    pieces.append(
                        f"""
                        <line
                            x1="{x1}" y1="{y1}"
                            x2="{x2}" y2="{y2}"
                            class="connection"
                        />
                        """
                    )

    # Draw neuron nodes and labels.
    for layer_index, layer in enumerate(layers):

        x = layer["x"]
        color = layer["color"]

        for cx, cy in positions[layer_index]:

            pieces.append(
                f"""
                <circle
                    cx="{cx}" cy="{cy}" r="9"
                    fill="#ffffff"
                    stroke="{color}"
                    stroke-width="2.2"
                />
                """
            )

        pieces.append(
            f"""
            <text x="{x}" y="62"
                  text-anchor="middle"
                  class="layer-title">
                {layer["title"]}
            </text>

            <text x="{x}" y="86"
                  text-anchor="middle"
                  class="layer-subtitle">
                {layer["subtitle"]}
            </text>
            """
        )

    # Layer descriptions.
    descriptions = [
        (95, "Speed + mask"),
        (350, "Shared GRU cell"),
        (570, "32 hidden units"),
        (790, "Final state"),
        (1050, "3 predictions"),
    ]

    for x, text in descriptions:

        pieces.append(
            f"""
            <text x="{x}" y="360"
                  text-anchor="middle"
                  class="detail">
                {text}
            </text>
            """
        )

    # Prediction outputs.
    for (x, y), label in zip(
        positions[-1],
        ["15 min", "30 min", "60 min"],
    ):

        pieces.append(
            f"""
            <text x="1073" y="{y + 4}"
                  class="output-label">
                {label}
            </text>
            """
        )

    # Recurrent flow.
    pieces.append(
        """
        <path
            d="M 350 390 C 430 425, 700 425, 790 390"
            fill="none"
            stroke="#0f766e"
            stroke-width="2"
        />

        <path
            d="M 780 383 L 790 390 L 780 397"
            fill="none"
            stroke="#0f766e"
            stroke-width="2"
        />

        <text x="570" y="439"
              text-anchor="middle"
              class="detail">
            Shared recurrent weights across the 12 input time steps
        </text>
        """
    )

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta name="viewport"
              content="width=device-width, initial-scale=1">

        <style>
            body {{
                margin: 0;
                padding: 0;
                background: #ffffff;
                font-family: Arial, sans-serif;
            }}

            .network-panel {{
                width: 100%;
                background: #ffffff;
                border-radius: 10px;
                overflow-x: auto;
            }}

            svg {{
                display: block;
                width: 100%;
                min-width: 760px;
                height: auto;
                background: #ffffff;
            }}

            .connection {{
                stroke: #b7c6d5;
                stroke-width: 1.15;
                opacity: 0.75;
            }}

            .layer-title {{
                fill: #14243a;
                font-size: 14px;
                font-weight: 700;
                letter-spacing: 0.7px;
            }}

            .layer-subtitle {{
                fill: #64748b;
                font-size: 11px;
            }}

            .detail {{
                fill: #536579;
                font-size: 11px;
            }}

            .output-label {{
                fill: #9a6700;
                font-size: 10px;
                font-weight: 700;
            }}
        </style>
    </head>

    <body>
        <div class="network-panel">

            <svg viewBox="0 0 {width} {height}"
                 role="img"
                 aria-label="FedRoad GRU architecture">

                {''.join(pieces)}

            </svg>

        </div>
    </body>
    </html>
    """


# ============================================================
# 8. HEADER
# ============================================================

render_html(
    st,
    """
    <div class="hero">

        <div class="eyebrow">
            FEDERATED TRAFFIC INTELLIGENCE
        </div>

        <h1>FedRoad</h1>

        <p>
            Deep-learning traffic-speed forecasting
            with GRU neural networks and Flower FedAvg.
        </p>

    </div>
    """,
)

st.caption(
    "METR-LA benchmark · 207 sensors · "
    "5-minute sampling · 60-minute input history · "
    "15 / 30 / 60-minute forecasting horizons"
)


# ============================================================
# 9. LOAD DATA
# ============================================================

try:
    with st.spinner("Preparing verified traffic dataset..."):
        traffic = load_traffic_data()

except Exception as exc:

    st.error(f"Unable to load METR-LA: {exc}")

    st.info(
        "Run `python -m scripts.download_data` "
        "from the project root."
    )

    st.stop()


central_metadata = read_json(CENTRAL_METADATA_PATH)
fed_metadata = read_json(FED_METADATA_PATH)


# ============================================================
# 10. MODEL AVAILABILITY
# ============================================================

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


# ============================================================
# 11. SYSTEM OVERVIEW
# ============================================================

st.subheader("System overview")

overview = st.columns(5)

metric_card(
    overview[0],
    "Traffic sensors",
    f"{len(traffic.sensor_ids):,}",
    "METR-LA road sensors",
)

metric_card(
    overview[1],
    "Time steps",
    f"{len(traffic.timestamps):,}",
    "Five-minute readings",
)

metric_card(
    overview[2],
    "Unavailable data",
    f"{traffic.zero_percent:.2f}%",
    "Masked during evaluation",
)

metric_card(
    overview[3],
    "Federated clients",
    fed_metadata.get("num_clients", 4),
    "Simulated clients",
)

metric_card(
    overview[4],
    "FedAvg rounds",
    fed_metadata.get("num_server_rounds", 10),
    "Completed aggregation rounds",
)


with st.expander("Dataset and model configuration"):

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown("**Dataset**")

        st.write("METR-LA")
        st.write(f"Rows: {len(traffic.timestamps):,}")
        st.write(f"Sensors: {len(traffic.sensor_ids)}")

        st.write(
            f"Unavailable readings: {traffic.zero_percent:.2f}%"
        )

    with c2:

        st.markdown("**Centralized GRU**")

        st.write("One GRU layer")
        st.write("32 hidden units")
        st.write("Three forecast outputs")

        st.write(
            f"Epochs: {central_metadata.get('epochs_run', '32')}"
        )

        st.write(
            f"Best epoch: {central_metadata.get('best_epoch', '25')}"
        )

    with c3:

        st.markdown("**Federated learning**")

        st.write("Framework: Flower")
        st.write("Strategy: FedAvg")

        st.write(
            f"Clients: {fed_metadata.get('num_clients', 4)}"
        )

        st.write(
            f"Rounds: {fed_metadata.get('num_server_rounds', 10)}"
        )

        st.write(
            f"Local epochs: {fed_metadata.get('local_epochs', 1)}"
        )


st.divider()


# ============================================================
# 12. TRAFFIC FORECAST
# ============================================================

st.subheader("Traffic forecasting")

st.markdown(
    """
    <div class="section-note">
        Choose a trained model, sensor and historical test window.
        The application forecasts future speeds and compares them
        with recorded observations.
    </div>
    """,
    unsafe_allow_html=True,
)

control_cols = st.columns([1.1, 1.1, 2.0])

with control_cols[0]:

    selected_model = st.selectbox(
        "Forecasting model",
        predictor_options,
        index=predictor_options.index(default_predictor),
    )

with control_cols[1]:

    selected_sensor = st.selectbox(
        "Traffic sensor ID",
        traffic.sensor_ids,
        index=0,
    )

sensor_index = traffic.sensor_ids.index(selected_sensor)

maximum_start = max(
    0,
    len(traffic.x_test) - INPUT_STEPS - max(HORIZONS),
)

with control_cols[2]:

    test_start = st.slider(
        "Historical test-window position",
        min_value=0,
        max_value=int(maximum_start),
        value=min(200, int(maximum_start)),
        help=(
            "Select the starting position of the "
            "60-minute historical input sequence."
        ),
    )


# ============================================================
# 13. INPUT PREPARATION
# ============================================================

test_offset = (
    len(traffic.timestamps) - len(traffic.x_test)
)

history_end = test_start + INPUT_STEPS

history_scaled = traffic.x_test[
    test_start:history_end,
    sensor_index,
]

history_mask = traffic.m_test[
    test_start:history_end,
    sensor_index,
]

sensor_mean = float(traffic.mean[sensor_index])
sensor_std = float(traffic.std[sensor_index])

history_raw = (
    history_scaled * sensor_std + sensor_mean
)

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

latest_speed = (
    float(history_raw[valid_history[-1]])
    if len(valid_history)
    else float("nan")
)

input_coverage = float(
    history_mask.mean() * 100
)


# ============================================================
# 14. FUTURE TARGETS
# ============================================================

future_indices = (
    test_start
    + INPUT_STEPS
    - 1
    + np.asarray(HORIZONS)
)

future_times = traffic.timestamps[
    test_offset + future_indices
]

future_mask = traffic.m_test[
    future_indices,
    sensor_index,
]

future_scaled = traffic.x_test[
    future_indices,
    sensor_index,
]

future_actual = np.where(
    future_mask > 0.5,
    future_scaled * sensor_std + sensor_mean,
    np.nan,
)


# ============================================================
# 15. MODEL INFERENCE
# ============================================================

hidden_states = None
model_error = None

if selected_model == "Persistence":

    if len(valid_history):

        latest_normalized = float(
            history_scaled[valid_history[-1]]
        )

        predictions_normalized = np.repeat(
            latest_normalized,
            len(HORIZONS),
        )

    else:

        predictions_normalized = np.full(
            len(HORIZONS),
            np.nan,
        )

else:

    checkpoint_path = model_paths[selected_model]

    try:

        model = load_gru_model(
            str(checkpoint_path.resolve()),
            checkpoint_path.stat().st_mtime,
        )

        model_input = np.stack(
            [history_scaled, history_mask],
            axis=-1,
        ).astype(np.float32)

        input_tensor = torch.from_numpy(
            model_input
        ).unsqueeze(0)

        with torch.inference_mode():

            recurrent_output, _ = model.gru(input_tensor)

            hidden_states = (
                recurrent_output[0].cpu().numpy()
            )

            predictions_normalized = (
                model(input_tensor)[0].cpu().numpy()
            )

    except Exception as exc:

        model_error = str(exc)

        predictions_normalized = np.full(
            len(HORIZONS),
            np.nan,
        )


predictions_raw = (
    predictions_normalized * sensor_std + sensor_mean
)


# ============================================================
# 16. INPUT QUALITY INFORMATION
# ============================================================

st.caption(
    f"Sensor {selected_sensor} · "
    f"Input period: {history_times[0]} to {history_times[-1]} · "
    f"Observed input coverage: {input_coverage:.1f}%"
)

if model_error:

    st.error(
        f"Unable to load or execute {selected_model}: {model_error}"
    )

if input_coverage < 100:

    st.warning(
        f"Only {input_coverage:.1f}% of input readings were observed. "
        "Missing inputs are imputed using training-set statistics, "
        "while their observation masks are retained."
    )

if not len(valid_history):

    st.warning(
        "No valid observed reading is available for this sensor "
        "in the selected window."
    )


# ============================================================
# 17. FORECAST CARDS
# ============================================================

forecast_columns = st.columns(3)

for index, horizon in enumerate([15, 30, 60]):

    value = float(predictions_raw[index])

    if np.isfinite(value):

        description = ""

        if np.isfinite(latest_speed):

            difference = value - latest_speed

            description = (
                f"{difference:+.2f} compared with latest observation"
            )

        metric_card(
            forecast_columns[index],
            f"Forecast · +{horizon} minutes",
            f"{value:.2f}",
            description,
        )

    else:

        metric_card(
            forecast_columns[index],
            f"Forecast · +{horizon} minutes",
            "Unavailable",
        )


# ============================================================
# 18. FORECAST CHART
# ============================================================

forecast_fig = go.Figure()

forecast_fig.add_trace(
    go.Scatter(
        x=history_times,
        y=history_plot,
        mode="lines+markers",
        name="Observed input history",
        connectgaps=False,
        line=dict(color="#526579", width=2.5),
        marker=dict(size=5),
    )
)

forecast_origin = history_times[-1]

forecast_origin_value = (
    float(history_raw[-1])
    if history_mask[-1] > 0.5
    else None
)

forecast_fig.add_trace(
    go.Scatter(
        x=[forecast_origin, *list(future_times)],
        y=[forecast_origin_value, *list(predictions_raw)],
        mode="lines+markers",
        name=f"{selected_model} forecast",
        connectgaps=False,
        line=dict(
            color="#0f766e",
            width=2.8,
            dash="dash",
        ),
        marker=dict(
            size=10,
            symbol="diamond",
        ),
    )
)

if np.isfinite(future_actual).any():

    forecast_fig.add_trace(
        go.Scatter(
            x=future_times,
            y=future_actual,
            mode="markers",
            name="Observed future",
            marker=dict(
                color="#c0842d",
                size=12,
                symbol="circle-open",
                line=dict(width=2),
            ),
        )
    )

forecast_fig.add_vline(
    x=forecast_origin,
    line_dash="dot",
    line_color="#94a3b8",
    annotation_text="Forecast origin",
)

forecast_fig.update_layout(
    title="Historical observations and predicted traffic speed",
    xaxis_title="Timestamp",
    yaxis_title="Speed (dataset units)",
)

st.plotly_chart(
    style_chart(forecast_fig, 430),
    use_container_width=True,
)

st.caption(
    "Predictions are generated at +15, +30 and +60 minutes. "
    "The connecting line is only a visual guide, not a series "
    "of additional five-minute predictions."
)


# ============================================================
# 19. FORECAST DETAIL TABLE
# ============================================================

forecast_table = pd.DataFrame({
    "Forecast horizon": [
        f"+{h} min" for h in [15, 30, 60]
    ],
    "Target timestamp": [
        str(t) for t in future_times
    ],
    "Predicted speed": predictions_raw,
    "Observed speed": future_actual,
})

forecast_table["Absolute error"] = np.where(
    np.isfinite(future_actual)
    & np.isfinite(predictions_raw),
    np.abs(predictions_raw - future_actual),
    np.nan,
)

st.dataframe(
    forecast_table.round(3),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    f"Observed future targets: "
    f"{int(np.sum(future_mask > 0.5))}/3. "
    "The model produces point forecasts, not calibrated prediction intervals."
)


st.divider()


# ============================================================
# 20. NEURAL NETWORK ARCHITECTURE
# ============================================================

st.subheader("Neural network architecture")

st.markdown(
    """
    <div class="section-note">
        Structural view of the GRU neural network and its
        actual hidden-state activations.
    </div>
    """,
    unsafe_allow_html=True,
)

components.html(
    build_gru_diagram(),
    height=480,
    scrolling=False,
)

st.caption(
    "Input: 12 time steps × 2 features → one GRU layer "
    "with 32 hidden units → linear output with three forecasts. "
    "Repeated GRU columns represent the same recurrent cell "
    "at different time steps, not independent hidden layers."
)


# ============================================================
# 21. ACTUAL GRU ACTIVATIONS
# ============================================================

if (
    selected_model in model_paths
    and hidden_states is not None
):

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    architecture_cols = st.columns(4)

    metric_card(
        architecture_cols[0],
        "Input tensor",
        "1 × 12 × 2",
        "One sensor sequence",
    )

    metric_card(
        architecture_cols[1],
        "GRU hidden units",
        "32",
        "One recurrent layer",
    )

    metric_card(
        architecture_cols[2],
        "Forecast outputs",
        "3",
        "15 / 30 / 60 minutes",
    )

    metric_card(
        architecture_cols[3],
        "Trainable parameters",
        f"{parameter_count:,}",
        "Selected checkpoint",
    )

    time_labels = [
        f"t-{(INPUT_STEPS - 1 - i) * 5} min"
        for i in range(INPUT_STEPS)
    ]

    activation_fig = px.imshow(
        hidden_states.T,
        x=time_labels,
        y=[
            f"Unit {i + 1}"
            for i in range(hidden_states.shape[1])
        ],
        aspect="auto",
        origin="lower",
        color_continuous_scale="RdBu_r",
        labels={
            "x": "Input time step",
            "y": "GRU hidden unit",
            "color": "Activation",
        },
        title="Actual GRU hidden-state activation matrix",
    )

    st.plotly_chart(
        style_chart(activation_fig, 430),
        use_container_width=True,
    )

    final_activation = hidden_states[-1]

    activation_data = pd.DataFrame({
        "Hidden unit": [
            f"Unit {i + 1}"
            for i in range(len(final_activation))
        ],
        "Activation": final_activation,
    })

    activation_bar = px.bar(
        activation_data,
        x="Hidden unit",
        y="Activation",
        title="Final recurrent state across hidden units",
        color_discrete_sequence=["#0f766e"],
    )

    st.plotly_chart(
        style_chart(activation_bar, 320),
        use_container_width=True,
    )

    st.caption(
        "These are actual hidden-state activations from the selected "
        "trained GRU. They are diagnostic values, not feature-importance "
        "scores or causal explanations."
    )

else:

    st.info(
        "Select a trained GRU model to inspect its actual hidden-state "
        "activations. The architecture diagram remains visible "
        "independently."
    )


st.divider()


# ============================================================
# 22. MODEL PERFORMANCE COMPARISON
# ============================================================

st.subheader("Model performance comparison")

st.markdown(
    """
    <div class="section-note">
        Held-out test results for persistence, centralized GRU
        and federated GRU. Lower MAE and RMSE indicate better forecasts.
    </div>
    """,
    unsafe_allow_html=True,
)

comparison = read_csv(COMPARISON_PATH)

if comparison is None or comparison.empty:

    st.warning(
        "Comparison results are unavailable. "
        "Run `python scripts/compare_experiments.py`."
    )

else:

    required_columns = {
        "Model",
        "Horizon (minutes)",
        "MAE",
        "RMSE",
        "Valid targets",
    }

    missing = required_columns - set(comparison.columns)

    if missing:

        st.error(
            f"Missing comparison columns: {sorted(missing)}"
        )

    else:

        for column in [
            "Horizon (minutes)",
            "MAE",
            "RMSE",
            "Valid targets",
        ]:

            comparison[column] = pd.to_numeric(
                comparison[column],
                errors="coerce",
            )

        selected_metric = st.radio(
            "Evaluation metric",
            ["MAE", "RMSE"],
            horizontal=True,
        )

        comparison_fig = px.bar(
            comparison,
            x="Horizon (minutes)",
            y=selected_metric,
            color="Model",
            barmode="group",
            text_auto=".3f",
            category_orders={
                "Horizon (minutes)": [15, 30, 60]
            },
            labels={
                "Horizon (minutes)": "Forecast horizon",
                selected_metric: (
                    f"{selected_metric} (dataset speed units)"
                ),
            },
            title=f"Held-out test {selected_metric} by horizon",
            color_discrete_sequence=[
                "#64748b",
                "#0f766e",
                "#c0842d",
            ],
        )

        comparison_fig.update_layout(
            xaxis_type="category",
        )

        st.plotly_chart(
            style_chart(comparison_fig, 420),
            use_container_width=True,
        )

        display_columns = [
            column
            for column in [
                "Model",
                "Horizon (minutes)",
                "MAE",
                "RMSE",
                "Valid targets",
                "MAE improvement vs persistence (%)",
                "MAE difference vs centralized (%)",
            ]
            if column in comparison.columns
        ]

        st.dataframe(
            comparison[display_columns].round(3),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download model comparison CSV",
            data=comparison.to_csv(index=False).encode("utf-8"),
            file_name="fedroad_model_comparison.csv",
            mime="text/csv",
        )

        # Centralized vs FedAvg comparison
        central_rows = comparison[
            comparison["Model"] == "Centralized shared GRU"
        ]

        fed_rows = comparison[
            comparison["Model"] == "FedAvg GRU"
        ]

        if not central_rows.empty and not fed_rows.empty:

            joined = central_rows[
                ["Horizon (minutes)", "MAE"]
            ].merge(
                fed_rows[
                    ["Horizon (minutes)", "MAE"]
                ],
                on="Horizon (minutes)",
                suffixes=("_central", "_fed"),
            )

            joined["Gap (%)"] = (
                (
                    joined["MAE_fed"]
                    - joined["MAE_central"]
                )
                / joined["MAE_central"]
            ) * 100

            gap_columns = st.columns(3)

            for index, (_, row) in enumerate(
                joined.sort_values(
                    "Horizon (minutes)"
                ).iterrows()
            ):

                if index >= 3:
                    break

                render_html(
                    gap_columns[index],
                    f"""
                    <div class="metric-card">

                        <div class="metric-label">
                            FedAvg MAE gap ·
                            {int(row['Horizon (minutes)'])} min
                        </div>

                        <div class="metric-value">
                            {row['Gap (%)']:+.2f}%
                        </div>

                        <div class="metric-description">
                            Positive means higher error than
                            centralized GRU.
                        </div>

                    </div>
                    """,
                )

            st.info(
                "Centralized and federated models used different "
                "training budgets. This comparison is informative, "
                "but it is not a perfectly controlled head-to-head experiment."
            )


st.divider()


# ============================================================
# 23. FEDERATED TRAINING PROGRESS
# ============================================================

st.subheader("Federated training progress")

st.markdown(
    """
    <div class="section-note">
        Training and validation convergence recorded during
        the Flower FedAvg experiment.
    </div>
    """,
    unsafe_allow_html=True,
)

rounds = read_csv(ROUND_PATH)

if rounds is None or rounds.empty:

    st.info(
        "Federated round history is unavailable. "
        "The held-out test comparison remains available above."
    )

else:

    required_round_columns = {
        "round",
        "train_mae_normalized",
        "val_mae_normalized",
    }

    if not required_round_columns.issubset(rounds.columns):

        st.warning(
            "Federated round CSV does not contain the expected columns."
        )

    else:

        rounds = rounds.sort_values("round")

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
            "train_mae_normalized": "Training MAE",
            "val_mae_normalized": "Validation MAE",
        })

        round_fig = px.line(
            round_long,
            x="round",
            y="Normalized MAE",
            color="Series",
            markers=True,
            title="FedAvg convergence across server rounds",
            color_discrete_sequence=[
                "#64748b",
                "#0f766e",
            ],
        )

        round_fig.update_layout(
            xaxis_title="Federated server round",
            yaxis_title="Normalized masked MAE",
        )

        st.plotly_chart(
            style_chart(round_fig, 360),
            use_container_width=True,
        )

        initial_val = float(
            rounds["val_mae_normalized"].iloc[0]
        )

        final_val = float(
            rounds["val_mae_normalized"].iloc[-1]
        )

        reduction = (
            (initial_val - final_val) / initial_val * 100
            if initial_val > 0
            else 0.0
        )

        progress_columns = st.columns(3)

        metric_card(
            progress_columns[0],
            "Recorded rounds",
            int(rounds["round"].nunique()),
            "Completed Flower rounds",
        )

        metric_card(
            progress_columns[1],
            "Final validation MAE",
            f"{final_val:.4f}",
            "Normalized units",
        )

        metric_card(
            progress_columns[2],
            "Validation reduction",
            f"{reduction:.2f}%",
            "Round 1 to final round",
        )


st.divider()


# ============================================================
# 24. METHODOLOGY
# ============================================================

with st.expander("Methodology, model settings and limitations"):

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### Dataset and preprocessing")

        st.markdown(
            f"""
            - Dataset: **METR-LA**
            - Observations: **{len(traffic.timestamps):,}**
            - Sensors: **{len(traffic.sensor_ids)}**
            - Sampling: **5 minutes**
            - Input history: **60 minutes**
            - Unavailable readings: **{traffic.zero_percent:.2f}%**
            - Chronological split: **70% / 10% / 20%**
            - Normalization fitted on training data only.
            - Observation masks retained during training and evaluation.
            """
        )

        st.markdown("### Neural network")

        st.markdown(
            """
            - Framework: PyTorch
            - Model: Sensor-level GRU
            - Input: Speed and observation mask
            - GRU layers: 1
            - Hidden units: 32
            - Output values: 3
            - Horizons: 15, 30 and 60 minutes
            """
        )

    with col2:

        st.markdown("### Federated learning")

        st.markdown(
            f"""
            - Framework: Flower
            - Aggregation: FedAvg
            - Simulated clients: **{fed_metadata.get('num_clients', 4)}**
            - Server rounds: **{fed_metadata.get('num_server_rounds', 10)}**
            - Local epochs: **{fed_metadata.get('local_epochs', 1)}**
            - Batch size: **{fed_metadata.get('batch_size', 64)}**
            - Clients train locally; the coordinator aggregates model updates.
            """
        )

        st.markdown("### Limitations")

        st.markdown(
            """
            - Predictions are based on historical held-out data.
            - This is not a live traffic service.
            - Predictions are point estimates, not calibrated intervals.
            - Clients are simulated in one runtime.
            - Differential privacy and secure aggregation are not implemented.
            - Centralized and federated training budgets differ.
            - The directed graph GRU remains experimental until
              its full-data benchmark is completed.
            """
        )


st.caption(
    "FedRoad · Research prototype · "
    "Historical traffic benchmark, not a live traffic service."
)
