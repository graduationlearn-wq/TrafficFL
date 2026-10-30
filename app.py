"""Streamlit interface for multi-target future forecasting and historical backtesting."""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

from main import (
    CHECKPOINT,
    DEVICE,
    FEATURES,
    HORIZONS,
    INPUT_STEPS,
    TrafficGRU,
)

st.set_page_config(
    page_title="TrafficFL | Multi-Target Forecasting",
    page_icon="🚦",
    layout="wide",
)

HORIZON_MINUTES = [step * 5 for step in HORIZONS]
CHART_FEATURES = [
    {
        "name": "Traffic volume",
        "column": "flow_veh_5min",
        "feature_index": 0,
        "unit": "Vehicles per 5 minutes",
        "scale": 1.0,
        "padding_floor": 3.0,
    },
    {
        "name": "Road occupancy",
        "column": "occupancy_fraction",
        "feature_index": 1,
        "unit": "Occupancy (%)",
        "scale": 100.0,
        "padding_floor": 0.2,
    },
    {
        "name": "Traffic speed",
        "column": "speed_mph",
        "feature_index": 2,
        "unit": "Speed (mph)",
        "scale": 1.0,
        "padding_floor": 0.5,
    },
]


@st.cache_resource
def load_model():
    if not CHECKPOINT.exists():
        raise FileNotFoundError("Model checkpoint not found. Run `python main.py train` first.")
    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    model = TrafficGRU().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def congestion_label(predicted_speed: float, free_flow_speed: float) -> str:
    """Project heuristic based on predicted speed relative to sensor reference speed."""
    ratio = predicted_speed / max(float(free_flow_speed), 1.0)
    if ratio >= 0.70:
        return "Free-flowing"
    if ratio >= 0.40:
        return "Moderate congestion"
    return "Heavy congestion"


def prepare_sensor_frame(uploaded, sensor_id):
    frame = pd.read_csv(uploaded)
    required = {"timestamp", "sensor_id", "flow_veh_5min", "speed_mph"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("CSV is missing required columns: " + ", ".join(sorted(missing)))
    if "occupancy_fraction" not in frame.columns and "occupancy_pct" not in frame.columns:
        raise ValueError("CSV must contain occupancy_fraction (0-1) or occupancy_pct (0-100).")

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["sensor_id"] = frame["sensor_id"].astype(str)
    frame = frame[frame["sensor_id"] == str(sensor_id)].copy()
    if frame.empty:
        raise ValueError("The selected sensor is not present in the uploaded CSV.")

    frame = frame.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    for column in ["flow_veh_5min", "speed_mph"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if "occupancy_fraction" in frame.columns:
        frame["occupancy_fraction"] = pd.to_numeric(frame["occupancy_fraction"], errors="coerce")
    else:
        supplied_occupancy = pd.to_numeric(frame["occupancy_pct"], errors="coerce")
        finite_values = supplied_occupancy[np.isfinite(supplied_occupancy)]
        # Accept legacy project exports where the 0-1 fraction was labelled occupancy_pct.
        if finite_values.empty or finite_values.max() <= 1.0:
            frame["occupancy_fraction"] = supplied_occupancy
        else:
            frame["occupancy_fraction"] = supplied_occupancy / 100.0

    frame = frame.dropna(subset=["timestamp"]).reset_index(drop=True)
    if len(frame) < INPUT_STEPS + max(HORIZONS):
        raise ValueError("At least 24 five-minute rows for the selected sensor are required.")

    gaps = frame["timestamp"].diff().dropna()
    if not gaps.eq(pd.Timedelta(minutes=5)).all():
        raise ValueError("Selected sensor timestamps must be continuous five-minute intervals with no gaps.")
    return frame


def transform_values(frame, checkpoint):
    raw = frame[["flow_veh_5min", "occupancy_fraction", "speed_mph"]].to_numpy(
        dtype=np.float32, copy=True
    )
    means = checkpoint["means"].cpu().numpy()
    medians = checkpoint["medians"].cpu().numpy()
    stds = checkpoint["stds"].cpu().numpy()
    observed = np.isfinite(raw) & (raw >= 0)
    for feature in range(3):
        raw[~observed[:, feature], feature] = medians[feature]
    normalized = (raw - means.reshape(1, 3)) / stds.reshape(1, 3)
    return normalized.astype(np.float32), observed


def predict_windows(model, normalized_windows, checkpoint):
    with torch.no_grad():
        tensor = torch.from_numpy(np.asarray(normalized_windows, dtype=np.float32)).to(DEVICE)
        prediction = model(tensor).cpu().numpy()
    means = checkpoint["means"].cpu().numpy().reshape(1, 1, 3)
    stds = checkpoint["stds"].cpu().numpy().reshape(1, 1, 3)
    prediction = prediction * stds + means
    prediction[:, :, 0] = np.maximum(prediction[:, :, 0], 0.0)
    prediction[:, :, 1] = np.clip(prediction[:, :, 1], 0.0, 1.0)
    prediction[:, :, 2] = np.maximum(prediction[:, :, 2], 0.0)
    return prediction


def get_targets(frame_values, origins):
    offsets = np.asarray(HORIZONS, dtype=np.int64)
    return np.stack([frame_values[origins + offset] for offset in offsets], axis=1)


def build_result_rows(prediction, actual, free_speed):
    rows = []
    for h_index, minutes in enumerate(HORIZON_MINUTES):
        pred_flow, pred_occupancy, pred_speed = prediction[h_index]
        row = {
            "Forecast": f"+{minutes} min",
            "Traffic volume (vehicles / 5 min)": pred_flow,
            "Traffic flow (vehicles / hour)": pred_flow * 12.0,
            "Road occupancy (%)": pred_occupancy * 100.0,
            "Traffic speed (mph)": pred_speed,
            "Congestion level": congestion_label(pred_speed, free_speed),
        }
        if actual is not None:
            act_flow, act_occupancy, act_speed = actual[h_index]
            row.update(
                {
                    "Actual volume (vehicles / 5 min)": act_flow,
                    "Actual occupancy (%)": act_occupancy * 100.0 if np.isfinite(act_occupancy) else np.nan,
                    "Actual speed (mph)": act_speed,
                    "Actual congestion": congestion_label(act_speed, free_speed)
                    if np.isfinite(act_speed)
                    else "Unavailable",
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def show_forecast_table(result):
    formatted = result.copy()
    numeric = formatted.select_dtypes(include=[np.number]).columns
    formatted[numeric] = formatted[numeric].round(2)
    st.dataframe(formatted, width="stretch", hide_index=True)


def _visible_y_range(series_values, padding_floor, zero_based):
    """Return a padded y-range so small changes are visible without changing data."""
    arrays = []
    for values in series_values:
        if values is None:
            continue
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        array = array[np.isfinite(array)]
        if array.size:
            arrays.append(array)

    if not arrays:
        return [0.0, 1.0]

    all_values = np.concatenate(arrays)
    low = float(np.min(all_values))
    high = float(np.max(all_values))
    span = high - low
    padding = max(span * 0.18, float(padding_floor))
    upper = high + padding

    if zero_based:
        return [0.0, max(upper, padding_floor * 2.0)]

    # Traffic measurements are non-negative. Keep a little space below the minimum,
    # but do not force the axis to zero unless the observed values approach zero.
    lower = max(0.0, low - padding)
    if upper - lower < padding_floor * 2.0:
        center = (low + high) / 2.0
        lower = max(0.0, center - padding_floor)
        upper = center + padding_floor
    return [lower, upper]


def build_forecast_figure(
    frame,
    origin_index,
    prediction,
    feature_config,
    actual_future=None,
    zero_based_y=False,
    chart_mode="future",
):
    """Build one interactive chart for one measured traffic variable."""
    feature_index = feature_config["feature_index"]
    scale = feature_config["scale"]
    column = feature_config["column"]
    title = feature_config["name"]
    unit = feature_config["unit"]

    history_start = max(0, origin_index - INPUT_STEPS + 1)
    history_times = frame.loc[history_start:origin_index, "timestamp"].tolist()
    history_values = (
        frame.loc[history_start:origin_index, column].to_numpy(dtype=np.float64) * scale
    )

    origin_time = pd.Timestamp(frame.loc[origin_index, "timestamp"])
    if actual_future is None:
        forecast_times = [origin_time + pd.Timedelta(minutes=minutes) for minutes in HORIZON_MINUTES]
    else:
        future_indices = origin_index + np.asarray(HORIZONS, dtype=np.int64)
        forecast_times = frame.loc[future_indices, "timestamp"].tolist()

    forecast_values = np.asarray(prediction[:, feature_index], dtype=np.float64) * scale
    actual_values = None
    if actual_future is not None:
        actual_values = np.asarray(actual_future[:, feature_index], dtype=np.float64) * scale

    y_range = _visible_y_range(
        [history_values, forecast_values, actual_values],
        feature_config["padding_floor"],
        zero_based_y,
    )

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=history_times,
            y=history_values,
            mode="lines+markers",
            name="Observed history",
            line={"shape": "spline", "smoothing": 0.45, "width": 3},
            marker={"size": 7},
            connectgaps=False,
            hovertemplate=f"%{{x|%d %b %H:%M}}<br>%{{y:.3f}} {unit}<extra>Observed history</extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=forecast_times,
            y=forecast_values,
            mode="lines+markers",
            name="GRU forecast (+15 / +30 / +60 min)",
            line={"shape": "spline", "smoothing": 0.55, "dash": "dash", "width": 3},
            marker={"size": 11, "symbol": "diamond"},
            hovertemplate=f"%{{x|%d %b %H:%M}}<br>%{{y:.3f}} {unit}<extra>GRU forecast</extra>",
        )
    )

    if actual_values is not None:
        figure.add_trace(
            go.Scatter(
                x=forecast_times,
                y=actual_values,
                mode="markers+lines",
                name="Recorded future",
                line={"width": 2, "dash": "dot"},
                marker={"size": 10, "symbol": "x"},
                connectgaps=False,
                hovertemplate=f"%{{x|%d %b %H:%M}}<br>%{{y:.3f}} {unit}<extra>Recorded future</extra>",
            )
        )

    figure.add_shape(
        type="line",
        x0=origin_time,
        x1=origin_time,
        y0=0,
        y1=1,
        yref="paper",
        line={"dash": "dot", "width": 1.5, "color": "#AAAAAA"},
    )
    figure.add_annotation(
        x=origin_time,
        y=1,
        yref="paper",
        text="Forecast origin",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font={"size": 11},
    )

    if chart_mode == "future":
        chart_title = f"{title}: recent observations and future forecast"
    else:
        chart_title = f"{title}: historical forecast vs recorded values"

    x_end = max(forecast_times)
    figure.update_layout(
        title={"text": chart_title, "x": 0.01, "xanchor": "left"},
        height=360,
        margin={"l": 20, "r": 25, "t": 65, "b": 35},
        template="plotly_dark",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        xaxis={
            "title": "Time",
            "type": "date",
            "range": [history_times[0], x_end],
            "tickformat": "%d %b<br>%H:%M",
            "dtick": 15 * 60 * 1000,
            "showgrid": True,
            "gridcolor": "rgba(180,180,180,0.15)",
            "zeroline": False,
        },
        yaxis={
            "title": unit,
            "range": y_range,
            "fixedrange": False,
            "showgrid": True,
            "gridcolor": "rgba(180,180,180,0.15)",
            "zeroline": False,
        },
    )
    return figure


def show_forecast_charts(frame, origin, prediction, actual=None, zero_based_y=False, chart_mode="future"):
    for config in CHART_FEATURES:
        figure = build_forecast_figure(
            frame=frame,
            origin_index=origin,
            prediction=prediction,
            feature_config=config,
            actual_future=actual,
            zero_based_y=zero_based_y,
            chart_mode=chart_mode,
        )
        st.plotly_chart(figure, width="stretch", config={"displaylogo": False})


st.title("🚦 TrafficFL: Multi-Target Traffic Forecasting")
st.caption("GRU + four simulated federated clients using PeMSD4 traffic measurements")

try:
    model, checkpoint = load_model()
except Exception as error:
    st.error(str(error))
    st.info("Run `python main.py inspect`, then `python main.py train` from the project folder.")
    st.stop()

st.success(f"Loaded best federated GRU checkpoint from communication round {checkpoint['best_round']}.")
st.markdown(
    "The GRU learns **flow, occupancy and speed**. Traffic volume is the same five-minute flow count, "
    "shown in vehicle-count form; hourly flow is derived by multiplying by 12. Congestion is derived "
    "from predicted speed and is not a separate neural-network output."
)

uploaded = st.file_uploader("Upload a traffic CSV", type=["csv"])
if uploaded is None:
    st.info(
        "CSV format: timestamp, sensor_id, flow_veh_5min, occupancy_fraction, speed_mph. "
        "Run `python main.py export --sensor 0` to create a real-data example CSV."
    )
    st.stop()

try:
    raw_preview = pd.read_csv(uploaded)
    if "sensor_id" not in raw_preview.columns:
        raise ValueError("CSV must contain a sensor_id column.")
    sensor_options = sorted(raw_preview["sensor_id"].dropna().astype(str).unique().tolist())
    if not sensor_options:
        raise ValueError("No sensor IDs were found in the CSV.")
    sensor_id = st.selectbox("Select sensor", sensor_options)
    uploaded.seek(0)
    frame = prepare_sensor_frame(uploaded, sensor_id)
    normalized, observed = transform_values(frame, checkpoint)
    st.caption(f"Selected sensor: {sensor_id} · {len(frame):,} five-minute observations")
except Exception as error:
    st.error(str(error))
    st.stop()

mode = st.radio("Choose analysis", ["Future prediction", "Historical backtesting"], horizontal=True)
zero_based_y = st.checkbox(
    "Use zero-based Y-axis (less zoomed)",
    value=False,
    help="By default, each chart zooms to the selected sensor's observed/predicted range so small changes are easier to see.",
)
st.caption(
    "The default charts use a zoomed Y-axis to make variation visible. The data is not changed. "
    "Enable the zero-based option when you want to compare charts using a baseline of zero."
)

sensor_number = int(sensor_id) if sensor_id.isdigit() else -1
free_speeds = checkpoint["free_flow_speed"].cpu().numpy()
free_speed = float(free_speeds[sensor_number]) if 0 <= sensor_number < len(free_speeds) else float(np.median(free_speeds))

if mode == "Future prediction":
    st.subheader("Future traffic forecast")
    if st.button("Generate future forecast", type="primary"):
        origin = len(frame) - 1
        last_window = normalized[-INPUT_STEPS:]
        prediction = predict_windows(model, last_window[None, :, :], checkpoint)[0]
        result = build_result_rows(prediction, None, free_speed)
        show_forecast_table(result)
        st.caption(
            f"Forecast origin: {frame['timestamp'].iloc[origin]} · Uses the last {INPUT_STEPS} observations "
            "(60 minutes). Actual future values are not available yet, so MAE/RMSE cannot be calculated."
        )
        st.subheader("Future prediction charts")
        show_forecast_charts(
            frame=frame,
            origin=origin,
            prediction=prediction,
            actual=None,
            zero_based_y=zero_based_y,
            chart_mode="future",
        )
        st.info(
            "The dashed forecast line connects the model's three outputs (+15, +30 and +60 minutes) "
            "for visual guidance. The GRU predicts these three horizons; it does not generate a new "
            "prediction at every five-minute interval."
        )
        st.download_button(
            "Download forecasts as CSV",
            result.to_csv(index=False).encode("utf-8"),
            file_name=f"traffic_forecast_sensor_{sensor_id}.csv",
            mime="text/csv",
        )
else:
    st.subheader("Historical forecast comparison")
    st.write(
        "Choose a historical forecast origin. The model uses the previous 12 rows and predicts the "
        "next 15, 30 and 60 minutes; those predictions are then compared with recorded values."
    )
    values = frame[["flow_veh_5min", "occupancy_fraction", "speed_mph"]].to_numpy(dtype=np.float32)
    first_origin = INPUT_STEPS - 1
    last_origin = len(frame) - max(HORIZONS) - 1
    origin = st.slider(
        "Input window ends at row",
        min_value=first_origin,
        max_value=last_origin,
        value=min(max(first_origin + 100, first_origin), last_origin),
    )
    if st.button("Run historical forecast", type="primary"):
        x = normalized[origin - INPUT_STEPS + 1 : origin + 1]
        prediction = predict_windows(model, x[None, :, :], checkpoint)[0]
        actual = values[origin + np.asarray(HORIZONS)]
        actual = np.where(np.isfinite(actual) & (actual >= 0), actual, np.nan)
        result = build_result_rows(prediction, actual, free_speed)
        show_forecast_table(result)
        st.write(f"Forecast origin: **{frame['timestamp'].iloc[origin]}**")

        st.subheader("Historical charts: prediction vs actual")
        show_forecast_charts(
            frame=frame,
            origin=origin,
            prediction=prediction,
            actual=actual,
            zero_based_y=zero_based_y,
            chart_mode="historical",
        )
        st.caption(
            "Historical observations use a solid line, GRU horizon forecasts use a dashed line, "
            "and recorded future values use X markers. All three variables have independent Y-axis scales."
        )

        error_rows = []
        for h_index, minutes in enumerate(HORIZON_MINUTES):
            for feature_index, feature in enumerate(FEATURES):
                if np.isfinite(actual[h_index, feature_index]):
                    metric_scale = 100.0 if feature_index == 1 else 1.0
                    error_rows.append(
                        {
                            "Horizon": f"{minutes} min",
                            "Target": feature,
                            "Unit": "percentage points" if feature_index == 1 else ("vehicles/5 min" if feature_index == 0 else "mph"),
                            "Absolute error": abs(float(prediction[h_index, feature_index] - actual[h_index, feature_index])) * metric_scale,
                        }
                    )
        if error_rows:
            st.subheader("Absolute prediction errors")
            st.dataframe(pd.DataFrame(error_rows).round(3), width="stretch", hide_index=True)

        st.download_button(
            "Download historical comparison as CSV",
            result.to_csv(index=False).encode("utf-8"),
            file_name=f"historical_forecast_sensor_{sensor_id}.csv",
            mime="text/csv",
        )

    st.divider()
    st.subheader("Evaluate all available historical origins for this sensor")
    st.caption("This computes MAE and RMSE over all rolling forecast windows in the uploaded file.")
    if st.button("Evaluate all historical windows"):
        origins = np.arange(first_origin, last_origin + 1, dtype=np.int64)
        windows = np.stack([normalized[i - INPUT_STEPS + 1 : i + 1] for i in origins])
        actual_all = get_targets(values, origins)
        observed_targets = np.isfinite(actual_all) & (actual_all >= 0)
        actual_filled = np.where(observed_targets, actual_all, 0.0)
        predictions = []
        for start in range(0, len(windows), 512):
            predictions.append(predict_windows(model, windows[start : start + 512], checkpoint))
        predictions = np.concatenate(predictions, axis=0)

        rows = []
        for horizon_index, minutes in enumerate(HORIZON_MINUTES):
            for feature_index, feature in enumerate(FEATURES):
                mask = observed_targets[:, horizon_index, feature_index]
                if not mask.any():
                    continue
                difference = predictions[:, horizon_index, feature_index][mask] - actual_filled[:, horizon_index, feature_index][mask]
                last_values = windows[:, -1, feature_index] * float(checkpoint["stds"][feature_index]) + float(checkpoint["means"][feature_index])
                baseline_difference = last_values[mask] - actual_filled[:, horizon_index, feature_index][mask]
                metric_scale = 100.0 if feature_index == 1 else 1.0
                gru_mae = float(np.mean(np.abs(difference))) * metric_scale
                persistence_mae = float(np.mean(np.abs(baseline_difference))) * metric_scale
                rows.append(
                    {
                        "Horizon": f"{minutes} min",
                        "Target": feature,
                        "Unit": "percentage points" if feature_index == 1 else ("vehicles/5 min" if feature_index == 0 else "mph"),
                        "GRU_MAE": gru_mae,
                        "GRU_RMSE": np.sqrt(np.mean(np.square(difference))) * metric_scale,
                        "Persistence_MAE": persistence_mae,
                        "Persistence_RMSE": np.sqrt(np.mean(np.square(baseline_difference))) * metric_scale,
                        "MAE_improvement_percent": 100.0 * (persistence_mae - gru_mae) / persistence_mae if persistence_mae > 0 else 0.0,
                        "Valid_targets": int(mask.sum()),
                    }
                )
        metrics = pd.DataFrame(rows)
        st.dataframe(metrics.round(3), width="stretch", hide_index=True)
        st.download_button(
            "Download full backtest metrics",
            metrics.to_csv(index=False).encode("utf-8"),
            file_name=f"historical_backtest_metrics_sensor_{sensor_id}.csv",
            mime="text/csv",
        )
