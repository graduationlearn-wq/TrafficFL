# TrafficFL — Multi-Target Traffic Forecasting

A small local Python project for a student prototype. It uses a GRU deep-learning model and a four-client **simulated** Federated Averaging (FedAvg) training loop.

## What the model predicts

The project uses **PeMSD4**, a public California PeMS traffic dataset. The published data shape is `16,992 time steps × 307 sensors × 3 channels`, sampled every five minutes. The channel order used by this project is:

1. `flow`: total vehicle flow in one five-minute interval (vehicles / 5 min).
2. `occupancy`: detector occupancy (percent).
3. `speed`: average traffic speed (mph).

The GRU learns these three measured variables. It uses the previous 12 time steps (60 minutes) to predict all three variables at +15, +30 and +60 minutes.

**Volume and flow are not separate independent channels in this dataset.** The model's flow channel is the number of vehicles counted over five minutes; the app shows it as five-minute traffic volume and also displays an hourly-equivalent flow rate (`vehicles / 5 min × 12`). This hourly value is a unit conversion, not a fourth model output.

**Congestion is derived from the predicted speed**, using a project-defined sensor-relative rule: speed at least 70% of that sensor's training 85th-percentile speed = free-flowing; 40–70% = moderate; below 40% = heavy. This is an illustrative rule for the prototype, not an official PeMS congestion label and not a separately trained output.

## Project files

```text
TrafficFL_Multitarget/
├── main.py           # dataset download, inspection, GRU, FedAvg, train and test
├── app.py            # Streamlit future forecast + historical backtesting
├── requirements.txt
├── README.md
├── data/
│   └── raw/           # PEMS04.npz is downloaded here
├── artifacts/         # best trained model checkpoint
└── results/           # training history and test metrics
```

## Run locally in VS Code (Windows PowerShell)

Open this folder in VS Code and run these commands in the VS Code terminal. No VM, Docker or batch files are required.

### Step 1 — Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### Step 2 — Download, verify and inspect the dataset

```powershell
python main.py inspect
```

The dataset downloader retries failed requests and resumes from `data/raw/PEMS04.npz.part`. It verifies SHA-256 before accepting the file. If your internet connection drops, rerun the same command; it will try to resume the partial download.

### Step 3 — Train the GRU with simulated FedAvg

```powershell
python main.py train
```

Defaults: four simulated clients, one local epoch per client per communication round, maximum 20 rounds, validation-based early stopping. It samples up to 800 evenly spaced training windows per sensor to keep CPU use manageable. Scalers are fitted on training rows only. The best validation checkpoint is saved to `artifacts/traffic_gru_fedavg.pt`.

### Step 4 — Evaluate the held-out test split

```powershell
python main.py test
```

Outputs `results/test_metrics.csv` with MAE/RMSE for flow, occupancy and speed at all three horizons, plus a persistence baseline comparison.

### Step 5 — Export a real sensor CSV for the app

```powershell
python main.py export --sensor 0
```

This creates `data/example_sensor_0.csv` using genuine PeMSD4 observations for one sensor, in the upload format expected by Streamlit.

### Step 6 — Start Streamlit

```powershell
python -m streamlit run app.py
```

Upload `data/example_sensor_0.csv`, select a sensor, and choose:
- **Future prediction**: predicts from the last 12 rows.
- **Historical backtesting**: chooses an earlier forecast origin and compares predictions with subsequent actual observations. The all-windows option calculates MAE/RMSE and a persistence comparison for that uploaded sensor.

## Data download and provenance

The archive is `PEMS04.npz` from the ASTGNN authors' public repository. It contains the PeMSD4 array under the key `data`; expected shape `(16992, 307, 3)`. The code validates the exact archive SHA-256 before loading.

- ASTGNN repository: https://github.com/guoshnBJTU/ASTGNN
- PEMS04 dataset path: https://raw.githubusercontent.com/guoshnBJTU/ASTGNN/main/data/PEMS04/PEMS04.npz
- ASTGCN dataset description: https://github.com/wanhuaiyu/ASTGCN
- Caltrans PeMS overview: https://dot.ca.gov/programs/traffic-operations/mpr/pems-source

## Important limitations

- Four clients are simulated sequentially on one computer; they are not four real traffic agencies.
- The current GRU is temporal and sensor-level. It does not explicitly model spatial links between neighboring sensors.
- The model learns three measured channels: flow, occupancy and speed. Volume is another presentation of five-minute flow; hourly flow is derived by multiplying by 12.
- Congestion is a transparent rule derived from predicted speed, not an independently trained label.
- No differential privacy or secure aggregation is implemented.
- The upload app expects a long-format CSV with columns: `timestamp`, `sensor_id`, `flow_veh_5min`, `occupancy_pct`, `speed_mph`.
