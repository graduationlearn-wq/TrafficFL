# FedRoad Experiment Results

## Test-set comparison

MAE and RMSE are in the dataset's original speed units. Lower values indicate better forecasts.

| Horizon | Model | MAE | RMSE | Valid targets | MAE improvement vs persistence | MAE difference vs centralized |
|---:|---|---:|---:|---:|---:|---:|
| 15 min | Persistence | 3.492 | 6.400 | 1,242,568 | 0.00% | +12.76% |
| 15 min | Centralized shared GRU | 3.097 | 6.110 | 1,242,568 | 11.32% | +0.00% |
| 15 min | FedAvg GRU | 3.147 | 6.160 | 1,242,568 | 9.87% | +1.63% |
| 30 min | Persistence | 4.191 | 8.022 | 1,242,550 | 0.00% | +11.41% |
| 30 min | Centralized shared GRU | 3.762 | 7.585 | 1,242,550 | 10.24% | +0.00% |
| 30 min | FedAvg GRU | 3.820 | 7.630 | 1,242,550 | 8.86% | +1.53% |
| 60 min | Persistence | 5.347 | 10.294 | 1,242,514 | 0.00% | +12.32% |
| 60 min | Centralized shared GRU | 4.760 | 9.457 | 1,242,514 | 10.96% | +0.00% |
| 60 min | FedAvg GRU | 4.853 | 9.434 | 1,242,514 | 9.23% | +1.95% |

## Interpretation and limitations

- Persistence is the simple forecasting baseline.
- The centralized GRU and FedAvg GRU use the same sensor-level model family and held-out test horizons.
- The centralized run used validation-based early stopping; the federated run used 10 server rounds with one local epoch per round. Training budgets therefore differ.
- The federated clients are simulated in one runtime. This experiment does not establish a deployment-level privacy guarantee.
- Use validation data for model selection. Preserve the test results as a final evaluation rather than repeatedly tuning against them.

## Source files

- `centralized_metrics.csv`
- `centralized_metadata.json`
- `federated_metrics.csv`
- `fedavg_metadata.json`
