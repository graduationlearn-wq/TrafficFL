# FedRoad Experiment Results

## Main held-out test comparison

MAE and RMSE use the dataset's original speed units. Lower values indicate better predictions. Unavailable target readings are excluded.

| Horizon | Model | MAE | RMSE | Valid targets |
|---:|---|---:|---:|---:|
| 15 min | Persistence | 3.492 | 6.400 | 1,242,568 |
| 15 min | Centralized shared GRU | 3.097 | 6.110 | 1,242,568 |
| 15 min | FedAvg GRU | 3.147 | 6.160 | 1,242,568 |
| 30 min | Persistence | 4.191 | 8.022 | 1,242,550 |
| 30 min | Centralized shared GRU | 3.762 | 7.585 | 1,242,550 |
| 30 min | FedAvg GRU | 3.820 | 7.630 | 1,242,550 |
| 60 min | Persistence | 5.347 | 10.294 | 1,242,514 |
| 60 min | Centralized shared GRU | 4.760 | 9.457 | 1,242,514 |
| 60 min | FedAvg GRU | 4.853 | 9.434 | 1,242,514 |

## Recorded FedAvg validation curve

The following values were transcribed from the successful 10-round Flower console output. Validation MAE is shown in normalized units; original-unit validation MAE is included for context.

| Round | Train MAE (normalized) | Validation MAE (normalized) | Validation MAE (original units) |
|---:|---:|---:|---:|
| 1 | 0.44308 | 0.38970 | 3.63476 |
| 2 | 0.39911 | 0.38559 | 3.58586 |
| 3 | 0.39794 | 0.38446 | 3.57254 |
| 4 | 0.39712 | 0.38351 | 3.56040 |
| 5 | 0.39619 | 0.38285 | 3.55534 |
| 6 | 0.39517 | 0.38183 | 3.54532 |
| 7 | 0.39403 | 0.38095 | 3.53659 |
| 8 | 0.39301 | 0.38000 | 3.52731 |
| 9 | 0.39219 | 0.37959 | 3.52408 |
| 10 | 0.39153 | 0.37914 | 3.51932 |

## Interpretation

- FedAvg improves over persistence on MAE at all three tested horizons (about 9.87%, 8.86%, and 9.23%).
- FedAvg MAE is about 1.63%, 1.53%, and 1.95% higher than the centralized GRU at the corresponding horizons.
- The centralized and federated training budgets differ. These results are useful as a first benchmark, not a controlled claim of superiority.
- The directed graph GRU is a separate experimental model and is not represented in the table because only smoke-test runs have been completed.
- The four clients were simulated in one runtime. No secure aggregation or differential privacy was implemented.
