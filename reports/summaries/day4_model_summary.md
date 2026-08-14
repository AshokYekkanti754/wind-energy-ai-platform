# Model Summary — Wind Generation & Demand Forecasting (Day 3–4)

## Approach
Single-turbine (WTG_001) daily generation forecasting, predicting next-day total generation (kWh)
from same-day weather/operational context plus lagged generation history. Chronological
train/val/test split (no shuffling) to avoid leakage. Baseline trained Day 3; hyperparameters
tuned Day 4 via chronological grid search (train on train, select on val, confirm on untouched test).

A second model forecasts next-day demand from a **documented synthetic proxy** (temperature-driven
cooling load + weekday multiplier) — no real load/consumption dataset was available among the
provided files. This must be labeled as a proxy wherever shown in the demo.

## Features used
Weather: mean/max wind speed, wind direction, ambient temperature, air density.
Time: day-of-year (cyclical), month, day-of-week, weekend flag.
History: 1/2/7-day generation lags, 7/30-day rolling means, 7-day wind rolling mean.
(18 features total for the generation model.)

## Accuracy achieved (test set, 2025-11-10 to 2025-12-30)

| Model | MAE (kWh) | RMSE (kWh) | MAPE |
|---|---|---|---|
| Day 3 baseline | 1,363 | 1,658 | 9.0% |
| Day 4 tuned    | 992 | 1,368 | 6.5% |
| Demand proxy (Day 4) | 842 | 1,074 | 4.2% |

Best hyperparameters (selected on validation set): {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.03}

## Error analysis — known limitation
The model systematically underpredicted the Nov–Dec test period in the Day 3 baseline (mean
residual ≈ +1,140 kWh/day). Root cause: site generation bottoms out around Sep–Oct then *rises*
through Nov–Dec as winter wind increases, but training data (through Sep 20) never included a
prior winter recovery period to learn from. Tuning improved this (mean residual now
720 kWh) but did not eliminate it — **this is a data-coverage
limitation, not a modeling one.** More historical years (as the technical report recommends: 3-5
years minimum for SCADA) would resolve it directly.

## What is prototype-level vs. needing further development
- Single turbine only — a farm-wide model needs SCADA for all turbines, which this dataset doesn't include.
- Wind-only — no solar generation data was provided.
- Demand is a labeled synthetic proxy, not real grid load.
- One year of training history — insufficient to learn year-over-year seasonal recovery patterns.
