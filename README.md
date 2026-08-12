# Wind Farm AI Energy Intelligence Platform — Prototype

Proof-of-concept build for the CubeAISolutions proposal pitch. Built over a 10-day
intern sprint (Days 1–10). Not production scale — a convincing, working demo of
every module described in the technical report.

## 1. Project Architecture

```
Raw Datasets (12 files)
      │
      ▼
Data Layer (ingest, validate, clean)
      │
      ▼
Feature Layer (weather + SCADA features, power curve lookups)
      │
      ├──► Forecasting Models (wind speed / power / demand) ─┐
      ├──► Anomaly Detection (expected vs actual power)       │
      ├──► Predictive Maintenance (failure risk)               ├─► Multi-Agent Layer
      ├──► Layout / Wake Optimization                          │   (Forecast, Optimization,
      └──► Grid & Curtailment Loss Attribution ─────────────────►  Maintenance, Grid, Carbon Agents)
                                                                │
                                                                ▼
                                                    GenAI Energy Copilot (LLM + context pipeline)
                                                                │
                                                                ▼
                                                    Streamlit Dashboard (single demo surface)
```

This mirrors Section 15 ("Digital Twin Architecture") and Section 18 ("Suggested AI
Platform Modules") of the technical report, scaled down to what 12 CSV files and
10 days can realistically support.

## 2. Folder Structure

```
wind-energy-ai-platform/
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example              # ANTHROPIC_API_KEY etc. (never commit .env)
├── config/
│   └── config.yaml           # paths, model params, turbine assumptions
├── data/
│   ├── raw/                  # original CSVs, untouched, read-only
│   ├── interim/              # cleaned/validated intermediate data
│   ├── processed/            # model-ready feature tables
│   └── external/             # anything pulled later (e.g. live weather API)
├── notebooks/                # exploratory + narrative work, numbered by day
│   ├── 00_environment_check.ipynb
│   ├── 01_data_inventory_and_exploration.ipynb
│   ├── 02_data_cleaning_and_validation.ipynb        (Day 2)
│   ├── 03_feature_engineering.ipynb                 (Day 3)
│   ├── 04_forecasting_baseline_model.ipynb          (Day 3)
│   ├── 05_forecasting_refinement_demand.ipynb       (Day 4)
│   ├── 06_genai_copilot.ipynb                       (Day 5)
│   ├── 07_multi_agent_demo.ipynb                    (Day 6)
│   ├── 08_battery_grid_optimization.ipynb           (Day 7)
│   └── 09_integration_walkthrough.ipynb             (Day 9)
├── src/                       # reusable code, imported by notebooks + dashboard
│   ├── data/                  # loaders, validators, cleaners
│   ├── features/               # feature engineering functions
│   ├── models/                 # train/predict wrappers (xgboost, lightgbm)
│   ├── agents/                  # ForecastAgent, OptimizationAgent, GridAgent, etc.
│   ├── optimization/            # PuLP battery/grid optimization logic
│   └── utils/                   # config loader, logging, metrics
├── models/                    # saved model artifacts (.pkl/.joblib) — gitignored
├── dashboard/
│   ├── app.py                  # Streamlit entry point (Day 8)
│   └── pages/                  # dashboard panels
├── reports/
│   ├── figures/                # exported charts for the proposal deck
│   └── summaries/               # one-pagers, model summary (Day 4, Day 10)
├── tests/                      # sanity checks on data + models
└── logs/                       # agent decision logs (Day 6 explainability)
```

**Why `src/` + `notebooks/` split:** notebooks are for exploration and the demo
narrative; anything reused across notebooks or by the dashboard/copilot (loaders,
feature functions, agent classes) lives in `src/` and gets imported. This is what
lets Day 8's dashboard and Day 5's copilot reuse the Day 3 model without copy-paste.

## 3. Dataset Inventory (what's actually in `data/raw/`)

| # | File | Rows | Role |
|---|------|------|------|
| 1 | `Dataset-4 Long term wind climate dataset 20 years 10min.csv` | ~1.05M | Long-term wind climate (site_id, hub-height wind speed, shear, TI) |
| 2 | `Dataset-5 Turbine power curve dataset.csv` | ~305 | Wind speed → expected power lookup per turbine model |
| 3 | `Dataset-6 Wind farm scada 1year.csv` | 52,560 | 10-min SCADA — the core forecasting + anomaly dataset |
| 4 | `Dataset-7 Turbine alarm event dataset 3year 50turbines 150k events.csv` | 150,000 | Alarm/event log — root-cause + anomaly labels |
| 5 | `Dataset-8 Wind turbine maintenance dataset 3year 50turbines 25000 records.csv` | 25,000 | Failure/work-order history — predictive maintenance labels |
| 6 | `Dataset-9 Grid and Curtailment dataset 10000 rows.csv` | 10,000 | Curtailment, energy-not-delivered — Grid Agent input |
| 7 | `Dtaset-10 GIS Terrain Dataset.csv` | 100 | Static terrain/spatial features |
| 8 | `Dataset-11 (A) Actual Wind Farm Layout.csv` | 100 | Real turbine positions for the demo farm |
| 9 | `Dataset-11 (B) AI Candidate Layouts 10000.csv` | 10,000 | Candidate layouts with gross/net AEP, wake loss — layout optimizer training data |
| 10 | `Dataset-11 (C) Layout Position Features 10000.csv` | 10,000 | Layout spatial features |
| 11 | `Dataset-12 weather forecast 10000 rows.csv` | 10,000 | Forecast vs actual weather + power forecast error |
| 12 | `wind_energy_dataset_100- 10000 Row.csv` | 10,000 | Simple flat table (Date_Time, Turbine_ID, Wind Speed, Power) — good smoke-test dataset |

There is **no separate demand/load dataset** — Day 4's "demand forecasting" will
need a proxy (documented in the notebook, per the work plan's own suggestion).

## 4. How This Maps to the 10-Day Plan

- **Day 1 (this step):** environment setup + data inventory/exploration notebook.
- **Day 2:** cleaning/validation rules from Report §17, merged model-ready table.
- **Days 3–4:** `src/models/` + `src/features/` power forecasting model.
- **Day 5:** `src/agents/` copilot context pipeline using Anthropic API.
- **Day 6:** agent classes wired into a pipeline, logged to `logs/`.
- **Day 7:** `src/optimization/` PuLP battery scheduling + grid scenarios.
- **Day 8:** `dashboard/app.py` Streamlit app.
- **Day 9–10:** integration, docs, `reports/summaries/`.
