"""
Assembles the data context the GenAI Energy Copilot needs to answer operator
questions. Per the work plan: "feed the LLM relevant data" rather than
letting it guess. Three of the four pieces below come from real data;
battery status is fully mocked since no battery/BESS dataset exists.

REAL:    generation forecast + actuals (from Day 3/4 models + SCADA)
REAL:    turbine health ranking (from Dataset-7 alarms + Dataset-8 maintenance,
         covering all 50 turbines -- SCADA telemetry only exists for WTG_001,
         but alarm/maintenance history is genuine for the whole fleet)
MOCKED:  battery/BESS status (no such dataset was provided anywhere)
"""
import numpy as np
import pandas as pd
from pathlib import Path

from src.utils.config_loader import load_config, raw_path, interim_path, processed_path
from src.data.loaders import load_alarms, load_maintenance
from src.models.forecasting_model_utils import load_model


def get_generation_context(cfg: dict | None = None, models_dir: str = "models") -> dict:
    """
    Real data: latest actual generation/wind/power-loss for WTG_001, plus a
    next-day forecast from the Day 4 tuned model and the demand proxy model.
    "Today" in this demo is anchored to the last date in the SCADA file
    (2025-12-31), since there's no live feed -- state that plainly if asked.
    """
    cfg = cfg or load_config()
    project_root = Path(cfg["paths"]["raw_dir"]).parent.parent if False else Path.cwd()

    scada = pd.read_csv(interim_path("scada_wtg001_cleaned_2025.csv", cfg), parse_dates=["timestamp"])
    daily_features = pd.read_csv(processed_path("daily_generation_features.csv", cfg), parse_dates=["timestamp"])

    anchor_date = scada["timestamp"].max().normalize()
    yesterday = anchor_date - pd.Timedelta(days=1)

    day_rows = scada[scada["timestamp"].dt.normalize() == anchor_date]
    actual_generation_kwh = day_rows["energy_generated_kwh"].sum()
    actual_power_loss_kwh = day_rows["power_loss_kw"].sum() / 6  # kW at 10-min steps -> kWh
    alarm_minutes_today = int(day_rows["active_alarm"].sum()) * 10
    mean_wind_today = day_rows["wind_speed_mps"].mean()

    # seasonal comparison: same week-of-year average from the full dataset, excluding today
    week_of_year = anchor_date.isocalendar().week
    seasonal_baseline = scada[scada["timestamp"].dt.isocalendar().week == week_of_year]
    seasonal_avg_wind = seasonal_baseline["wind_speed_mps"].mean()

    # trailing 7-day average generation (excluding today) -- lets the copilot state
    # plainly whether generation actually rose or fell, instead of just reciting numbers
    trailing_window = daily_features[
        (daily_features["timestamp"] < anchor_date) & (daily_features["timestamp"] >= anchor_date - pd.Timedelta(days=7))
    ]
    trailing_7d_avg_generation_kwh = trailing_window["generation_kwh"].mean() if len(trailing_window) else None
    generation_vs_trailing_avg_pct = (
        round(float((actual_generation_kwh - trailing_7d_avg_generation_kwh) / trailing_7d_avg_generation_kwh * 100), 1)
        if trailing_7d_avg_generation_kwh else None
    )

    gen_model_path = Path(models_dir) / "next_day_generation_model_tuned.joblib"
    gen_model, gen_meta = load_model(gen_model_path)
    latest_feature_row = daily_features.iloc[[-1]][gen_meta["feature_cols"]]
    next_day_generation_forecast_kwh = float(gen_model.predict(latest_feature_row)[0])

    demand_model_path = Path(models_dir) / "next_day_demand_proxy_model.joblib"
    next_day_demand_forecast_kwh = None
    if demand_model_path.exists():
        demand_model, demand_meta = load_model(demand_model_path)
        available_cols = [c for c in demand_meta["feature_cols"] if c in daily_features.columns]
        if len(available_cols) == len(demand_meta["feature_cols"]):
            next_day_demand_forecast_kwh = float(
                demand_model.predict(daily_features.iloc[[-1]][demand_meta["feature_cols"]])[0]
            )

    return {
        "anchor_date": str(anchor_date.date()),
        "turbine_id": "WTG_001",
        "actual_generation_today_kwh": round(float(actual_generation_kwh), 1),
        "actual_power_loss_today_kwh": round(float(actual_power_loss_kwh), 1),
        "alarm_minutes_today": alarm_minutes_today,
        "mean_wind_speed_today_mps": round(float(mean_wind_today), 2),
        "seasonal_avg_wind_speed_mps": round(float(seasonal_avg_wind), 2),
        "wind_vs_seasonal_pct": round(float((mean_wind_today - seasonal_avg_wind) / seasonal_avg_wind * 100), 1),
        "trailing_7d_avg_generation_kwh": round(float(trailing_7d_avg_generation_kwh), 1) if trailing_7d_avg_generation_kwh else None,
        "generation_vs_trailing_avg_pct": generation_vs_trailing_avg_pct,
        "next_day_generation_forecast_kwh": round(next_day_generation_forecast_kwh, 1),
        "next_day_demand_forecast_kwh": round(next_day_demand_forecast_kwh, 1) if next_day_demand_forecast_kwh else None,
        "demand_is_synthetic_proxy": True,
        "model_test_mae_kwh": gen_meta["test_metrics"]["MAE"],
    }


def get_turbine_health_ranking(cfg: dict | None = None, anchor_date: str | None = None,
                                 window_days: int = 90, top_n: int = 5) -> pd.DataFrame:
    """
    REAL data: ranks all 50 turbines by a simple, transparent risk score built
    from recent maintenance and alarm history (Report Model 4 territory --
    this is a rule-based stand-in, not a trained failure-prediction model,
    and should be described to the operator as such).
    """
    cfg = cfg or load_config()
    maintenance = load_maintenance(cfg)
    alarms = load_alarms(cfg)

    anchor = pd.Timestamp(anchor_date) if anchor_date else maintenance["failure_date"].max()
    window_start = anchor - pd.Timedelta(days=window_days)

    recent_maint = maintenance[(maintenance["failure_date"] >= window_start) & (maintenance["failure_date"] <= anchor)].copy()
    recent_alarms = alarms[(alarms["event_timestamp"] >= window_start) & (alarms["event_timestamp"] <= anchor)]

    severity_weight = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
    recent_maint["sev_score"] = recent_maint["failure_severity"].map(severity_weight)

    risk = recent_maint.groupby("turbine_id").agg(
        n_failures=("work_order_id", "count"),
        total_downtime_hours=("downtime_hours", "sum"),
        max_severity_score=("sev_score", "max"),
        last_failure_date=("failure_date", "max"),
        most_common_component=("component", lambda s: s.mode().iloc[0] if not s.mode().empty else None),
    ).reset_index()

    alarm_counts = recent_alarms.groupby("turbine_id").size().rename(f"n_alarms_{window_days}d")
    risk = risk.merge(alarm_counts, on="turbine_id", how="left").fillna({f"n_alarms_{window_days}d": 0})

    risk["risk_score"] = (
        risk["n_failures"] * 2
        + risk["total_downtime_hours"] * 0.1
        + risk["max_severity_score"] * 5
        + risk[f"n_alarms_{window_days}d"] * 0.05
    )
    return risk.sort_values("risk_score", ascending=False).head(top_n).reset_index(drop=True)


def get_mock_battery_status(n_batteries: int = 4, random_seed: int = 42) -> pd.DataFrame:
    """
    MOCKED -- no battery/BESS dataset exists among the provided files. This
    is placeholder data for demonstrating the copilot's dispatch-question
    handling only. Must be labeled "mocked" wherever displayed or spoken
    about in the demo, per the work plan's own instruction for this item.
    """
    rng = np.random.default_rng(random_seed)
    battery_ids = [f"BESS_{i+1:02d}" for i in range(n_batteries)]
    soc_pct = rng.uniform(40, 95, n_batteries).round(1)
    health_pct = rng.uniform(85, 99, n_batteries).round(1)
    cycles = rng.integers(200, 1800, n_batteries)

    df = pd.DataFrame({
        "battery_id": battery_ids,
        "state_of_charge_pct": soc_pct,
        "health_pct": health_pct,
        "cycle_count": cycles,
    })
    # simple dispatch heuristic: discharge highest (SOC x health) first --
    # maximizes usable energy while favoring healthier packs
    df["dispatch_priority_score"] = (df["state_of_charge_pct"] * df["health_pct"] / 100).round(1)
    return df.sort_values("dispatch_priority_score", ascending=False).reset_index(drop=True)


def assemble_full_context(cfg: dict | None = None) -> dict:
    """One call that gathers everything the copilot's system prompt needs."""
    cfg = cfg or load_config()
    generation = get_generation_context(cfg)
    turbine_health = get_turbine_health_ranking(cfg, anchor_date=generation["anchor_date"])
    battery = get_mock_battery_status()
    return {
        "generation": generation,
        "turbine_health_top5": turbine_health.to_dict(orient="records"),
        "battery_status_MOCKED": battery.to_dict(orient="records"),
    }
