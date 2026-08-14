"""
No real demand/load dataset exists among the 12 provided files (confirmed
Day 1). Per the work plan's own instruction ("add a demand forecasting
target using historical load data, or a reasonable proxy dataset if real
data isn't available"), this module generates a **documented synthetic
proxy** -- it is not real grid demand and must be labeled as such wherever
it's used (dashboard, copilot, proposal appendix).

Proxy design rationale:
- Ambient temperature at this site ranges ~23-31 C (confirmed from SCADA) --
  a tropical/sub-tropical range where demand is driven by COOLING load
  (air conditioning), not heating. So demand rises with temperature above
  a comfort threshold, it does not have the U-shape you'd model in a
  temperate climate.
- Weekdays carry higher industrial/commercial load than weekends.
- A base load floor represents residential/always-on consumption.
- Gaussian noise represents everything else this simple model doesn't
  capture (unmodeled demand drivers) -- kept deliberately modest so the
  proxy remains learnable, since the point is to demonstrate a working
  demand-forecasting pipeline, not to claim real grid accuracy.
"""
import numpy as np
import pandas as pd


def generate_demand_proxy(
    daily: pd.DataFrame,
    temp_col: str = "mean_ambient_temp",
    day_of_week_col: str = "day_of_week",
    base_load_kwh: float = 18000.0,
    comfort_temp_c: float = 24.0,
    cooling_coefficient: float = 600.0,
    cooling_exponent: float = 1.3,
    weekday_multiplier: float = 1.08,
    weekend_multiplier: float = 0.92,
    noise_std_kwh: float = 800.0,
    random_seed: int = 42,
) -> pd.Series:
    """
    Returns a synthetic daily demand series (kWh) aligned to `daily`'s index.
    Requires `daily` to already have a day_of_week column (0=Mon..6=Sun);
    if missing, derives it from `timestamp`.
    """
    df = daily.copy()
    if day_of_week_col not in df.columns:
        df[day_of_week_col] = df["timestamp"].dt.dayofweek

    cooling_load = cooling_coefficient * np.clip(df[temp_col] - comfort_temp_c, 0, None) ** cooling_exponent
    is_weekend = df[day_of_week_col] >= 5
    day_multiplier = np.where(is_weekend, weekend_multiplier, weekday_multiplier)

    rng = np.random.default_rng(random_seed)
    noise = rng.normal(0, noise_std_kwh, size=len(df))

    demand = (base_load_kwh + cooling_load) * day_multiplier + noise
    return pd.Series(demand, index=df.index, name="demand_proxy_kwh")
