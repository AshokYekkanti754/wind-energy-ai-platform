"""
The Day 3/4 forecasting models predict daily totals. A battery schedule
needs hourly resolution. This module disaggregates a daily forecast into
an hourly profile using a REAL historical diurnal shape (same-month hourly
average from SCADA), scaled to match the forecast total -- not an
arbitrary invented curve.

Demand's hourly shape is necessarily synthetic (the daily demand figure is
already a proxy, see src/features/demand_proxy.py) -- built from a fixed,
documented cooling-load-driven curve (peak in the afternoon, tropical
climate) rather than a flat or randomly shaped profile.
"""
import numpy as np
import pandas as pd


def build_generation_hourly_profile(scada: pd.DataFrame, target_month: int,
                                     daily_forecast_kwh: float,
                                     timestamp_col: str = "timestamp",
                                     power_col: str = "active_power_kw") -> pd.Series:
    """
    Returns a 24-length Series (index 0-23, hour of day) of forecast kWh per
    hour, shaped like the real historical same-month average generation
    profile and scaled so the 24 values sum to `daily_forecast_kwh`.
    """
    df = scada.copy()
    df["hour"] = df[timestamp_col].dt.hour
    df["month"] = df[timestamp_col].dt.month

    month_data = df[df["month"] == target_month]
    if month_data.empty:
        month_data = df  # fallback to full-year average if the month has no data

    hourly_shape = month_data.groupby("hour")[power_col].mean()
    hourly_shape = hourly_shape.reindex(range(24), fill_value=hourly_shape.mean())

    normalized = hourly_shape / hourly_shape.sum()
    return (normalized * daily_forecast_kwh).rename("generation_kwh")


# Fixed, documented synthetic demand shape: cooling-load driven (tropical
# climate, matches the temperature-driven daily proxy in demand_proxy.py).
# Values are relative weights, not kWh -- normalized before use.
_DEMAND_SHAPE_WEIGHTS = np.array([
    0.55, 0.50, 0.48, 0.46, 0.46, 0.50,   # 00-05: overnight base load
    0.60, 0.70, 0.80, 0.90, 1.00, 1.10,   # 06-11: morning ramp-up
    1.25, 1.35, 1.40, 1.35, 1.25, 1.15,   # 12-17: afternoon cooling peak
    1.10, 1.05, 0.95, 0.80, 0.70, 0.60,   # 18-23: evening decline
])


def build_demand_hourly_profile(daily_forecast_kwh: float) -> pd.Series:
    """
    Returns a 24-length Series of SYNTHETIC forecast demand kWh per hour,
    shaped by a fixed cooling-load-driven curve, scaled so the 24 values
    sum to `daily_forecast_kwh`. This compounds an already-synthetic daily
    figure with a synthetic hourly shape -- treat the resulting curve as
    illustrative only, not a real load profile.
    """
    normalized = _DEMAND_SHAPE_WEIGHTS / _DEMAND_SHAPE_WEIGHTS.sum()
    return pd.Series(normalized * daily_forecast_kwh, index=range(24), name="demand_kwh")
