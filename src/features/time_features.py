"""
Feature engineering functions shared by the feature-engineering notebook,
the forecasting model, and (later) the dashboard/copilot. Keeping these here
instead of duplicating logic in each notebook means Day 3's features and
Day 8's dashboard can never silently drift apart.
"""
import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """
    Adds calendar + cyclical time features. Cyclical (sin/cos) encodings avoid
    the false discontinuity a raw hour/day-of-year value creates at midnight
    or year-end (hour 23 and hour 0 are adjacent, but 23 != 0 numerically).
    """
    df = df.copy()
    ts = df[timestamp_col]

    df["hour"] = ts.dt.hour
    df["day_of_year"] = ts.dt.dayofyear
    df["month"] = ts.dt.month
    df["day_of_week"] = ts.dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

    return df


def add_rolling_features(df: pd.DataFrame, col: str, windows: dict[str, int],
                          min_periods: int = 1) -> pd.DataFrame:
    """
    Adds rolling-mean columns for `col` over each named window.
    windows example for 10-min data: {"1h": 6, "24h": 144, "7d": 1008}
    Assumes df is already sorted chronologically.
    """
    df = df.copy()
    for label, periods in windows.items():
        df[f"{col}_rolling_mean_{label}"] = df[col].rolling(periods, min_periods=min_periods).mean()
    return df


def build_daily_generation_table(scada: pd.DataFrame,
                                  timestamp_col: str = "timestamp",
                                  energy_col: str = "energy_generated_kwh") -> pd.DataFrame:
    """
    Aggregates 10-minute SCADA to one row per day: total generation plus
    daily-mean weather/operational context. This is the base table for
    next-day generation forecasting (Day 3 goal, per the work plan).
    """
    daily = (
        scada.set_index(timestamp_col)
        .resample("D")
        .agg(
            generation_kwh=(energy_col, "sum"),
            mean_wind_speed=("wind_speed_mps", "mean"),
            max_wind_speed=("wind_speed_mps", "max"),
            mean_wind_direction=("wind_direction_deg", "mean"),
            mean_ambient_temp=("ambient_temperature_c", "mean"),
            mean_air_density=("air_density_kg_m3", "mean"),
            availability_pct=("availability_status", lambda s: (s == "Available").mean()),
        )
        .reset_index()
    )
    return daily


def add_daily_lag_and_target(daily: pd.DataFrame, target_horizon_days: int = 1) -> pd.DataFrame:
    """
    Adds lag/rolling features of past generation (leakage-safe: every lag and
    rolling feature is shifted so it only uses information available *before*
    the day being predicted) and the forecast target itself.
    """
    df = daily.copy()

    df["day_of_year"] = df["timestamp"].dt.dayofyear
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

    df["gen_lag_1d"] = df["generation_kwh"].shift(1)
    df["gen_lag_2d"] = df["generation_kwh"].shift(2)
    df["gen_lag_7d"] = df["generation_kwh"].shift(7)
    df["gen_rolling_mean_7d"] = df["generation_kwh"].shift(1).rolling(7).mean()
    df["gen_rolling_mean_30d"] = df["generation_kwh"].shift(1).rolling(30).mean()
    df["wind_rolling_mean_7d"] = df["mean_wind_speed"].shift(1).rolling(7).mean()

    df[f"target_next_{target_horizon_days}day_generation_kwh"] = df["generation_kwh"].shift(-target_horizon_days)

    return df
