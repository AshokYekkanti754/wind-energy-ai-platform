"""
Data validation rules taken directly from the technical report, Section 17
(Data Quality and Governance). Each function returns a boolean mask of
INVALID rows (True = problem) so callers can decide whether to drop, flag,
or investigate.
"""
import pandas as pd


def invalid_wind_speed(df: pd.DataFrame, col: str = "wind_speed_mps") -> pd.Series:
    """Wind speed must be >= 0."""
    return df[col] < 0


def invalid_wind_direction(df: pd.DataFrame, col: str = "wind_direction_deg") -> pd.Series:
    """Wind direction must be within [0, 360]."""
    return ~df[col].between(0, 360)


def invalid_humidity(df: pd.DataFrame, col: str = "humidity_pct") -> pd.Series:
    """Humidity must be within [0, 100]."""
    return ~df[col].between(0, 100)


def power_exceeds_rated_capacity(df: pd.DataFrame, rated_power_kw: float,
                                  power_col: str = "active_power_kw") -> pd.Series:
    """Active power should never exceed the turbine's rated capacity."""
    return df[power_col] > rated_power_kw


def rotor_stationary_but_generating(df: pd.DataFrame, rotor_col: str = "rotor_speed_rpm",
                                     power_col: str = "active_power_kw") -> pd.Series:
    """Rotor speed = 0 while power > 0 is a sensor inconsistency."""
    return (df[rotor_col] == 0) & (df[power_col] > 0)


def duplicate_timestamp_rows(df: pd.DataFrame, keys: list[str]) -> pd.Series:
    """Flags duplicate (timestamp, turbine_id) rows -- keep='first' marks later dupes True."""
    return df.duplicated(subset=keys, keep="first")


def run_all_checks(df: pd.DataFrame, rated_power_kw: float) -> pd.DataFrame:
    """
    Returns a summary DataFrame: one row per check, with the count of rows
    that fail it. Does not modify the input.
    """
    checks = {}
    if "wind_speed_mps" in df.columns:
        checks["negative_wind_speed"] = invalid_wind_speed(df).sum()
    if "wind_direction_deg" in df.columns:
        checks["wind_direction_out_of_range"] = invalid_wind_direction(df).sum()
    if "active_power_kw" in df.columns:
        checks["power_exceeds_rated_capacity"] = power_exceeds_rated_capacity(df, rated_power_kw).sum()
    if {"rotor_speed_rpm", "active_power_kw"}.issubset(df.columns):
        checks["rotor_stationary_but_generating"] = rotor_stationary_but_generating(df).sum()
    if {"timestamp", "turbine_id"}.issubset(df.columns):
        checks["duplicate_timestamp_rows"] = duplicate_timestamp_rows(df, ["timestamp", "turbine_id"]).sum()
    checks["missing_values_total"] = int(df.isna().sum().sum())

    return pd.DataFrame({"check": list(checks.keys()), "failing_rows": list(checks.values())})
