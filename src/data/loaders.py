"""
Thin, typed loaders around the raw CSVs so notebooks and the dashboard don't
each re-implement parsing/encoding choices. Add to this file as later days
need new datasets -- don't duplicate read_csv calls across notebooks.
"""
import pandas as pd
from pathlib import Path
from src.utils.config_loader import load_config, raw_path


def load_scada(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    df = pd.read_csv(raw_path("scada", cfg), parse_dates=["timestamp"])
    return df.sort_values(["turbine_id", "timestamp"]).reset_index(drop=True)


def load_power_curve(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    return pd.read_csv(raw_path("power_curve", cfg))


def load_alarms(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    return pd.read_csv(raw_path("alarms", cfg),
                        parse_dates=["event_timestamp", "event_end_timestamp"])


def load_maintenance(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    return pd.read_csv(raw_path("maintenance", cfg), parse_dates=["failure_date"])


def load_grid(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    return pd.read_csv(raw_path("grid_curtailment", cfg), parse_dates=["timestamp"])


def load_interim(filename: str, cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    path = Path(cfg["paths"]["interim_dir"]) / filename
    parse_col = "timestamp" if "timestamp" in pd.read_csv(path, nrows=1).columns else None
    return pd.read_csv(path, parse_dates=[parse_col] if parse_col else None)
