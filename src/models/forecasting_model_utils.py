"""
Model-agnostic helpers for the forecasting work: chronological (not random)
train/val/test splitting, the three metrics the work plan specifies
(MAE, RMSE, MAPE), and joblib save/load so the dashboard and copilot can
reuse whatever gets trained in the notebooks.
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path


def chronological_split(df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15):
    """
    Splits a time-ordered DataFrame into train/val/test by position, not
    randomly -- required to avoid leaking future information into training,
    per the work plan's Day 3 instruction.
    """
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return (
        df.iloc[:train_end].reset_index(drop=True),
        df.iloc[train_end:val_end].reset_index(drop=True),
        df.iloc[val_end:].reset_index(drop=True),
    )


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """MAE, RMSE, and MAPE -- the three metrics named in the Day 3 work plan."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def save_model(model, path: Path, metadata: dict | None = None):
    """Saves a model (and optional metadata dict, e.g. feature list + metrics) with joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metadata": metadata or {}}, path)


def load_model(path: Path):
    """Returns (model, metadata) as saved by save_model."""
    bundle = joblib.load(path)
    return bundle["model"], bundle["metadata"]
