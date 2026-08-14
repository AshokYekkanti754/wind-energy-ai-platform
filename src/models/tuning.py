"""
Hyperparameter search for time-series models must not use random k-fold CV
-- that would let the model train on future days and validate on past ones.
This module does a simple grid search that trains only on `train`, scores
only on `val` (both already chronologically split), and returns the best
params plus the full results table for transparency.
"""
import itertools
import numpy as np
import pandas as pd


def chronological_grid_search(model_factory, param_grid: dict,
                                X_train, y_train, X_val, y_val) -> tuple[dict, pd.DataFrame]:
    """
    model_factory: a callable that takes **params and returns an unfitted model
                   (e.g. lambda **p: xgb.XGBRegressor(**p, random_state=42))
    param_grid: dict of param_name -> list of values to try, e.g.
                {"max_depth": [3, 5], "learning_rate": [0.03, 0.1]}

    Returns (best_params, results_df) where results_df has one row per
    combination tried, sorted by validation MAE ascending.
    """
    keys = list(param_grid.keys())
    combinations = list(itertools.product(*param_grid.values()))

    rows = []
    param_dicts = []  # kept separately, with original types, since a param_grid
                        # value of None (e.g. max_depth=None) forces pandas to
                        # upcast an otherwise-int column to float64 -- reading
                        # "best params" back out of the DataFrame would silently
                        # hand ints back as np.float64 and break strict sklearn
                        # parameter validation.
    for combo in combinations:
        params = dict(zip(keys, combo))
        param_dicts.append(params)
        model = model_factory(**params)
        model.fit(X_train, y_train)
        pred_val = model.predict(X_val)
        mae = np.mean(np.abs(np.asarray(y_val) - pred_val))
        rmse = np.sqrt(np.mean((np.asarray(y_val) - pred_val) ** 2))
        rows.append({**{k: str(v) for k, v in params.items()}, "val_MAE": mae, "val_RMSE": rmse})

    results_df = pd.DataFrame(rows).sort_values("val_MAE").reset_index(drop=True)
    best_idx = int(np.argmin([r["val_MAE"] for r in rows]))
    best_params = param_dicts[best_idx]
    return best_params, results_df
