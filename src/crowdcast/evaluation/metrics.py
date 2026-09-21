"""Regression metrics for a 0-100 target."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, park_id: np.ndarray | None = None) -> dict[str, float]:
    """MAE, RMSE, R2 and Spearman rho (predictions are clipped to 0-100 first); optionally the median per-park MAE."""
    y_true = np.asarray(y_true, dtype=float)
    pred = np.clip(np.asarray(y_pred, dtype=float), 0, 100)
    err = pred - y_true
    out = {
        "MAE": float(np.abs(err).mean()),
        "RMSE": float(np.sqrt((err**2).mean())),
        "R2": float(1 - (err**2).sum() / ((y_true - y_true.mean()) ** 2).sum()),
        "rho": float(spearmanr(y_true, pred)[0]),
    }
    if park_id is not None:
        out["parkMAE_med"] = float(pd.Series(np.abs(err)).groupby(np.asarray(park_id)).mean().median())
    return out
