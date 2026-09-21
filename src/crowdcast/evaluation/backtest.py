"""Back-test the gated model over a held-out period.

Refits on all labels up to ``cutoff`` and scores the following window, reporting accuracy by horizon (how many days of recent
labels are assumed known) and by gate group (parks served by the park model vs the fallback).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from crowdcast.evaluation.metrics import regression_metrics
from crowdcast.features.columns import DRIFT_LAGS
from crowdcast.models.config import ModelConfig
from crowdcast.models.gated import GatedCrowdModel


@dataclass
class BacktestResult:
    model: GatedCrowdModel
    test: pd.DataFrame  # the scored rows (feature frame)
    by_horizon: pd.DataFrame  # gated parks, one row per horizon
    by_group: pd.DataFrame  # gate groups incl. reference predictors
    predictions: pd.DataFrame  # long-horizon predictions + actuals, one row per park-day


def _row(name: str, y: np.ndarray, pred: np.ndarray) -> dict[str, object]:
    if len(y) == 0:
        return {"setup": name, "rows": 0, "MAE": float("nan"), "R2": float("nan")}
    m = regression_metrics(y, pred)
    return {"setup": name, "rows": len(y), "MAE": round(m["MAE"], 2), "R2": round(m["R2"], 3)}


def run_backtest(frame: pd.DataFrame, cutoff: str, test_end: str, config: ModelConfig | None = None) -> BacktestResult:
    """Fit to ``cutoff``, score ``(cutoff, test_end]``."""
    model = GatedCrowdModel(config).fit(frame, cutoff)
    test = frame[(frame["date"] > cutoff) & (frame["date"] <= test_end)]
    y = test["crowd_percent"].to_numpy()
    long_horizon = model.predict(test)
    gated = ~long_horizon["low_confidence"].to_numpy()

    horizons = [("long horizon (no recent labels)", None)] + [(f"labels through t-{h}", h) for h in DRIFT_LAGS]
    rows = []
    for name, lag in horizons:
        pred = model.predict(test, lag=lag)["prediction"].to_numpy()
        rows.append(_row(name, y[gated], pred[gated]))
    by_horizon = pd.DataFrame(rows)

    p = long_horizon["prediction"].to_numpy()
    all_full = np.clip(model.predict_with("full", test), 0, 100)
    all_fallback = np.clip(model.predict_with("fallback", test), 0, 100)
    groups = [
        _row("all parks: gated system", y, p),
        _row("all parks: full model, no gate", y, all_full),
        _row("gated-in parks: park model (used)", y[gated], p[gated]),
        _row("gated-in parks: fallback would give", y[gated], all_fallback[gated]),
        _row("short-history parks: fallback (used)", y[~gated], p[~gated]),
        _row("short-history parks: full model would give", y[~gated], all_full[~gated]),
        _row("short-history parks: guess 50", y[~gated], np.full((~gated).sum(), 50.0)),
    ]
    predictions = long_horizon.assign(crowd_percent=y)
    return BacktestResult(model, test, by_horizon, pd.DataFrame(groups), predictions)
