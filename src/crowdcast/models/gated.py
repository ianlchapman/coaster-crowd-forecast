"""The gated crowd model.

* Parks with at least ``min_years`` of labelled history (measured at the training cutoff) are served by the **full** model:
  LightGBM with the park id, prior-year level, weather and (optionally) recent drift.
* Everything else, including parks never seen in training, gets a **fallback** model without the park id or any
  history-based features, and every row it produces is flagged ``low_confidence``.

One set of trees handles every horizon. During training, drift columns and weather are blanked at random, so at prediction time
the caller states what is available: ``lag`` (labels known through ``t - lag``) and whether weather is present.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from crowdcast.features.calendar import in_covid
from crowdcast.features.columns import DRIFT_LAGS, FALLBACK, FULL, WEATHER
from crowdcast.models.config import ModelConfig

log = logging.getLogger(__name__)


def blank_drift(features: pd.DataFrame, lag: int | None) -> pd.DataFrame:
    """Labels known through ``t - lag``: keep drift columns with lag >= ``lag`` and blank the rest (``None``: blank all)."""
    out = features.copy()
    for h in DRIFT_LAGS:
        if lag is None or h < lag:
            out[f"drift{h}"] = np.nan
    return out


class GatedCrowdModel:
    """Fit on a labelled feature frame (see :func:`crowdcast.features.build.build_feature_frame`), predict any frame with the same columns."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        self.full: lgb.LGBMRegressor | None = None
        self.fallback: lgb.LGBMRegressor | None = None
        self.history_years: pd.Series = pd.Series(dtype=float)
        self.cutoff: pd.Timestamp | None = None

    # ------------------------------------------------------------------ training
    def fit(self, frame: pd.DataFrame, cutoff: str | pd.Timestamp) -> GatedCrowdModel:
        cfg = self.config
        cutoff = pd.Timestamp(cutoff)
        upto = frame[frame["date"] <= cutoff]
        train = upto[~in_covid(upto["date"])] if cfg.exclude_covid else upto
        self.cutoff = cutoff
        self.history_years = upto.groupby("park_id").size() / 365.25  # labelled days incl. the COVID window

        rng = np.random.default_rng(cfg.seed)
        x_full = train[FULL].copy()
        as_of = rng.choice([0, *DRIFT_LAGS], len(train))  # 0 = no recent labels; else labels known through t - lag
        for h in DRIFT_LAGS:
            x_full.loc[(as_of == 0) | (as_of > h), f"drift{h}"] = np.nan
        x_fallback = train[FALLBACK].copy()
        if cfg.wx_blank:
            blank = rng.random(len(train)) < cfg.wx_blank
            x_full.loc[blank, WEATHER] = np.nan
            x_fallback.loc[blank, WEATHER] = np.nan
        y = train["crowd_percent"]
        self.full = lgb.LGBMRegressor(**cfg.lightgbm).fit(x_full, y)
        self.fallback = lgb.LGBMRegressor(**cfg.lightgbm).fit(x_fallback, y)
        log.info(
            "fitted on %d rows to %s; %d/%d parks gated in",
            len(train),
            cutoff.date(),
            int((self.history_years >= cfg.min_years).sum()),
            len(self.history_years),
        )
        return self

    # ---------------------------------------------------------------- prediction
    def _check_fitted(self) -> tuple[lgb.LGBMRegressor, lgb.LGBMRegressor]:
        if self.full is None or self.fallback is None:
            raise RuntimeError("model is not fitted")
        return self.full, self.fallback

    def predict(self, frame: pd.DataFrame, lag: int | None = None) -> pd.DataFrame:
        """Predict ``crowd_percent`` (clipped to 0-100).

        ``lag=None`` is a long-horizon prediction (drift blanked); ``lag in DRIFT_LAGS`` uses drift columns known at that lag and adds
        ``drift_effect`` (prediction with drift minus without; NaN for fallback parks, which have no drift input).
        """
        full, fallback = self._check_fitted()
        years = frame["park_id"].map(self.history_years).fillna(0).to_numpy()
        gated = years >= self.config.min_years
        pred = np.empty(len(frame))
        base = np.full(len(frame), np.nan)
        if gated.any():
            x = frame.loc[gated, FULL]
            pred[gated] = full.predict(blank_drift(x, lag))
            base[gated] = pred[gated] if lag is None else full.predict(blank_drift(x, None))
        if (~gated).any():
            pred[~gated] = fallback.predict(frame.loc[~gated, FALLBACK])
        out = pd.DataFrame(
            {
                "park_id": frame["park_id"].to_numpy(),
                "date": frame["date"].to_numpy(),
                "prediction": np.clip(pred, 0, 100).round(1),
                "model": np.where(gated, "park", "fallback"),
                "history_years": np.round(years, 2),
                "low_confidence": ~gated,
            },
            index=frame.index,
        )
        if lag is not None:
            out["drift_effect"] = (pred - base).round(1)
        return out

    def predict_with(self, which: str, frame: pd.DataFrame, lag: int | None = None) -> np.ndarray:
        """Raw (unclipped) predictions of one sub-model for every row: ``"full"`` or ``"fallback"``. Used in evaluation."""
        full, fallback = self._check_fitted()
        if which == "full":
            return np.asarray(full.predict(blank_drift(frame[FULL], lag)), dtype=float)
        return np.asarray(fallback.predict(frame[FALLBACK]), dtype=float)

    # --------------------------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self, "lightgbm": lgb.__version__}, path)

    @staticmethod
    def load(path: str | Path) -> GatedCrowdModel:
        """Load a model saved by :meth:`save`. Only load files you produced yourself (joblib/pickle can execute code)."""
        blob: dict[str, Any] = joblib.load(path)
        if blob.get("lightgbm") != lgb.__version__:
            log.warning("model was saved with lightgbm %s, running %s", blob.get("lightgbm"), lgb.__version__)
        model = blob["model"]
        assert isinstance(model, GatedCrowdModel)
        return model

    def feature_importance(self, top: int = 15) -> pd.Series:
        """Share (%) of total split gain per feature in the full model."""
        full, _ = self._check_fitted()
        gain = pd.Series(full.booster_.feature_importance("gain"), index=full.booster_.feature_name())
        return (100 * gain / gain.sum()).sort_values(ascending=False).head(top).round(2)
