"""Feature-driven is_open / opens / closes model: LightGBM classifier + two regressors on the frame from
:func:`crowdcast.features.status_build.build_status_frame`.

One stage, not gated by history length like :class:`crowdcast.models.gated.GatedCrowdModel` — LightGBM
handles the NaN prior-year features a brand-new park gets natively, falling back on calendar/holiday/park
features alone. Revisit gating if evaluation shows new parks need it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from crowdcast.features.calendar import in_covid
from crowdcast.features.status import minutes_to_hhmm
from crowdcast.features.status_build import HOURS_FEATURES, STATUS_FEATURES
from crowdcast.models.config import ModelConfig


class StatusModel:
    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        self.is_open: lgb.LGBMClassifier | None = None
        self.opens: lgb.LGBMRegressor | None = None
        self.closes: lgb.LGBMRegressor | None = None

    def fit(self, frame: pd.DataFrame, cutoff: str | pd.Timestamp) -> StatusModel:
        upto = frame[frame["date"] <= pd.Timestamp(cutoff)]
        train = upto[~in_covid(upto["date"])] if self.config.exclude_covid else upto
        self.is_open = lgb.LGBMClassifier(**self.config.lightgbm).fit(train[STATUS_FEATURES], train["is_open"])
        open_rows = train[train["is_open"]]
        self.opens = lgb.LGBMRegressor(**self.config.lightgbm).fit(open_rows[HOURS_FEATURES], open_rows["open_min"])
        self.closes = lgb.LGBMRegressor(**self.config.lightgbm).fit(open_rows[HOURS_FEATURES], open_rows["close_min"])
        return self

    def _check_fitted(self) -> tuple[lgb.LGBMClassifier, lgb.LGBMRegressor, lgb.LGBMRegressor]:
        if self.is_open is None or self.opens is None or self.closes is None:
            raise RuntimeError("model is not fitted")
        return self.is_open, self.opens, self.closes

    def predict_is_open(self, frame: pd.DataFrame) -> np.ndarray:
        """Just the is_open classifier -- the field it beats the lookup heuristic on in every backtest
        year (see docs/planning/opening-hours-forecast.md); ``opens``/``closes`` still come from the
        lookup in ``pipeline.forecast()``, which wins those fields."""
        is_open_clf, _, _ = self._check_fitted()
        return np.asarray(is_open_clf.predict(frame[STATUS_FEATURES])).astype(bool)

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        is_open_clf, opens_reg, closes_reg = self._check_fitted()
        is_open = np.asarray(is_open_clf.predict(frame[STATUS_FEATURES])).astype(bool)
        # real schedules sit on a 30-minute grid (see docs/planning/opening-hours-forecast.md); round the
        # regressor's continuous output onto it so near-miss predictions still land on the actual value.
        open_min = np.round(np.asarray(opens_reg.predict(frame[HOURS_FEATURES]), dtype=float) / 30) * 30
        close_min = np.round(np.asarray(closes_reg.predict(frame[HOURS_FEATURES]), dtype=float) / 30) * 30
        out = pd.DataFrame(
            {
                "park_id": frame["park_id"].to_numpy(),
                "date": frame["date"].to_numpy(),
                "is_open": is_open,
                "open_min": open_min,
                "close_min": close_min,
            },
            index=frame.index,
        )
        out["opens"] = minutes_to_hhmm(out["open_min"]).where(out["is_open"])
        out["closes"] = minutes_to_hhmm(out["close_min"]).where(out["is_open"])
        return out

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self, "lightgbm": lgb.__version__}, path)

    @staticmethod
    def load(path: str | Path) -> StatusModel:
        """Only load files you produced yourself (joblib/pickle can execute code)."""
        blob: dict[str, Any] = joblib.load(path)
        model = blob["model"]
        assert isinstance(model, StatusModel)
        return model
