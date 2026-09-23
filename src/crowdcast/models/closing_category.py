"""Closing time as a park-relative day-type classifier instead of free regression.

Regressing ``close_min`` directly rarely lands on the park's actual (discrete) schedule value -- see
docs/planning/opening-hours-forecast.md's v2 write-up. This reframes it: classify each open day into one
of four day-types -- ``event`` (an event was on) else ``short``/``normal``/``long``, defined per park as
terciles of *that park's own* historical non-event close times (so "long" means different clock times at
different parks) -- then decode the predicted category back to a clock time using that park's own median
close time observed in that category. Both the tercile boundaries and the decode table are fit from
training data only, so nothing about the held-out window leaks in.
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
from crowdcast.features.status_build import CATEGORY_FEATURES
from crowdcast.models.config import ModelConfig

CATEGORIES = ("short", "normal", "long", "event")
#: event groups plausibly tied to *extended closing hours*; excludes ev_early_extra (early entry/extra
#: magic hours affects opening, not closing) so "just an early-entry day" doesn't get lumped in as "event".
CLOSING_EVENT_GROUPS = ["ev_halloween", "ev_christmas", "ev_summer", "ev_festival", "ev_ticketed"]


def _is_closing_event(rows: pd.DataFrame) -> pd.Series:
    return rows[CLOSING_EVENT_GROUPS].any(axis=1)


def _categorize(open_rows: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Category per row, plus the per-park [q33, q67] boundary table used to build it (non-event rows only)."""
    is_event = _is_closing_event(open_rows)
    non_event = open_rows[~is_event]
    long = non_event.groupby("park_id")["close_min"].quantile(np.array([1 / 3, 2 / 3])).rename("value").reset_index()
    long.columns = ["park_id", "q", "value"]
    q = long.pivot_table(index="park_id", columns="q", values="value")
    q.columns = ["q33", "q67"]

    cat = pd.Series("normal", index=open_rows.index, dtype="object")
    cat[is_event] = "event"
    rest = ~is_event
    q33, q67 = open_rows["park_id"].map(q["q33"]), open_rows["park_id"].map(q["q67"])
    cat[rest & (open_rows["close_min"] <= q33)] = "short"
    cat[rest & (open_rows["close_min"] > q67)] = "long"
    return cat, q


class ClosingCategoryModel:
    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        self.clf: lgb.LGBMClassifier | None = None
        self.park_category_close: dict[tuple[int, str], float] = {}
        self.park_median_close: dict[int, float] = {}
        self.category_global_offset: dict[str, float] = {}
        self.global_median_close: float = 0.0

    def fit(self, frame: pd.DataFrame, cutoff: str | pd.Timestamp) -> ClosingCategoryModel:
        upto = frame[frame["date"] <= pd.Timestamp(cutoff)]
        train = upto[~in_covid(upto["date"])] if self.config.exclude_covid else upto
        open_rows = train[train["is_open"] & train["close_min"].notna()]

        cat, _ = _categorize(open_rows)
        self.clf = lgb.LGBMClassifier(**self.config.lightgbm).fit(open_rows[CATEGORY_FEATURES], cat)

        labelled = open_rows.assign(close_category=cat)
        by_park_category = (
            labelled.groupby(["park_id", "close_category"], observed=True)["close_min"].median().reset_index()
        )
        self.park_category_close = {
            (int(pid), str(c)): float(v)
            for pid, c, v in zip(
                by_park_category["park_id"],
                by_park_category["close_category"],
                by_park_category["close_min"],
                strict=True,
            )
        }
        by_park = open_rows.groupby("park_id")["close_min"].median().reset_index()
        self.park_median_close = {
            int(pid): float(v) for pid, v in zip(by_park["park_id"], by_park["close_min"], strict=True)
        }
        self.global_median_close = float(open_rows["close_min"].median())
        by_category = (
            labelled.groupby("close_category", observed=True)["close_min"].median() - self.global_median_close
        ).reset_index()
        self.category_global_offset = {
            str(c): float(v) for c, v in zip(by_category["close_category"], by_category["close_min"], strict=True)
        }
        return self

    def _decode(self, park_id: pd.Series, category: np.ndarray) -> np.ndarray:
        """Category -> clock time: this park's own median in that category, else this park's median close
        shifted by the category's average effect across all parks, else the global median close."""
        out = np.empty(len(park_id), dtype=float)
        for i, (pid, cat) in enumerate(zip(park_id, category, strict=True)):
            if (pid, cat) in self.park_category_close:
                out[i] = self.park_category_close[(pid, cat)]
            else:
                base = self.park_median_close.get(pid, self.global_median_close)
                out[i] = base + self.category_global_offset.get(cat, 0.0)
        return out

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.clf is None:
            raise RuntimeError("model is not fitted")
        category = np.asarray(self.clf.predict(frame[CATEGORY_FEATURES]))
        close_min = self._decode(frame["park_id"], category)
        out = pd.DataFrame(
            {
                "park_id": frame["park_id"].to_numpy(),
                "date": frame["date"].to_numpy(),
                "close_category": category,
                "close_min": close_min,
            },
            index=frame.index,
        )
        out["closes"] = minutes_to_hhmm(out["close_min"])
        return out

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self, "lightgbm": lgb.__version__}, path)

    @staticmethod
    def load(path: str | Path) -> ClosingCategoryModel:
        """Only load files you produced yourself (joblib/pickle can execute code)."""
        blob: dict[str, Any] = joblib.load(path)
        model = blob["model"]
        assert isinstance(model, ClosingCategoryModel)
        return model
