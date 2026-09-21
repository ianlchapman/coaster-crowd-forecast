"""Baselines and alternative estimators the gated LightGBM model is compared against.

Each entry has ``fit_predict(frame, train_mask, eval_mask) -> predictions`` so experiments can treat them uniformly.
Every model is fitted on the rows selected by ``train_mask`` only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from crowdcast.features.columns import BASE, CATEGORICAL, HOLIDAY, PRIOR_YEAR

TARGET = "crowd_percent"
#: Feature sets used in the model comparison (park identity with / without last-year features, and without park identity).
F_ID = [*BASE, "months_open", "southern", "latitude", "longitude", *CATEGORICAL, "park_cat"]
F_ID_PY = [*F_ID, *PRIOR_YEAR]
F_NOID = [*BASE, "months_open", "southern", "latitude", "longitude", *CATEGORICAL]
F_NO_HOLIDAY = [f for f in F_ID if f not in HOLIDAY]


class Model(Protocol):
    def fit_predict(self, frame: pd.DataFrame, train: pd.Series, evaluate: pd.Series) -> np.ndarray: ...


@dataclass
class GlobalMean:
    def fit_predict(self, frame: pd.DataFrame, train: pd.Series, evaluate: pd.Series) -> np.ndarray:
        return np.full(int(evaluate.sum()), frame.loc[train, TARGET].mean())


@dataclass
class GroupMean:
    """Mean of the target per group (e.g. park x weekday), falling back to the park mean, then 50."""

    keys: Sequence[str]

    def fit_predict(self, frame: pd.DataFrame, train: pd.Series, evaluate: pd.Series) -> np.ndarray:
        keys = list(self.keys)
        means = frame[train].groupby(keys, observed=True)[TARGET].mean().rename("m").reset_index()
        pred = frame.loc[evaluate, keys].merge(means, on=keys, how="left")["m"]
        park_mean = frame[train].groupby("park_id")[TARGET].mean()
        fallback = frame.loc[evaluate, "park_id"].map(park_mean).to_numpy()
        return pred.fillna(pd.Series(fallback, index=pred.index)).fillna(50).to_numpy()


@dataclass
class SameWeekdayLastYear:
    """The park's value 52 weeks earlier; where that is missing, the park x weekday mean."""

    def fit_predict(self, frame: pd.DataFrame, train: pd.Series, evaluate: pd.Series) -> np.ndarray:
        fallback = GroupMean(["park_id", "dow"]).fit_predict(frame, train, evaluate)
        last_year = frame.loc[evaluate, "py_same_wd"].to_numpy()
        return np.where(np.isnan(last_year), fallback, last_year)


@dataclass
class LightGBM:
    features: Sequence[str]
    params: dict[str, Any] = field(default_factory=dict)

    def fit_predict(self, frame: pd.DataFrame, train: pd.Series, evaluate: pd.Series) -> np.ndarray:
        base = {"n_estimators": 600, "learning_rate": 0.03, "num_leaves": 63, "min_child_samples": 30, "subsample": 0.8, "subsample_freq": 1,
                "colsample_bytree": 0.8, "reg_lambda": 1.0, "verbose": -1, "n_jobs": 8, "cat_smooth": 10}  # fmt: skip
        model = lgb.LGBMRegressor(**{**base, **self.params}).fit(
            frame.loc[train, list(self.features)], frame.loc[train, TARGET]
        )
        return np.asarray(model.predict(frame.loc[evaluate, list(self.features)]), dtype=float)


def _dense(frame: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    """Numeric matrix for sklearn: categoricals as integer codes, NaN as -1 (tree models cope with both)."""
    x = frame[list(features)].copy()
    for col in x.columns:
        if isinstance(x[col].dtype, pd.CategoricalDtype):
            x[col] = x[col].cat.codes
    return x.fillna(-1)


@dataclass
class SklearnTrees:
    estimator: Any
    features: Sequence[str]

    def fit_predict(self, frame: pd.DataFrame, train: pd.Series, evaluate: pd.Series) -> np.ndarray:
        self.estimator.fit(_dense(frame[train], self.features), frame.loc[train, TARGET])
        return np.asarray(self.estimator.predict(_dense(frame[evaluate], self.features)), dtype=float)


@dataclass
class RidgePerPark:
    """Linear model with park x weekday / month / holiday interactions as one-hot terms."""

    alpha: float = 10.0
    numeric: Sequence[str] = field(
        default_factory=lambda: [
            *HOLIDAY,
            "open_min",
            "close_min",
            "open_hours",
            "is_weekend",
            "doy_sin",
            "doy_cos",
            "covid",
            "has_event",
        ]
    )

    def _design(self, frame: pd.DataFrame) -> pd.DataFrame:
        x = frame[["park_id", "dow", "month", *self.numeric]].copy()
        pid = x["park_id"].astype(str)
        x["park_dow"] = pid + "_" + x["dow"].astype(str)
        x["park_month"] = pid + "_" + x["month"].astype(str)
        x["park_school"] = pid + "_s" + (frame["school_holiday"] > 0.2).astype(int).astype(str)
        x["park_nat"] = pid + "_n" + (frame["national_holiday"] > 0.2).astype(int).astype(str)
        for col in ("park_id", "dow", "month"):
            x[col] = x[col].astype(str)
        return x

    def fit_predict(self, frame: pd.DataFrame, train: pd.Series, evaluate: pd.Series) -> np.ndarray:
        categorical = ["park_id", "dow", "month", "park_dow", "park_month", "park_school", "park_nat"]
        transform = ColumnTransformer(
            [("c", OneHotEncoder(handle_unknown="ignore"), categorical), ("n", StandardScaler(), list(self.numeric))]
        )
        pipe = make_pipeline(transform, Ridge(alpha=self.alpha))
        pipe.fit(self._design(frame[train]).fillna(0), frame.loc[train, TARGET])
        return np.asarray(pipe.predict(self._design(frame[evaluate]).fillna(0)), dtype=float)


def default_models() -> dict[str, Model]:
    """The comparison set (baselines, linear, tree ensembles, LightGBM variants)."""
    return {
        "B0 global mean": GlobalMean(),
        "B1 park mean": GroupMean(["park_id"]),
        "B2 park x weekday mean": GroupMean(["park_id", "dow"]),
        "B3 park x weekday x month mean": GroupMean(["park_id", "dow", "month"]),
        "B4 same weekday last year": SameWeekdayLastYear(),
        "M1 Ridge (park x weekday/month/holiday)": RidgePerPark(10.0),
        "M2 RandomForest": SklearnTrees(
            RandomForestRegressor(n_estimators=200, min_samples_leaf=5, max_features=0.5, n_jobs=8, random_state=0),
            F_ID,
        ),
        "M3 ExtraTrees": SklearnTrees(
            ExtraTreesRegressor(n_estimators=200, min_samples_leaf=5, max_features=0.7, n_jobs=8, random_state=0), F_ID
        ),
        "M4 LightGBM (park id, no last-year)": LightGBM(F_ID),
        "M5 LightGBM (park id + last-year)": LightGBM(F_ID_PY),
        "M6 LightGBM (no park id)": LightGBM(F_NOID),
        "M7 LightGBM (no holiday features)": LightGBM(F_NO_HOLIDAY),
    }
