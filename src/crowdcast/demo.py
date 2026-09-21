"""End-to-end run on synthetic data: build features, back-test, then build and score future rows.

Exercises the same code paths as the real pipeline (enhance -> features -> gated model -> future rows -> forecast) without any
external data or network, so it doubles as a smoke test. The numbers say nothing about real parks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from crowdcast.data.synthetic import SyntheticData, make_synthetic
from crowdcast.evaluation.backtest import BacktestResult, run_backtest
from crowdcast.features.build import build_feature_frame, park_table
from crowdcast.features.future import build_future_rows
from crowdcast.models.config import ModelConfig
from crowdcast.models.gated import GatedCrowdModel
from crowdcast.scoring.daily import DailyScores
from crowdcast.scoring.enhance import enhance_calendar
from crowdcast.weather.features import derive_weather_features

TIERS = ["local", "regional", "continental"]


@dataclass(frozen=True)
class SyntheticInputs:
    frame: pd.DataFrame  # labelled feature frame
    scores: DailyScores
    weather: pd.DataFrame  # wx_* features with wx_source, incl. the future window
    raw: SyntheticData


def synthetic_scores(s: SyntheticData) -> DailyScores:
    """Holiday scores of the synthetic parks in the same container the real pipeline uses."""
    ids = np.sort(s.parks["id"].to_numpy())
    dates = pd.DatetimeIndex(sorted(s.holiday_scores["date"].unique()))
    hs = s.holiday_scores.sort_values(["park_id", "date"])
    n_dates = len(dates)
    nat = hs["national_score"].to_numpy(dtype=np.float32).reshape(len(ids), n_dates)
    sch = hs["school_score"].to_numpy(dtype=np.float32).reshape(len(ids), n_dates)
    return DailyScores(ids, dates, nat, sch)


def build_synthetic_inputs(
    n_parks: int = 6, start: str = "2018-01-01", end: str = "2024-12-31", seed: int = 0
) -> SyntheticInputs:
    s = make_synthetic(n_parks, start, end, seed)
    scores = synthetic_scores(s)
    enriched = pd.DataFrame(
        {
            "park_id": s.parks["id"],
            "country_iso2": s.parks["country"],
            "tier": [TIERS[i % len(TIERS)] for i in range(len(s.parks))],
        }
    )
    enhanced = enhance_calendar(s.crowd, scores)
    weather = derive_weather_features(s.weather)
    frame = build_feature_frame(enhanced, park_table(s.parks, enriched), weather)
    return SyntheticInputs(frame, scores, weather.assign(wx_source="forecast"), s)


def run_demo(n_parks: int = 6, config: ModelConfig | None = None) -> tuple[BacktestResult, pd.DataFrame]:
    """Back-test on the last year of synthetic data, then score the 16 days after the last label."""
    inputs = build_synthetic_inputs(n_parks)
    cfg = config or ModelConfig(
        lightgbm={"n_estimators": 150, "num_leaves": 31, "min_child_samples": 20, "verbose": -1, "n_jobs": 2}
    )
    result = run_backtest(inputs.frame, cutoff="2023-12-31", test_end="2024-12-31", config=cfg)
    model = GatedCrowdModel(cfg).fit(inputs.frame, "2024-12-31")
    future = build_future_rows(inputs.frame, inputs.scores, inputs.weather, end="2025-01-16")
    forecast = model.predict(future, lag=1).assign(days_ahead=future["days_ahead"].to_numpy())
    return result, forecast
