import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import ExtraTreesRegressor

from crowdcast.demo import SyntheticInputs
from crowdcast.evaluation.forecast_weather import RAW_COLUMNS, with_forecast_weather, without_weather
from crowdcast.features.columns import WEATHER
from crowdcast.models.zoo import (
    F_ID,
    F_ID_PY,
    F_NOID,
    GlobalMean,
    GroupMean,
    LightGBM,
    RidgePerPark,
    SameWeekdayLastYear,
    SklearnTrees,
    default_models,
)


@pytest.fixture(scope="module")
def split(inputs: SyntheticInputs):
    f = inputs.frame
    return f, f["date"] <= "2022-12-31", f["date"] > "2022-12-31"


def test_baselines_are_sensible(split):
    f, train, ev = split
    y = f.loc[ev, "crowd_percent"].to_numpy()
    mean = GlobalMean().fit_predict(f, train, ev)
    by_dow = GroupMean(["park_id", "dow"]).fit_predict(f, train, ev)
    last_year = SameWeekdayLastYear().fit_predict(f, train, ev)
    assert np.allclose(mean, f.loc[train, "crowd_percent"].mean())
    assert np.abs(by_dow - y).mean() < np.abs(mean - y).mean()  # the weekday pattern is real in the synthetic data
    assert not np.isnan(last_year).any() and len(last_year) == int(ev.sum())


def test_group_mean_falls_back_for_unseen_groups(split):
    f, train, ev = split
    unseen = f.copy()
    unseen.loc[ev, "dow"] = 99  # a weekday never seen in training
    pred = GroupMean(["park_id", "dow"]).fit_predict(unseen, train, ev)
    park_means = f[train].groupby("park_id")["crowd_percent"].mean()
    assert np.allclose(pred, f.loc[ev, "park_id"].map(park_means).to_numpy())


def test_learned_models_beat_the_mean_and_feature_sets_are_consistent(split):
    f, train, ev = split
    y = f.loc[ev, "crowd_percent"].to_numpy()
    base = np.abs(GlobalMean().fit_predict(f, train, ev) - y).mean()
    small = {"n_estimators": 40, "num_leaves": 15, "min_child_samples": 20, "n_jobs": 2}
    for model in (
        LightGBM(F_ID_PY, small),
        SklearnTrees(ExtraTreesRegressor(n_estimators=30, min_samples_leaf=5, n_jobs=2, random_state=0), F_ID),
        RidgePerPark(10.0),
    ):
        assert np.abs(model.fit_predict(f, train, ev) - y).mean() < base
    assert set(F_NOID) < set(F_ID) and set(F_ID) < set(F_ID_PY) and "park_cat" not in F_NOID
    assert len(default_models()) == 12


def test_forecast_weather_swaps_same_day_values_and_recomputes_flags(inputs: SyntheticInputs):
    rows = inputs.frame[inputs.frame["date"] > "2023-06-01"].head(6).copy()
    archived = pd.DataFrame(
        [
            {
                "park_id": r.park_id,
                "date": r.date,
                "lead": 7,
                **{c: 25.0 if c == "wx_temp_max" else 12.0 for c in RAW_COLUMNS},
            }
            for r in rows.itertuples()
        ]
    )
    out, ok = with_forecast_weather(rows, archived, 7)
    assert ok.all()
    assert (
        (out["wx_temp_max"] == 25.0).all() and (out["wx_hot"] == 0).all() and (out["wx_heavy_rain"] == 1).all()
    )  # 12 mm rain
    clim = rows["wx_temp_max"] - rows["wx_temp_anom"]
    np.testing.assert_allclose(out["wx_temp_anom"], 25.0 - clim)  # anomaly against the same climatology
    np.testing.assert_array_equal(out["wx_precip_prev3"], rows["wx_precip_prev3"])  # look-back stays observed
    assert (rows["wx_temp_max"] != 25.0).any()  # the input frame is not modified


def test_missing_forecasts_keep_observed_weather_and_are_reported(inputs: SyntheticInputs):
    rows = inputs.frame[inputs.frame["date"] > "2023-06-01"].head(4).copy()
    archived = pd.DataFrame(
        {"park_id": [rows.iloc[0].park_id], "date": [rows.iloc[0].date], "lead": [1], **dict.fromkeys(RAW_COLUMNS, 1.0)}
    )
    out, ok = with_forecast_weather(rows, archived, 1)
    assert ok.tolist() == [True, False, False, False]
    np.testing.assert_array_equal(out["wx_temp_max"].iloc[1:], rows["wx_temp_max"].iloc[1:])


def test_without_weather_blanks_every_weather_column(inputs: SyntheticInputs):
    out = without_weather(inputs.frame.head(5))
    assert out[WEATHER].isna().all().all() and inputs.frame[WEATHER].head(5).notna().any().any()
