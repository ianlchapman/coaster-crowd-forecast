import numpy as np
import pandas as pd
import pytest

from crowdcast.demo import SyntheticInputs
from crowdcast.evaluation.backtest import run_backtest
from crowdcast.features.columns import DRIFT, FALLBACK, FULL, WEATHER
from crowdcast.models.config import DEFAULT_CONFIG_PATH, ModelConfig
from crowdcast.models.gated import GatedCrowdModel, blank_drift


def test_yaml_config_matches_python_defaults():
    from_yaml = ModelConfig.from_yaml(DEFAULT_CONFIG_PATH)
    assert from_yaml == ModelConfig()


def test_predictions_are_valid_and_aligned(inputs: SyntheticInputs, model: GatedCrowdModel):
    test = inputs.frame[inputs.frame["date"] > "2022-12-31"]
    out = model.predict(test)
    assert out.index.equals(test.index)
    assert out["prediction"].between(0, 100).all() and out["prediction"].notna().all()
    assert (out["model"] == "park").all() and not out["low_confidence"].any()
    assert (out["history_years"] > 3).all()


def test_short_history_parks_use_the_fallback_and_are_flagged(inputs: SyntheticInputs):
    frame = inputs.frame
    # park 1 only has 40 days of labels before the cutoff -> below the 1-year gate
    keep = (frame["park_id"] != 1) | (frame["date"] > "2022-11-20")
    model = GatedCrowdModel(
        ModelConfig(lightgbm={"n_estimators": 30, "num_leaves": 7, "min_child_samples": 20, "verbose": -1, "n_jobs": 2})
    )
    model.fit(frame[keep], "2022-12-31")
    out = model.predict(frame[keep & (frame["date"] > "2022-12-31")])
    assert out.loc[out["park_id"] == 1, "low_confidence"].all()
    assert (out.loc[out["park_id"] == 1, "model"] == "fallback").all()
    assert not out.loc[out["park_id"] != 1, "low_confidence"].any()
    unseen = frame[(frame["park_id"] == 1)].assign(park_id=999).head(5)  # a park the model has never seen
    assert model.predict(unseen)["low_confidence"].all()


def test_training_ignores_rows_after_the_cutoff(inputs: SyntheticInputs, fast_config: ModelConfig):
    f = inputs.frame
    scrambled = f.copy()
    late = scrambled["date"] > "2022-06-30"
    scrambled.loc[late, "crowd_percent"] = 100 - scrambled.loc[late, "crowd_percent"]
    a = GatedCrowdModel(fast_config).fit(f, "2022-06-30")
    b = GatedCrowdModel(fast_config).fit(scrambled, "2022-06-30")
    probe = f[f["date"] > "2022-12-31"].head(200)
    np.testing.assert_array_equal(a.predict(probe)["prediction"], b.predict(probe)["prediction"])


def test_covid_window_is_excluded_from_training(inputs: SyntheticInputs, fast_config: ModelConfig):
    f = inputs.frame.copy()
    covid = (f["date"] >= "2020-03-01") & (f["date"] <= "2021-12-31")
    f.loc[covid, "crowd_percent"] = 1.0  # poison the window; an excluded window cannot affect the fit
    poisoned = GatedCrowdModel(fast_config).fit(f, "2022-12-31")
    clean = GatedCrowdModel(fast_config).fit(inputs.frame, "2022-12-31")
    probe = inputs.frame[inputs.frame["date"] > "2022-12-31"].head(200)
    np.testing.assert_allclose(poisoned.predict(probe)["prediction"], clean.predict(probe)["prediction"])


def test_long_horizon_equals_predicting_with_drift_removed(inputs: SyntheticInputs, model: GatedCrowdModel):
    test = inputs.frame[inputs.frame["date"] > "2022-12-31"]
    no_drift = test.copy()
    no_drift[DRIFT] = np.nan
    np.testing.assert_array_equal(model.predict(test)["prediction"], model.predict(no_drift)["prediction"])


def test_blank_drift_keeps_only_columns_known_at_the_lag(inputs: SyntheticInputs):
    x = inputs.frame[FULL].head(50).fillna(0.0)
    assert blank_drift(x, 1)[DRIFT].notna().all().all()
    kept = blank_drift(x, 7)
    assert kept["drift1"].isna().all() and kept["drift7"].notna().all() and kept["drift14"].notna().all()
    assert blank_drift(x, 14)[["drift1", "drift7"]].isna().all().all()
    assert blank_drift(x, None)[DRIFT].isna().all().all()
    assert x[DRIFT].notna().all().all()  # the input is not modified


def test_weather_blanked_training_still_predicts_without_weather(inputs: SyntheticInputs, model: GatedCrowdModel):
    test = inputs.frame[inputs.frame["date"] > "2022-12-31"].copy()
    test[WEATHER] = np.nan
    assert model.predict(test)["prediction"].notna().all()
    assert set(WEATHER) <= set(FULL) and set(WEATHER) <= set(FALLBACK)


def test_model_learns_the_weekday_pattern(inputs: SyntheticInputs, model: GatedCrowdModel):
    test = inputs.frame[inputs.frame["date"] > "2022-12-31"]
    pred = model.predict(test)["prediction"].to_numpy()
    y = test["crowd_percent"].to_numpy()
    assert np.abs(pred - y).mean() < np.abs(y - y.mean()).mean() * 0.8  # clearly better than predicting the mean
    weekend = test["is_weekend"].to_numpy() == 1
    assert pred[weekend].mean() > pred[~weekend].mean() + 3


def test_save_load_roundtrip(tmp_path, inputs: SyntheticInputs, model: GatedCrowdModel):
    path = tmp_path / "m.joblib"
    model.save(path)
    again = GatedCrowdModel.load(path)
    probe = inputs.frame.tail(100)
    pd.testing.assert_frame_equal(model.predict(probe, lag=7), again.predict(probe, lag=7))


def test_unfitted_model_raises(inputs: SyntheticInputs):
    with pytest.raises(RuntimeError, match="not fitted"):
        GatedCrowdModel().predict(inputs.frame.head(3))


def test_backtest_reports_all_horizons_and_groups(inputs: SyntheticInputs, fast_config: ModelConfig):
    r = run_backtest(inputs.frame, "2022-12-31", "2023-12-31", fast_config)
    assert r.by_horizon["setup"].tolist()[0].startswith("long horizon") and len(r.by_horizon) == 4
    assert {"all parks: gated system", "short-history parks: guess 50"} <= set(r.by_group["setup"])
    assert r.predictions["crowd_percent"].notna().all()
