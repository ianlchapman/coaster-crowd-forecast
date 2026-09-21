"""The file-based pipeline on synthetic inputs: raw loader files on disk -> enhanced calendar -> features -> model -> forecast."""

import numpy as np
import pandas as pd

from crowdcast import pipeline
from crowdcast.demo import build_synthetic_inputs
from crowdcast.models.gated import GatedCrowdModel


def test_enhanced_calendar_is_written_and_reloadable(data_dir):
    paths, s = data_dir
    enhanced = pipeline.make_enhanced_calendar(paths)
    assert paths.enhanced_calendar.exists()
    assert {"national_holiday", "school_holiday", "days_until_school_holiday"} <= set(enhanced.columns)
    assert (enhanced["status"] != "closed").all() and len(enhanced) < len(s.crowd)


def test_training_frame_from_files_matches_the_in_memory_pipeline(data_dir):
    paths, _ = data_dir
    pipeline.make_enhanced_calendar(paths)
    frame = pipeline.load_training_frame(paths)
    memory = build_synthetic_inputs(n_parks=3, start="2019-01-01", end="2022-12-31", seed=2).frame
    assert frame.shape == memory.shape
    np.testing.assert_allclose(frame["crowd_percent"], memory["crowd_percent"])
    np.testing.assert_allclose(frame["wx_temp_max"], memory["wx_temp_max"], atol=1e-6)
    np.testing.assert_allclose(frame["school_holiday"], memory["school_holiday"], atol=1e-4)


def test_forecast_from_files_with_a_faked_weather_api(data_dir, monkeypatch, fast_config):
    paths, s = data_dir
    pipeline.make_enhanced_calendar(paths)
    frame = pipeline.load_training_frame(paths)
    model = GatedCrowdModel(fast_config).fit(frame, "2021-12-31")
    fc_days = pd.date_range("2022-11-01", "2023-01-05")
    fake = s.weather[s.weather["date"].isin(fc_days)]
    monkeypatch.setattr(pipeline, "fetch_forecast", lambda coords, cache, refresh=False: fake)
    out = pipeline.forecast(paths, model, end="2023-01-05")
    assert {"prediction", "days_ahead", "weather", "open_last_year", "drift_effect", "low_confidence"} <= set(
        out.columns
    )
    assert out["date"].min() > frame["date"].max() and out["date"].max() == pd.Timestamp("2023-01-05")
    assert out["prediction"].between(0, 100).all()
    assert set(out["weather"]) <= {"archive", "forecast", "none"}
