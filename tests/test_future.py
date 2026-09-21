import numpy as np
import pandas as pd
import pytest

from crowdcast.demo import SyntheticInputs
from crowdcast.features.columns import DRIFT, FALLBACK, FULL
from crowdcast.features.future import build_future_rows
from crowdcast.models.gated import GatedCrowdModel


def test_future_rows_have_every_model_column_and_start_after_the_last_label(inputs: SyntheticInputs):
    f = build_future_rows(inputs.frame, inputs.scores, inputs.weather, end="2024-01-10")
    assert set(FULL) | set(FALLBACK) <= set(f.columns)
    assert f["date"].min() == inputs.frame["date"].max() + pd.Timedelta(days=1)
    assert f["date"].max() == pd.Timestamp("2024-01-10")
    assert (f["days_ahead"] >= 1).all() and not f.duplicated(["park_id", "date"]).any()
    assert f["park_cat"].dtype == "category"


def test_future_calendar_holidays_and_prior_year_match_real_features(inputs: SyntheticInputs):
    """Rebuild a labelled summer window as if it were unknown and compare with its true features."""
    frame = inputs.frame
    cutoff, end = pd.Timestamp("2023-06-30"), pd.Timestamp("2023-08-15")
    built = build_future_rows(frame, inputs.scores, inputs.weather, end=end, asof=cutoff)
    real = frame[(frame["date"] > cutoff) & (frame["date"] <= end)].merge(
        built, on=["park_id", "date"], suffixes=("", "_built")
    )
    assert len(real) > 100
    exact = ["dow", "month", "week", "year", "doy_sin", "months_open", "southern", "national_holiday", "school_holiday",
             "days_until_national_holiday_log", "days_since_school_holiday_log", "py_same_wd", "py_wd_mean3", "wx_temp_max", "wx_precip_prev3"]  # fmt: skip
    for col in exact:
        np.testing.assert_allclose(
            real[col].astype(float), real[col + "_built"].astype(float), atol=1e-3, equal_nan=True, err_msg=col
        )


def test_drift_is_only_available_while_labels_are_recent_enough(inputs: SyntheticInputs):
    f = build_future_rows(inputs.frame, inputs.scores, inputs.weather, end="2024-01-14")
    park = f[f["park_id"] == f["park_id"].min()].sort_values("date")
    ahead = park["days_ahead"].to_numpy()
    for h in (1, 7, 14):
        known = park[f"drift{h}"].notna().to_numpy()
        assert not known[ahead > h].any(), f"drift{h} must be blank when the window would end after the last label"
    assert park.loc[ahead == 1, "drift1"].notna().all() or park.loc[ahead == 1, "drift14"].notna().all()


def test_dates_without_weather_are_marked_and_still_predictable(inputs: SyntheticInputs, model: GatedCrowdModel):
    weather = inputs.weather[inputs.weather["date"] <= "2023-12-31"]  # nothing for the future window
    f = build_future_rows(inputs.frame, inputs.scores, weather, end="2024-01-05")
    assert (f["wx_source"] == "none").all() and f["wx_temp_max"].isna().all()
    assert model.predict(f, lag=1)["prediction"].notna().all()


def test_labels_beyond_asof_never_leak_into_future_rows(inputs: SyntheticInputs):
    frame = inputs.frame
    asof = pd.Timestamp("2023-06-30")
    tampered = frame.copy()
    tampered.loc[tampered["date"] > asof, "crowd_percent"] = 0.0
    a = build_future_rows(frame, inputs.scores, inputs.weather, end="2023-07-30", asof=asof)
    b = build_future_rows(tampered, inputs.scores, inputs.weather, end="2023-07-30", asof=asof)
    pd.testing.assert_frame_equal(
        a[[*DRIFT, "py_same_wd", "open_last_year"]], b[[*DRIFT, "py_same_wd", "open_last_year"]]
    )
    assert np.isfinite(a["days_ahead"]).all()


def test_open_last_year_flags_reopening_dates(inputs: SyntheticInputs):
    f = build_future_rows(inputs.frame, inputs.scores, inputs.weather, end="2024-01-14")
    assert set(f["open_last_year"].unique()) <= {0, 1}
    assert f["open_last_year"].sum() > 0


@pytest.mark.parametrize("bad_end", ["2023-12-01"])
def test_end_before_last_label_yields_no_rows(inputs: SyntheticInputs, bad_end: str):
    assert len(build_future_rows(inputs.frame, inputs.scores, inputs.weather, end=bad_end)) == 0
