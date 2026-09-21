import numpy as np
import pandas as pd
import pytest

from crowdcast.demo import SyntheticInputs
from crowdcast.features.build import add_holiday_distance_logs
from crowdcast.features.calendar import add_calendar_features, in_covid
from crowdcast.features.columns import DRIFT_LAGS, FALLBACK, FULL
from crowdcast.features.history import add_drift, add_prior_year
from crowdcast.features.hours_events import add_events, add_hours


def test_feature_frame_has_every_model_column(inputs: SyntheticInputs):
    f = inputs.frame
    assert set(FULL) | set(FALLBACK) <= set(f.columns)
    assert f["crowd_percent"].notna().all() and f["crowd_percent"].between(0, 100).all()
    assert not f.duplicated(["park_id", "date"]).any()
    assert f["park_cat"].dtype == "category" and f["tier"].dtype == "category"


def test_closed_and_predicted_rows_never_become_labels(inputs: SyntheticInputs):
    raw = inputs.raw.crowd
    assert (raw["status"] == "closed").any()
    n_open_labelled = int(((raw["status"] == "open") & raw["crowd_percent"].notna()).sum())
    assert len(inputs.frame) == n_open_labelled


def test_calendar_features():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-03-01", "2021-12-31", "2022-01-01"]),
            "latitude": [10.0, -5.0, 10.0],
            "first_year": 2019,
            "first_month": 1,
        }
    )
    add_calendar_features(df)
    assert df["covid"].tolist() == [1, 1, 0]
    assert df["southern"].tolist() == [0, 1, 0]
    assert df["dow"].tolist() == [6, 4, 5] and df["is_weekend"].tolist() == [1, 0, 1]
    assert df["months_open"].tolist() == [14, 35, 36]
    assert in_covid(pd.Series(pd.to_datetime(["2019-12-31"]))).tolist() == [False]


def test_hours_handle_midnight_close_and_missing():
    df = pd.DataFrame({"park_id": [1, 1, 1], "opens": ["10:00", "10:00", None], "closes": ["18:00", "01:00", None]})
    add_hours(df)
    assert df["open_hours"].tolist()[:2] == [8.0, 15.0]  # closing after midnight wraps
    assert df.loc[2, "open_min"] == 600  # filled with the park median


def test_event_flags():
    df = pd.DataFrame({"events": pd.array(["Halloween Nights", None, "Summer Festival"], dtype="string")})
    add_events(df)
    assert df["has_event"].tolist() == [1, 0, 1]
    assert df["ev_halloween"].tolist() == [1, 0, 0]
    assert df[["ev_summer", "ev_festival"]].iloc[2].tolist() == [1, 1]


def test_holiday_distance_logs_are_capped():
    df = pd.DataFrame({"days_until_national_holiday": [0, 10, 1000]})
    add_holiday_distance_logs(df)
    assert df["days_until_national_holiday_log"].tolist() == pytest.approx([0.0, np.log1p(10), np.log1p(180)])


def _labelled(n_days: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n_days)
    return pd.DataFrame({"park_id": 1, "date": dates, "crowd_percent": rng.integers(1, 100, n_days).astype(float)})


def test_prior_year_uses_same_weekday_one_year_back():
    df = add_prior_year(_labelled())
    t = pd.Timestamp("2023-03-10")
    row = df[df["date"] == t].iloc[0]
    lag = df.set_index("date")["crowd_percent"]
    assert row["py_same_wd"] == lag[t - pd.Timedelta(days=364)]
    assert row["py_wd_mean3"] == pytest.approx(np.mean([lag[t - pd.Timedelta(days=d)] for d in (357, 364, 371)]))
    assert (t - pd.Timedelta(days=364)).dayofweek == t.dayofweek
    assert df.loc[df["date"] < "2022-12-27", "py_same_wd"].isna().all()  # nothing to look back on in the first year


@pytest.mark.parametrize("lag", DRIFT_LAGS)
def test_drift_never_uses_labels_after_t_minus_lag(lag: int):
    """Perturb every label from day T on: drift{lag} must not change for any day t with t - lag < T."""
    base = add_prior_year(_labelled())
    cut = pd.Timestamp("2023-04-01")
    changed = base.copy()
    changed.loc[changed["date"] >= cut, "crowd_percent"] += 25
    a, b = add_drift(base), add_drift(changed)
    unaffected = a["date"] < cut + pd.Timedelta(days=lag)
    both = a.loc[unaffected, f"drift{lag}"].to_numpy(), b.loc[unaffected, f"drift{lag}"].to_numpy()
    np.testing.assert_allclose(both[0], both[1], equal_nan=True)
    affected = a["date"] >= cut + pd.Timedelta(days=lag + 7)
    assert not np.allclose(a.loc[affected, f"drift{lag}"].dropna(), b.loc[affected, f"drift{lag}"].dropna())
