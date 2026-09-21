import numpy as np
import pandas as pd
import pytest

from crowdcast.scoring import CalendarMatrices, DailyScores, compute_daily_scores
from crowdcast.scoring.daily import distances_to_holidays
from crowdcast.scoring.enhance import enhance_calendar


def _matrices() -> CalendarMatrices:
    dates = pd.date_range("2024-01-01", "2024-01-31")
    nat = np.zeros((3, len(dates)), dtype=np.float32)
    sch = np.zeros_like(nat)
    nat[0, 0] = 1.0  # region A: 1 Jan national
    nat[1, 0] = 1.0
    sch[0, 10:15] = 1.0  # region A: school 11-15 Jan
    sch[2, 12:20] = 0.5  # region C: half of pupils off 13-20 Jan
    return CalendarMatrices(
        np.array(["A", "B", "C"]), dates, nat, sch, np.array([2024]), np.ones((3, 1), bool), np.ones((3, 1), bool)
    )


def _weights() -> pd.DataFrame:
    rows = [(1, "A", "national", 0.6), (1, "B", "national", 0.4), (1, "A", "school", 0.5), (1, "C", "school", 0.5),
            (2, "B", "national", 1.0), (2, "C", "school", 1.0)]  # fmt: skip
    return pd.DataFrame(rows, columns=["park_id", "region_code", "type", "weight"])


def test_scores_are_weighted_sums_in_unit_interval():
    s = compute_daily_scores(_weights(), _matrices())
    assert s.national.min() >= 0 and s.national.max() <= 1 and s.school.max() <= 1
    p1 = list(s.park_ids).index(1)
    assert s.national[p1, 0] == pytest.approx(1.0)  # both of park 1's national regions are off on 1 Jan
    assert s.school[p1, 12] == pytest.approx(0.5 * 1.0 + 0.5 * 0.5)


def test_unknown_region_in_weights_is_rejected():
    w = pd.concat([_weights(), pd.DataFrame([(3, "ZZ", "national", 1.0)], columns=_weights().columns)])
    with pytest.raises(ValueError, match="missing from the calendar"):
        compute_daily_scores(w, _matrices())


def test_distances_to_holidays():
    score = np.array([0, 0, 0.5, 0, 0, 0, 0.9, 0], dtype=float)
    until, since = distances_to_holidays(score, threshold=0.2)
    np.testing.assert_array_equal(until, [2, 1, 0, 3, 2, 1, 0, np.nan])  # no holiday after the last day -> NaN
    np.testing.assert_array_equal(since, [np.nan, np.nan, 0, 1, 2, 3, 0, 1])


def test_holiday_features_lookup_and_errors(tmp_path):
    s = compute_daily_scores(_weights(), _matrices())
    f = s.holiday_features([1, 1, 2], pd.to_datetime(["2024-01-01", "2024-01-13", "2024-01-15"]))
    assert f.loc[0, "national_holiday"] == pytest.approx(1.0) and f.loc[0, "days_until_national_holiday"] == 0
    assert f.loc[1, "days_since_national_holiday"] == 12
    with pytest.raises(KeyError):
        s.holiday_features([99], pd.to_datetime(["2024-01-01"]))
    with pytest.raises(ValueError, match="outside"):
        s.holiday_features([1], pd.to_datetime(["2030-01-01"]))
    path = tmp_path / "scores.npz"
    s.save(path)
    again = DailyScores.load(path)
    np.testing.assert_array_equal(again.national, s.national)


def test_calendar_matrices_roundtrip(tmp_path):
    m = _matrices()
    m.save(tmp_path / "m.npz")
    back = CalendarMatrices.load(tmp_path / "m.npz")
    np.testing.assert_array_equal(back.school, m.school)
    assert list(back.regions) == ["A", "B", "C"]


def test_enhance_drops_closed_and_predicted_rows():
    s = compute_daily_scores(_weights(), _matrices())
    dates = pd.date_range("2024-01-01", periods=4)
    crowd = pd.DataFrame(
        {"park_id": 1, "date": dates, "status": ["open", "closed", "open", "open"], "crowd_percent": [10, np.nan, 30, 40],
         "predicted": [False, False, True, False], "opens": "10:00", "closes": "18:00", "events": pd.array([None] * 4, dtype="string")}
    )  # fmt: skip
    out = enhance_calendar(crowd, s)
    assert out["date"].tolist() == [dates[0], dates[3]]
    assert {"national_holiday", "school_holiday", "days_until_school_holiday"} <= set(out.columns)
