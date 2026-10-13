import pandas as pd

from crowdcast.features.status import add_is_open, minutes_to_hhmm


def _calendar(rows: list[tuple[int, str, str]]) -> pd.DataFrame:
    """rows of (park_id, date, status)."""
    df = pd.DataFrame(rows, columns=["park_id", "date", "status"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_same_weekday_last_year_wins_over_other_fallbacks():
    last = pd.Timestamp("2024-01-08")  # a Monday
    recent_mondays = pd.date_range("2023-10-16", "2024-01-08", freq="7D")  # all open -> weekday rate says "open"
    cal = _calendar(
        [(1, "2023-01-16", "closed")]  # exactly 364 days before the target row's date
        + [(1, str(d.date()), "open") for d in recent_mondays]
    )
    rows = pd.DataFrame({"park_id": [1], "date": [pd.Timestamp("2023-01-16") + pd.Timedelta(days=364)]})
    out = add_is_open(rows, cal, last)
    assert out["is_open"].tolist() == [False]


def test_falls_back_to_park_weekday_open_rate_when_no_last_year_row():
    last = pd.Timestamp("2024-03-01")
    mondays = pd.date_range("2023-12-04", "2024-02-26", freq="7D")  # 13 Mondays, all closed
    cal = _calendar([(1, str(d.date()), "closed") for d in mondays])
    rows = pd.DataFrame({"park_id": [1], "date": [pd.Timestamp("2024-03-04")]})  # a Monday, no matching date - 364d row
    out = add_is_open(rows, cal, last)
    assert out["is_open"].tolist() == [False]


def test_brand_new_park_with_no_history_defaults_to_open():
    last = pd.Timestamp("2024-01-01")
    cal = _calendar([(1, "2024-01-01", "open")])  # unrelated park has history; park 2 has none
    rows = pd.DataFrame({"park_id": [2], "date": [pd.Timestamp("2024-01-05")]})
    out = add_is_open(rows, cal, last)
    assert out["is_open"].tolist() == [True]


def test_minutes_to_hhmm_round_trips_and_blanks_missing():
    minutes = pd.Series([0.0, 90.0, 600.0, float("nan")])
    out = minutes_to_hhmm(minutes)
    assert out.tolist()[:3] == ["00:00", "01:30", "10:00"]
    assert pd.isna(out.iloc[3])


def _calendar_with_hours(rows: list[tuple[int, str, str, str | None, str | None]]) -> pd.DataFrame:
    """rows of (park_id, date, status, opens, closes)."""
    df = pd.DataFrame(rows, columns=["park_id", "date", "status", "opens", "closes"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_known_schedule_uses_calendar_hours_and_closures_only_when_complete():
    from crowdcast.features.status import known_schedule

    cal = _calendar_with_hours(
        [
            (1, "2024-06-01", "open", "09:30", "18:00"),
            (1, "2024-06-02", "closed", None, None),
            (1, "2024-06-03", "open", None, None),  # open but no hours: not enough to settle the day
            (1, "2024-06-04", "unknown", None, None),
        ]
    )
    rows = pd.DataFrame(
        {
            "park_id": [1] * 5,
            "date": pd.to_datetime(["2024-06-01", "2024-06-02", "2024-06-03", "2024-06-04", "2024-06-05"]),
        }
    )
    out = known_schedule(rows, cal)
    assert out["known"].tolist() == [True, True, False, False, False]  # last: no calendar row at all
    assert out["is_open"].tolist()[:2] == [True, False]
    assert out.loc[0, "open_min"] == 9 * 60 + 30 and out.loc[0, "close_min"] == 18 * 60
    assert out[["open_min", "close_min"]].iloc[1:].isna().all().all()


def test_apply_known_hours_only_touches_known_open_days():
    from crowdcast.features.status import apply_known_hours

    rows = pd.DataFrame(
        {"open_min": [600.0, 600.0, 600.0], "close_min": [1080.0, 1080.0, 1080.0], "open_hours": [8.0, 8.0, 8.0]}
    )
    known = pd.DataFrame(
        {
            "known": [True, True, False],
            "is_open": [True, False, False],
            "open_min": [480.0, float("nan"), float("nan")],
            "close_min": [60.0, float("nan"), float("nan")],  # closes 01:00, after midnight
        }
    )
    out = apply_known_hours(rows, known)
    assert out.loc[0, ["open_min", "close_min", "open_hours"]].tolist() == [480.0, 60.0, 17.0]
    assert out.loc[1:, ["open_min", "close_min", "open_hours"]].eq(rows.loc[1:]).all().all()
    assert rows.loc[0, "open_min"] == 600.0  # input not mutated
