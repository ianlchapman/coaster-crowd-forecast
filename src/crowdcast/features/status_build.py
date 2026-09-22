"""Feature frame for the is_open / opening-hours model.

Unlike :func:`crowdcast.features.build.build_feature_frame` (crowd_percent, open days only), this keeps
every open-or-closed park-day and adds year-on-year trend features for status and hours: same weekday
one year earlier, plus the 3-week-window mean around it (51/52/53 weeks back), via the same
:func:`crowdcast.features.history.prior_year_frame` the crowd model uses for ``py_same_wd``/``py_wd_mean3``.
The ~364-day lookback is far longer than any realistic held-out/forecast window, so unlike drift it needs
no ``asof`` leakage guard: build once over the full history, then split by date for training/evaluation.
"""

from __future__ import annotations

import pandas as pd

from crowdcast.features.build import add_holiday_distance_logs
from crowdcast.features.calendar import add_calendar_features
from crowdcast.features.columns import CALENDAR, CATEGORICAL, EVENTS, HOLIDAY, PARK
from crowdcast.features.history import prior_year_frame
from crowdcast.features.hours_events import add_events
from crowdcast.scoring.daily import DailyScores

STATUS_FEATURES = [*CALENDAR, *HOLIDAY, *PARK, "park_cat", "py_is_open_same_wd", "py_is_open_wd_mean3"]
HOURS_FEATURES = [*STATUS_FEATURES, "py_open_min", "py_open_min_wd_mean3", "py_close_min", "py_close_min_wd_mean3"]
#: adds the event-day flags (known on the day itself, not ahead of time -- see build_future_rows's
#: "copied from date - 364" note) for the closing-category classifier, which leans on "is this an event day".
CATEGORY_FEATURES = [*HOURS_FEATURES, *EVENTS]


def _add_prior_year(df: pd.DataFrame, value_col: str, same_wd_col: str, wd_mean3_col: str) -> None:
    labels = df.set_index(["park_id", "date"])[value_col].astype(float)
    py = prior_year_frame(labels, df["park_id"], df["date"])
    df[same_wd_col] = py["py_same_wd"].to_numpy()
    df[wd_mean3_col] = py["py_wd_mean3"].to_numpy()


def build_status_frame(raw_calendar: pd.DataFrame, parks: pd.DataFrame, scores: DailyScores) -> pd.DataFrame:
    """``raw_calendar``: unfiltered ``park_id, date, status, opens, closes`` (open + closed, incl. from the raw loader).

    ``parks``: :func:`crowdcast.features.build.park_table` output. ``scores``: holiday scores for the same parks.
    """
    df = raw_calendar[raw_calendar["status"].isin(["open", "closed"])].copy()
    df["is_open"] = df["status"].eq("open")
    df = df.merge(parks, on="park_id", how="left").sort_values(["park_id", "date"]).reset_index(drop=True)
    add_calendar_features(df)

    holiday = scores.holiday_features(df["park_id"], df["date"])
    for col in holiday.columns:
        df[col] = holiday[col].to_numpy()
    add_holiday_distance_logs(df)
    df = df.drop(columns=[c for c in holiday.columns if c.startswith("days_") and not c.endswith("_log")])

    opens = pd.to_datetime(df["opens"], format="%H:%M", errors="coerce")
    closes = pd.to_datetime(df["closes"], format="%H:%M", errors="coerce")
    df["open_min"] = opens.dt.hour * 60 + opens.dt.minute
    df["close_min"] = closes.dt.hour * 60 + closes.dt.minute

    _add_prior_year(df, "is_open", "py_is_open_same_wd", "py_is_open_wd_mean3")
    _add_prior_year(df, "open_min", "py_open_min", "py_open_min_wd_mean3")
    _add_prior_year(df, "close_min", "py_close_min", "py_close_min_wd_mean3")
    add_events(df)

    for col in CATEGORICAL:
        df[col] = df[col].astype("category")
    df["park_cat"] = df["park_id"].astype("category")
    return df.drop(columns=["events"], errors="ignore")
