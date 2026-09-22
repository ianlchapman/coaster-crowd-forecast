"""Is-open lookup for future dates and minute-of-day -> "HH:MM" formatting for forecast output.

Closed days are dropped from the crowd-percent training frame entirely (``scoring/enhance.py``), so this
reads the raw calendar (which still has them) separately. Mirrors the hours-imputation fallback chain in
``features/future.py::_add_hours_and_events``: same weekday last year, else the park+weekday open rate
over recent history, else the park's all-time open rate.
"""

from __future__ import annotations

import pandas as pd

LAST_YEAR = pd.Timedelta(days=364)
HISTORY_DAYS = 84


def add_is_open(rows: pd.DataFrame, status_calendar: pd.DataFrame, last: pd.Timestamp) -> pd.DataFrame:
    """Add a boolean ``is_open`` column to ``rows`` (needs ``park_id``, ``date``)."""
    cal = status_calendar.loc[status_calendar["date"] <= last, ["park_id", "date", "status"]].copy()
    cal["is_open"] = cal["status"].eq("open").astype(float)
    dow = rows["date"].dt.dayofweek

    last_year = cal.set_index(["park_id", "date"])["is_open"].reindex(
        pd.MultiIndex.from_arrays([rows["park_id"], rows["date"] - LAST_YEAR])
    )
    recent = cal[cal["date"] > last - pd.Timedelta(days=HISTORY_DAYS)]
    weekday_rate = recent.groupby(["park_id", recent["date"].dt.dayofweek])["is_open"].mean()
    weekday_fallback = weekday_rate.reindex(pd.MultiIndex.from_arrays([rows["park_id"], dow]))
    park_fallback = rows["park_id"].map(cal.groupby("park_id")["is_open"].mean())

    prob = (
        pd.Series(last_year.to_numpy(), index=rows.index)
        .fillna(pd.Series(weekday_fallback.to_numpy(), index=rows.index))
        .fillna(park_fallback)
        .fillna(1.0)  # no history at all (brand-new park): assume open
    )
    rows["is_open"] = prob.to_numpy() >= 0.5
    return rows


def minutes_to_hhmm(minutes: pd.Series) -> pd.Series:
    """Minute-of-day -> ``"HH:MM"`` string, ``pd.NA`` where ``minutes`` is NaN."""
    m = minutes.round().astype("Int64") % (24 * 60)
    hh = (m // 60).astype("string").str.zfill(2)
    mm = (m % 60).astype("string").str.zfill(2)
    return (hh + ":" + mm).where(minutes.notna())
