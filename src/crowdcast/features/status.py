"""Is-open lookup for future dates and minute-of-day -> "HH:MM" formatting for forecast output.

Closed days are dropped from the crowd-percent training frame entirely (``scoring/enhance.py``), so this
reads the raw calendar (which still has them) separately. Mirrors the hours-imputation fallback chain in
``features/future.py::_add_hours_and_events``: same weekday last year, else the park+weekday open rate
over recent history, else the park's all-time open rate.
"""

from __future__ import annotations

import pandas as pd

from crowdcast.features.history import prior_year_frame

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


def add_is_open_prior_year(rows: pd.DataFrame, status_calendar: pd.DataFrame, last: pd.Timestamp) -> pd.DataFrame:
    """``py_is_open_same_wd``/``py_is_open_wd_mean3`` for future rows, computed the same way
    :func:`crowdcast.features.status_build.build_status_frame` computes them for training -- needed to
    feed :class:`crowdcast.models.status.StatusModel` at prediction time."""
    cal = status_calendar.loc[status_calendar["date"] <= last]
    labels = cal.set_index(["park_id", "date"])["status"].eq("open").astype(float)
    py = prior_year_frame(labels, rows["park_id"], rows["date"])
    rows["py_is_open_same_wd"] = py["py_same_wd"].to_numpy()
    rows["py_is_open_wd_mean3"] = py["py_wd_mean3"].to_numpy()
    return rows


def known_schedule(rows: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """The schedule the scraped calendar already shows for each ``rows`` park-day (same index as ``rows``).

    Columns: ``known`` (the calendar settles this day), ``is_open``, ``open_min``, ``close_min``.
    A day is known when the calendar lists it as ``closed``, or as ``open`` with both an opening and a closing time.
    Anything else (no row, ``unknown``/``no_data``, open without hours) is not known and is left to the model.
    """
    cal = calendar.set_index(["park_id", "date"])[["status", "opens", "closes"]].reindex(
        pd.MultiIndex.from_arrays([rows["park_id"], rows["date"]])
    )
    opens = pd.to_datetime(cal["opens"], format="%H:%M", errors="coerce")
    closes = pd.to_datetime(cal["closes"], format="%H:%M", errors="coerce")
    open_with_hours = cal["status"].eq("open") & opens.notna() & closes.notna()
    closed = cal["status"].eq("closed")
    return pd.DataFrame(
        {
            "known": (open_with_hours | closed).to_numpy(),
            "is_open": open_with_hours.to_numpy(),
            "open_min": (opens.dt.hour * 60 + opens.dt.minute).to_numpy(),
            "close_min": (closes.dt.hour * 60 + closes.dt.minute).to_numpy(),
        },
        index=rows.index,
    )


def apply_known_hours(rows: pd.DataFrame, known: pd.DataFrame) -> pd.DataFrame:
    """Give the crowd model the scraped hours for days the calendar already settles as open.

    ``build_future_rows`` fills ``open_min``/``close_min``/``open_hours`` with last year's value (or a median), which is
    all that is known for most future days. Where the calendar has the real schedule, the model should see that instead
    -- it is trained on each day's actual recorded hours.
    """
    sched = (known["known"] & known["is_open"]).to_numpy()
    rows = rows.copy()
    rows.loc[sched, "open_min"] = known.loc[sched, "open_min"].to_numpy()
    rows.loc[sched, "close_min"] = known.loc[sched, "close_min"].to_numpy()
    hours = (rows["close_min"] - rows["open_min"]) / 60
    hours = hours.where(hours >= 0, hours + 24)  # closes after midnight
    rows.loc[sched, "open_hours"] = hours[sched]
    return rows


def minutes_to_hhmm(minutes: pd.Series) -> pd.Series:
    """Minute-of-day -> ``"HH:MM"`` string, ``pd.NA`` where ``minutes`` is NaN."""
    m = minutes.round().astype("Int64") % (24 * 60)
    hh = (m // 60).astype("string").str.zfill(2)
    mm = (m % 60).astype("string").str.zfill(2)
    return (hh + ":" + mm).where(minutes.notna())
