"""Feature rows for dates that have no labels yet (nowcast + forecast).

Pure function of already-loaded inputs (no network, no files): the caller supplies the labelled history, holiday scores and
weather. Blocks and where they come from:

    calendar / park attributes   computed from the date and the park table
    holiday scores + distances   ``DailyScores.holiday_features`` (same code path as training)
    opening hours                same weekday-aligned day last year (date - 364), else the park's median for that weekday
                                 over its last 12 labelled weeks
    events                       not known ahead of time: flags copied from date - 364, else 0
    prior-year (py_*)            labelled history at date - 357 / 364 / 371
    drift1/7/14                  rolling gap series, only where its window ends on or before the park's last label
    weather                      supplied ``wx_*`` features (forecast for future days, archive for older ones)

``asof`` pretends labels stop at that date, which lets the builder be validated against days whose real features are known.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crowdcast.features.build import HOLIDAY_DISTANCE_CAP
from crowdcast.features.calendar import add_calendar_features
from crowdcast.features.columns import DRIFT_LAGS, EVENTS, HOURS
from crowdcast.features.history import drift_series, prior_year_frame
from crowdcast.scoring.daily import DailyScores

ACTIVE_WINDOW_DAYS = 60  # a park is "active" if it has a label in this many days before the last label
HOURS_HISTORY_DAYS = 84
LAST_YEAR = pd.Timedelta(days=364)


def _active_parks(labelled: pd.DataFrame, last: pd.Timestamp) -> np.ndarray:
    return labelled.loc[labelled["date"] > last - pd.Timedelta(days=ACTIVE_WINDOW_DAYS), "park_id"].unique()


def _add_hours_and_events(rows: pd.DataFrame, labelled: pd.DataFrame, last: pd.Timestamp) -> pd.DataFrame:
    recent = labelled[labelled["date"] > last - pd.Timedelta(days=HOURS_HISTORY_DAYS)]
    by_weekday = recent.groupby(["park_id", "dow"])[HOURS].median().reset_index()
    rows = rows.merge(by_weekday, on=["park_id", "dow"], how="left")
    park_median = labelled.groupby("park_id")[HOURS].median()
    last_year = labelled.set_index(["park_id", "date"])[[*HOURS, *EVENTS]].reindex(
        pd.MultiIndex.from_arrays([rows["park_id"], rows["date"] - LAST_YEAR])
    )
    for col in HOURS:
        rows[col] = (
            pd.Series(last_year[col].to_numpy(), index=rows.index)
            .fillna(rows[col])
            .fillna(rows["park_id"].map(park_median[col]))
        )
    for col in EVENTS:
        rows[col] = pd.Series(last_year[col].to_numpy(), index=rows.index).fillna(0).astype(int)
    return rows


def _add_drift(rows: pd.DataFrame, labelled: pd.DataFrame) -> pd.DataFrame:
    gap = labelled["crowd_percent"] - labelled["py_same_wd"]
    last_label = labelled.groupby("park_id")["date"].max()
    for h in DRIFT_LAGS:
        rows[f"drift{h}"] = np.nan
    for pid, group in labelled.groupby("park_id"):
        park_last = pd.Timestamp(last_label.loc[pid])
        series = pd.Series(gap.loc[group.index].to_numpy(), index=group["date"].to_numpy())
        rolling = drift_series(series, park_last)
        mask = (rows["park_id"] == pid).to_numpy()
        for h in DRIFT_LAGS:
            end = rows.loc[mask, "date"] - pd.Timedelta(days=h)
            known = (end <= park_last).to_numpy()
            rows.loc[mask, f"drift{h}"] = np.where(known, rolling.reindex(end.to_numpy()).to_numpy(), np.nan)
    return rows


def build_future_rows(
    labelled: pd.DataFrame,
    scores: DailyScores,
    weather: pd.DataFrame,
    end: str | pd.Timestamp,
    asof: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """One row per active park and date from the day after the last label to ``end``.

    ``labelled``: feature frame from :func:`crowdcast.features.build.build_feature_frame` (must include ``py_same_wd``).
    ``weather``: ``park_id, date, wx_*, wx_source`` (``wx_source`` is 'archive'/'forecast'; dates without weather get 'none').
    Adds ``open_last_year`` (open on the same weekday a year earlier), ``days_ahead`` and ``wx_source``.
    """
    lab = labelled if asof is None else labelled[labelled["date"] <= pd.Timestamp(asof)]
    last = lab["date"].max()
    dates = pd.date_range(last + pd.Timedelta(days=1), pd.Timestamp(end))
    active = _active_parks(lab, last)
    attrs = (
        lab.sort_values("date").drop_duplicates("park_id", keep="last").set_index("park_id").loc[active]
        [["company_name", "country_iso2", "tier", "latitude", "longitude", "first_year", "first_month"]].reset_index()
    )  # fmt: skip
    rows = (
        pd.MultiIndex.from_product([active, dates], names=["park_id", "date"])
        .to_frame(index=False)
        .merge(attrs, on="park_id")
    )
    add_calendar_features(rows)
    rows["covid"] = 0

    holiday = scores.holiday_features(rows["park_id"], rows["date"])
    for col in holiday.columns:
        rows[col] = holiday[col].to_numpy()
        if col.startswith("days_"):
            rows[col + "_log"] = np.log1p(rows[col].clip(upper=HOLIDAY_DISTANCE_CAP))
    rows = rows.drop(columns=[c for c in holiday.columns if c.startswith("days_") and not c.endswith("_log")])

    rows = _add_hours_and_events(rows, lab, last)
    labels = lab.set_index(["park_id", "date"])["crowd_percent"]
    py = prior_year_frame(labels, rows["park_id"], rows["date"])
    rows["py_same_wd"], rows["py_wd_mean3"] = py["py_same_wd"].to_numpy(), py["py_wd_mean3"].to_numpy()
    rows["open_last_year"] = (
        labels.reindex(pd.MultiIndex.from_arrays([rows["park_id"], rows["date"] - LAST_YEAR]))
        .notna()
        .astype(int)
        .to_numpy()
    )
    rows = _add_drift(rows, lab)
    rows["days_ahead"] = (rows["date"] - rows["park_id"].map(lab.groupby("park_id")["date"].max())).dt.days

    rows = rows.merge(weather, on=["park_id", "date"], how="left")
    rows["wx_source"] = rows["wx_source"].fillna("none")
    for col in ("company_name", "country_iso2", "tier"):
        rows[col] = rows[col].astype(labelled[col].dtype)
    rows["park_cat"] = pd.Categorical(rows["park_id"], categories=labelled["park_cat"].cat.categories)
    return rows.drop(columns=["first_year", "first_month"])
