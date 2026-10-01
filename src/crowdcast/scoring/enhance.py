"""Add the holiday columns to the raw crowd calendar."""

from __future__ import annotations

import logging

import pandas as pd

from crowdcast.scoring.daily import DailyScores

log = logging.getLogger(__name__)


def enhance_calendar(crowd: pd.DataFrame, scores: DailyScores) -> pd.DataFrame:
    """Drop closed and loader-predicted rows, then add ``national_holiday``, ``school_holiday`` and ``days_until/since_*``.

    Rows the loader marked ``predicted`` are estimates, not observations, so they never become training labels.
    Parks without holiday scores (e.g. permanently closed, so gone from the loader's park list while their history
    stays in the calendar) are dropped with a warning rather than failing the run.
    """
    keep = (crowd["status"] != "closed") & ~crowd["predicted"].astype(str).str.lower().eq("true")
    rows = crowd[keep]
    unscored = ~rows["park_id"].isin(scores.park_ids)
    if unscored.any():
        log.warning(
            "dropping %d rows for parks without holiday scores: %s",
            unscored.sum(),
            sorted(rows.loc[unscored, "park_id"].unique())[:10],
        )
        rows = rows[~unscored]
    rows = rows.reset_index(drop=True)
    holiday = scores.holiday_features(rows["park_id"], rows["date"])
    return pd.concat([rows, holiday], axis=1)
