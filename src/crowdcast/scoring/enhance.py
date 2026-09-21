"""Add the holiday columns to the raw crowd calendar."""

from __future__ import annotations

import pandas as pd

from crowdcast.scoring.daily import DailyScores


def enhance_calendar(crowd: pd.DataFrame, scores: DailyScores) -> pd.DataFrame:
    """Drop closed and loader-predicted rows, then add ``national_holiday``, ``school_holiday`` and ``days_until/since_*``.

    Rows the loader marked ``predicted`` are estimates, not observations, so they never become training labels.
    """
    keep = (crowd["status"] != "closed") & ~crowd["predicted"].astype(str).str.lower().eq("true")
    rows = crowd[keep].reset_index(drop=True)
    holiday = scores.holiday_features(rows["park_id"], rows["date"])
    return pd.concat([rows, holiday], axis=1)
