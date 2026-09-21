"""Date-derived features and the COVID exclusion window."""

from __future__ import annotations

import numpy as np
import pandas as pd

COVID_START = "2020-03-01"
COVID_END = "2021-12-31"


def in_covid(dates: pd.Series) -> pd.Series:
    """True for dates in the COVID disruption window (attendance there does not describe normal seasonality)."""
    return (dates >= COVID_START) & (dates <= COVID_END)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add day-of-week, month, ISO week, year, annual sin/cos, park age in months, hemisphere and the COVID flag.

    Needs columns ``date``, ``latitude``, ``first_year``, ``first_month``.
    """
    dt = df["date"].dt
    df["dow"] = dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["month"] = dt.month
    df["week"] = dt.isocalendar().week.astype(int).values
    df["year"] = dt.year
    doy = dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["months_open"] = (df["year"] - df["first_year"]) * 12 + (df["month"] - df["first_month"])
    df["southern"] = (df["latitude"] < 0).astype(int)
    df["covid"] = in_covid(df["date"]).astype(int)
    return df
