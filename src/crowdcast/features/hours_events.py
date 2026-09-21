"""Opening hours and event flags from the loader's free-text columns."""

from __future__ import annotations

import pandas as pd

from crowdcast.features.columns import EVENT_GROUPS

HOURS_COLS = ["open_min", "close_min", "open_hours"]
EVENT_COLS = ["has_event", *EVENT_GROUPS]


def add_hours(df: pd.DataFrame) -> pd.DataFrame:
    """``opens``/``closes`` ("HH:MM") -> minutes since midnight and open duration; missing values use the park's median."""
    opens = pd.to_datetime(df["opens"], format="%H:%M", errors="coerce")
    closes = pd.to_datetime(df["closes"], format="%H:%M", errors="coerce")
    df["open_min"] = opens.dt.hour * 60 + opens.dt.minute
    df["close_min"] = closes.dt.hour * 60 + closes.dt.minute
    df["open_hours"] = (df["close_min"] - df["open_min"]) / 60
    df.loc[df["open_hours"] < 0, "open_hours"] += 24  # closes after midnight
    for col in HOURS_COLS:
        df[col] = df[col].fillna(df.groupby("park_id")[col].transform("median"))
    return df


def add_events(df: pd.DataFrame) -> pd.DataFrame:
    """``has_event`` plus keyword-group flags from the day's event text."""
    text = df["events"].fillna("").astype(str).str.lower()
    df["has_event"] = (text != "").astype(int)
    for name, pattern in EVENT_GROUPS.items():
        df[name] = text.str.contains(pattern, regex=True).astype(int)
    return df
