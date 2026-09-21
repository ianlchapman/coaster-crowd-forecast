"""Readers for the three files produced by the queue-times loader (see docs/DATA.md for the contract)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from crowdcast.data.schema import CROWD_CALENDAR, EVENTS, PARKS, validate


def load_crowd_calendar(path: str | Path) -> pd.DataFrame:
    """Daily crowd table: one row per park-day with status, ``crowd_percent`` (0-100, NaN if unknown), hours and event text."""
    df = pd.read_csv(path, dtype={"events": "string", "opens": "string", "closes": "string"}, parse_dates=["date"])
    if df["predicted"].dtype == object:
        df["predicted"] = df["predicted"].astype(str).str.lower().eq("true")
    return validate(df, CROWD_CALENDAR)


def load_parks(path: str | Path) -> pd.DataFrame:
    """Park master table (``id`` is the park key; renamed to ``park_id`` by downstream code)."""
    return validate(pd.read_csv(path), PARKS)


def load_events(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return validate(df, EVENTS)
