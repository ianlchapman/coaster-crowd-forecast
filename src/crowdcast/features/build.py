"""Assemble the model-ready feature frame from the enhanced crowd calendar, park tables and weather features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from crowdcast.features.calendar import add_calendar_features
from crowdcast.features.columns import CATEGORICAL
from crowdcast.features.history import add_drift, add_prior_year
from crowdcast.features.hours_events import add_events, add_hours

PARK_COLUMNS = ["park_id", "company_name", "country_iso2", "tier", "latitude", "longitude", "first_year", "first_month"]
HOLIDAY_DISTANCE_CAP = 180  # days; beyond this "far from a holiday" is all the same


def park_table(parks: pd.DataFrame, enriched: pd.DataFrame) -> pd.DataFrame:
    """Park attributes: master table (``id`` -> ``park_id``) joined with country code and tier from the enriched table."""
    p = parks.rename(columns={"id": "park_id"}).merge(
        enriched[["park_id", "country_iso2", "tier"]], on="park_id", how="left"
    )
    return p[PARK_COLUMNS]


def add_holiday_distance_logs(df: pd.DataFrame) -> pd.DataFrame:
    """``days_until/since_*`` (whole days, capped) -> ``*_log`` features."""
    for col in [c for c in df.columns if c.startswith("days_") and not c.endswith("_log")]:
        df[col + "_log"] = np.log1p(df[col].clip(upper=HOLIDAY_DISTANCE_CAP))
    return df


def build_feature_frame(
    enhanced: pd.DataFrame,
    parks: pd.DataFrame,
    weather_features: pd.DataFrame | None = None,
    with_history: bool = True,
) -> pd.DataFrame:
    """One row per open, labelled park-day with every model feature.

    Parameters
    ----------
    enhanced
        Crowd calendar with holiday columns (``national_holiday``, ``school_holiday``, ``days_until/since_*``); see
        :func:`crowdcast.scoring.enhance.enhance_calendar`.
    parks
        Output of :func:`park_table`.
    weather_features
        Optional ``park_id, date, wx_*`` table (:func:`crowdcast.weather.features.derive_weather_features`).
    with_history
        Also add prior-year and drift features (they need the labels of earlier days in the same frame).
    """
    df = enhanced
    if "predicted" in df.columns:
        df = df[~df["predicted"].astype(str).str.lower().eq("true")]
    df = df[(df["status"] == "open") & df["crowd_percent"].notna()].drop(
        columns=[c for c in ("status", "predicted") if c in df.columns]
    )
    df = df.merge(parks, on="park_id", how="left").sort_values(["park_id", "date"]).reset_index(drop=True)
    add_calendar_features(df)
    add_hours(df)
    add_events(df)
    add_holiday_distance_logs(df)
    if with_history:
        add_prior_year(df)
    df = df.drop(columns=["events", "opens", "closes"])
    for col in CATEGORICAL:
        df[col] = df[col].astype("category")
    df["park_cat"] = df["park_id"].astype("category")
    if weather_features is not None:
        df = df.merge(weather_features, on=["park_id", "date"], how="left")
    if with_history:
        df = add_drift(df)
    return df
