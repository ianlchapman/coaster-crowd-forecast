"""Archived forecasts: what the model said N days ahead (Open-Meteo previous-runs API), aggregated to daily.

Used to score the crowd model on the weather that would actually have been available at forecast time, not the weather that
happened. The API only serves hourly ``<var>_previous_dayN``; aggregates use the same definitions as the archive variables.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from crowdcast.weather.client import PREVIOUS_RUNS_URL, OpenMeteoClient
from crowdcast.weather.grid import add_cell

LEADS = (1, 7)
HOURLY_VARIABLES = [
    "temperature_2m", "apparent_temperature", "precipitation", "snowfall", "wind_speed_10m", "wind_gusts_10m",
    "sunshine_duration", "weather_code",
]  # fmt: skip


def aggregate_hourly(hourly: dict, leads: tuple[int, ...] = LEADS) -> pd.DataFrame:
    """Hourly ``<var>_previous_day<lead>`` arrays -> one row per (date, lead) with ``wx_*`` daily aggregates."""
    h = pd.DataFrame(hourly)
    h["date"] = pd.to_datetime(h["time"].str[:10])
    out = []
    for lead in leads:
        by = h["date"]

        def v(name: str, lead: int = lead) -> pd.Series:
            return h[f"{name}_previous_day{lead}"]

        precip = v("precipitation")
        agg = pd.DataFrame(
            {
                "wx_temp_max": v("temperature_2m").groupby(by).max(),
                "wx_temp_min": v("temperature_2m").groupby(by).min(),
                "wx_feels_max": v("apparent_temperature").groupby(by).max(),
                "wx_precip_mm": precip.groupby(by).sum(min_count=1),
                "wx_snow_cm": v("snowfall").groupby(by).sum(min_count=1),
                "wx_precip_hours": (precip >= 0.1).astype(float).where(precip.notna()).groupby(by).sum(min_count=1),
                "wx_wind_max": v("wind_speed_10m").groupby(by).max(),
                "wx_gust_max": v("wind_gusts_10m").groupby(by).max(),
                "wx_sun_hours": v("sunshine_duration").groupby(by).sum(min_count=1) / 3600,
                "wx_code": v("weather_code").groupby(by).max(),  # proxy for the daily "worst" code
            }
        )
        out.append(agg.assign(lead=lead).reset_index())
    return pd.concat(out, ignore_index=True)


def fetch_previous_runs(
    parks: pd.DataFrame,
    start: str,
    end: str,
    cache_dir: Path,
    client: OpenMeteoClient | None = None,
    leads: tuple[int, ...] = LEADS,
    pause: float = 1.5,
) -> pd.DataFrame:
    """Archived-forecast daily weather per park for ``start..end`` (one request per grid cell, cached per cell)."""
    client = client or OpenMeteoClient()
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    for cell, grp in add_cell(parks).groupby("cell"):
        cache = cache_dir / f"cell_{cell}.csv"
        if not cache.exists():
            hourly = ",".join(f"{v}_previous_day{lead}" for lead in leads for v in HOURLY_VARIABLES)
            params = {
                "latitude": f"{grp['latitude'].mean():.4f}",
                "longitude": f"{grp['longitude'].mean():.4f}",
                "start_date": start,
                "end_date": end,
                "hourly": hourly,
                "timezone": "auto",
            }
            aggregate_hourly(client.get_json(PREVIOUS_RUNS_URL, params)["hourly"], leads).to_csv(cache, index=False)
            time.sleep(pause)
        cell_frame = pd.read_csv(cache, parse_dates=["date"])
        frames.extend(cell_frame.assign(park_id=pid) for pid in grp["park_id"])
    return pd.concat(frames, ignore_index=True)
