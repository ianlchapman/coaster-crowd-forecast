"""Recent and future daily weather from the Open-Meteo forecast API, merged with the archive."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from crowdcast.weather.client import DAILY_VARIABLES, FORECAST_URL, OpenMeteoClient
from crowdcast.weather.grid import add_cell

PAST_DAYS = 92  # the API's maximum look-back; covers the gap between the archive's end and today
FORECAST_DAYS = 16
CACHE_HOURS = 3


def _fetch_cell(client: OpenMeteoClient, lat: float, lon: float, cache_file: Path, refresh: bool) -> dict:
    fresh = cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < CACHE_HOURS * 3600
    if fresh and not refresh:
        return json.loads(cache_file.read_text())
    params = {
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "daily": ",".join(DAILY_VARIABLES),
        "past_days": PAST_DAYS,
        "forecast_days": FORECAST_DAYS,
        "timezone": "auto",
    }
    data = client.get_json(FORECAST_URL, params)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(data))
    return data


def fetch_forecast(
    parks: pd.DataFrame, cache_dir: Path, client: OpenMeteoClient | None = None, refresh: bool = False
) -> pd.DataFrame:
    """Daily weather from ``today - 92`` to ``today + 15`` for each park in ``parks`` (``park_id, latitude, longitude``)."""
    client = client or OpenMeteoClient()
    frames: list[pd.DataFrame] = []
    for cell, grp in add_cell(parks).groupby("cell"):
        data = _fetch_cell(
            client, grp["latitude"].mean(), grp["longitude"].mean(), cache_dir / f"fc_{cell}.json", refresh
        )
        daily = pd.DataFrame(data["daily"]).rename(columns={"time": "date"})
        daily["date"] = pd.to_datetime(daily["date"])
        frames.extend(daily.assign(park_id=pid) for pid in grp["park_id"])
    return pd.concat(frames, ignore_index=True)


def combine_archive_and_forecast(archive: pd.DataFrame, forecast: pd.DataFrame) -> pd.DataFrame:
    """Prefer archive values; use forecast-API values only for dates the archive lacks. Adds ``wx_source``."""
    arch = archive[archive["park_id"].isin(forecast["park_id"].unique())].assign(wx_source="archive")
    seen = pd.MultiIndex.from_frame(arch[["park_id", "date"]])
    fresh = forecast[~pd.MultiIndex.from_frame(forecast[["park_id", "date"]]).isin(seen)].assign(wx_source="forecast")
    return pd.concat([arch, fresh], ignore_index=True).sort_values(["park_id", "date"]).reset_index(drop=True)
