"""Historical daily weather (Open-Meteo archive) per park, cached one file per park so interrupted runs resume."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

from crowdcast.weather.client import ARCHIVE_URL, DAILY_VARIABLES, OpenMeteoClient
from crowdcast.weather.grid import add_cell

log = logging.getLogger(__name__)
ARCHIVE_LAG_DAYS = 3  # the archive trails today by a few days
PAD_BEFORE_DAYS = 7  # so lagged features exist at the start of a park's history
PAD_AFTER_DAYS = 3


def fetch_archive(
    parks: pd.DataFrame,
    spans: pd.DataFrame,
    cache_dir: Path,
    client: OpenMeteoClient | None = None,
    refresh: bool = False,
    pause: float = 1.5,
) -> None:
    """Download weather for every park that is not cached yet.

    ``parks``: ``park_id, latitude, longitude``. ``spans``: index ``park_id`` with ``min``/``max`` labelled dates.
    """
    client = client or OpenMeteoClient()
    cache_dir.mkdir(parents=True, exist_ok=True)
    table = add_cell(parks.merge(spans, left_on="park_id", right_index=True))
    if not refresh:
        table = table[~table["park_id"].map(lambda i: (cache_dir / f"park_{i}.csv").exists())]
    today = pd.Timestamp.today().normalize()
    for n, (cell, grp) in enumerate(table.groupby("cell"), 1):
        start = (pd.Timestamp(grp["min"].min()) - pd.Timedelta(days=PAD_BEFORE_DAYS)).strftime("%Y-%m-%d")
        end = min(
            pd.Timestamp(grp["max"].max()) + pd.Timedelta(days=PAD_AFTER_DAYS),
            today - pd.Timedelta(days=ARCHIVE_LAG_DAYS),
        )
        log.info("[%d] cell %s parks %s", n, cell, list(grp["park_id"]))
        params = {
            "latitude": f"{grp['latitude'].mean():.4f}",
            "longitude": f"{grp['longitude'].mean():.4f}",
            "start_date": start,
            "end_date": end.strftime("%Y-%m-%d"),
            "daily": ",".join(DAILY_VARIABLES),
            "timezone": "auto",
        }
        daily = pd.DataFrame(client.get_json(ARCHIVE_URL, params)["daily"]).rename(columns={"time": "date"})
        for pid in grp["park_id"]:
            daily.assign(park_id=pid)[["park_id", *daily.columns]].to_csv(cache_dir / f"park_{pid}.csv", index=False)
        time.sleep(pause)


def load_archive(cache_dir: Path) -> pd.DataFrame:
    """All cached parks as one table: ``park_id, date`` + raw daily variables."""
    files = sorted(cache_dir.glob("park_*.csv"))
    if not files:
        raise FileNotFoundError(f"no cached weather in {cache_dir}; run `crowdcast weather-fetch` first")
    out = pd.concat([pd.read_csv(f, parse_dates=["date"]) for f in files], ignore_index=True)
    return out.drop(columns=["rain_sum", "shortwave_radiation_sum"], errors="ignore")
