"""Project paths and run settings.

All data lives under one directory (``data/`` in the repo by default, or ``$CROWDCAST_DATA_DIR``). Nothing under it is committed.

    data/raw/loader/     crowd-calendar.csv, park-events.csv, parks.csv   (output of the separate queue-times loader)
    data/raw/geo/        cities1000.txt, ne_admin1.geojson                 (downloaded, see docs/DATA.md)
    data/interim/        calendar source tables, weather caches
    data/processed/      park weights, holiday scores, enhanced calendar, weather, trained models
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DIR = REPO_ROOT / "reference"


def default_data_dir() -> Path:
    return Path(os.environ.get("CROWDCAST_DATA_DIR", REPO_ROOT / "data")).expanduser().resolve()


@dataclass(frozen=True)
class Paths:
    """Every file location the pipeline reads or writes, derived from one root."""

    root: Path = field(default_factory=default_data_dir)

    @property
    def raw_loader(self) -> Path:
        return self.root / "raw" / "loader"

    @property
    def raw_geo(self) -> Path:
        return self.root / "raw" / "geo"

    @property
    def interim(self) -> Path:
        return self.root / "interim"

    @property
    def processed(self) -> Path:
        return self.root / "processed"

    @property
    def models(self) -> Path:
        return self.processed / "models"

    # --- named files -------------------------------------------------------------------------------------------
    @property
    def crowd_calendar(self) -> Path:
        return self.raw_loader / "crowd-calendar.csv"

    @property
    def parks_csv(self) -> Path:
        return self.raw_loader / "parks.csv"

    @property
    def events_csv(self) -> Path:
        return self.raw_loader / "park-events.csv"

    @property
    def enhanced_calendar(self) -> Path:
        return self.processed / "enhanced-crowd-calendar.csv"

    @property
    def weather_archive(self) -> Path:
        """One csv per park (raw Open-Meteo archive)."""
        return self.interim / "weather" / "archive"

    @property
    def weather_forecast_cache(self) -> Path:
        return self.interim / "weather" / "forecast"

    @property
    def weather_previous_runs(self) -> Path:
        return self.interim / "weather" / "previous_runs"

    @property
    def parks_dir(self) -> Path:
        return self.processed / "parks"

    @property
    def parks_enriched(self) -> Path:
        return self.parks_dir / "parks_enriched.csv"

    @property
    def regions(self) -> Path:
        return self.parks_dir / "regions.csv"

    @property
    def candidate_markets(self) -> Path:
        return self.parks_dir / "candidate_markets.csv"

    @property
    def calendars_dir(self) -> Path:
        return self.interim / "calendars"

    @property
    def calendar_matrices(self) -> Path:
        return self.processed / "holidays" / "calendar_matrices.npz"

    @property
    def park_region_weights(self) -> Path:
        return self.processed / "holidays" / "park_region_weights.csv"

    @property
    def park_strength(self) -> Path:
        return self.processed / "holidays" / "park_strength.csv"

    @property
    def daily_scores(self) -> Path:
        return self.processed / "holidays" / "park_daily_scores.npz"

    @property
    def model_file(self) -> Path:
        return self.models / "crowd_model.joblib"

    @property
    def status_model_file(self) -> Path:
        return self.models / "status_model.joblib"

    def ensure(self, *dirs: Path) -> None:
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
