"""Group parks into ~0.25 degree grid cells so parks that share a reanalysis cell share one download."""

from __future__ import annotations

import pandas as pd

GRID_DEGREES = 0.25


def add_cell(parks: pd.DataFrame, grid: float = GRID_DEGREES) -> pd.DataFrame:
    """Add a ``cell`` key (``"<lat idx>_<lon idx>"``) to a table with ``latitude`` and ``longitude``."""
    lat = (parks["latitude"] / grid).round().astype(int).astype(str)
    lon = (parks["longitude"] / grid).round().astype(int).astype(str)
    return parks.assign(cell=lat + "_" + lon)
