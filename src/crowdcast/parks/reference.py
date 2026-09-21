"""Loaders for the hand-curated tables in ``reference/`` (see reference/README.md for provenance)."""

from __future__ import annotations

import pandas as pd

from crowdcast.config import REFERENCE_DIR

TIERS = ("local", "regional", "continental", "intercontinental")


def load_tiers() -> pd.DataFrame:
    df = pd.read_csv(REFERENCE_DIR / "park_tiers.csv")
    bad = set(df["tier"]) - set(TIERS)
    if bad:
        raise ValueError(f"park_tiers.csv: unknown tiers {sorted(bad)}")
    return df


def load_coordinate_fixes() -> pd.DataFrame:
    return pd.read_csv(REFERENCE_DIR / "park_coordinate_fixes.csv")


def load_countries() -> pd.DataFrame:
    return pd.read_csv(REFERENCE_DIR / "countries.csv")


def load_region_population() -> pd.DataFrame:
    return pd.read_csv(REFERENCE_DIR / "region_population.csv")
