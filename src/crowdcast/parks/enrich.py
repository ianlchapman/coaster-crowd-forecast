"""Attach each park to a region (point-in-polygon on Natural Earth admin-1), a tier and corrected coordinates."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
from shapely import STRtree
from shapely.geometry import Point, shape

from crowdcast.parks.geo import feature_target

log = logging.getLogger(__name__)

#: The loader's ``parks.csv`` uses country names; this maps them to ISO codes for the sanity check against the polygon hit.
COUNTRY_NAME_TO_ISO2 = {
    "United States": "US", "Canada": "CA", "China": "CN", "France": "FR", "Germany": "DE", "United Kingdom": "GB",
    "Japan": "JP", "Belgium": "BE", "Netherlands": "NL", "South Korea": "KR", "Spain": "ES", "Australia": "AU",
    "Denmark": "DK", "Italy": "IT", "Sweden": "SE", "Hong Kong": "HK", "Austria": "AT", "Brazil": "BR", "Poland": "PL",
    "Malaysia": "MY", "Mexico": "MX", "Saudi Arabia": "SA",
}  # fmt: skip
NEAREST_TOLERANCE_NOTE = "nearest polygon used for coastal points"
COLUMNS = [
    "park_id",
    "name",
    "company_name",
    "country_iso2",
    "region_code",
    "latitude",
    "longitude",
    "first_year",
    "first_month",
    "tier",
    "tier_reason",
]


def enrich_parks(
    parks: pd.DataFrame,
    geojson: Path,
    regions: pd.DataFrame,
    tiers: pd.DataFrame,
    fixes: pd.DataFrame,
    countries: pd.DataFrame,
) -> pd.DataFrame:
    """One row per park with region, tier and (corrected) coordinates. Logs any park whose polygon country disagrees with the loader."""
    iso = set(countries["iso2"])
    split = set(countries.loc[countries["split_admin1"], "iso2"])
    features = []
    for f in json.loads(Path(geojson).read_text())["features"]:
        target = feature_target(f["properties"], iso, split)
        if target:
            features.append((f["properties"]["iso_a2"], target[0], shape(f["geometry"])))
    tree = STRtree([g for *_, g in features])
    valid_regions = set(regions["region_code"])
    fix = fixes.set_index("park_id")[["latitude", "longitude"]].to_dict("index")
    tier = tiers.set_index("park_id")

    rows: list[list[Any]] = []
    park_rows: Iterable[Any] = parks.itertuples(index=False)  # dynamic namedtuples
    for p in park_rows:
        pid = int(p.id)
        lat, lon = (
            (fix[pid]["latitude"], fix[pid]["longitude"]) if pid in fix else (float(p.latitude), float(p.longitude))
        )
        point = Point(lon, lat)
        hit = list(tree.query(point, predicate="within"))
        if not hit:
            hit = [int(tree.nearest(point))]
            log.info("park %s %s: nearest-polygon fallback", pid, p.name)
        cc, code, _ = features[hit[0]]
        if cc != COUNTRY_NAME_TO_ISO2.get(p.country):
            log.warning("park %s %s: loader country %s but coordinates fall in %s", pid, p.name, p.country, cc)
        if code not in valid_regions:
            raise ValueError(f"park {pid} maps to region {code} which is not in the region table")
        reason = str(tier.loc[pid, "reason"]) + (" [coords corrected]" if pid in fix else "")
        rows.append(
            [
                pid,
                p.name,
                p.company_name,
                cc,
                code,
                lat,
                lon,
                p.first_year,
                p.first_month,
                tier.loc[pid, "tier"],
                reason,
            ]
        )
    return pd.DataFrame(rows, columns=COLUMNS).sort_values("park_id").reset_index(drop=True)
