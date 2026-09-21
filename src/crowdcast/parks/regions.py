"""Build the region table (admin-1 for large countries, country level otherwise) and assign GeoNames places to regions.

Population comes from the hand-entered reference tables (GeoNames city sums double-count boroughs); GeoNames places are used for
population-weighted centroids and to spread a region's population over its geography when computing distances to a park.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pycountry
from shapely import STRtree
from shapely.geometry import Point, shape

from crowdcast.parks.geo import feature_target

log = logging.getLogger(__name__)

NAME_OVERRIDE = {
    "HK": "Hong Kong",
    "MO": "Macao",
    "TW": "Taiwan",
    "KR": "South Korea",
    "RU": "Russia",
    "TR": "Turkey",
    "VN": "Vietnam",
}
MIN_REGION_POPULATION = 1000


def load_admin1(
    geojson: Path, countries: set[str], split: set[str]
) -> tuple[dict[str, list[tuple[str, object]]], dict[str, str]]:
    """Admin-1 polygons per country as ``(region_code, geometry)``, plus the first seen name per region code."""
    by_country: dict[str, list[tuple[str, object]]] = defaultdict(list)
    names: dict[str, str] = {}
    for feature in json.loads(Path(geojson).read_text())["features"]:
        target = feature_target(feature["properties"], countries, split)
        if target is None:
            continue
        code, name = target
        by_country[feature["properties"]["iso_a2"]].append((code, shape(feature["geometry"])))
        if name:
            names.setdefault(code, name)
    return by_country, names


def load_places(path: Path, countries: set[str]) -> pd.DataFrame:
    """GeoNames ``cities1000`` places in the given countries: ``country, lat, lon, pop``."""
    cols = {8: "country", 4: "lat", 5: "lon", 14: "pop"}
    df = pd.read_csv(
        path, sep="\t", header=None, usecols=list(cols), names=None, dtype=str, quoting=3, keep_default_na=False
    )
    df = df.rename(columns=cols)
    df = df[df["country"].isin(countries)].copy()
    df["lat"], df["lon"] = df["lat"].astype(float), df["lon"].astype(float)
    df["pop"] = pd.to_numeric(df["pop"], errors="coerce").fillna(0).astype(int)
    return df.reset_index(drop=True)


def assign_places(
    places: pd.DataFrame, admin1: dict[str, list[tuple[str, object]]], split: set[str], order: list[str]
) -> pd.DataFrame:
    """Give every place a region code: its own country for unsplit countries, else the admin-1 polygon it falls in (nearest as fallback)."""
    out = []
    for cc in order:
        sub = places[places["country"] == cc]
        if sub.empty:
            log.warning("no places for %s", cc)
            continue
        if cc not in split:
            out.append(sub.assign(region_code=cc))
            continue
        geoms = admin1[cc]
        tree = STRtree([g for _, g in geoms])
        points = [Point(lon, lat) for lat, lon in zip(sub["lat"], sub["lon"], strict=True)]
        hits = tree.query(points, predicate="within")
        first: dict[int, int] = {}
        for pi, gi in zip(hits[0], hits[1], strict=True):
            first.setdefault(int(pi), int(gi))
        region = [geoms[first.get(i, int(tree.nearest(pt)))][0] for i, pt in enumerate(points)]
        out.append(sub.assign(region_code=region))
    return pd.concat(out, ignore_index=True)


def _centroid(lat: np.ndarray, lon: np.ndarray, weight: np.ndarray) -> tuple[float, float]:
    la, lo = np.radians(lat), np.radians(lon)
    x, y, z = (
        (np.cos(la) * np.cos(lo) * weight).sum(),
        (np.cos(la) * np.sin(lo) * weight).sum(),
        (np.sin(la) * weight).sum(),
    )
    return float(np.degrees(np.arctan2(z, np.hypot(x, y)))), float(np.degrees(np.arctan2(y, x)))


def build_regions(
    geojson: Path, places_file: Path, countries: pd.DataFrame, region_population: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(regions, places_assigned)``.

    ``regions``: ``region_code, country, name, latitude, longitude, population`` (population-weighted centroid).
    ``places_assigned``: ``region_code, lat, lon, pop`` for the kept regions (input to the market prior).
    """
    iso = list(countries["iso2"])
    split = set(countries.loc[countries["split_admin1"], "iso2"])
    country_pop = dict(zip(countries["iso2"], countries["population_millions"], strict=True))
    admin1, names = load_admin1(geojson, set(iso), split)
    placed = assign_places(load_places(places_file, set(iso)), admin1, split, iso)
    placed["w"] = placed["pop"].clip(lower=1)

    reg_pop_k = dict(zip(region_population["region_code"], region_population["population_thousands"], strict=True))
    rows = []
    for code, g in placed.groupby("region_code"):
        cc = str(code)[:2]
        lat, lon = _centroid(g["lat"].to_numpy(), g["lon"].to_numpy(), g["w"].to_numpy())
        if cc in split:
            total_k = sum(v for k, v in reg_pop_k.items() if k[:2] == cc)
            pop = reg_pop_k[str(code)] * 1e3 * country_pop[cc] * 1e6 / (total_k * 1e3)
        else:
            pop = country_pop[cc] * 1e6
        name = names.get(str(code)) or NAME_OVERRIDE.get(str(code)) or _country_name(str(code))
        if code == cc and cc in NAME_OVERRIDE:
            name = NAME_OVERRIDE[cc]
        rows.append([code, cc, name, round(lat, 4), round(lon, 4), int(round(pop))])
    regions = pd.DataFrame(rows, columns=["region_code", "country", "name", "latitude", "longitude", "population"])
    regions = regions[regions["population"] >= MIN_REGION_POPULATION].reset_index(drop=True)
    kept = placed[placed["region_code"].isin(regions["region_code"])]
    return regions, kept[["region_code", "lat", "lon", "w"]].rename(columns={"w": "pop"}).reset_index(drop=True)


def _country_name(code: str) -> str:
    country = pycountry.countries.get(alpha_2=code)
    return str(country.name) if country is not None else code
