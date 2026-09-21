"""Prior over each park's source markets.

    prior_weight(park, region) ~ boost * region_pop * (pop-weighted mean over the region's places of decay(distance park -> place)) * affluence

* ``decay = exp(-d / L)``; for intercontinental parks ``0.5 exp(-d / L) + 0.5 / (1 + d / L)^3`` (heavy tail).
* ``L`` by tier: local 150, regional 600, continental 2500, intercontinental 6000 km (per-park override in ``park_tiers.csv``).
* Own-country boost: x3 local, x6 regional, x10 continental, x6 intercontinental.
* ``affluence``: crude 4-step travel-propensity of the source country (``reference/countries.csv``).
* Keep regions up to 99% cumulative weight (max 150) and renormalise to sum 1.

Every number here is an assumption. The prior only shapes the starting point; the weight-fitting step blends it with each park's own data.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from crowdcast.parks.geo import haversine_km

DECAY_KM = {"local": 150, "regional": 600, "continental": 2500, "intercontinental": 6000}
OWN_COUNTRY_BOOST = {"local": 3.0, "regional": 6.0, "continental": 10.0, "intercontinental": 6.0}
CUMULATIVE_MASS = 0.99
MAX_REGIONS = 150


def candidate_markets(
    parks: pd.DataFrame, regions: pd.DataFrame, places: pd.DataFrame, countries: pd.DataFrame, tiers: pd.DataFrame
) -> pd.DataFrame:
    """``park_id, region_code, distance_km, prior_weight`` (weights sum to 1 per park)."""
    codes = list(regions["region_code"])
    index = {c: i for i, c in enumerate(codes)}
    pop = regions["population"].to_numpy(dtype=float)
    rlat, rlon = regions["latitude"].to_numpy(), regions["longitude"].to_numpy()
    rcountry = regions["country"].to_numpy()
    affluence = (
        regions["country"].map(dict(zip(countries["iso2"], countries["affluence"], strict=True))).to_numpy(dtype=float)
    )
    p_lat, p_lon, p_pop = places["lat"].to_numpy(), places["lon"].to_numpy(), places["pop"].to_numpy(dtype=float)
    p_region = places["region_code"].map(index).to_numpy()
    place_total = np.bincount(p_region, weights=p_pop, minlength=len(codes))
    override = tiers.set_index("park_id")["decay_km_override"].dropna().to_dict()

    rows: list[tuple[Any, ...]] = []
    park_rows: Iterable[Any] = parks.itertuples(index=False)  # dynamic namedtuples
    for park in park_rows:
        tier = park.tier
        scale = override.get(park.park_id) or DECAY_KM[tier]
        dist = haversine_km(park.latitude, park.longitude, p_lat, p_lon)
        decay = np.exp(-dist / scale)
        if tier == "intercontinental":
            decay = 0.5 * decay + 0.5 / (1 + dist / scale) ** 3
        weighted = np.bincount(p_region, weights=p_pop * decay, minlength=len(codes))
        w = pop * weighted / np.maximum(place_total, 1e-9) * affluence
        w = w * np.where(rcountry == park.country_iso2, OWN_COUNTRY_BOOST[tier], 1.0)
        w = w / w.sum()
        order = np.argsort(-w)
        n = min(MAX_REGIONS, int(np.searchsorted(np.cumsum(w[order]), CUMULATIVE_MASS)) + 1)
        keep = order[:n]
        wk = np.round(w[keep] / w[keep].sum(), 8)
        wk[0] += 1 - wk.sum()
        dk = haversine_km(park.latitude, park.longitude, rlat[keep], rlon[keep])
        rows.extend(
            (park.park_id, codes[k], round(float(d), 1), float(x)) for k, x, d in zip(keep, wk, dk, strict=True)
        )
    return pd.DataFrame(rows, columns=["park_id", "region_code", "distance_km", "prior_weight"])
