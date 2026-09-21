"""Geography helpers: mapping Natural Earth admin-1 features to the region level used here, and great-circle distance."""

from __future__ import annotations

from typing import Any

import numpy as np

EARTH_RADIUS_KM = 6371.0088

# Overseas territories that are not part of the mainland market
FR_DROP = {"FR-GUF", "FR-MTQ", "FR-GUA", "FR-LRE", "FR-MAY"}
ES_MAP = {"ES.MU": "ES-MC", "ES.PM": "ES-IB", "ES.LO": "ES-RI", "ES.NA": "ES-NC", "ES.CT": "ES-CT", "ES.VC": "ES-VC"}
GB_MAP = {
    "England": ("GB-ENG", "England"),
    "Scotland": ("GB-SCT", "Scotland"),
    "Wales": ("GB-WLS", "Wales"),
    "Northern Ireland": ("GB-NIR", "Northern Ireland"),
}
BE_MAP = {
    "Flemish": ("BE-VLG", "Flanders"),
    "Walloon": ("BE-WAL", "Wallonia"),
    "Capital Region": ("BE-BRU", "Brussels-Capital"),
}


def feature_target(props: dict[str, Any], countries: set[str], split: set[str]) -> tuple[str, str | None] | None:
    """Map a Natural Earth admin-1 feature to ``(region_code, name)`` at the target level, or ``None`` to drop it."""
    cc = props["iso_a2"]
    if cc not in countries:
        return None
    if cc not in split:
        return (cc, None)
    iso = (props["iso_3166_2"] or "").strip()
    if cc == "FR":
        code = (props["region_cod"] or "").strip()
        return None if code in FR_DROP else (code, props["region"])
    if cc == "ES":
        raw = props["region_cod"]
        if props["name"] == "Ceuta":
            return ("ES-CE", "Ceuta")
        if props["name"] == "Melilla":
            return ("ES-ML", "Melilla")
        return (ES_MAP.get(raw, raw.replace(".", "-")), props["region"])
    if cc == "IT":
        return (props["region_cod"], props["region"])
    if cc == "GB":
        return GB_MAP[props["geonunit"]]
    if cc == "BE":
        return BE_MAP[props["region"]]
    if cc in ("AU", "CN"):
        return None if "~" in iso else (iso, props["name"])
    if cc == "NL":
        return None if iso.startswith("NL-BQ") else (iso, props["name"])
    return (iso, props["name"])


def haversine_km(lat1: Any, lon1: Any, lat2: Any, lon2: Any) -> Any:
    """Great-circle distance in km; arguments broadcast (degrees)."""
    a, b = np.radians(lat1), np.radians(lat2)
    dphi, dlam = b - a, np.radians(lon2) - np.radians(lon1)
    h = np.sin(dphi / 2) ** 2 + np.cos(a) * np.cos(b) * np.sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(h))
