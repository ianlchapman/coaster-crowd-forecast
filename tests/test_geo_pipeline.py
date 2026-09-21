"""Regions, park enrichment and calendar merge on tiny hand-made inputs (no real geodata needed)."""

import json

import numpy as np
import pandas as pd
import pycountry
import pytest

from crowdcast.calendars.merge import merge_calendars
from crowdcast.parks.enrich import enrich_parks
from crowdcast.parks.regions import build_regions


def _square(lon0, lat0, lon1, lat1):
    return {"type": "Polygon", "coordinates": [[[lon0, lat0], [lon1, lat0], [lon1, lat1], [lon0, lat1], [lon0, lat0]]]}


@pytest.fixture()
def geodata(tmp_path):
    def feat(cc, iso, name, geom):
        return {
            "type": "Feature",
            "geometry": geom,
            "properties": {
                "iso_a2": cc,
                "iso_3166_2": iso,
                "name": name,
                "region_cod": iso,
                "region": name,
                "geonunit": name,
            },
        }

    gj = {"type": "FeatureCollection", "features": [
        feat("DE", "DE-BY", "Bavaria", _square(10, 47, 14, 50)),
        feat("DE", "DE-BW", "Baden-Wurttemberg", _square(7, 47, 10, 50)),
        feat("NL", "NL-NH", "North Holland", _square(3, 50, 7, 53)),
    ]}  # fmt: skip
    geojson = tmp_path / "admin1.geojson"
    geojson.write_text(json.dumps(gj))
    rows = []
    for lat, lon, cc, pop in [
        (48.1, 11.5, "DE", 1_500_000),
        (48.8, 9.2, "DE", 600_000),
        (49.5, 8.0, "DE", 100_000),
        (52.4, 4.9, "NL", 800_000),
        (60.0, 30.0, "NL", 5),
    ]:
        fields = [""] * 19
        fields[4], fields[5], fields[8], fields[14] = str(lat), str(lon), cc, str(pop)
        rows.append("\t".join(fields))
    places = tmp_path / "cities.txt"
    places.write_text("\n".join(rows))
    countries = pd.DataFrame(
        {
            "iso2": ["DE", "NL"],
            "population_millions": [83.5, 17.9],
            "split_admin1": [True, False],
            "affluence": [1.0, 1.0],
        }
    )
    region_pop = pd.DataFrame({"region_code": ["DE-BY", "DE-BW"], "population_thousands": [13370, 11280]})
    return geojson, places, countries, region_pop


def test_regions_use_admin1_for_split_countries_and_rescale_population(geodata):
    geojson, places, countries, region_pop = geodata
    regions, placed = build_regions(geojson, places, countries, region_pop)
    assert regions["region_code"].tolist() == ["DE-BW", "DE-BY", "NL"]
    de = regions[regions["country"] == "DE"]["population"].sum()
    assert de == pytest.approx(83.5e6, rel=1e-6)  # admin-1 estimates are rescaled to the country total
    assert regions.set_index("region_code").loc["NL", "population"] == 17_900_000
    by = regions.set_index("region_code").loc["DE-BY"]
    assert 47 < by["latitude"] < 50 and 10 < by["longitude"] < 14  # centroid inside its polygon
    assert set(placed["region_code"]) == {"DE-BW", "DE-BY", "NL"}
    assert (
        placed.loc[placed["region_code"] == "DE-BW", "pop"].sum() == 700_000 - 100_000 + 100_000
    )  # the Mannheim-area place falls in BW


def test_parks_map_to_regions_with_nearest_fallback_and_coordinate_fixes(geodata):
    geojson, places, countries, region_pop = geodata
    regions, _ = build_regions(geojson, places, countries, region_pop)
    parks = pd.DataFrame(
        {"id": [1, 2, 3], "name": ["Inside", "Coast", "Wrong coords"], "company_name": "c", "country": ["Germany", "Netherlands", "Germany"],
         "latitude": [48.5, 52.0, 0.0], "longitude": [12.0, 2.9, 0.0], "first_year": 2020, "first_month": 1}
    )  # fmt: skip
    tiers = pd.DataFrame({"park_id": [1, 2, 3], "tier": ["local", "regional", "local"], "reason": ["a", "b", "c"]})
    fixes = pd.DataFrame({"park_id": [3], "latitude": [48.0], "longitude": [8.5], "note": "fix"})
    out = enrich_parks(parks, geojson, regions, tiers, fixes, countries)
    assert out.set_index("park_id")["region_code"].to_dict() == {1: "DE-BY", 2: "NL", 3: "DE-BW"}
    assert out.loc[out["park_id"] == 3, "tier_reason"].iloc[0] == "c [coords corrected]"
    assert out.loc[out["park_id"] == 3, "latitude"].iloc[0] == 48.0


def test_merge_calendars_on_tiny_tables(tmp_path):
    cols = ["region_code", "date", "type", "name", "share", "source", "confidence", "notes"]
    row = lambda code, date, kind, share: (code, date, kind, "x", share, "s", "high", "")  # noqa: E731
    eu = pd.DataFrame([row("NL", "2024-01-01", "national", 1.0), row("DE", "2024-07-10", "school", 0.5), row("DE-BY", "2024-07-10", "school", 0.5),
                       row("NL-NH", "2024-07-10", "school", 1.0)], columns=cols)  # fmt: skip
    empty = pd.DataFrame(columns=cols)
    for name, df in (
        ("national_and_eu_school", eu),
        ("school_gapfill_eu", empty),
        ("school_non_eu", empty),
        ("school_non_eu_verified", empty),
    ):
        df.to_csv(tmp_path / f"{name}.csv", index=False)
    pd.DataFrame(columns=["region_code", "year", "old_name", "new_name", "reason"]).to_csv(
        tmp_path / "school_non_eu_replaces.csv", index=False
    )
    regions = pd.DataFrame({"region_code": ["DE-BW", "DE-BY", "NL"], "country": ["DE", "DE", "NL"]})
    res = merge_calendars(tmp_path, regions)
    m = res.matrices
    day = m.dates.get_loc(pd.Timestamp("2024-07-10"))
    r = {c: i for i, c in enumerate(m.regions)}
    assert m.school[r["DE-BW"], day] == pytest.approx(0.5)  # country-wide row reaches every DE region
    assert m.school[r["DE-BY"], day] == pytest.approx(0.75)  # 1 - (1 - 0.5)(1 - 0.5): overlap combines, not sums
    n_children = len([s for s in pycountry.subdivisions.get(country_code="NL") if s.parent_code is None])
    assert m.school[r["NL"], day] == pytest.approx(
        1 / n_children, rel=1e-3
    )  # a child of a single-region country counts as a fraction
    assert m.national[r["NL"], m.dates.get_loc(pd.Timestamp("2024-01-01"))] == pytest.approx(1.0)
    assert (m.national[r["DE-BY"]] > 0).sum() > 5  # Germany had no national rows: filled from the holidays package
    assert res.coverage.query("region_code == 'NL' and year == 2024")["school_covered"].iloc[0]
    assert not res.sparse.empty and np.isfinite(res.sparse[["national", "school"]].to_numpy()).all()
