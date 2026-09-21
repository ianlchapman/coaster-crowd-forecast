import numpy as np
import pandas as pd
import pytest

from crowdcast.calendars.merge import _combine_overlaps, fill_national, remap_subregions
from crowdcast.parks.geo import feature_target, haversine_km
from crowdcast.parks.markets import candidate_markets
from crowdcast.parks.reference import load_countries, load_region_population, load_tiers


def test_haversine_known_distances():
    assert haversine_km(51.5074, -0.1278, 48.8566, 2.3522) == pytest.approx(343.6, abs=1.5)  # London - Paris
    assert haversine_km(10.0, 20.0, 10.0, 20.0) == 0.0
    d = haversine_km(np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0, 90.0]), np.array([180.0, 0.0]))
    assert d == pytest.approx([20015.1, 10007.5], abs=1.0)  # antipode and quarter circle


def test_feature_target_maps_admin1_to_the_region_level():
    countries, split = {"GB", "FR", "DE", "NL"}, {"GB", "FR", "DE"}
    assert feature_target(
        {"iso_a2": "GB", "geonunit": "Scotland", "iso_3166_2": "", "name": "x"}, countries, split
    ) == ("GB-SCT", "Scotland")
    assert (
        feature_target(
            {"iso_a2": "FR", "region_cod": "FR-GUF", "iso_3166_2": "", "region": "x", "name": "x"}, countries, split
        )
        is None
    )  # overseas
    assert feature_target({"iso_a2": "NL", "iso_3166_2": "NL-NH", "name": "x"}, countries, split) == (
        "NL",
        None,
    )  # unsplit country -> country level
    assert (
        feature_target({"iso_a2": "US", "iso_3166_2": "US-CA", "name": "x"}, countries, split) is None
    )  # not in scope


def test_reference_tables_are_consistent():
    countries, tiers, pop = load_countries(), load_tiers(), load_region_population()
    assert countries["iso2"].is_unique and countries["affluence"].between(0, 1).all()
    assert tiers["park_id"].is_unique and tiers["park_id"].notna().all()
    split_countries = set(countries.loc[countries["split_admin1"], "iso2"])
    assert {c[:2] for c in pop["region_code"]} <= split_countries  # population only for admin-1 split countries


def _toy_markets():
    regions = pd.DataFrame(
        {"region_code": ["GB", "FR", "US"], "country": ["GB", "FR", "US"], "name": ["a", "b", "c"], "latitude": [52.0, 46.0, 39.0],
         "longitude": [-1.0, 2.0, -98.0], "population": [68_000_000, 68_000_000, 340_000_000]}
    )  # fmt: skip
    places = pd.DataFrame(
        {
            "region_code": ["GB", "FR", "US"],
            "lat": [52.0, 46.0, 39.0],
            "lon": [-1.0, 2.0, -98.0],
            "pop": [1000, 1000, 1000],
        }
    )
    countries = pd.DataFrame({"iso2": ["GB", "FR", "US"], "affluence": [1.0, 1.0, 1.0]})
    tiers = pd.DataFrame({"park_id": [1, 2], "decay_km_override": [np.nan, np.nan]})
    parks = pd.DataFrame(
        {
            "park_id": [1, 2],
            "latitude": [52.5, 52.5],
            "longitude": [-1.5, -1.5],
            "country_iso2": ["GB", "GB"],
            "tier": ["local", "intercontinental"],
        }
    )
    return parks, regions, places, countries, tiers


def test_market_prior_sums_to_one_and_favours_the_home_country_for_local_parks():
    m = candidate_markets(*_toy_markets())
    assert m.groupby("park_id")["prior_weight"].sum().round(6).eq(1.0).all()
    local = m[m["park_id"] == 1].set_index("region_code")["prior_weight"]
    far = m[m["park_id"] == 2].set_index("region_code")["prior_weight"]
    assert local["GB"] > 0.9 and local["GB"] > far["GB"]  # intercontinental parks draw more from far away
    assert far["US"] > local.get("US", 0.0)  # (regions beyond 99% cumulative weight are dropped for local parks)


def test_overlapping_calendar_rows_combine_as_one_minus_product():
    df = pd.DataFrame({"region_code": ["A", "A", "A"], "date": pd.to_datetime(["2024-01-01"] * 2 + ["2024-01-02"]), "type": "school",
                       "name": ["x", "y", "x"], "share": [0.5, 0.5, 0.3], "confidence": "high"})  # fmt: skip
    g = _combine_overlaps(df).set_index("date")["s"]
    assert g[pd.Timestamp("2024-01-01")] == pytest.approx(0.75)  # not 1.0 (a sum) and not 0.5 (a max)
    assert g[pd.Timestamp("2024-01-02")] == pytest.approx(0.3)


def test_subregion_remapping():
    df = pd.DataFrame({"region_code": ["PL-02", "FR-57", "FR-6AE", "DE-BY"], "date": pd.Timestamp("2024-01-01"), "type": "school",
                       "name": "x", "share": [1.0, 1.0, 1.0, 1.0], "confidence": "high"})  # fmt: skip
    out = remap_subregions(df)
    assert out["region_code"].tolist() == ["PL-DS", "FR-GES", "FR-GES", "DE-BY"]
    assert out["share"].round(3).tolist()[1:3] == [0.55, 0.55]
    assert out["share"].max() < 1.0  # clipped so log1p(-share) is finite


def test_national_fill_covers_countries_without_rows_and_skips_sundays():
    df = pd.DataFrame(
        {
            "region_code": ["DE"],
            "date": pd.Timestamp("2024-01-01"),
            "type": "national",
            "name": "x",
            "share": 1.0,
            "confidence": "high",
        }
    )
    out = fill_national(df, {"DE", "GR"})
    added = out[out["region_code"] == "GR"]
    assert len(added) > 100 and (added["date"].dt.dayofweek != 6).all() and (added["confidence"] == "medium").all()
    assert (out["region_code"] == "DE").sum() == 1  # countries that already have national rows are untouched
