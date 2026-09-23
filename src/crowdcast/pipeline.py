"""Glue between files on disk and the pure feature/model code: load inputs, build frames, forecast."""

from __future__ import annotations

import logging

import pandas as pd

from crowdcast.config import Paths
from crowdcast.data.loaders import load_crowd_calendar, load_parks
from crowdcast.features.build import build_feature_frame, park_table
from crowdcast.features.future import build_future_rows
from crowdcast.features.status import add_is_open_prior_year, minutes_to_hhmm
from crowdcast.models.gated import GatedCrowdModel
from crowdcast.models.status import StatusModel
from crowdcast.scoring.daily import DailyScores
from crowdcast.scoring.enhance import enhance_calendar
from crowdcast.weather.archive import load_archive
from crowdcast.weather.features import derive_weather_features
from crowdcast.weather.forecast import combine_archive_and_forecast, fetch_forecast

log = logging.getLogger(__name__)


def make_enhanced_calendar(paths: Paths) -> pd.DataFrame:
    """Raw loader calendar + holiday scores -> ``data/processed/enhanced-crowd-calendar.csv``."""
    enhanced = enhance_calendar(load_crowd_calendar(paths.crowd_calendar), DailyScores.load(paths.daily_scores))
    paths.ensure(paths.enhanced_calendar.parent)
    enhanced.to_csv(paths.enhanced_calendar, index=False)
    return enhanced


def load_parks_table(paths: Paths) -> pd.DataFrame:
    return park_table(load_parks(paths.parks_csv), pd.read_csv(paths.parks_enriched))


def load_training_frame(paths: Paths) -> pd.DataFrame:
    """Every open, labelled park-day with all features (calendar, holidays, hours, events, weather, prior-year, drift)."""
    enhanced = pd.read_csv(paths.enhanced_calendar, parse_dates=["date"], dtype={"events": "string"})
    weather = derive_weather_features(load_archive(paths.weather_archive))
    return build_feature_frame(enhanced, load_parks_table(paths), weather)


def forecast(
    paths: Paths,
    model: GatedCrowdModel,
    status_model: StatusModel,
    end: str | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Predict every active park from the day after the last label to ``end`` (default: today + 15, the weather forecast horizon).

    ``is_open`` comes from ``status_model`` (wins every backtest year, see docs/planning/opening-hours-
    forecast.md); ``opens``/``closes`` still come from the same-weekday-last-year lookup, which wins those
    fields in most/all backtest years -- a blend of the two approaches, not a full replacement.
    """
    labelled = load_training_frame(paths)
    end = end or str((pd.Timestamp.today().normalize() + pd.Timedelta(days=15)).date())
    active = labelled.loc[labelled["date"] > labelled["date"].max() - pd.Timedelta(days=60), "park_id"].unique()
    coords = pd.read_csv(paths.parks_enriched).set_index("park_id").loc[active, ["latitude", "longitude"]].reset_index()

    archive = load_archive(paths.weather_archive)
    combined = combine_archive_and_forecast(
        archive, fetch_forecast(coords, paths.weather_forecast_cache, refresh=refresh)
    )
    features = derive_weather_features(combined.drop(columns=["wx_source"]))
    weather = features.merge(combined[["park_id", "date", "wx_source"]], on=["park_id", "date"], how="left")

    rows = build_future_rows(labelled, DailyScores.load(paths.daily_scores), weather, end)
    status_calendar = load_crowd_calendar(paths.crowd_calendar)[["park_id", "date", "status"]]
    rows = add_is_open_prior_year(rows, status_calendar, labelled["date"].max())
    is_open = pd.Series(status_model.predict_is_open(rows), index=rows.index)

    pred = model.predict(rows, lag=1)  # lag=1 keeps every drift column; those without labels are already blank
    pred = pred.assign(
        is_open=is_open.to_numpy(),
        opens=minutes_to_hhmm(rows["open_min"]).where(is_open).to_numpy(),
        closes=minutes_to_hhmm(rows["close_min"]).where(is_open).to_numpy(),
        days_ahead=rows["days_ahead"].to_numpy(),
        weather=rows["wx_source"].to_numpy(),
        open_last_year=rows["open_last_year"].to_numpy(),
    )
    names = load_parks(paths.parks_csv)[["id", "name"]].rename(columns={"id": "park_id", "name": "park_name"})
    pred = pred.merge(names, on="park_id", how="left")
    front = ["park_id", "park_name", "date", "is_open", "opens", "closes"]
    return pred[front + [c for c in pred.columns if c not in front]]


# --------------------------------------------------------------------------------------- data-pipeline steps
def build_parks(paths: Paths) -> None:
    """Regions, park -> region mapping and the market prior -> ``data/processed/parks/``."""
    from crowdcast.parks.enrich import enrich_parks
    from crowdcast.parks.markets import candidate_markets
    from crowdcast.parks.reference import (
        load_coordinate_fixes,
        load_countries,
        load_region_population,
        load_tiers,
    )
    from crowdcast.parks.regions import build_regions

    countries, tiers = load_countries(), load_tiers()
    geojson, places = paths.raw_geo / "ne_admin1.geojson", paths.raw_geo / "cities1000.txt"
    regions, placed = build_regions(geojson, places, countries, load_region_population())
    parks = enrich_parks(load_parks(paths.parks_csv), geojson, regions, tiers, load_coordinate_fixes(), countries)
    markets = candidate_markets(parks, regions, placed, countries, tiers)
    paths.ensure(paths.parks_dir)
    regions.to_csv(paths.regions, index=False)
    parks.to_csv(paths.parks_enriched, index=False)
    markets.to_csv(paths.candidate_markets, index=False)
    log.info("regions %d, parks %d, market rows %d", len(regions), len(parks), len(markets))


def merge_calendar_sources(paths: Paths) -> None:
    """Source calendar tables (``scripts/calendars``) -> region x date matrices."""
    from crowdcast.calendars.merge import merge_calendars

    result = merge_calendars(paths.calendars_dir, pd.read_csv(paths.regions))
    paths.ensure(paths.calendar_matrices.parent)
    result.matrices.save(paths.calendar_matrices)
    result.sparse.to_csv(paths.calendar_matrices.with_name("combined_calendar.csv"), index=False)
    result.coverage.to_csv(paths.calendar_matrices.with_name("combined_coverage.csv"), index=False)
    log.info("rows dropped below region level: %s", dict(sorted(result.dropped.items(), key=lambda kv: -kv[1])[:8]))


def fit_holiday_weights(paths: Paths, workers: int = 6, placebo_shift: int = 0) -> None:
    """Per-park region weights for national and school holidays (slow: about a minute on 6 cores)."""
    from crowdcast.data.loaders import load_events
    from crowdcast.scoring.daily import CalendarMatrices
    from crowdcast.scoring.weights import WeightConfig, WeightFitter, fit_all

    fitter = WeightFitter(
        pd.read_csv(paths.parks_enriched),
        pd.read_csv(paths.candidate_markets),
        CalendarMatrices.load(paths.calendar_matrices),
        load_crowd_calendar(paths.crowd_calendar),
        load_events(paths.events_csv),
        WeightConfig(workers=workers, placebo_shift=placebo_shift),
    )
    weights, strength = fit_all(fitter)
    paths.ensure(paths.park_region_weights.parent)
    weights.to_csv(paths.park_region_weights, index=False)
    strength.to_csv(paths.park_strength, index=False)


def score_holidays(paths: Paths) -> DailyScores:
    """Park-level daily holiday scores from the weights and calendar matrices."""
    from crowdcast.scoring.daily import CalendarMatrices, compute_daily_scores

    scores = compute_daily_scores(
        pd.read_csv(paths.park_region_weights), CalendarMatrices.load(paths.calendar_matrices)
    )
    scores.save(paths.daily_scores)
    return scores
