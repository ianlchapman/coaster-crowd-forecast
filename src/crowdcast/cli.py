"""Command line interface: ``crowdcast <command>``. Run ``crowdcast --help`` for the list."""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from crowdcast.config import Paths
from crowdcast.evaluation.splits import SplitDates
from crowdcast.models.config import ModelConfig

log = logging.getLogger("crowdcast")


def _cmd_demo(args: argparse.Namespace) -> int:
    from crowdcast.demo import run_demo

    result, forecast = run_demo(args.parks)
    print("Back-test on synthetic data (numbers are not real)\n")
    print(result.by_horizon.to_string(index=False), "\n")
    print(result.by_group.to_string(index=False), "\n")
    print(f"Forecast of the days after the last label: {len(forecast)} rows, first rows:\n")
    print(forecast.head().to_string(index=False))
    return 0


def _cmd_enhance(args: argparse.Namespace) -> int:
    from crowdcast.pipeline import make_enhanced_calendar

    paths = Paths()
    out = make_enhanced_calendar(paths)
    print(f"wrote {paths.enhanced_calendar} ({len(out):,} rows)")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from crowdcast.models.gated import GatedCrowdModel
    from crowdcast.pipeline import load_training_frame

    paths = Paths()
    frame = load_training_frame(paths)
    cutoff = args.cutoff or str(frame["date"].max().date())
    model = GatedCrowdModel(ModelConfig.from_yaml(args.config) if args.config else ModelConfig()).fit(frame, cutoff)
    model.save(paths.model_file)
    gated = int((model.history_years >= model.config.min_years).sum())
    print(
        f"saved {paths.model_file}: trained to {cutoff}; {gated}/{len(model.history_years)} parks pass the {model.config.min_years:g}y gate"
    )
    return 0


def _cmd_train_status(args: argparse.Namespace) -> int:
    from crowdcast.data.loaders import load_crowd_calendar, load_parks
    from crowdcast.features.build import park_table
    from crowdcast.features.status_build import build_status_frame
    from crowdcast.models.status import StatusModel
    from crowdcast.scoring.daily import DailyScores

    paths = Paths()
    raw = load_crowd_calendar(paths.crowd_calendar)
    parks = park_table(load_parks(paths.parks_csv), pd.read_csv(paths.parks_enriched))
    frame = build_status_frame(raw, parks, DailyScores.load(paths.daily_scores))
    cutoff = args.cutoff or str(frame["date"].max().date())
    model = StatusModel(ModelConfig.from_yaml(args.config) if args.config else ModelConfig()).fit(frame, cutoff)
    model.save(paths.status_model_file)
    print(f"saved {paths.status_model_file}: trained to {cutoff}")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from crowdcast.evaluation.backtest import run_backtest
    from crowdcast.pipeline import load_training_frame

    dates = SplitDates()
    frame = load_training_frame(Paths())
    result = run_backtest(frame, args.cutoff or dates.val_end, args.test_end or dates.test_end)
    print(result.by_horizon.to_string(index=False), "\n")
    print(result.by_group.to_string(index=False), "\n")
    print("Top features (share of split gain, %):")
    print(result.model.feature_importance(10).to_string())
    if args.out:
        result.predictions.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    from crowdcast.models.gated import GatedCrowdModel
    from crowdcast.pipeline import load_training_frame

    paths = Paths()
    frame = load_training_frame(paths)
    if args.start:
        frame = frame[frame["date"] >= args.start]
    pred = GatedCrowdModel.load(paths.model_file).predict(frame, lag=args.lag)
    pred.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(pred):,} rows, {int(pred['low_confidence'].sum()):,} low-confidence")
    return 0


def _cmd_forecast(args: argparse.Namespace) -> int:
    from crowdcast.models.gated import GatedCrowdModel
    from crowdcast.models.status import StatusModel
    from crowdcast.pipeline import forecast

    paths = Paths()
    pred = forecast(
        paths,
        GatedCrowdModel.load(paths.model_file),
        StatusModel.load(paths.status_model_file),
        end=args.to,
        refresh=args.refresh,
    )
    pred.to_csv(args.out, index=False)
    print(
        f"wrote {args.out}: {len(pred):,} rows, {pred['park_id'].nunique()} parks, {pred['date'].min().date()}..{pred['date'].max().date()}"
    )
    print("weather:", pred["weather"].value_counts().to_dict())
    if (pred["weather"] == "none").any():
        print("note: rows with no weather (beyond the forecast horizon) are predicted without it and are less accurate")
    return 0


def _cmd_weather_fetch(args: argparse.Namespace) -> int:
    from crowdcast.data.loaders import load_crowd_calendar
    from crowdcast.weather.archive import fetch_archive

    paths = Paths()
    parks = pd.read_csv(paths.parks_enriched)[["park_id", "latitude", "longitude"]]
    dates = load_crowd_calendar(paths.crowd_calendar).groupby("park_id")["date"].agg(["min", "max"])
    fetch_archive(parks, dates, paths.weather_archive, refresh=args.refresh)
    print(f"weather cached in {paths.weather_archive}")
    return 0


def _cmd_parks_build(args: argparse.Namespace) -> int:
    from crowdcast.pipeline import build_parks

    build_parks(Paths())
    return 0


def _cmd_calendars_merge(args: argparse.Namespace) -> int:
    from crowdcast.pipeline import merge_calendar_sources

    merge_calendar_sources(Paths())
    return 0


def _cmd_holidays_fit(args: argparse.Namespace) -> int:
    from crowdcast.pipeline import fit_holiday_weights

    fit_holiday_weights(Paths(), workers=args.workers, placebo_shift=args.placebo_shift)
    return 0


def _cmd_holidays_score(args: argparse.Namespace) -> int:
    from crowdcast.pipeline import score_holidays

    scores = score_holidays(Paths())
    print(f"scored {len(scores.park_ids)} parks, {scores.dates[0].date()}..{scores.dates[-1].date()}")
    return 0


def _cmd_weather_previous_runs(args: argparse.Namespace) -> int:
    from crowdcast.weather.previous_runs import fetch_previous_runs

    paths = Paths()
    parks = pd.read_csv(paths.parks_enriched)[["park_id", "latitude", "longitude"]]
    frame = fetch_previous_runs(parks, args.start, args.end, paths.weather_previous_runs / "cells")
    out = paths.weather_previous_runs / "prev_runs_daily.csv"
    frame.to_csv(out, index=False)
    print(f"wrote {out} ({len(frame):,} rows)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="crowdcast", description="Forecast theme-park crowd levels.")
    p.add_argument("-v", "--verbose", action="store_true", help="log progress")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("demo", help="end-to-end run on synthetic data (no data or network needed)")
    s.add_argument("--parks", type=int, default=6)
    s.set_defaults(func=_cmd_demo)

    s = sub.add_parser("enhance", help="add holiday scores to the raw crowd calendar")
    s.set_defaults(func=_cmd_enhance)

    s = sub.add_parser("train", help="fit the gated model on all labels up to --cutoff and save it")
    s.add_argument("--cutoff", help="last training date (default: last labelled day)")
    s.add_argument("--config", type=Path, help="YAML model config (default configs/model.yaml values)")
    s.set_defaults(func=_cmd_train)

    s = sub.add_parser(
        "train-status", help="fit the is_open/opens/closes model on all status history up to --cutoff and save it"
    )
    s.add_argument("--cutoff", help="last training date (default: last status day)")
    s.add_argument("--config", type=Path, help="YAML model config (default configs/model.yaml values)")
    s.set_defaults(func=_cmd_train_status)

    s = sub.add_parser("evaluate", help="back-test: fit to --cutoff, score the following window")
    s.add_argument("--cutoff")
    s.add_argument("--test-end")
    s.add_argument("--out", type=Path, help="write per-row predictions to this CSV")
    s.set_defaults(func=_cmd_evaluate)

    s = sub.add_parser("predict", help="score labelled rows with the saved model")
    s.add_argument("out", type=Path)
    s.add_argument("--from", dest="start")
    s.add_argument("--lag", type=int, choices=[1, 7, 14], help="labels known through t-LAG (default: long horizon)")
    s.set_defaults(func=_cmd_predict)

    s = sub.add_parser("forecast", help="predict the days after the last label using forecast weather")
    s.add_argument("out", type=Path)
    s.add_argument("--to", help="last date (default: today + 15)")
    s.add_argument("--refresh", action="store_true", help="ignore the 3-hour forecast cache")
    s.set_defaults(func=_cmd_forecast)

    s = sub.add_parser(
        "weather-fetch", help="download historical weather for every park (Open-Meteo, cached, resumable)"
    )
    s.add_argument("--refresh", action="store_true")
    s.set_defaults(func=_cmd_weather_fetch)

    s = sub.add_parser(
        "weather-previous-runs", help="download archived forecasts (lead 1 and 7 days) for scoring on forecast weather"
    )
    s.add_argument("--start", default="2025-08-25")
    s.add_argument("--end", default="2026-08-31")
    s.set_defaults(func=_cmd_weather_previous_runs)

    s = sub.add_parser(
        "parks-build", help="regions, park->region mapping and market prior from reference tables + geodata"
    )
    s.set_defaults(func=_cmd_parks_build)

    s = sub.add_parser(
        "calendars-merge", help="merge the source holiday calendars (scripts/calendars) into region x date matrices"
    )
    s.set_defaults(func=_cmd_calendars_merge)

    s = sub.add_parser("holidays-fit", help="fit per-park national/school holiday region weights")
    s.add_argument("--workers", type=int, default=6)
    s.add_argument("--placebo-shift", type=int, default=0, help="validation only: shift calendars by N days")
    s.set_defaults(func=_cmd_holidays_fit)

    s = sub.add_parser("holidays-score", help="turn the weights into daily park holiday scores")
    s.set_defaults(func=_cmd_holidays_score)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )
    warnings.filterwarnings("ignore", category=FutureWarning)
    try:
        return int(args.func(args))
    except FileNotFoundError as err:
        print(f"error: {err}\nSee docs/DATA.md for where each input file goes.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
