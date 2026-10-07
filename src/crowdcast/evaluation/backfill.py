"""Replay past forecasts so the accuracy log has history before live logging started.

For each ``made_on`` date the model predicts the dates ``lead`` days ahead using only what was knowable then (drift features cut to
the labels known at that lead). The model is refit on labels up to the day before every ``refit_days``-th ``made_on`` and reused for
the days in between, so it can be up to ``refit_days - 1`` days staler than the live model (slightly pessimistic, never leaky). The rows are tagged ``source="backtest"``; they are
honest out-of-sample predictions but a replay, so the weather is approximated:

* leads 1-3 use the archived 1-day-ahead weather forecast, leads 4-16 the 7-day-ahead one (dates outside the archive keep the
  weather that occurred, which is slightly optimistic);
* longer leads have the weather blanked, as in live forecasts beyond the forecast horizon.
"""

from __future__ import annotations

import logging

import pandas as pd

from crowdcast.evaluation.accuracy import BACKTEST, LEADS, LOG_COLUMNS
from crowdcast.evaluation.forecast_weather import with_forecast_weather, without_weather
from crowdcast.models.config import ModelConfig
from crowdcast.models.gated import GatedCrowdModel

log = logging.getLogger(__name__)

WEATHER_HORIZON = 16  # days a live forecast has real weather for (docs/RESULTS.md)


def _weather_for_lead(rows: pd.DataFrame, lead: int, archived: pd.DataFrame | None) -> pd.DataFrame:
    if lead > WEATHER_HORIZON:
        return without_weather(rows)
    if archived is None or archived.empty:
        return rows
    return with_forecast_weather(rows, archived, 1 if lead <= 3 else 7)[0]


def backfill_log(
    frame: pd.DataFrame,
    made_on: list[pd.Timestamp],
    leads: tuple[int, ...] = LEADS,
    archived: pd.DataFrame | None = None,
    config: ModelConfig | None = None,
    refit_days: int = 7,
) -> pd.DataFrame:
    """Backtest predictions in the accuracy-log format, refitting every ``refit_days`` days of ``made_on`` dates."""
    out = []
    model: GatedCrowdModel | None = None
    refit_on: pd.Timestamp | None = None
    for day in sorted(made_on):
        if model is None or refit_on is None or (day - refit_on).days >= refit_days:
            model = GatedCrowdModel(config).fit(frame, day - pd.Timedelta(days=1))
            refit_on = day
        for lead in leads:
            rows = frame[frame["date"] == day + pd.Timedelta(days=lead)]
            if rows.empty:
                continue
            # labels are known through made_on - 1, i.e. lead + 1 days before the target
            pred = model.predict(_weather_for_lead(rows, lead, archived), lag=lead + 1)
            out.append(pred.assign(made_on=day, lead_days=lead, source=BACKTEST))
        log.info("backtested forecasts made on %s", day.date())
    if not out:
        return pd.DataFrame(columns=LOG_COLUMNS)
    df = pd.concat(out, ignore_index=True)
    return df[LOG_COLUMNS].sort_values(["made_on", "lead_days", "park_id"]).reset_index(drop=True)
