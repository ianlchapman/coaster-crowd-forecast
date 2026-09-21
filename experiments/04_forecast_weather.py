"""Score the test year with archived-forecast weather (Open-Meteo previous-runs, lead 1 and 7 days) instead of observed weather.

Requires ``crowdcast weather-previous-runs`` first. Also compares a model trained on observed weather only against one trained
with 25% of rows weather-blanked, on rows with no weather at all (dates beyond the 16-day forecast horizon).
"""

from __future__ import annotations

import pandas as pd

from _common import load_frame, out_dir, timed
from crowdcast.config import Paths
from crowdcast.evaluation.forecast_weather import with_forecast_weather, without_weather
from crowdcast.evaluation.metrics import regression_metrics
from crowdcast.evaluation.splits import SplitDates
from crowdcast.models.config import ModelConfig
from crowdcast.models.gated import GatedCrowdModel


def main() -> None:
    frame, dates, paths = load_frame(), SplitDates(), Paths()
    archived = pd.read_csv(paths.weather_previous_runs / "prev_runs_daily.csv", parse_dates=["date"])
    test = frame[(frame["date"] > dates.val_end) & (frame["date"] <= dates.test_end)]
    rows = []
    for label, cfg in (
        ("trained on observed weather only", ModelConfig(wx_blank=0.0)),
        ("trained with 25% weather blanked", ModelConfig()),
    ):
        with timed(label):
            model = GatedCrowdModel(cfg).fit(frame, dates.val_end)
        gated = test[test["park_id"].map(model.history_years).fillna(0) >= cfg.min_years]
        scenarios = {"observed": gated, "forecast, 1 day ahead": with_forecast_weather(gated, archived, 1)[0],
                     "forecast, 7 days ahead": with_forecast_weather(gated, archived, 7)[0], "no weather": without_weather(gated)}  # fmt: skip
        for lag in (None, 1):
            for weather, x in scenarios.items():
                pred = model.predict(x, lag=lag)["prediction"].to_numpy()
                m = regression_metrics(gated["crowd_percent"].to_numpy(), pred)
                rows.append(
                    {
                        "model": label,
                        "labels": "none" if lag is None else f"through t-{lag}",
                        "weather": weather,
                        "rows": len(gated),
                        "MAE": round(m["MAE"], 2),
                        "R2": round(m["R2"], 3),
                    }
                )
    result = pd.DataFrame(rows)
    result.to_csv(out_dir() / "forecast_weather.csv", index=False)
    print(result.pivot_table(index=["model", "weather"], columns="labels", values="MAE").to_string())


if __name__ == "__main__":
    main()
