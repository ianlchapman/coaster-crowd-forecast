"""Does weather help, and which part of it? Feature sets: none / same-day / + past week / + look-ahead (tomorrow's weather).

Look-ahead is included only to show it adds nothing over same-day weather; it is not used by the deployed model (it would not
be known beyond the forecast horizon).
"""

from __future__ import annotations

import pandas as pd

from _common import load_frame, out_dir, timed
from crowdcast.evaluation.metrics import regression_metrics
from crowdcast.evaluation.splits import SplitDates, time_split
from crowdcast.features.calendar import in_covid
from crowdcast.models.zoo import F_ID_PY, LightGBM
from crowdcast.pipeline import Paths
from crowdcast.weather.archive import load_archive
from crowdcast.weather.features import derive_weather_features

SAME_DAY = ["wx_temp_max", "wx_temp_min", "wx_feels_max", "wx_precip_mm", "wx_snow_cm", "wx_precip_hours", "wx_wind_max", "wx_gust_max",
            "wx_code", "wx_sun_hours", "wx_wet", "wx_heavy_rain", "wx_hot", "wx_cold", "wx_freezing", "wx_temp_anom", "wx_precip_anom"]  # fmt: skip
PAST = ["wx_precip_prev3", "wx_wet_days_prev7"]
AHEAD = ["wx_precip_next3", "wx_temp_max_next1", "wx_precip_next1"]
PARAMS = {"num_leaves": 127, "min_child_samples": 50}


def main() -> None:
    frame = load_frame()
    extra = derive_weather_features(load_archive(Paths().weather_archive), lookahead=True)[["park_id", "date", *AHEAD]]
    frame = frame.merge(extra, on=["park_id", "date"], how="left")
    train, val, test = time_split(frame, SplitDates())
    covid = in_covid(frame["date"])
    sets = {
        "no weather": F_ID_PY,
        "+ same-day weather": [*F_ID_PY, *SAME_DAY],
        "+ same-day + past week": [*F_ID_PY, *SAME_DAY, *PAST],
        "+ all, incl. next-days": [*F_ID_PY, *SAME_DAY, *PAST, *AHEAD],
    }
    rows = []
    for split, fit_on, evaluate in (("validation", train & ~covid, val), ("test", (train | val) & ~covid, test)):
        for name, features in sets.items():
            with timed(f"{split} {name}"):
                pred = LightGBM(features, PARAMS).fit_predict(frame, fit_on, evaluate)
            m = regression_metrics(frame.loc[evaluate, "crowd_percent"].to_numpy(), pred)
            rows.append({"split": split, "features": name, **{k: round(v, 3) for k, v in m.items()}})
            print(rows[-1], flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(out_dir() / "weather_ablation.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
