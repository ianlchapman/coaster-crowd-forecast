"""Synthetic stand-in for the loader output.

The real data comes from queue-times.com via a separate loader and is not redistributable, so tests, CI and ``crowdcast demo``
run on this generator instead. It has the same columns as the real files and a crowd signal with the structure the model is
meant to learn: a weekday pattern, a seasonal curve, school/national holiday uplift, rain/temperature effects and park-level
differences, plus noise. The numbers are not real and carry no information about any park.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

WEEKDAY_EFFECT = np.array([-8.0, -10.0, -6.0, -2.0, 6.0, 16.0, 12.0])  # Mon..Sun
COUNTRIES = [
    ("GB", 52.0, -1.5),
    ("US", 28.4, -81.5),
    ("DE", 48.3, 7.7),
    ("FR", 48.9, 2.8),
    ("JP", 35.6, 139.9),
    ("AU", -28.0, 153.4),
]


@dataclass(frozen=True)
class SyntheticData:
    crowd: pd.DataFrame  # same columns as crowd-calendar.csv (labels stop at ``end``)
    parks: pd.DataFrame  # same columns as parks.csv
    weather: pd.DataFrame  # park_id, date + raw Open-Meteo daily variables (runs ``future_days`` past ``end``)
    holiday_scores: pd.DataFrame  # park_id, date, national_score, school_score (same window as the weather)


def _parks(n_parks: int, rng: np.random.Generator) -> pd.DataFrame:
    country = [COUNTRIES[i % len(COUNTRIES)] for i in range(n_parks)]
    return pd.DataFrame(
        {
            "id": np.arange(1, n_parks + 1),
            "name": [f"Park {i}" for i in range(1, n_parks + 1)],
            "company_name": [f"Company {i % 3}" for i in range(n_parks)],
            "country": [c[0] for c in country],
            "latitude": [c[1] + rng.normal(0, 0.2) for c in country],
            "longitude": [c[2] + rng.normal(0, 0.2) for c in country],
            "first_year": 2018,
            "first_month": 1,
        }
    )


def make_synthetic(
    n_parks: int = 6, start: str = "2018-01-01", end: str = "2024-12-31", seed: int = 0, future_days: int = 16
) -> SyntheticData:
    """Crowd labels run to ``end``; weather and holiday scores run ``future_days`` further, like a forecast horizon."""
    rng = np.random.default_rng(seed)
    all_dates = pd.date_range(start, pd.Timestamp(end) + pd.Timedelta(days=future_days))
    parks = _parks(n_parks, rng)
    crowd_rows, wx_rows, hol_rows = [], [], []
    n = len(all_dates)
    doy = all_dates.dayofyear.to_numpy()
    month = all_dates.month.to_numpy()
    day = all_dates.day.to_numpy()
    dow = all_dates.dayofweek.to_numpy()
    for _, p in parks.iterrows():
        south = p["latitude"] < 0
        season = 14 * np.sin(2 * np.pi * (doy - (20 if south else 110)) / 365.25)
        if south:
            school = (np.isin(month, [12, 1]) | ((month == 7) & (day > 5) & (day < 20))).astype(float)
        else:
            school = (np.isin(month, [7, 8]) | ((month == 12) & (day > 20))).astype(float)
        national = np.zeros(n)
        national[((month == 12) & (day == 25)) | ((month == 1) & (day == 1))] = 1.0
        temp = 12 + 12 * np.sin(2 * np.pi * (doy - (200 if south else 110)) / 365.25) + rng.normal(0, 3, n)
        precip = np.where(rng.random(n) < 0.25, rng.gamma(2.0, 3.0, n), 0.0)
        signal = (
            50
            + rng.normal(0, 4)  # park level
            + WEEKDAY_EFFECT[dow]
            + season
            + 12 * school
            + 8 * national
            - 0.6 * np.clip(precip, 0, 20)
            + 0.2 * (temp - 15)
            + rng.normal(0, 9, n)
        )
        # crowd_percent is a per-park rank-normalised score, so mirror that
        crowd = np.clip(np.round(pd.Series(signal).rank(pct=True).to_numpy() * 100), 1, 100)
        closed = np.isin(month, [11, 12, 1, 2]) & ~np.isin(dow, [4, 5, 6])
        crowd_rows.append(
            pd.DataFrame(
                {
                    "park_id": p["id"],
                    "date": all_dates,
                    "status": np.where(closed, "closed", "open"),
                    "crowd_percent": np.where(closed, np.nan, crowd),
                    "predicted": False,
                    "opens": "10:00",
                    "closes": np.where(dow >= 5, "19:00", "18:00"),
                    "events": pd.Series(np.where((month == 10) & (day > 15), "Halloween Nights", ""), dtype="string"),
                }
            )
        )
        wx_rows.append(
            pd.DataFrame(
                {
                    "park_id": p["id"],
                    "date": all_dates,
                    "temperature_2m_max": temp + 4,
                    "temperature_2m_min": temp - 4,
                    "apparent_temperature_max": temp + 3,
                    "precipitation_sum": precip,
                    "snowfall_sum": 0.0,
                    "precipitation_hours": np.where(precip > 0, np.minimum(24, precip * 1.5), 0.0),
                    "wind_speed_10m_max": np.abs(rng.normal(15, 6, n)),
                    "wind_gusts_10m_max": np.abs(rng.normal(30, 10, n)),
                    "sunshine_duration": np.clip(rng.normal(6, 3, n), 0, 14) * 3600,
                    "weather_code": np.where(precip > 1, 61, 1),
                }
            )
        )
        hol_rows.append(
            pd.DataFrame({"park_id": p["id"], "date": all_dates, "national_score": national, "school_score": school})
        )
    crowd_all = pd.concat(crowd_rows, ignore_index=True)
    crowd_all = crowd_all[crowd_all["date"] <= pd.Timestamp(end)].reset_index(drop=True)
    return SyntheticData(
        crowd_all, parks, pd.concat(wx_rows, ignore_index=True), pd.concat(hol_rows, ignore_index=True)
    )
