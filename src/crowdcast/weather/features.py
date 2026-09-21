"""Turn raw daily Open-Meteo variables into the ``wx_*`` model features.

Same-day values, simple threshold flags, anomalies against each park's own climatology, and two look-back context features
(rain in the previous 3 days, wet days in the previous 7). Look-ahead features are optional and off by default: the model is
meant for forecasting, and tomorrow's weather is only known at forecast time up to the forecast horizon.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RENAME = {
    "temperature_2m_max": "wx_temp_max",
    "temperature_2m_min": "wx_temp_min",
    "apparent_temperature_max": "wx_feels_max",
    "precipitation_sum": "wx_precip_mm",
    "snowfall_sum": "wx_snow_cm",
    "precipitation_hours": "wx_precip_hours",
    "wind_speed_10m_max": "wx_wind_max",
    "wind_gusts_10m_max": "wx_gust_max",
    "weather_code": "wx_code",
}
WET_MM, HEAVY_RAIN_MM, HOT_C, COLD_C, FREEZING_C = 1.0, 10.0, 30.0, 5.0, 0.0
CLIMATOLOGY_HALF_WINDOW = 7  # days either side of the day-of-year


def _flag(cond: pd.Series, valid: pd.Series) -> pd.Series:
    return cond.astype(float).where(valid)


def seasonal_climatology(values: np.ndarray, day_of_year: np.ndarray) -> np.ndarray:
    """Mean of ``values`` per day of year (all years), smoothed over +-7 days with wrap-around; returned per input row."""
    by_day = pd.Series(values).groupby(day_of_year).mean().reindex(range(1, 366))
    k = CLIMATOLOGY_HALF_WINDOW
    wrapped = pd.concat([by_day.iloc[-k:], by_day, by_day.iloc[:k]]).rolling(2 * k + 1, center=True).mean().iloc[k:-k]
    return wrapped.reindex(day_of_year).to_numpy()


def derive_weather_features(raw: pd.DataFrame, lookahead: bool = False) -> pd.DataFrame:
    """``raw``: ``park_id, date`` + Open-Meteo daily variables. Returns ``park_id, date, wx_*``.

    Anomalies use the climatology of the rows passed in, so pass the park's full history (not a short slice).
    """
    w = raw.rename(columns=RENAME).sort_values(["park_id", "date"]).reset_index(drop=True)
    w["date"] = pd.to_datetime(w["date"])
    w["wx_sun_hours"] = w["sunshine_duration"] / 3600
    w = w.drop(columns=["sunshine_duration"])
    w["wx_wet"] = _flag(w["wx_precip_mm"] >= WET_MM, w["wx_precip_mm"].notna())
    w["wx_heavy_rain"] = _flag(w["wx_precip_mm"] >= HEAVY_RAIN_MM, w["wx_precip_mm"].notna())
    w["wx_hot"] = _flag(w["wx_temp_max"] >= HOT_C, w["wx_temp_max"].notna())
    w["wx_cold"] = _flag(w["wx_temp_max"] <= COLD_C, w["wx_temp_max"].notna())
    w["wx_freezing"] = _flag(w["wx_temp_min"] <= FREEZING_C, w["wx_temp_min"].notna())

    doy = w["date"].dt.dayofyear.clip(upper=365).to_numpy()
    for col, out in (("wx_temp_max", "wx_temp_anom"), ("wx_precip_mm", "wx_precip_anom")):
        clim = np.full(len(w), np.nan)
        for _, idx in w.groupby("park_id").indices.items():
            clim[idx] = seasonal_climatology(w[col].to_numpy()[idx], doy[idx])
        w[out] = w[col] - clim

    by_park = w.groupby("park_id")
    w["wx_precip_prev3"] = by_park["wx_precip_mm"].transform(lambda s: s.shift(1).rolling(3, min_periods=2).sum())
    w["wx_wet_days_prev7"] = by_park["wx_wet"].transform(lambda s: s.shift(1).rolling(7, min_periods=5).sum())
    if lookahead:
        w["wx_precip_next3"] = by_park["wx_precip_mm"].transform(
            lambda s: s[::-1].shift(1).rolling(3, min_periods=2).sum()[::-1]
        )
        w["wx_temp_max_next1"] = by_park["wx_temp_max"].shift(-1)
        w["wx_precip_next1"] = by_park["wx_precip_mm"].shift(-1)
    keep = ["park_id", "date", *[c for c in w.columns if c.startswith("wx_")]]
    return w[keep]
