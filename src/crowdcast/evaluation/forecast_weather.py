"""Score a fitted model with archived-forecast weather instead of the weather that actually occurred.

Live predictions use forecast weather, which is noisier than the reanalysis the model was trained on. This swaps the same-day
weather columns for what a forecast issued ``lead`` days earlier said, recomputes the flags and anomalies, and leaves the
look-back features (past days are known live) as actuals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crowdcast.features.columns import WEATHER

RAW_COLUMNS = [
    "wx_temp_max", "wx_temp_min", "wx_feels_max", "wx_precip_mm", "wx_snow_cm", "wx_precip_hours", "wx_wind_max", "wx_gust_max",
    "wx_sun_hours", "wx_code",
]  # fmt: skip


def with_forecast_weather(rows: pd.DataFrame, archived: pd.DataFrame, lead: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Return ``(rows with forecast weather, mask of rows where a full forecast was available)``.

    Rows with any missing forecast variable keep their actual weather (so the estimate is slightly optimistic when the mask is
    not all True).
    """
    fc = archived[archived["lead"] == lead].set_index(["park_id", "date"])[RAW_COLUMNS]
    out = rows.copy()
    forecast = fc.reindex(pd.MultiIndex.from_arrays([out["park_id"], out["date"]]))
    clim_temp = out["wx_temp_max"] - out["wx_temp_anom"]  # climatology recovered from the actual-weather columns
    clim_precip = out["wx_precip_mm"] - out["wx_precip_anom"]
    ok = forecast.notna().all(axis=1).to_numpy()
    for col in RAW_COLUMNS:
        out.loc[ok, col] = forecast[col].to_numpy()[ok]
    out["wx_wet"] = (out["wx_precip_mm"] >= 1).astype(float)
    out["wx_heavy_rain"] = (out["wx_precip_mm"] >= 10).astype(float)
    out["wx_hot"] = (out["wx_temp_max"] >= 30).astype(float)
    out["wx_cold"] = (out["wx_temp_max"] <= 5).astype(float)
    out["wx_freezing"] = (out["wx_temp_min"] <= 0).astype(float)
    out["wx_temp_anom"] = out["wx_temp_max"] - clim_temp
    out["wx_precip_anom"] = out["wx_precip_mm"] - clim_precip
    return out, ok


def without_weather(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    out[WEATHER] = np.nan
    return out
