"""Feature-set definitions. The order matters (LightGBM indexes features by position), so keep these lists stable."""

from __future__ import annotations

EVENT_GROUPS: dict[str, str] = {
    "ev_early_extra": r"early entry|extra hours|extended|magic hours|extra magic",
    "ev_halloween": r"halloween|brick or treat|fright|scream|haunt|boo",
    "ev_christmas": r"christmas|winter|holiday|noel|yule|santa",
    "ev_summer": r"summer",
    "ev_festival": r"festival|oktoberfest|celebration|spectacular|fest\b",
    "ev_ticketed": r"ticketed|after hours|special",
}

CALENDAR = ["dow", "is_weekend", "month", "doy_sin", "doy_cos", "week", "year"]
HOLIDAY = [
    "national_holiday",
    "school_holiday",
    "days_until_national_holiday_log",
    "days_since_national_holiday_log",
    "days_until_school_holiday_log",
    "days_since_school_holiday_log",
]
HOURS = ["open_min", "close_min", "open_hours"]
EVENTS = ["has_event", *EVENT_GROUPS]
PARK = ["months_open", "southern", "latitude", "longitude", "company_name", "country_iso2", "tier"]
PRIOR_YEAR = ["py_same_wd", "py_wd_mean3"]
WEATHER = [
    "wx_temp_max", "wx_temp_min", "wx_feels_max", "wx_precip_mm", "wx_snow_cm", "wx_precip_hours", "wx_wind_max",
    "wx_gust_max", "wx_code", "wx_sun_hours", "wx_wet", "wx_heavy_rain", "wx_hot", "wx_cold", "wx_freezing",
    "wx_temp_anom", "wx_precip_anom", "wx_precip_prev3", "wx_wet_days_prev7",
]  # fmt: skip

DRIFT_LAGS = (1, 7, 14)
DRIFT_WINDOW = 28
DRIFT = [f"drift{h}" for h in DRIFT_LAGS]

BASE = [*CALENDAR, *HOLIDAY, *HOURS, *EVENTS, "covid"]
#: Model for parks with enough history: knows the park id, last year's level and (optionally) recent drift.
FULL = [*BASE, *PARK, "park_cat", *PRIOR_YEAR, *WEATHER, *DRIFT]
#: Fallback for parks with too little history: no park id, no prior-year, no drift.
FALLBACK = [*BASE, *PARK, *WEATHER]
CATEGORICAL = ["company_name", "country_iso2", "tier"]
