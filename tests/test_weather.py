import json
import urllib.error

import numpy as np
import pandas as pd
import pytest

from crowdcast.weather.client import DailyLimitError, OpenMeteoClient
from crowdcast.weather.features import derive_weather_features, seasonal_climatology
from crowdcast.weather.grid import add_cell
from crowdcast.weather.previous_runs import HOURLY_VARIABLES, aggregate_hourly


class _Resp:
    def __init__(self, payload):
        self._b = json.dumps(payload)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, *a):
        return self._b.encode()


def _http_error(code: int, body: str = "") -> urllib.error.HTTPError:
    import io

    return urllib.error.HTTPError("http://x", code, "err", {}, io.BytesIO(body.encode()))  # type: ignore[arg-type]


def _client(script):
    """A client whose transport plays back ``script`` (exceptions are raised, dicts are returned as JSON)."""
    calls, sleeps = [], []

    def opener(url, timeout=None):
        calls.append(url)
        item = script.pop(0)
        if isinstance(item, Exception):
            raise item
        return _Resp(item)

    return OpenMeteoClient(opener=opener, sleep=sleeps.append, max_attempts=5), calls, sleeps


def test_hourly_rate_limit_is_waited_out():
    client, calls, sleeps = _client([_http_error(429, "Hourly API request limit exceeded"), {"ok": 1}])
    assert client.get_json("http://x", {"a": 1}) == {"ok": 1}
    assert sleeps == [600] and len(calls) == 2


def test_daily_rate_limit_raises_instead_of_retrying():
    client, calls, _ = _client([_http_error(429, "Daily API request limit exceeded")])
    with pytest.raises(DailyLimitError):
        client.get_json("http://x", {})
    assert len(calls) == 1


def test_server_errors_and_empty_bodies_back_off_then_succeed():
    client, calls, sleeps = _client([_http_error(503), ValueError("Expecting value"), {"ok": 2}])
    assert client.get_json("http://x", {}) == {"ok": 2}
    assert len(sleeps) == 2 and len(calls) == 3


def test_client_errors_are_not_retried():
    client, calls, _ = _client([_http_error(400, "bad request")])
    with pytest.raises(urllib.error.HTTPError):
        client.get_json("http://x", {})
    assert len(calls) == 1


def test_gives_up_after_max_attempts():
    client, _, _ = _client([_http_error(503)] * 5)
    with pytest.raises(RuntimeError, match="gave up"):
        client.get_json("http://x", {})


def test_grid_cell_shared_by_nearby_parks():
    parks = pd.DataFrame({"park_id": [1, 2, 3], "latitude": [48.80, 48.85, 40.0], "longitude": [2.77, 2.80, -74.0]})
    cells = add_cell(parks)["cell"]
    assert cells[0] == cells[1] != cells[2]


def _raw(n=730, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.date_range("2021-01-01", periods=n)
    t = 15 + 10 * np.sin(2 * np.pi * (d.dayofyear - 100) / 365.25)
    return pd.DataFrame(
        {
            "park_id": 1, "date": d, "temperature_2m_max": t, "temperature_2m_min": t - 8, "apparent_temperature_max": t,
            "precipitation_sum": np.where(rng.random(n) < 0.3, 5.0, 0.0), "snowfall_sum": 0.0, "precipitation_hours": 1.0,
            "wind_speed_10m_max": 10.0, "wind_gusts_10m_max": 20.0, "sunshine_duration": 3600.0 * 5, "weather_code": 1,
        }
    )  # fmt: skip


def test_weather_flags_and_units():
    raw = _raw().iloc[:5].copy()
    raw["precipitation_sum"] = [0.0, 1.0, 9.9, 10.0, np.nan]
    raw["temperature_2m_max"] = [35.0, 30.0, 5.0, 4.9, 20.0]
    raw["temperature_2m_min"] = [10.0, 0.0, -1.0, 3.0, 8.0]
    f = derive_weather_features(raw)
    assert f["wx_wet"].tolist()[:4] == [0, 1, 1, 1] and np.isnan(f["wx_wet"].iloc[4])
    assert f["wx_heavy_rain"].tolist()[:4] == [0, 0, 0, 1]
    assert f["wx_hot"].tolist() == [1, 1, 0, 0, 0]
    assert f["wx_cold"].tolist() == [0, 0, 1, 1, 0]  # <= 5 C
    assert f["wx_freezing"].tolist() == [0, 1, 1, 0, 0]  # min <= 0 C
    assert f["wx_sun_hours"].tolist() == [5.0] * 5


def test_lookback_features_never_see_today_or_future():
    raw = _raw(30)
    raw["precipitation_sum"] = 0.0
    raw.loc[10, "precipitation_sum"] = 20.0
    f = derive_weather_features(raw)
    assert f.loc[10, "wx_precip_prev3"] == 0  # today's rain is not in "previous 3 days"
    assert (
        f.loc[11, "wx_precip_prev3"] == 20 and f.loc[13, "wx_precip_prev3"] == 20 and f.loc[14, "wx_precip_prev3"] == 0
    )
    assert "wx_precip_next1" not in f.columns  # look-ahead is opt-in
    assert "wx_precip_next1" in derive_weather_features(raw, lookahead=True).columns


def test_seasonal_climatology_wraps_around_new_year():
    doy = np.tile(np.arange(1, 366), 3)
    values = np.where((doy > 355) | (doy < 10), 10.0, 0.0)
    clim = seasonal_climatology(values, doy)
    assert clim[doy == 1][0] > 4  # smoothing reaches across 31 Dec -> 1 Jan
    assert np.isclose(clim[doy == 183][0], 0)


def test_previous_runs_daily_aggregation():
    hours = pd.date_range("2026-01-01", periods=48, freq="h")
    payload = {"time": hours.strftime("%Y-%m-%dT%H:%M").tolist()}
    for lead in (1, 7):
        for v in HOURLY_VARIABLES:
            payload[f"{v}_previous_day{lead}"] = [1.0] * 48
        payload[f"temperature_2m_previous_day{lead}"] = list(np.arange(48, dtype=float))
        payload[f"precipitation_previous_day{lead}"] = [0.5] * 24 + [0.0] * 24
    out = aggregate_hourly(payload)
    day1 = out[(out["lead"] == 1) & (out["date"] == "2026-01-01")].iloc[0]
    assert day1["wx_temp_max"] == 23 and day1["wx_temp_min"] == 0
    assert day1["wx_precip_mm"] == 12.0 and day1["wx_precip_hours"] == 24
    assert day1["wx_sun_hours"] == pytest.approx(24 / 3600)
