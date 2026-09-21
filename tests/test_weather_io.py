import numpy as np
import pandas as pd
import pytest

from crowdcast.weather.archive import fetch_archive, load_archive
from crowdcast.weather.client import DAILY_VARIABLES
from crowdcast.weather.forecast import combine_archive_and_forecast, fetch_forecast
from crowdcast.weather.previous_runs import HOURLY_VARIABLES, fetch_previous_runs


class FakeClient:
    """Answers Open-Meteo calls with plausible synthetic payloads and records what was asked."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def get_json(self, url: str, params: dict) -> dict:
        self.calls.append((url, params))
        if "hourly" in params:
            hours = pd.date_range(
                params["start_date"], pd.Timestamp(params["end_date"]) + pd.Timedelta(hours=23), freq="h"
            )
            out = {"time": hours.strftime("%Y-%m-%dT%H:%M").tolist()}
            for name in params["hourly"].split(","):
                out[name] = [1.0] * len(hours)
            return {"hourly": out}
        if "past_days" in params:
            today = pd.Timestamp.today().normalize()
            days = pd.date_range(
                today - pd.Timedelta(days=int(params["past_days"])),
                today + pd.Timedelta(days=int(params["forecast_days"]) - 1),
            )
        else:
            days = pd.date_range(params["start_date"], params["end_date"])
        return {
            "daily": {
                "time": days.strftime("%Y-%m-%d").tolist(),
                **{v: np.linspace(0, 10, len(days)).tolist() for v in DAILY_VARIABLES},
            }
        }


PARKS = pd.DataFrame({"park_id": [1, 2, 3], "latitude": [48.80, 48.85, 40.0], "longitude": [2.77, 2.80, -74.0]})


def test_archive_is_fetched_once_per_grid_cell_cached_and_resumable(tmp_path):
    spans = pd.DataFrame(
        {"min": pd.Timestamp("2024-01-10"), "max": pd.Timestamp("2024-03-01")},
        index=pd.Index([1, 2, 3], name="park_id"),
    )
    client = FakeClient()
    fetch_archive(PARKS, spans, tmp_path, client=client, pause=0)
    assert len(client.calls) == 2  # parks 1 and 2 share a 0.25 degree cell
    assert client.calls[0][1]["start_date"] == "2024-01-03"  # padded 7 days before the first labelled day
    archive = load_archive(tmp_path)
    assert sorted(archive["park_id"].unique()) == [1, 2, 3] and {"date", "temperature_2m_max"} <= set(archive.columns)
    fetch_archive(PARKS, spans, tmp_path, client=client, pause=0)
    assert len(client.calls) == 2  # everything cached: no new requests


def test_load_archive_without_cache_explains_what_to_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="crowdcast weather-fetch"):
        load_archive(tmp_path)


def test_forecast_covers_past_and_future_and_is_cached(tmp_path):
    client = FakeClient()
    fc = fetch_forecast(PARKS, tmp_path, client=client)
    today = pd.Timestamp.today().normalize()
    assert fc["date"].min() == today - pd.Timedelta(days=92) and fc["date"].max() == today + pd.Timedelta(days=15)
    assert len(client.calls) == 2 and fc["park_id"].nunique() == 3
    fetch_forecast(PARKS, tmp_path, client=client)
    assert len(client.calls) == 2  # served from the 3-hour cache
    fetch_forecast(PARKS, tmp_path, client=client, refresh=True)
    assert len(client.calls) == 4


def test_archive_wins_over_forecast_where_both_exist():
    dates = pd.date_range("2024-01-01", periods=5)
    archive = pd.DataFrame({"park_id": 1, "date": dates[:3], "temperature_2m_max": [1.0, 2.0, 3.0]})
    forecast = pd.DataFrame({"park_id": 1, "date": dates[1:], "temperature_2m_max": [90.0, 91.0, 92.0, 93.0]})
    out = combine_archive_and_forecast(archive, forecast)
    assert out["temperature_2m_max"].tolist() == [1.0, 2.0, 3.0, 92.0, 93.0]
    assert out["wx_source"].tolist() == ["archive"] * 3 + ["forecast"] * 2


def test_previous_runs_are_aggregated_cached_and_assigned_to_every_park(tmp_path):
    client = FakeClient()
    out = fetch_previous_runs(PARKS, "2026-01-01", "2026-01-03", tmp_path, client=client, pause=0)
    assert len(client.calls) == 2 and set(out["lead"]) == {1, 7} and out["park_id"].nunique() == 3
    assert len(out) == 3 * 3 * 2  # parks x days x leads
    hourly = client.calls[0][1]["hourly"].split(",")
    assert len(hourly) == 2 * len(HOURLY_VARIABLES)
    fetch_previous_runs(PARKS, "2026-01-01", "2026-01-03", tmp_path, client=client, pause=0)
    assert len(client.calls) == 2
