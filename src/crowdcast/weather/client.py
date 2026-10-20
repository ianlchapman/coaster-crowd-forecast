"""Minimal Open-Meteo HTTP client with retry, back-off and rate-limit handling.

Free-tier limits are 600 calls/minute, 5,000/hour and 10,000/day, weighted by variables x days. An hourly-limit response
(HTTP 429) is waited out; a daily-limit response raises :class:`DailyLimitError` because retrying today cannot succeed.
Transport failures (timeouts, TLS handshakes, resets) are retried only ``max_network_failures`` times in a row, so a
bad route fails in minutes instead of spinning for hours. Transport (``opener``) and sleeping (``sleep``) are injectable so the retry logic is unit-testable without a network.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any

log = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

#: Daily variables used everywhere (10 variables = 1 weight unit per 14 days requested).
DAILY_VARIABLES = [
    "temperature_2m_max", "temperature_2m_min", "apparent_temperature_max", "precipitation_sum", "snowfall_sum",
    "precipitation_hours", "wind_speed_10m_max", "wind_gusts_10m_max", "sunshine_duration", "weather_code",
]  # fmt: skip

HOURLY_WAIT_SECONDS = 600


class DailyLimitError(RuntimeError):
    """The API's daily quota is exhausted; results already cached are kept, retry tomorrow."""


class OpenMeteoClient:
    def __init__(
        self,
        opener: Callable[..., Any] = urllib.request.urlopen,
        sleep: Callable[[float], None] = time.sleep,
        max_attempts: int = 40,
        timeout: float = 30.0,
        max_network_failures: int = 6,
    ) -> None:
        self._open = opener
        self._sleep = sleep
        self.max_attempts = max_attempts
        self.timeout = timeout
        self.max_network_failures = max_network_failures

    def get_json(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        full = url + "?" + urllib.parse.urlencode(params)
        network_failures = 0
        for attempt in range(1, self.max_attempts + 1):
            try:
                with self._open(full, timeout=self.timeout) as resp:
                    return dict(json.load(resp))
            except urllib.error.HTTPError as err:
                body = err.read().decode("utf8", "ignore")
                if err.code == 429:
                    if "Daily" in body:
                        raise DailyLimitError("Open-Meteo daily limit reached; retry tomorrow") from err
                    log.warning("hourly rate limit hit, waiting %ss", HOURLY_WAIT_SECONDS)
                    self._sleep(HOURLY_WAIT_SECONDS)
                    continue
                if err.code not in (500, 502, 503, 504):
                    raise
                self._sleep(5 * attempt)
            except (
                urllib.error.URLError,
                TimeoutError,
                ValueError,
            ) as err:  # ValueError: empty/invalid JSON (throttling)
                log.warning("request failed (%s: %s), retrying", type(err).__name__, err)
                network_failures += 1
                if network_failures >= self.max_network_failures:
                    raise RuntimeError(f"gave up after {network_failures} network failures: {full}") from err
                self._sleep(min(10 * network_failures, 60))
        raise RuntimeError(f"gave up after {self.max_attempts} attempts: {full}")
