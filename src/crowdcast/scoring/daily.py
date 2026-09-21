"""Daily park holiday scores, and the holiday features derived from them.

The same :meth:`DailyScores.holiday_features` serves historical rows (training) and future rows (forecasting), so the two
cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

HOLIDAY_THRESHOLD = 0.2  # a park-day with score above this counts as "a holiday day" for the distance features
SCORE_DECIMALS = 4


@dataclass(frozen=True)
class CalendarMatrices:
    """Region x date holiday intensity (0-1) for national and school holidays, plus which region-years are covered."""

    regions: np.ndarray  # (R,) region codes
    dates: pd.DatetimeIndex  # (D,)
    national: np.ndarray  # (R, D) float32
    school: np.ndarray  # (R, D) float32
    years: np.ndarray  # (Y,)
    covered_national: np.ndarray  # (R, Y) bool
    covered_school: np.ndarray  # (R, Y) bool

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, regions=self.regions, dates=self.dates.values.astype("datetime64[D]"), nat=self.national, sch=self.school,
            years=self.years, cov_nat=self.covered_national, cov_sch=self.covered_school,
        )  # fmt: skip

    @classmethod
    def load(cls, path: str | Path) -> CalendarMatrices:
        z = np.load(path, allow_pickle=False)
        return cls(
            z["regions"], pd.DatetimeIndex(z["dates"]), z["nat"], z["sch"], z["years"], z["cov_nat"], z["cov_sch"]
        )


def distances_to_holidays(score: np.ndarray, threshold: float = HOLIDAY_THRESHOLD) -> tuple[np.ndarray, np.ndarray]:
    """Days until the next / since the last holiday day (score > threshold), 0 on the day itself, NaN if none in range."""
    n = len(score)
    flag = score > threshold
    idx = np.arange(n)
    nxt = np.minimum.accumulate(np.where(flag, idx, 10**9)[::-1])[::-1]
    prv = np.maximum.accumulate(np.where(flag, idx, -1))
    until = np.where(nxt < 10**9, nxt - idx, np.nan)
    since = np.where(prv >= 0, idx - prv, np.nan)
    return until, since


@dataclass
class DailyScores:
    """National and school holiday scores for every park on every date of the calendar window."""

    park_ids: np.ndarray  # (P,)
    dates: pd.DatetimeIndex  # (D,)
    national: np.ndarray  # (P, D) float32
    school: np.ndarray  # (P, D) float32
    _distances: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = field(default_factory=dict, repr=False)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, ids=self.park_ids, dates=self.dates.values.astype("datetime64[D]"), nat=self.national, sch=self.school
        )

    @classmethod
    def load(cls, path: str | Path) -> DailyScores:
        z = np.load(path, allow_pickle=False)
        return cls(z["ids"], pd.DatetimeIndex(z["dates"]), z["nat"], z["sch"])

    @property
    def last_date(self) -> pd.Timestamp:
        return self.dates[-1]

    def _dist(self, row: int, kind: str) -> tuple[np.ndarray, np.ndarray]:
        key = (row, kind)
        if key not in self._distances:
            self._distances[key] = distances_to_holidays(self.national[row] if kind == "national" else self.school[row])
        return self._distances[key]

    def holiday_features(
        self, park_id: np.ndarray | pd.Series, date: np.ndarray | pd.Series | pd.DatetimeIndex
    ) -> pd.DataFrame:
        """Holiday columns for each ``(park_id, date)`` pair: scores and days until / since the nearest holiday day.

        Raises ``KeyError`` for an unknown park and ``ValueError`` for a date outside the calendar window.
        """
        park_id = np.asarray(park_id)
        col = self.dates.get_indexer(pd.DatetimeIndex(date))
        if (col < 0).any():
            raise ValueError(
                f"dates outside the holiday calendar window {self.dates[0].date()}..{self.dates[-1].date()}"
            )
        pos = pd.Series(np.arange(len(self.park_ids)), index=self.park_ids)
        unknown = sorted(set(park_id) - set(pos.index))
        if unknown:
            raise KeyError(f"no holiday scores for parks {unknown[:5]}")
        out = {k: np.full(len(park_id), np.nan) for k in (
            "national_holiday", "school_holiday", "days_until_national_holiday", "days_since_national_holiday",
            "days_until_school_holiday", "days_since_school_holiday")}  # fmt: skip
        for pid in np.unique(park_id):
            m = park_id == pid
            row = int(pos[pid])
            out["national_holiday"][m] = self.national[row, col[m]]
            out["school_holiday"][m] = self.school[row, col[m]]
            for kind in ("national", "school"):
                until, since = self._dist(row, kind)
                out[f"days_until_{kind}_holiday"][m] = until[col[m]]
                out[f"days_since_{kind}_holiday"][m] = since[col[m]]
        df = pd.DataFrame(out)
        df["national_holiday"] = df["national_holiday"].round(SCORE_DECIMALS)
        df["school_holiday"] = df["school_holiday"].round(SCORE_DECIMALS)
        return df


def compute_daily_scores(weights: pd.DataFrame, matrices: CalendarMatrices) -> DailyScores:
    """Weighted sum of regional holiday intensities for every park.

    ``weights``: ``park_id, region_code, type, weight`` (weights sum to 1 per park and type).
    """
    region_index = {r: i for i, r in enumerate(matrices.regions)}
    park_ids = np.sort(weights["park_id"].unique())
    pidx = {p: i for i, p in enumerate(park_ids)}
    unknown = sorted(set(weights["region_code"]) - set(region_index))
    if unknown:
        raise ValueError(f"weights reference regions missing from the calendar: {unknown[:5]}")
    scores = {}
    for kind, matrix in (("national", matrices.national), ("school", matrices.school)):
        w = np.zeros((len(park_ids), len(matrices.regions)), dtype=np.float32)
        sub = weights[weights["type"] == kind]
        w[sub["park_id"].map(pidx).to_numpy(), sub["region_code"].map(region_index).to_numpy()] = sub[
            "weight"
        ].to_numpy()
        scores[kind] = np.clip(w @ matrix, 0, 1).astype(np.float32)
    return DailyScores(park_ids, matrices.dates, scores["national"], scores["school"])
