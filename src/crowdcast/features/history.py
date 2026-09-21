"""Features built from a park's own past labels: prior-year level and recent drift.

Both are leakage-safe by construction: the prior-year lags look 357-371 days back, and drift at lag ``h`` only uses
labels dated on or before ``t - h`` (so a forecast made with labels known through ``t - h`` can compute it).
"""

from __future__ import annotations

import pandas as pd

from crowdcast.features.columns import DRIFT_LAGS, DRIFT_WINDOW

PRIOR_YEAR_LAGS = (357, 364, 371)  # 51, 52 and 53 weeks: same weekday, one year earlier (weekday-aligned)


def prior_year_frame(labels: pd.Series, park_id: pd.Series, date: pd.Series) -> pd.DataFrame:
    """Prior-year level for each row: ``py_same_wd`` (t-364) and ``py_wd_mean3`` (mean of t-357, t-364, t-371).

    ``labels`` is indexed by ``(park_id, date)``. Rows whose earlier date has no label get NaN.
    """
    cols = {}
    for lag in PRIOR_YEAR_LAGS:
        key = pd.MultiIndex.from_arrays([park_id, date - pd.Timedelta(days=lag)])
        cols[lag] = labels.reindex(key).to_numpy()
    out = pd.DataFrame({"py_same_wd": cols[364]}, index=park_id.index)
    out["py_wd_mean3"] = pd.DataFrame({lag: cols[lag] for lag in PRIOR_YEAR_LAGS}, index=park_id.index).mean(axis=1)
    return out


def add_prior_year(df: pd.DataFrame) -> pd.DataFrame:
    labels = df.set_index(["park_id", "date"])["crowd_percent"]
    py = prior_year_frame(labels, df["park_id"], df["date"])
    df["py_same_wd"] = py["py_same_wd"].to_numpy()
    df["py_wd_mean3"] = py["py_wd_mean3"].to_numpy()
    return df


def drift_series(gap: pd.Series, last_label: pd.Timestamp | None = None) -> pd.Series:
    """Rolling mean of ``gap`` (actual minus same-weekday-last-year) over a daily calendar; index is the date."""
    end = last_label if last_label is not None else gap.index.max()
    daily = gap.reindex(pd.date_range(gap.index.min(), end))
    return daily.rolling(DRIFT_WINDOW, min_periods=7).mean()


def add_drift(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``drift{1,7,14}``: the park's mean (actual - same weekday last year) over the 28 days ending ``lag`` days before.

    Rows are labelled days, so the value at ``t`` is taken from the rolling series at ``t - lag``.
    """
    gap = df["crowd_percent"] - df["py_same_wd"]
    parts = []
    horizon = max(DRIFT_LAGS)
    for pid, idx in df.groupby("park_id").groups.items():
        s = pd.Series(gap.loc[idx].to_numpy(), index=df.loc[idx, "date"].to_numpy())
        roll = drift_series(s, s.index.max() + pd.Timedelta(days=horizon))
        parts.append(
            pd.DataFrame(
                {"park_id": pid, "date": roll.index, **{f"drift{h}": roll.shift(h).to_numpy() for h in DRIFT_LAGS}}
            )
        )
    return df.merge(pd.concat(parts), on=["park_id", "date"], how="left")
