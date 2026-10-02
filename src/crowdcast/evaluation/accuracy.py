"""Live accuracy tracking: log each day's forecast as it was made, then score it once the actual crowd level is known.

The deployed model is retrained daily, so re-predicting past dates later would be in-sample. Instead, every run appends
the forecast it just made (at a few lead times) to a log, and ``score`` joins that log to the actuals that have since
arrived. The log is point-in-time, so the scores are honest out-of-sample numbers.

    made_on  park_id  date  lead_days  prediction  low_confidence      <- the log (one row per park, date and lead)
"""

from __future__ import annotations

import pandas as pd

LEADS = (1, 7, 14, 30, 90)
WINDOWS = (7, 14, 30, 90)
HEADLINE_LEAD = 7
TOLERANCE = 10.0  # "within +-10 points"

LOG_COLUMNS = ["made_on", "park_id", "date", "lead_days", "prediction", "low_confidence"]
DETAIL_COLUMNS = [*LOG_COLUMNS, "actual", "error", "abs_error"]
SUMMARY_COLUMNS = ["window_days", "lead_days", "days_covered", "n", "accuracy_pct", "within_10_pct", "mae", "bias"]


def snapshot(forecast: pd.DataFrame, made_on: str | pd.Timestamp, leads: tuple[int, ...] = LEADS) -> pd.DataFrame:
    """Rows of ``forecast`` (``crowdcast forecast`` output) whose date is exactly ``lead`` days after ``made_on``."""
    made = pd.Timestamp(made_on).normalize()
    df = forecast.dropna(subset=["prediction"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    df["lead_days"] = (df["date"] - made).dt.days
    df = df[df["lead_days"].isin(leads)]
    if "low_confidence" not in df.columns:
        df["low_confidence"] = False
    df["made_on"] = made
    return df[LOG_COLUMNS].sort_values(["lead_days", "park_id"]).reset_index(drop=True)


def append_log(log: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    """Add ``new`` to ``log``; re-running a day replaces that day's rows instead of duplicating them."""
    if log is None or log.empty:
        return new.reset_index(drop=True)
    log = log.assign(made_on=pd.to_datetime(log["made_on"]), date=pd.to_datetime(log["date"]))
    out = pd.concat([log[~log["made_on"].isin(new["made_on"].unique())], new], ignore_index=True)
    return out.sort_values(["made_on", "lead_days", "park_id"]).reset_index(drop=True)


def actuals(crowd: pd.DataFrame, asof: str | pd.Timestamp) -> pd.DataFrame:
    """Observed ``crowd_percent`` for open park-days strictly before ``asof`` (loader-predicted rows are not actuals)."""
    c = crowd[(crowd["status"] == "open") & ~crowd["predicted"].astype(bool) & crowd["crowd_percent"].notna()]
    c = c[pd.to_datetime(c["date"]) < pd.Timestamp(asof).normalize()]
    return c[["park_id", "date", "crowd_percent"]].rename(columns={"crowd_percent": "actual"})


def score(
    log: pd.DataFrame, observed: pd.DataFrame, asof: str | pd.Timestamp, max_window: int = max(WINDOWS)
) -> pd.DataFrame:
    """Predictions for dates in the last ``max_window`` days before ``asof``, with the actual and the error."""
    asof = pd.Timestamp(asof).normalize()
    lg = log.assign(date=pd.to_datetime(log["date"]), made_on=pd.to_datetime(log["made_on"]))
    lg = lg[lg["date"] >= asof - pd.Timedelta(days=max_window)]
    obs = observed.assign(date=pd.to_datetime(observed["date"]))
    df = lg.merge(obs, on=["park_id", "date"], how="inner")
    df["prediction"] = df["prediction"].clip(0, 100)
    df["error"] = df["prediction"] - df["actual"]
    df["abs_error"] = df["error"].abs()
    return df[DETAIL_COLUMNS].sort_values(["date", "park_id", "lead_days"]).reset_index(drop=True)


def daily_by_park(detail: pd.DataFrame, leads: tuple[int, ...] = LEADS) -> pd.DataFrame:
    """Full-history wide table, one row per park and date: ``actual`` plus ``pred_<lead>d`` for each lead (blank if none)."""
    cols = ["park_id", "date", "actual", *[f"pred_{n}d" for n in leads]]
    if detail.empty:
        return pd.DataFrame(columns=cols)
    wide = detail.pivot_table(index=["park_id", "date"], columns="lead_days", values="prediction", aggfunc="last")
    wide.columns = [f"pred_{int(n)}d" for n in wide.columns]
    actual = detail.groupby(["park_id", "date"])["actual"].first()
    out = wide.join(actual).reset_index()
    for c in cols:
        if c not in out.columns:
            out[c] = float("nan")
    out = out[cols].sort_values(["date", "park_id"]).reset_index(drop=True)
    out[cols[2:]] = out[cols[2:]].round(2)
    return out


def daily_network(by_park: pd.DataFrame) -> pd.DataFrame:
    """One row per date, averaged over parks: ``n_parks`` with an actual, mean ``actual`` and mean ``pred_<lead>d``.

    Each column averages the parks that have a value for it, so a lead that only covers some parks is not diluted.
    """
    if by_park.empty:
        return pd.DataFrame(columns=["date", "n_parks", *by_park.columns[2:]])
    g = by_park.drop(columns="park_id").groupby("date")
    out = g.mean().round(2)
    out.insert(0, "n_parks", g["actual"].count())
    return out.reset_index()


def _stats(g: pd.DataFrame) -> dict[str, float]:
    mae = float(g["abs_error"].mean())
    return {
        "days_covered": int(g["date"].nunique()),
        "n": len(g),
        "accuracy_pct": round(100 - mae, 1),
        "within_10_pct": round(100 * float((g["abs_error"] <= TOLERANCE).mean()), 1),
        "mae": round(mae, 2),
        "bias": round(float(g["error"].mean()), 2),
    }


def summarise(
    detail: pd.DataFrame, asof: str | pd.Timestamp, windows: tuple[int, ...] = WINDOWS, by_park: bool = False
) -> pd.DataFrame:
    """Accuracy per (trailing window of target dates, lead time). ``accuracy_pct`` is 100 - mean absolute miss."""
    asof = pd.Timestamp(asof).normalize()
    keys = ["park_id"] if by_park else []
    rows = []
    for w in windows:
        d = detail[pd.to_datetime(detail["date"]) >= asof - pd.Timedelta(days=w)]
        for key, g in d.groupby([*keys, "lead_days"]):
            key = key if isinstance(key, tuple) else (key,)
            rows.append({**dict(zip([*keys, "lead_days"], key, strict=True)), "window_days": w, **_stats(g)})
    cols = [*keys, *SUMMARY_COLUMNS]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)[cols].sort_values([*keys, "window_days", "lead_days"]).reset_index(drop=True)


def headline(summary: pd.DataFrame, window: int = 90, lead: int = HEADLINE_LEAD) -> str | None:
    """e.g. ``84% accuracy, last 90 days | 41% of park-days within +-10 points | average miss 16.0 pts``."""
    row = summary[(summary["window_days"] == window) & (summary["lead_days"] == lead)]
    if row.empty:
        return None
    r = row.iloc[0]
    return (
        f"{r['accuracy_pct']:.0f}% accuracy, last {window} days ({int(r['days_covered'])} days of data, "
        f"{lead}-day-ahead forecasts) | {r['within_10_pct']:.0f}% of park-days within +-{TOLERANCE:.0f} points | "
        f"average miss {r['mae']:.1f} pts"
    )


def parse_leads(text: str) -> tuple[int, ...]:
    leads = tuple(sorted({int(x) for x in text.split(",") if x.strip()}))
    if not leads or min(leads) < 0:
        raise ValueError("--leads needs comma-separated non-negative integers, e.g. 1,7,14,30,90")
    return leads


__all__ = [
    "HEADLINE_LEAD",
    "LEADS",
    "WINDOWS",
    "actuals",
    "append_log",
    "daily_by_park",
    "daily_network",
    "headline",
    "parse_leads",
    "score",
    "snapshot",
    "summarise",
]
