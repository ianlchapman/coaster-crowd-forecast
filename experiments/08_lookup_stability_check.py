"""Is the v1 lookup's backtest score (05_opening_hours_eval.py) a real, stable number, or an artifact of
having only tested one 90-day window? Reruns the same heuristic + metrics across several historical
90-day windows spread over the last few years and reports whether the numbers hold up.
"""

from __future__ import annotations

import pandas as pd
from _common import load_frame

from crowdcast.config import Paths
from crowdcast.data.loaders import load_crowd_calendar
from crowdcast.features.future import build_future_rows
from crowdcast.features.status import add_is_open, minutes_to_hhmm
from crowdcast.scoring.daily import DailyScores
from crowdcast.weather.archive import load_archive
from crowdcast.weather.features import derive_weather_features

WINDOW_DAYS = 90
CUTOFFS_YEARS_AGO = (0.25, 1, 2, 3)  # 0.25y = the same window 05_ used


def evaluate_window(paths: Paths, labelled: pd.DataFrame, scores, weather, raw, cutoff: pd.Timestamp) -> dict:
    end = cutoff + pd.Timedelta(days=WINDOW_DAYS)
    rows = build_future_rows(labelled, scores, weather, end=end, asof=cutoff)
    rows = add_is_open(rows, raw[["park_id", "date", "status"]], cutoff)
    rows["pred_opens"] = minutes_to_hhmm(rows["open_min"]).where(rows["is_open"])
    rows["pred_closes"] = minutes_to_hhmm(rows["close_min"]).where(rows["is_open"])

    truth = raw[raw["status"].isin(["open", "closed"])][["park_id", "date", "status", "opens", "closes"]]
    truth = truth.assign(actual_open=truth["status"] == "open")
    merged = rows.merge(truth, on=["park_id", "date"], how="inner")
    if merged.empty:
        return {}

    acc = (merged["is_open"] == merged["actual_open"]).mean()
    tn = int((~merged["is_open"] & ~merged["actual_open"]).sum())
    fp = int((merged["is_open"] & ~merged["actual_open"]).sum())
    fn = int((~merged["is_open"] & merged["actual_open"]).sum())
    closed_precision = tn / (tn + fn) if (tn + fn) else float("nan")
    closed_recall = tn / (tn + fp) if (tn + fp) else float("nan")

    open_rows = merged[merged["actual_open"]]
    exact_opens = (open_rows["pred_opens"] == open_rows["opens"]).mean()
    exact_closes = (open_rows["pred_closes"] == open_rows["closes"]).mean()

    return {
        "cutoff": str(cutoff.date()),
        "n": len(merged),
        "is_open_acc": round(acc, 4),
        "actual_open_rate": round(merged["actual_open"].mean(), 3),
        "closed_precision": round(closed_precision, 3),
        "closed_recall": round(closed_recall, 3),
        "opens_exact": round(exact_opens, 3),
        "closes_exact": round(exact_closes, 3),
    }


if __name__ == "__main__":
    paths = Paths()
    labelled = load_frame()
    last = labelled["date"].max()
    scores = DailyScores.load(paths.daily_scores)
    weather = derive_weather_features(load_archive(paths.weather_archive))
    weather["wx_source"] = "archive"
    raw = load_crowd_calendar(paths.crowd_calendar)

    results = []
    for years_ago in CUTOFFS_YEARS_AGO:
        cutoff = last - pd.Timedelta(days=int(years_ago * 365.25) + WINDOW_DAYS)
        r = evaluate_window(paths, labelled, scores, weather, raw, cutoff)
        if r:
            r["years_ago"] = years_ago
            results.append(r)

    out = pd.DataFrame(results)[["years_ago", "cutoff", "n", "is_open_acc", "actual_open_rate", "closed_precision", "closed_recall", "opens_exact", "closes_exact"]]
    print(out.to_string(index=False))
    print("\nspread (max - min) across windows:")
    for col in ["is_open_acc", "closed_precision", "closed_recall", "opens_exact", "closes_exact"]:
        print(f"  {col}: {out[col].max() - out[col].min():.3f}")
