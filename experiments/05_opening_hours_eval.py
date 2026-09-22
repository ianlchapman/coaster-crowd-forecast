"""Backtest the is_open/opens/closes lookup heuristic (features/status.py) against real historical data.

Holds out the last 90 days of real, labelled history as if unknown (asof=cutoff), predicts is_open/opens/
closes for that window the same way pipeline.forecast() does, then compares against what actually
happened (the raw calendar, which still has closed days that the crowd-percent training frame drops).
"""

from __future__ import annotations

import pandas as pd
from _common import load_frame, out_dir

from crowdcast.config import Paths
from crowdcast.data.loaders import load_crowd_calendar
from crowdcast.features.future import build_future_rows
from crowdcast.features.status import add_is_open, minutes_to_hhmm
from crowdcast.scoring.daily import DailyScores
from crowdcast.weather.archive import load_archive
from crowdcast.weather.features import derive_weather_features

HOLD_OUT_DAYS = 90
BUCKETS = ([0, 7, 30, 90], ["1-7d", "8-30d", "31-90d"])


def build_held_out_predictions(paths: Paths, labelled: pd.DataFrame) -> pd.DataFrame:
    last = labelled["date"].max()
    cutoff = last - pd.Timedelta(days=HOLD_OUT_DAYS)
    scores = DailyScores.load(paths.daily_scores)
    weather = derive_weather_features(load_archive(paths.weather_archive))
    weather["wx_source"] = "archive"

    rows = build_future_rows(labelled, scores, weather, end=last, asof=cutoff)
    status_calendar = load_crowd_calendar(paths.crowd_calendar)[["park_id", "date", "status"]]
    rows = add_is_open(rows, status_calendar, cutoff)
    rows["pred_opens"] = minutes_to_hhmm(rows["open_min"]).where(rows["is_open"])
    rows["pred_closes"] = minutes_to_hhmm(rows["close_min"]).where(rows["is_open"])
    print(f"data through {last.date()}, held out from {cutoff.date()} ({HOLD_OUT_DAYS}d): {len(rows)} rows, {rows['park_id'].nunique()} parks")
    return rows


def attach_ground_truth(paths: Paths, rows: pd.DataFrame) -> pd.DataFrame:
    truth = load_crowd_calendar(paths.crowd_calendar)[["park_id", "date", "status", "opens", "closes"]]
    truth = truth[truth["status"].isin(["open", "closed"])]  # drop unknown/no_data: rare, ambiguous ground truth
    truth["actual_open"] = truth["status"] == "open"
    merged = rows.merge(truth, on=["park_id", "date"], how="inner")
    print(f"rows with usable ground truth: {len(merged)}")
    return merged


def score_is_open(merged: pd.DataFrame) -> None:
    tp = int((merged["is_open"] & merged["actual_open"]).sum())
    tn = int((~merged["is_open"] & ~merged["actual_open"]).sum())
    fp = int((merged["is_open"] & ~merged["actual_open"]).sum())
    fn = int((~merged["is_open"] & merged["actual_open"]).sum())
    base_rate = merged["actual_open"].mean()
    acc = (tp + tn) / len(merged)
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    closed_recall = tn / (tn + fp) if (tn + fp) else float("nan")
    closed_precision = tn / (tn + fn) if (tn + fn) else float("nan")

    print("\n=== is_open flag ===")
    print(f"actual open-rate: {base_rate:.3f}  |  naive majority-class baseline accuracy: {max(base_rate, 1 - base_rate):.4f}")
    print(f"accuracy: {acc:.4f}   open-class precision/recall: {precision:.4f}/{recall:.4f}")
    print(f"closed-class precision/recall: {closed_precision:.4f}/{closed_recall:.4f}  (confusion tp={tp} tn={tn} fp={fp} fn={fn})")

    print("by days_ahead bucket:")
    bins, labels = BUCKETS
    merged = merged.assign(bucket=pd.cut(merged["days_ahead"], bins=bins, labels=labels))
    for b, g in merged.groupby("bucket", observed=True):
        print(f"  {b:>7}: n={len(g):6d}  accuracy={(g['is_open'] == g['actual_open']).mean():.4f}  actual_open_rate={g['actual_open'].mean():.3f}")


def score_hours(merged: pd.DataFrame) -> None:
    open_rows = merged[merged["actual_open"]].copy()
    parsed = pd.to_datetime(open_rows["opens"], format="%H:%M", errors="coerce")
    open_rows["actual_open_min"] = parsed.dt.hour * 60 + parsed.dt.minute
    parsed = pd.to_datetime(open_rows["closes"], format="%H:%M", errors="coerce")
    open_rows["actual_close_min"] = parsed.dt.hour * 60 + parsed.dt.minute
    open_rows = open_rows.dropna(subset=["actual_open_min", "actual_close_min"])

    opens_err = (open_rows["open_min"] - open_rows["actual_open_min"]).abs()
    closes_err = (open_rows["close_min"] - open_rows["actual_close_min"]).abs()
    print(f"\n=== opens/closes (n={len(open_rows)} actually-open days with a recorded schedule) ===")
    print(
        f"opens : MAE={opens_err.mean():.1f}min  median={opens_err.median():.1f}min  "
        f"exact={(open_rows['pred_opens'] == open_rows['opens']).mean():.3f}  within30min={(opens_err <= 30).mean():.3f}"
    )
    print(
        f"closes: MAE={closes_err.mean():.1f}min  median={closes_err.median():.1f}min  "
        f"exact={(open_rows['pred_closes'] == open_rows['closes']).mean():.3f}  within30min={(closes_err <= 30).mean():.3f}"
    )

    print("by days_ahead bucket:")
    bins, labels = BUCKETS
    open_rows = open_rows.assign(bucket=pd.cut(open_rows["days_ahead"], bins=bins, labels=labels))
    for b, g in open_rows.groupby("bucket", observed=True):
        oe, ce = (g["open_min"] - g["actual_open_min"]).abs(), (g["close_min"] - g["actual_close_min"]).abs()
        print(f"  {b:>7}: n={len(g):6d}  opens_MAE={oe.mean():5.1f}min  closes_MAE={ce.mean():5.1f}min  opens_exact={(g['pred_opens'] == g['opens']).mean():.3f}")


if __name__ == "__main__":
    paths = Paths()
    labelled = load_frame()
    rows = build_held_out_predictions(paths, labelled)
    merged = attach_ground_truth(paths, rows)
    score_is_open(merged)
    score_hours(merged)
    merged.to_csv(out_dir() / "opening_hours_eval.csv", index=False)
