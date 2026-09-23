"""Backtest the feature-driven StatusModel (LightGBM + year-on-year prior-year features) against real data,
same 90-day held-out window and metrics as the lookup heuristic in 05_opening_hours_eval.py, for a direct
before/after comparison.
"""

from __future__ import annotations

import pandas as pd

from _common import out_dir
from crowdcast.config import Paths
from crowdcast.data.loaders import load_crowd_calendar, load_parks
from crowdcast.features.build import park_table
from crowdcast.features.status_build import HOURS_FEATURES, STATUS_FEATURES, build_status_frame
from crowdcast.models.status import StatusModel
from crowdcast.scoring.daily import DailyScores

HOLD_OUT_DAYS = 90
BUCKETS = ([0, 7, 30, 90], ["1-7d", "8-30d", "31-90d"])


def score_is_open(test: pd.DataFrame, pred: pd.DataFrame, cutoff: pd.Timestamp) -> None:
    is_open, actual_open = pred["is_open"].to_numpy(), test["is_open"].to_numpy()
    tp = int((is_open & actual_open).sum())
    tn = int((~is_open & ~actual_open).sum())
    fp = int((is_open & ~actual_open).sum())
    fn = int((~is_open & actual_open).sum())
    base_rate = actual_open.mean()
    acc = (tp + tn) / len(test)
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    closed_precision = tn / (tn + fn) if (tn + fn) else float("nan")
    closed_recall = tn / (tn + fp) if (tn + fp) else float("nan")

    print("\n=== is_open flag (StatusModel) ===")
    print(
        f"actual open-rate: {base_rate:.3f}  |  naive majority-class baseline accuracy: {max(base_rate, 1 - base_rate):.4f}"
    )
    print(f"accuracy: {acc:.4f}   open-class precision/recall: {precision:.4f}/{recall:.4f}")
    print(
        f"closed-class precision/recall: {closed_precision:.4f}/{closed_recall:.4f}  (confusion tp={tp} tn={tn} fp={fp} fn={fn})"
    )

    bins, labels = BUCKETS
    days_ahead = (test["date"] - cutoff).dt.days
    g = pd.DataFrame(
        {
            "bucket": pd.cut(days_ahead, bins=bins, labels=labels),
            "hit": is_open == actual_open,
            "actual_open": actual_open,
        }
    )
    print("by days_ahead-from-cutoff bucket:")
    for b, gg in g.groupby("bucket", observed=True):
        print(
            f"  {b:>7}: n={len(gg):6d}  accuracy={gg['hit'].mean():.4f}  actual_open_rate={gg['actual_open'].mean():.3f}"
        )


def score_hours(test: pd.DataFrame, pred: pd.DataFrame) -> None:
    # only score hours where the park was actually open and had a recorded schedule, same universe as 05_
    both = (test["is_open"] & test["opens"].notna() & test["closes"].notna()).to_numpy()
    actual_open_min = test.loc[both, "open_min"].to_numpy()
    actual_close_min = test.loc[both, "close_min"].to_numpy()
    pred_open_min = pred.loc[both, "open_min"].to_numpy()
    pred_close_min = pred.loc[both, "close_min"].to_numpy()
    opens_err = pd.Series(pred_open_min - actual_open_min).abs()
    closes_err = pd.Series(pred_close_min - actual_close_min).abs()
    exact_opens = (pred.loc[both, "opens"].fillna("") == test.loc[both, "opens"].dt.strftime("%H:%M").fillna("")).mean()
    exact_closes = (
        pred.loc[both, "closes"].fillna("") == test.loc[both, "closes"].dt.strftime("%H:%M").fillna("")
    ).mean()

    print(f"\n=== opens/closes (StatusModel, n={both.sum()} actually-open days) ===")
    print(
        f"opens : MAE={opens_err.mean():.1f}min  median={opens_err.median():.1f}min  exact={exact_opens:.3f}  within30min={(opens_err <= 30).mean():.3f}"
    )
    print(
        f"closes: MAE={closes_err.mean():.1f}min  median={closes_err.median():.1f}min  exact={exact_closes:.3f}  within30min={(closes_err <= 30).mean():.3f}"
    )


if __name__ == "__main__":
    paths = Paths()
    raw = load_crowd_calendar(paths.crowd_calendar)
    parks = park_table(load_parks(paths.parks_csv), pd.read_csv(paths.parks_enriched))
    scores = DailyScores.load(paths.daily_scores)

    frame = build_status_frame(raw, parks, scores)
    last = frame["date"].max()
    cutoff = last - pd.Timedelta(days=HOLD_OUT_DAYS)
    print(f"data through {last.date()}, held out from {cutoff.date()} ({HOLD_OUT_DAYS}d)")

    model = StatusModel().fit(frame, cutoff)
    test = frame[frame["date"] > cutoff].copy()
    test["opens"] = pd.to_datetime(
        raw.set_index(["park_id", "date"])["opens"]
        .reindex(pd.MultiIndex.from_frame(test[["park_id", "date"]]))
        .to_numpy(),
        format="%H:%M",
        errors="coerce",
    )
    test["closes"] = pd.to_datetime(
        raw.set_index(["park_id", "date"])["closes"]
        .reindex(pd.MultiIndex.from_frame(test[["park_id", "date"]]))
        .to_numpy(),
        format="%H:%M",
        errors="coerce",
    )
    pred = model.predict(test)
    print(f"held-out rows: {len(test)}, parks: {test['park_id'].nunique()}")

    score_is_open(test, pred, cutoff)
    score_hours(test, pred)

    imp = pd.Series(model.is_open.booster_.feature_importance("gain"), index=STATUS_FEATURES)
    print(
        "\ntop is_open features (% gain):",
        (100 * imp / imp.sum()).sort_values(ascending=False).head(8).round(1).to_dict(),
    )
    imp = pd.Series(model.closes.booster_.feature_importance("gain"), index=HOURS_FEATURES)
    print(
        "top closes features (% gain):", (100 * imp / imp.sum()).sort_values(ascending=False).head(8).round(1).to_dict()
    )

    pred.assign(actual_is_open=test["is_open"].to_numpy()).to_csv(
        out_dir() / "opening_hours_model_eval.csv", index=False
    )
