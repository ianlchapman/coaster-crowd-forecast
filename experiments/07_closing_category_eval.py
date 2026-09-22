"""Backtest ClosingCategoryModel (park-relative short/normal/long/event classifier, decoded to a clock
time via each park's own history) against the same 90-day real-data window as 05_/06_, for a direct
three-way comparison on `closes`: lookup vs. free regression vs. this.
"""

from __future__ import annotations

import pandas as pd
from _common import out_dir

from crowdcast.config import Paths
from crowdcast.data.loaders import load_crowd_calendar, load_parks
from crowdcast.features.build import park_table
from crowdcast.features.status_build import build_status_frame
from crowdcast.models.closing_category import ClosingCategoryModel
from crowdcast.scoring.daily import DailyScores

HOLD_OUT_DAYS = 90

if __name__ == "__main__":
    paths = Paths()
    raw = load_crowd_calendar(paths.crowd_calendar)
    parks = park_table(load_parks(paths.parks_csv), pd.read_csv(paths.parks_enriched))
    scores = DailyScores.load(paths.daily_scores)

    frame = build_status_frame(raw, parks, scores)
    last = frame["date"].max()
    cutoff = last - pd.Timedelta(days=HOLD_OUT_DAYS)
    print(f"data through {last.date()}, held out from {cutoff.date()} ({HOLD_OUT_DAYS}d)")

    model = ClosingCategoryModel().fit(frame, cutoff)
    test = frame[(frame["date"] > cutoff) & frame["is_open"] & frame["close_min"].notna()].copy()
    pred = model.predict(test)
    print(f"held-out actually-open rows with a recorded schedule: {len(test)}")

    err = (pred["close_min"].to_numpy() - test["close_min"].to_numpy())
    err_abs = pd.Series(err).abs()
    exact = (pred["closes"].to_numpy() == test["closes"].to_numpy()).mean()
    print("\n=== closes (ClosingCategoryModel) ===")
    print(f"MAE={err_abs.mean():.1f}min  median={err_abs.median():.1f}min  exact={exact:.3f}  within30min={(err_abs <= 30).mean():.3f}")

    print("\npredicted category distribution:", pred["close_category"].value_counts(normalize=True).round(3).to_dict())

    imp = pd.Series(model.clf.booster_.feature_importance("gain"), index=model.clf.feature_name_)
    print("\ntop category features (% gain):", (100 * imp / imp.sum()).sort_values(ascending=False).head(8).round(1).to_dict())

    print("\nMAE by predicted category:")
    by_cat = pd.DataFrame({"cat": pred["close_category"].to_numpy(), "err": err_abs})
    print(by_cat.groupby("cat")["err"].agg(["count", "mean"]).round(1))

    pred.assign(actual_close_min=test["close_min"].to_numpy(), actual_closes=test["closes"].to_numpy()).to_csv(
        out_dir() / "closing_category_eval.csv", index=False
    )
