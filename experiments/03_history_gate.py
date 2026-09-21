"""How does accuracy depend on how much history a park has, and where should the "min years" gate sit?

Fits the deployed model once (to the validation end), scores the test year, and reports MAE by history bucket and by gate
threshold against the trivial guess of 50.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _common import load_frame, out_dir
from crowdcast.evaluation.splits import SplitDates
from crowdcast.models.config import ModelConfig
from crowdcast.models.gated import GatedCrowdModel

THRESHOLDS = [0, 0.5, 1, 1.5, 2, 3, 4, 5]
BUCKETS = [-0.01, 0, 0.5, 1, 1.5, 2, 3, 4, 20]
LABELS = ["none", "<0.5y", "0.5-1y", "1-1.5y", "1.5-2y", "2-3y", "3-4y", "4y+"]


def main() -> None:
    frame, dates = load_frame(), SplitDates()
    cfg = ModelConfig(min_years=0.0)  # gate open: the park model scores everyone, so history effects are visible
    model = GatedCrowdModel(cfg).fit(frame, dates.val_end)
    test = frame[(frame["date"] > dates.val_end) & (frame["date"] <= dates.test_end)].copy()
    test["pred"] = model.predict(test)["prediction"].to_numpy()
    test["ae"] = (test["pred"] - test["crowd_percent"]).abs()
    test["ae50"] = (50 - test["crowd_percent"]).abs()
    test["history_years"] = test["park_id"].map(model.history_years).fillna(0)

    rows = []
    for g in THRESHOLDS:
        s = test[test["history_years"] >= g]
        per_park = s.groupby("park_id")["ae"].mean()
        rows.append({"min_years": g, "parks": s["park_id"].nunique(), "parks_kept_pct": round(100 * s["park_id"].nunique() / test["park_id"].nunique()),
                     "rows_kept_pct": round(100 * len(s) / len(test)), "MAE": round(s["ae"].mean(), 2), "MAE_guess_50": round(s["ae50"].mean(), 2),
                     "parks_over_20_MAE": int((per_park > 20).sum())})  # fmt: skip
    by_threshold = pd.DataFrame(rows)
    bucket = pd.cut(test["history_years"], BUCKETS, labels=LABELS)
    by_bucket = (
        test.groupby(bucket, observed=True)
        .agg(parks=("park_id", "nunique"), rows=("ae", "size"), MAE=("ae", "mean"), guess_50=("ae50", "mean"))
        .round(1)
    )
    per_park = test.groupby("park_id").agg(years=("history_years", "first"), mae=("ae", "mean"), guess=("ae50", "mean"))
    print(by_threshold.to_string(index=False), "\n")
    print(by_bucket.to_string(), "\n")
    print("Spearman(history, MAE) across parks:", round(per_park[["years", "mae"]].corr("spearman").iloc[0, 1], 2))
    print("parks worse than guessing 50:", int((per_park["mae"] > per_park["guess"]).sum()), "of", len(per_park))
    by_threshold.to_csv(out_dir() / "history_gate_by_threshold.csv", index=False)
    by_bucket.to_csv(out_dir() / "history_gate_by_bucket.csv")
    np.save(out_dir() / "history_gate_park_mae.npy", per_park["mae"].to_numpy())


if __name__ == "__main__":
    main()
