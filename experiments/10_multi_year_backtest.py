"""Rolling-origin, full-year backtest: for each test year, train on *everything* before it (not a
rolling window -- StatusModel.fit and the lookup both already use all history up to the cutoff, COVID
rows excluded by default) and test on the full year that follows. Two things this checks:

1. Does the v1-lookup-wins-closes / v2-model-wins-is_open conclusion hold across several independent
   years, not just 2025 (09_2025_holdout_eval.py)?
2. Does more training history actually help? Each later test year trains on strictly more accumulated
   years of data, so if the "more data -> higher accuracy" intuition is right, later folds should score
   at least as well as earlier ones (all else -- underlying seasonality -- being roughly equal).
"""

from __future__ import annotations

import pandas as pd
from _common import load_frame, out_dir

from crowdcast.config import Paths
from crowdcast.data.loaders import load_crowd_calendar, load_parks
from crowdcast.features.build import park_table
from crowdcast.features.future import build_future_rows
from crowdcast.features.status import add_is_open, minutes_to_hhmm
from crowdcast.features.status_build import build_status_frame
from crowdcast.models.status import StatusModel
from crowdcast.scoring.daily import DailyScores
from crowdcast.weather.archive import load_archive
from crowdcast.weather.features import derive_weather_features

TEST_YEARS = [2019, 2022, 2023, 2024, 2025]  # skip 2020/2021: real closures that year aren't a fair "normal seasonality" test


def is_open_metrics(is_open: pd.Series, actual_open: pd.Series) -> dict:
    is_open, actual_open = is_open.to_numpy(), actual_open.to_numpy()
    tp, tn = int((is_open & actual_open).sum()), int((~is_open & ~actual_open).sum())
    fp, fn = int((is_open & ~actual_open).sum()), int((~is_open & actual_open).sum())
    return {
        "accuracy": round((tp + tn) / len(is_open), 4),
        "closed_precision": round(tn / (tn + fn), 3) if (tn + fn) else float("nan"),
        "closed_recall": round(tn / (tn + fp), 3) if (tn + fp) else float("nan"),
    }


def hours_metrics(pred_min: pd.Series, actual_min: pd.Series, pred_str: pd.Series, actual_str: pd.Series) -> dict:
    err_abs = (pred_min.to_numpy() - actual_min.to_numpy())
    err_abs = pd.Series(err_abs).abs()
    exact = (pred_str.fillna("").to_numpy() == actual_str.fillna("").to_numpy()).mean()
    return {"MAE": round(err_abs.mean(), 1), "exact": round(exact, 3)}


def run_fold(paths: Paths, labelled: pd.DataFrame, raw: pd.DataFrame, frame: pd.DataFrame, scores, weather, test_year: int) -> dict:
    cutoff = pd.Timestamp(f"{test_year - 1}-12-31")
    test_end = pd.Timestamp(f"{test_year}-12-31")
    train_years = labelled.loc[labelled["date"] <= cutoff, "date"].dt.year.nunique()

    rows = build_future_rows(labelled, scores, weather, end=test_end, asof=cutoff)
    rows = add_is_open(rows, raw[["park_id", "date", "status"]], cutoff)
    rows["pred_opens"] = minutes_to_hhmm(rows["open_min"]).where(rows["is_open"])
    rows["pred_closes"] = minutes_to_hhmm(rows["close_min"]).where(rows["is_open"])
    truth = raw[raw["status"].isin(["open", "closed"])][["park_id", "date", "status", "opens", "closes"]]
    truth = truth.assign(actual_open=truth["status"] == "open")
    v1 = rows.merge(truth, on=["park_id", "date"], how="inner")

    model2 = StatusModel().fit(frame, cutoff)
    test = frame[(frame["date"] > cutoff) & (frame["date"] <= test_end)]
    pred2 = model2.predict(test)

    v1_keyed = v1.set_index(["park_id", "date"])
    v2_keyed = pred2.set_index(["park_id", "date"])[["is_open", "open_min", "close_min", "opens", "closes"]]
    common = v1_keyed.index.intersection(v2_keyed.index)
    v1m, v2m = v1_keyed.loc[common], v2_keyed.loc[common]
    both = v1m["actual_open"].to_numpy()
    v1o, v2o = v1m[both], v2m[both]
    actual_open_min = pd.to_datetime(v1o["opens"], format="%H:%M", errors="coerce")
    actual_open_min = actual_open_min.dt.hour * 60 + actual_open_min.dt.minute
    actual_close_min = pd.to_datetime(v1o["closes"], format="%H:%M", errors="coerce")
    actual_close_min = actual_close_min.dt.hour * 60 + actual_close_min.dt.minute

    return {
        "test_year": test_year,
        "train_years": train_years,
        "n": len(common),
        "v1_is_open": is_open_metrics(v1m["is_open"], v1m["actual_open"]),
        "v2_is_open": is_open_metrics(v2m["is_open"], v1m["actual_open"]),
        "v1_opens": hours_metrics(v1o["open_min"], actual_open_min, v1o["pred_opens"], v1o["opens"]),
        "v2_opens": hours_metrics(v2o["open_min"], actual_open_min, v2o["opens"], v1o["opens"]),
        "v1_closes": hours_metrics(v1o["close_min"], actual_close_min, v1o["pred_closes"], v1o["closes"]),
        "v2_closes": hours_metrics(v2o["close_min"], actual_close_min, v2o["closes"], v1o["closes"]),
    }


if __name__ == "__main__":
    paths = Paths()
    raw = load_crowd_calendar(paths.crowd_calendar)
    parks = park_table(load_parks(paths.parks_csv), pd.read_csv(paths.parks_enriched))
    scores = DailyScores.load(paths.daily_scores)
    labelled = load_frame()
    weather = derive_weather_features(load_archive(paths.weather_archive))
    weather["wx_source"] = "archive"
    frame = build_status_frame(raw, parks, scores)

    results = [run_fold(paths, labelled, raw, frame, scores, weather, y) for y in TEST_YEARS]

    rows = []
    for r in results:
        rows.append(
            {
                "test_year": r["test_year"],
                "train_years": r["train_years"],
                "n": r["n"],
                "v1_is_open_acc": r["v1_is_open"]["accuracy"],
                "v2_is_open_acc": r["v2_is_open"]["accuracy"],
                "v1_closed_prec": r["v1_is_open"]["closed_precision"],
                "v2_closed_prec": r["v2_is_open"]["closed_precision"],
                "v1_opens_exact": r["v1_opens"]["exact"],
                "v2_opens_exact": r["v2_opens"]["exact"],
                "v1_closes_MAE": r["v1_closes"]["MAE"],
                "v2_closes_MAE": r["v2_closes"]["MAE"],
                "v1_closes_exact": r["v1_closes"]["exact"],
                "v2_closes_exact": r["v2_closes"]["exact"],
            }
        )
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv(out_dir() / "multi_year_backtest.csv", index=False)

    print("\ncorrelation of train_years with each metric (does more training data actually help?):")
    for col in [c for c in out.columns if c not in ("test_year", "train_years", "n")]:
        print(f"  {col}: {out['train_years'].corr(out[col]):+.2f}")
