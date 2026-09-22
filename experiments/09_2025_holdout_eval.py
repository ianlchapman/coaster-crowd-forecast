"""A more realistic split than the 90-day windows in 05_/06_/07_: train on everything through 2024
(COVID rows already excluded by ModelConfig.exclude_covid, the default), test on all of 2025. Runs all
three approaches (v1 lookup, v2 StatusModel, v3 ClosingCategoryModel) on the same split so the
earlier "is_open model wins, closes lookup wins" conclusion can be checked against a full year rather
than one 90-day window.
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
from crowdcast.models.closing_category import ClosingCategoryModel
from crowdcast.models.status import StatusModel
from crowdcast.scoring.daily import DailyScores
from crowdcast.weather.archive import load_archive
from crowdcast.weather.features import derive_weather_features

CUTOFF = pd.Timestamp("2024-12-31")
TEST_END = pd.Timestamp("2025-12-31")


def is_open_metrics(is_open: pd.Series, actual_open: pd.Series) -> dict:
    is_open, actual_open = is_open.to_numpy(), actual_open.to_numpy()
    tp = int((is_open & actual_open).sum())
    tn = int((~is_open & ~actual_open).sum())
    fp = int((is_open & ~actual_open).sum())
    fn = int((~is_open & actual_open).sum())
    return {
        "n": len(is_open),
        "actual_open_rate": round(actual_open.mean(), 3),
        "accuracy": round((tp + tn) / len(is_open), 4),
        "closed_precision": round(tn / (tn + fn), 3) if (tn + fn) else float("nan"),
        "closed_recall": round(tn / (tn + fp), 3) if (tn + fp) else float("nan"),
    }


def hours_metrics(pred_min: pd.Series, actual_min: pd.Series, pred_str: pd.Series, actual_str: pd.Series) -> dict:
    err = (pred_min.to_numpy() - actual_min.to_numpy())
    err_abs = pd.Series(err).abs()
    exact = (pred_str.fillna("").to_numpy() == actual_str.fillna("").to_numpy()).mean()
    return {"n": len(err_abs), "MAE": round(err_abs.mean(), 1), "exact": round(exact, 3), "within30min": round((err_abs <= 30).mean(), 3)}


if __name__ == "__main__":
    paths = Paths()
    raw = load_crowd_calendar(paths.crowd_calendar)
    parks = park_table(load_parks(paths.parks_csv), pd.read_csv(paths.parks_enriched))
    scores = DailyScores.load(paths.daily_scores)
    print(f"train: everything through {CUTOFF.date()} (COVID rows excluded)  |  test: {CUTOFF.date()} - {TEST_END.date()}")

    # --- v1: lookup heuristic --------------------------------------------------------------------------
    labelled = load_frame()
    weather = derive_weather_features(load_archive(paths.weather_archive))
    weather["wx_source"] = "archive"
    rows = build_future_rows(labelled, scores, weather, end=TEST_END, asof=CUTOFF)
    rows = add_is_open(rows, raw[["park_id", "date", "status"]], CUTOFF)
    rows["pred_opens"] = minutes_to_hhmm(rows["open_min"]).where(rows["is_open"])
    rows["pred_closes"] = minutes_to_hhmm(rows["close_min"]).where(rows["is_open"])

    truth = raw[raw["status"].isin(["open", "closed"])][["park_id", "date", "status", "opens", "closes"]]
    truth = truth.assign(actual_open=truth["status"] == "open")
    v1 = rows.merge(truth, on=["park_id", "date"], how="inner")
    print(f"\nv1 lookup: {len(v1)} held-out rows, {v1['park_id'].nunique()} parks")
    print("is_open:", is_open_metrics(v1["is_open"], v1["actual_open"]))
    v1_open = v1[v1["actual_open"]]
    print("opens  :", hours_metrics(v1_open["open_min"], pd.to_datetime(v1_open["opens"], format="%H:%M").dt.hour * 60 + pd.to_datetime(v1_open["opens"], format="%H:%M").dt.minute, v1_open["pred_opens"], v1_open["opens"]))
    v1_close_actual_min = pd.to_datetime(v1_open["closes"], format="%H:%M", errors="coerce")
    v1_close_actual_min = v1_close_actual_min.dt.hour * 60 + v1_close_actual_min.dt.minute
    print("closes :", hours_metrics(v1_open["close_min"], v1_close_actual_min, v1_open["pred_closes"], v1_open["closes"]))

    # --- v2: StatusModel (LightGBM classifier + regressors) --------------------------------------------
    frame = build_status_frame(raw, parks, scores)
    model2 = StatusModel().fit(frame, CUTOFF)
    test = frame[(frame["date"] > CUTOFF) & (frame["date"] <= TEST_END)].copy()
    pred2 = model2.predict(test)
    print(f"\nv2 StatusModel: {len(test)} held-out rows, {test['park_id'].nunique()} parks")
    print("is_open:", is_open_metrics(pred2["is_open"], test["is_open"]))
    both = test["is_open"].to_numpy()
    print("opens  :", hours_metrics(pred2.loc[both, "open_min"], test.loc[both, "open_min"], pred2.loc[both, "opens"], test.loc[both, "opens"]))
    print("closes :", hours_metrics(pred2.loc[both, "close_min"], test.loc[both, "close_min"], pred2.loc[both, "closes"], test.loc[both, "closes"]))

    # --- v3: ClosingCategoryModel (closes only) ---------------------------------------------------------
    model3 = ClosingCategoryModel().fit(frame, CUTOFF)
    test3 = test[test["is_open"] & test["close_min"].notna()]
    pred3 = model3.predict(test3)
    print(f"\nv3 ClosingCategoryModel: {len(test3)} held-out actually-open rows")
    print("closes :", hours_metrics(pred3["close_min"], test3["close_min"], pred3["closes"], test3["closes"]))

    # --- matched comparison: v1 and v2 don't cover the same park population by construction --------------
    # (v1's build_future_rows only includes parks with a recent crowd_percent label; v2's build_status_frame
    # includes any park with status history, even ones the crowd model never sees) -- restrict both to the
    # exact same (park_id, date) rows before comparing, otherwise "v2 wins" could just mean "easier parks".
    v1_keyed = v1.set_index(["park_id", "date"])
    v2_keyed = pred2.set_index(["park_id", "date"])[["is_open", "open_min", "close_min", "opens", "closes"]]
    common = v1_keyed.index.intersection(v2_keyed.index)
    v1m, v2m = v1_keyed.loc[common], v2_keyed.loc[common]
    print(f"\n=== matched comparison: {len(common)} (park_id, date) rows common to both v1 and v2's test sets ===")
    print("v1 is_open:", is_open_metrics(v1m["is_open"], v1m["actual_open"]))
    print("v2 is_open:", is_open_metrics(v2m["is_open"], v1m["actual_open"]))
    both_m = v1m["actual_open"].to_numpy()
    v1m_open, v2m_open = v1m[both_m], v2m[both_m]
    actual_open_min = pd.to_datetime(v1m_open["opens"], format="%H:%M", errors="coerce")
    actual_open_min = actual_open_min.dt.hour * 60 + actual_open_min.dt.minute
    actual_close_min = pd.to_datetime(v1m_open["closes"], format="%H:%M", errors="coerce")
    actual_close_min = actual_close_min.dt.hour * 60 + actual_close_min.dt.minute
    print("v1 opens  :", hours_metrics(v1m_open["open_min"], actual_open_min, v1m_open["pred_opens"], v1m_open["opens"]))
    print("v2 opens  :", hours_metrics(v2m_open["open_min"], actual_open_min, v2m_open["opens"], v1m_open["opens"]))
    print("v1 closes :", hours_metrics(v1m_open["close_min"], actual_close_min, v1m_open["pred_closes"], v1m_open["closes"]))
    print("v2 closes :", hours_metrics(v2m_open["close_min"], actual_close_min, v2m_open["closes"], v1m_open["closes"]))

    pd.DataFrame(
        {"cutoff": [str(CUTOFF.date())], "test_end": [str(TEST_END.date())]}
    ).to_csv(out_dir() / "2025_holdout_eval_meta.csv", index=False)
