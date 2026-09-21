"""Baselines vs Ridge vs forests vs LightGBM, chronologically: fit on train -> score validation; refit on train+validation -> score test.

Uses the calendar / holiday / hours / event / park / last-year features only (no weather), i.e. the model-comparison stage that
came before weather was added. COVID days are excluded from training for the learned models (baselines see them).
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from _common import load_frame, out_dir, timed
from crowdcast.evaluation.metrics import regression_metrics
from crowdcast.evaluation.splits import SplitDates, time_split
from crowdcast.features.calendar import in_covid
from crowdcast.models.zoo import F_ID, F_ID_PY, LightGBM, SklearnTrees, default_models

TUNING_GRID = [
    {"num_leaves": 15, "min_child_samples": 50},
    {"num_leaves": 31, "min_child_samples": 50},
    {"num_leaves": 63, "min_child_samples": 100},
    {"num_leaves": 127, "min_child_samples": 50},
    {"num_leaves": 31, "min_child_samples": 50, "n_estimators": 1200, "learning_rate": 0.015},
    {"num_leaves": 31, "min_child_samples": 200, "reg_lambda": 10},
]


def score(frame: pd.DataFrame, evaluate: pd.Series, pred: object) -> dict[str, float]:
    return regression_metrics(
        frame.loc[evaluate, "crowd_percent"].to_numpy(), pred, frame.loc[evaluate, "park_id"].to_numpy()
    )  # type: ignore[arg-type]


def main() -> None:
    frame = load_frame()
    train, val, test = time_split(frame, SplitDates())
    covid = in_covid(frame["date"])
    models = default_models()
    rows = []

    print("== validation (fit on train, COVID excluded for learned models)")
    for name, model in models.items():
        fit_on = train if name.startswith("B") else train & ~covid
        with timed(name):
            rows.append(
                {"split": "validation", "model": name, **score(frame, val, model.fit_predict(frame, fit_on, val))}
            )
        print(rows[-1], flush=True)

    print("== tuning LightGBM on validation")
    best = min(
        TUNING_GRID,
        key=lambda g: score(frame, val, LightGBM(F_ID_PY, g).fit_predict(frame, train & ~covid, val))["MAE"],
    )
    print("best params:", best)

    print("== test (refit on train + validation)")
    fit_on = (train | val) & ~covid
    tuned = {
        "B0 global mean": models["B0 global mean"],
        "B2 park x weekday mean": models["B2 park x weekday mean"],
        "B3 park x weekday x month mean": models["B3 park x weekday x month mean"],
        "B4 same weekday last year": models["B4 same weekday last year"],
        "M1 Ridge (park x weekday/month/holiday)": models["M1 Ridge (park x weekday/month/holiday)"],
        "M2 RandomForest": models["M2 RandomForest"],
        "M3 ExtraTrees": models["M3 ExtraTrees"],
        "M4t LightGBM tuned (park id, no last-year)": LightGBM(F_ID, best),
        "M5t LightGBM tuned (park id + last-year)": LightGBM(F_ID_PY, best),
        "M6 LightGBM (no park id)": models["M6 LightGBM (no park id)"],
    }
    preds = {}
    for name, model in tuned.items():
        mask = (train | val) if name.startswith("B") else fit_on
        with timed(name):
            preds[name] = model.fit_predict(frame, mask, test)
        rows.append({"split": "test", "model": name, **score(frame, test, preds[name])})
        print(rows[-1], flush=True)
    et = SklearnTrees(
        ExtraTreesRegressor(n_estimators=200, min_samples_leaf=5, max_features=0.7, n_jobs=8, random_state=0), F_ID
    )
    ensemble = 0.5 * et.fit_predict(frame, fit_on, test) + 0.5 * preds["M5t LightGBM tuned (park id + last-year)"]
    rows.append({"split": "test", "model": "ENS ExtraTrees + tuned LightGBM", **score(frame, test, ensemble)})

    result = pd.DataFrame(rows)
    result.to_csv(out_dir() / "model_comparison.csv", index=False)
    print(result.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
