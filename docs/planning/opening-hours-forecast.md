# Opening hours + is_open forecast — progress log

Branch: `feature/opening-hours-forecast`

Goal: forecast output gains `is_open`, `opens`, `closes` per park-day, alongside existing `crowd_percent`.

## Approach

Lookup-table heuristic, not a trained classifier — closed days are rare, sparse, and mostly
calendar-driven (weekday + season). Mirrors the existing hours-imputation fallback chain in
`features/future.py::_add_hours_and_events`:

1. same weekday, same date last year -> use its recorded status/hours
2. else park+weekday rate/median over trailing 84 days
3. else park all-time rate/median

Closed days were previously dropped everywhere (`scoring/enhance.py` filters `status != "closed"`
before the crowd-percent training frame is even built), so a separate status-calendar load path was
needed to retain them.

## Changes

- `src/crowdcast/features/status.py` (new): `add_is_open` fallback-chain lookup, `minutes_to_hhmm`
  formatter. Reads closed days straight from `load_crowd_calendar` (already returns them unfiltered) —
  no separate loader needed.
- `src/crowdcast/pipeline.py`: `forecast()` now emits `is_open`, `opens`, `closes`; blanks
  `opens`/`closes` when `is_open` is False. Output column order: `park_id, park_name, date, is_open,
  opens, closes, ...`.
- `tests/test_status.py` (new): fallback-chain cases + blank-when-closed check.
- `experiments/05_opening_hours_eval.py` (new): backtest against real data (see Evaluation below).

## Evaluation against real data

Real data lives in the sibling repo `../coaster-crowd-forecast.data/data` (point `CROWDCAST_DATA_DIR` at
it — no copying needed, `Paths` picks it up via env var). 124 parks, 2014-2026, pre-built model and
holiday scores already present.

Backtest method (`experiments/05_opening_hours_eval.py`): hold out the last 90 days of real labelled
history as if unknown (`asof=cutoff`), predict `is_open`/`opens`/`closes` for that window exactly as
`pipeline.forecast()` does, compare against what the raw calendar actually recorded. 8,969 held-out
park-days with usable ground truth (open/closed only; unknown/no_data statuses excluded, ~1.6% of rows).

**is_open**: 96.3% accuracy vs. a 95.4% naive "always open" baseline (open-rate is heavily imbalanced,
so accuracy alone understates the lift). Open-class precision/recall 0.986/0.975. Closed-class
precision/recall 0.579/0.705 — the heuristic catches ~70% of actual closures but over-predicts closures
too (42% of predicted-closed days were actually open). Accuracy is flat across horizon (95.6% at 1-7
days out, 96.9% at 31-90 days out) — the same-weekday-last-year lookup does the heavy lifting regardless
of how far ahead we're predicting.

**opens/closes** (8,537 actually-open days with a recorded schedule): opening time MAE 11.3min (89.1%
exact HH:MM match, 93.1% within 30min). Closing time is noticeably worse: MAE 35.6min, only 74.1% exact
match — closing times vary more (late nights, seasonal extended hours) than opening times, which the
weekday/last-year lookup doesn't fully capture. Both degrade mildly with horizon (opens MAE 8.8min at
1-7d -> 12.0min at 31-90d; closes 28.4min -> 36.9min).

**Takeaway**: the lookup heuristic is a solid floor for `is_open` and opening time, but closing time and
the closed-class precision are the weak points — worth targeting first if/when a trained classifier
replaces this (see Out of scope below).

## v2: feature-driven model (LightGBM), evaluated against the lookup

Built to test whether year-on-year trend features (the same `prior_year_frame` mechanism the
crowd_percent model already uses for `py_same_wd`/`py_wd_mean3`) beat the plain lookup, per user
request. New files:

- `src/crowdcast/features/status_build.py` — `build_status_frame`: one row per open-or-closed park-day
  (unlike `build_feature_frame`, keeps closed days) with calendar/holiday/park features plus
  `py_is_open_same_wd`/`py_is_open_wd_mean3` and the same for `open_min`/`close_min`. No `asof` leakage
  guard needed — the ~364-day lookback is far longer than any realistic held-out window.
- `src/crowdcast/models/status.py` — `StatusModel`: one `LGBMClassifier` (is_open) + two `LGBMRegressor`s
  (open_min/close_min, trained only on actually-open rows). Single-stage, not gated by park history length
  like `GatedCrowdModel` — untested whether new parks need that; LightGBM handles the NaN prior-year
  features they'd get. Regressor output is rounded to the nearest 30 minutes before formatting (real
  schedules sit on a 30-min grid — see the `opens`/`closes` value counts in the real data), which matters
  a lot for exact-match (see below).
- `experiments/06_opening_hours_model_eval.py` — same 90-day held-out backtest as
  `05_opening_hours_eval.py`, for direct comparison.

**Same 90-day real-data backtest, model vs. lookup heuristic:**

| metric | lookup (v1) | LightGBM (v2) |
|---|---|---|
| is_open accuracy | 96.3% | 96.4% |
| is_open closed-class precision/recall | 0.579 / 0.705 | **0.726 / 0.760** |
| opens MAE / exact match | 11.3min / 89.1% | **8.2min** / 85.1% |
| closes MAE / exact match | 35.6min / 74.1% | 39.0min / 52.9% |

Mixed result, not a clean win: **is_open is clearly better** (the closed-class numbers were the lookup's
known weak spot, and the model closes a good chunk of that gap — plausibly via `school_holiday` and
`doy_cos`, both in its top-8 features by gain, alongside `py_is_open_same_wd`/`py_is_open_wd_mean3`
which still dominate). **Opens is a modest win** on MAE, roughly a wash on exact match. **Closes is worse**
on both — before rounding, closes exact match was only 2.7% (continuous regression essentially never
lands on the discrete true value); rounding recovered it to 52.9%, still well below the lookup's 74.1%.
Closing time has more distinct values in the real data than opening time (see value counts run during
this eval) and the top features (`py_close_min_wd_mean3`, `py_close_min`, `park_cat`, `months_open`) are
the same drivers, so the regression framing itself — not missing features — looks like the limitation.

**Conclusion**: worth swapping `is_open` to the model; `opens` is a toss-up; keep the lookup for `closes`
until the hours prediction is reframed as classification over each park's own observed schedule values
rather than free regression (see follow-up below) — that's the natural next step given this result, not
more feature engineering.

## Out of scope (follow-ups)

- Wire `StatusModel` into `pipeline.forecast()` for `is_open` (validated win); decide on `opens`/`closes`
  once reframed as classification over each park's observed schedule values (see v2 conclusion above).
- History-length gating for `StatusModel` like `GatedCrowdModel` has for crowd_percent — untested whether
  brand-new parks (all-NaN prior-year features) need a simpler fallback model.
- One-off/irregular closures (e.g. unannounced single-day closure) not tied to weekday pattern.
- Crowd model itself doesn't yet skip/adjust prediction on predicted-closed days — the two fields
  are consumed together by the caller, not coupled internally.

## Docs to update once this lands

- `docs/METHODOLOGY.md` — new section on the hours/is_open lookup chain.
- `docs/MODEL_CARD.md` — note the heuristic (non-ML) nature of is_open/opens/closes, and the evaluation
  numbers above (especially the closed-class precision/recall and closing-time MAE limitations).
- `docs/DATA.md` — note `CROWDCAST_DATA_DIR` as the way to point at a real data checkout for evaluation.
- `TUTORIAL.md` — mention new forecast output columns.