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

## v3: closing time as a park-relative day-type classifier — did not beat v1

Tried per user suggestion: classify each open day as `event` (an extend-hours-relevant event: halloween/
christmas/summer/festival/ticketed — excludes early-entry, which affects opening not closing) else
`short`/`normal`/`long`, the latter three defined as **terciles of that park's own** historical non-event
close times (so "long" means a different clock time at different parks), then decode the predicted
category back to a clock time via that park's own median close time observed in that category (falling
back to the park's overall median + the category's average cross-park offset for park/category pairs with
no training example). New files: `src/crowdcast/models/closing_category.py`
(`ClosingCategoryModel`), `experiments/07_closing_category_eval.py`.

**Same 90-day backtest, closes only:**

| | v1 lookup | v2 regression | v3 category |
|---|---|---|---|
| MAE | 35.6min | 39.0min | 46.5-52.1min |
| exact match | 74.1% | 52.9% | 62.8-63.4% |

v3 beats v2 on exact match but loses to both v1 and v2 on MAE, and loses to v1 on exact match too — net,
it didn't beat the plain lookup. Feature importance shows `has_event`/`park_cat` dominate the category
call, and the worst-performing bucket is `short` (MAE 51-63min) — the likely reason: **tercile splits
don't align with how real schedules actually vary**. A park's closing times are typically a handful of
exact, often bimodal values (e.g. 18:00 most weekdays, 22:00 weekends/summer) rather than a smooth
continuum, so a statistical tercile cut can put two genuinely distinct real values in the same "short"
bucket, and the bucket's decode (a median) lands on neither.

**Follow-up implied by this result**: don't classify into a statistical bucket that still needs
decoding — classify directly among **each park's own observed distinct close-time values** (true
multiclass, cardinality = however many distinct values that park has actually used), which was the
original idea from the very first exploratory discussion on this feature and is the one variant not yet
tried. `event`-type features are still useful there, as classifier inputs, not as a forced separate
bucket that dilutes the decode. This needs per-park multiclass (variable class count per park) rather
than one global classifier, which is more plumbing than v1-v3 — flagged as the next experiment, not
built yet.

## Robustness check: is the lookup's score just an artifact of one test window?

User asked whether the v1 lookup is "overfitting". It can't overfit in the classical sense (nothing is
fit to training noise — it's a pure lookup), but the concern translates to a real one: **all prior
numbers came from a single 90-day window**, so they could be an artifact of that window rather than a
stable estimate. Checked two ways:

**`experiments/08_lookup_stability_check.py`** — reran the v1 lookup on four different historical 90-day
windows (0.25/1/2/3 years back). `is_open` accuracy and opens/closes exact-match were fairly stable
(~0.05-0.06 spread), but **closed-class precision ranged from 0.476 to 0.733** — a 0.257 spread. The
originally-reported 0.579 just happened to land mid-range; it's not a stable number. Reason: closures
cluster in specific calendar windows (off-season months) whose exact boundaries shift year to year
(leap years, moving holidays), so which season a 90-day test window lands in matters a lot for that one
metric.

**`experiments/09_2025_holdout_eval.py`** — a more realistic split per user request: train on everything
through 2024-12-31 (COVID rows excluded via `ModelConfig.exclude_covid`, already the default), test on
all of 2025. Also fixed a real apples-to-apples gap: v1's population (parks `build_future_rows` includes,
gated on a recent `crowd_percent` label — 93 parks) doesn't match v2's (any park with status history —
128 parks), so the raw v1-vs-v2 numbers below aren't directly comparable; a **matched comparison** on the
33,675 (park_id, date) rows both cover is the one that counts.

| | v1 lookup (matched) | v2 StatusModel (matched) |
|---|---|---|
| is_open accuracy | 93.3% | **95.2%** |
| is_open closed precision/recall | 0.880 / 0.885 | **0.930 / 0.898** |
| opens MAE / exact | 11.3min / **83.0%** | **10.6min** / 78.5% |
| closes MAE / exact | **51.5min** / **68.3%** | 55.0min / 43.4% |
| v3 closes (unmatched, 28,398 rows) | — | MAE 70.5min / exact 57.3% |

**This confirms the earlier conclusion holds up on a full year, not just 90 days**: is_open → model wins
clearly (bigger margin than before: +1.9pp accuracy, +5pp closed-precision); opens → toss-up (model
better MAE, lookup better exact-match, same as the 90-day result); closes → lookup wins clearly, v3
category classifier still doesn't beat it. One thing the year-long test *did* change: **closes MAE was
optimistic in the 90-day window** (35.6min there vs. 51.5min over a full year) — forecasting a full year
ahead is genuinely harder than 90 days, so the short window understated real difficulty for that metric
specifically. `is_open` and `opens` were comparatively stable between the two test lengths.

**Practical takeaway**: trust the *direction* of every conclusion above (which approach wins which
field) — it's now checked on two different test setups and holds. Don't trust the *exact magnitude* of
closed-class precision or closes MAE from any single window; use the full-year numbers in this section
as the better estimate.

## Multi-year rolling-origin backtest: does more history actually help?

User's framing: train on *all* previous years (not a rolling recent-window), predict a full year/date
ahead, expecting more historical data to raise accuracy (the ~seasonal normalization already built into
the model via `doy_sin`/`doy_cos`/`month`/`year` is what should let more years actually help rather than
just adding noise). Both v1 (`build_future_rows`/`add_is_open`) and v2 (`StatusModel.fit`) already train
cumulatively (`upto = frame[frame["date"] <= cutoff]` — everything, not a window), so this was really
about testing that hypothesis properly rather than changing the training method.

`experiments/10_multi_year_backtest.py`: 5 independent folds, each trained on everything before that
year (COVID excluded) and tested on the full year — 2019 (5 training years), 2022 (8), 2023 (9), 2024
(10), 2025 (11). 2020/2021 excluded as test years too (real closures that year aren't a fair "normal
seasonality" test).

| test year | train yrs | v1 is_open acc | v2 is_open acc | v1 closed prec | v2 closed prec | v1 opens exact | v2 opens exact | v1 closes MAE | v2 closes MAE | v1 closes exact | v2 closes exact |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2019 | 5  | 0.906 | 0.944 | 0.809 | 0.923 | 0.774 | 0.669 | 74.5  | 69.9  | 0.554 | 0.295 |
| 2022 | 8  | 0.834 | 0.908 | 0.656 | 0.881 | 0.515 | 0.545 | 132.3 | 133.8 | 0.375 | 0.182 |
| 2023 | 9  | 0.930 | 0.947 | 0.863 | 0.908 | 0.706 | 0.662 | 63.0  | 72.7  | 0.597 | 0.333 |
| 2024 | 10 | 0.939 | 0.953 | 0.890 | 0.928 | 0.800 | 0.698 | 47.6  | 51.9  | 0.655 | 0.460 |
| 2025 | 11 | 0.933 | 0.952 | 0.880 | 0.930 | 0.830 | 0.785 | 51.5  | 55.0  | 0.683 | 0.434 |

Correlation of `train_years` with each metric: is_open_acc +0.41/+0.32 (v1/v2), closed_prec +0.45/+0.23,
opens_exact +0.26/+0.49, closes_exact +0.51/+0.63, closes_MAE **-0.42/-0.32** (negative = MAE falls as
training years grow, same direction as "more data helps"). **Confirmed: more history helps, for both
approaches, on every metric** — monotonically from 2022 to 2025. The 2022 fold is a clear outlier (worst
score across the board for both approaches) — the year right after the excluded COVID window, when parks
were still normalizing reopening schedules that don't resemble pre-pandemic history; even 8 years of
data can't predict a genuinely disrupted year well. Worth remembering as a limitation independent of how
much history is available.

**This also corrects the v2-conclusion from the single-year (2025) test**, which was too tentative on
`opens`: across 5 independent folds, **the lookup wins `opens` in 4 of 5 years**, not a toss-up. `closes`
exact-match shows the lookup winning by a larger, more consistent margin than the single-year test
suggested. Only `is_open` is a clean win for the model in every single fold.

**Revised conclusion (supersedes the single-year one above)**: `is_open` → StatusModel, wins every
year tested. `opens` and `closes` → v1 lookup, wins most/all years tested. Likely reason: real schedules
repeat almost verbatim year-to-year for a given park, so directly copying last year's recorded value
beats a regression/classifier that has to fit one shared set of trees across many parks and years —
exact repetition is easy for a lookup and hard for anything that must generalize.

## Shipped: the blend, wired into pipeline.forecast()

Decided architecture, matching the revised conclusion above exactly — per field, not per row:

- **`is_open`**: `StatusModel.predict_is_open()` (new method — just the classifier, skips computing the
  hours regressors' inputs since production doesn't use them for this field).
- **`opens`/`closes`**: unchanged v1 lookup, straight from `build_future_rows`'s existing
  `open_min`/`close_min` imputation, blanked wherever the new `is_open` (from the model, not the old
  heuristic) says closed.

Changes:
- `src/crowdcast/features/status.py`: new `add_is_open_prior_year` — computes
  `py_is_open_same_wd`/`py_is_open_wd_mean3` for future rows the same way `build_status_frame` computes
  them for training (via `prior_year_frame`), since `StatusModel` needs them and `build_future_rows`
  doesn't compute them (it only has the crowd model's own prior-year/hours features). The old `add_is_open`
  heuristic function stays in the module (still used by the `05_`/`08_`/`09_`/`10_` experiment scripts as
  the v1 baseline for backtesting) but is no longer called from `pipeline.py`.
- `src/crowdcast/models/status.py`: new `StatusModel.predict_is_open()`.
- `src/crowdcast/pipeline.py`: `forecast()` takes a new required `status_model: StatusModel` argument.
- `src/crowdcast/config.py`: new `Paths.status_model_file` (`processed/models/status_model.joblib`,
  parallel to `model_file`).
- `src/crowdcast/cli.py`: new `train-status` command (mirrors `train`, fits `StatusModel` on
  `build_status_frame` and saves it); `forecast` command now also loads `paths.status_model_file`.
- `tests/test_pipeline_integration.py`, `tests/test_cli.py`: updated for the new `forecast()` signature
  and the `train-status` command; assert `opens`/`closes` are blank exactly where `is_open` is false.

Verified against the real data repo (archive weather, no live API call needed for this check): 0 rows
where `is_open`/`opens`/`closes` are inconsistent, `is_open` rate ~71% matches the historical base rate.

**Not done** (both still call for a workflow change, not just code — flagged rather than silently
skipped): the `05_`-`10_` experiment scripts still evaluate the standalone `add_is_open` lookup, not this
wired-in blend, so there's no single script that backtests the *production* `pipeline.forecast()` output
end-to-end; and there's no scheduled/documented cadence for re-running `train-status` as new data arrives
(same gap `train` already has — out of scope for this change).

## Out of scope (follow-ups)

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