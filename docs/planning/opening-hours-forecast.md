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

## Out of scope (follow-ups)

- Trained is_open classifier (vs. lookup) — closed-class precision (0.579) is the clearest opportunity.
- Better closing-time modeling (currently the weakest metric: 35.6min MAE vs 11.3min for opening time) —
  likely needs event/season features the lookup doesn't use (late nights, extended-hours events).
- One-off/irregular closures (e.g. unannounced single-day closure) not tied to weekday pattern.
- Crowd model itself doesn't yet skip/adjust prediction on predicted-closed days — the two fields
  are consumed together by the caller, not coupled internally.

## Docs to update once this lands

- `docs/METHODOLOGY.md` — new section on the hours/is_open lookup chain.
- `docs/MODEL_CARD.md` — note the heuristic (non-ML) nature of is_open/opens/closes, and the evaluation
  numbers above (especially the closed-class precision/recall and closing-time MAE limitations).
- `docs/DATA.md` — note `CROWDCAST_DATA_DIR` as the way to point at a real data checkout for evaluation.
- `TUTORIAL.md` — mention new forecast output columns.