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

- `src/crowdcast/data/loaders.py`: `load_status_calendar` — raw calendar incl. closed days.
- `src/crowdcast/features/status.py` (new): `add_is_open` fallback-chain lookup.
- `src/crowdcast/pipeline.py`: `forecast()` now emits `is_open`, `opens`, `closes`; blanks
  `opens`/`closes` when `is_open` is False.
- `tests/test_status.py` (new): fallback-chain cases + blank-when-closed check.

## Out of scope (follow-ups)

- Trained is_open classifier (vs. lookup).
- One-off/irregular closures (e.g. unannounced single-day closure) not tied to weekday pattern.
- Crowd model itself doesn't yet skip/adjust prediction on predicted-closed days — the two fields
  are consumed together by the caller, not coupled internally.

## Docs to update once this lands

- `docs/METHODOLOGY.md` — new section on the hours/is_open lookup chain.
- `docs/MODEL_CARD.md` — note the heuristic (non-ML) nature of is_open/opens/closes, limitations.
- `TUTORIAL.md` — mention new forecast output columns.