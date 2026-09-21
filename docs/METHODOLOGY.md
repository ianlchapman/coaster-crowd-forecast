# Methodology

## Target

`crowd_percent` (0-100) comes from the loader and is **rank-normalised per park over its whole history**: a value of 90 means the day was busier than 90% of that park's days. Consequences:

* Every park's mean is about 50, so park identity says almost nothing about the *level*; it matters for *shape* (which weekdays, seasons and holidays move that park).
* The ranking includes the test period, so scores are slightly optimistic (a rank computed only from data before a forecast date would be stricter). See the model card.

Only open, non-`predicted`, labelled days are used as targets.

## Splits and protocol

Everything is chronological.

| Split | Dates |
|---|---|
| train | up to 2024-12-31 |
| validation | 2025-01-01 to 2025-08-31 (choose models and hyper-parameters) |
| test | 2025-09-01 to 2026-08-31 (scored once, models refitted on train + validation) |

The COVID window (2020-03-01 to 2021-12-31) is excluded from training: attendance then does not describe normal seasonality. Excluding it was marginally better or equal on validation.

Back-tests (`crowdcast evaluate`) refit to a cutoff and score the following window.

## Features

| Group | Columns | Notes |
|---|---|---|
| Calendar | weekday, weekend, month, ISO week, year, sin/cos of day of year | seasonality is shared across hemispheres via `southern` |
| Holidays | national and school score (0-1); log days until / since the nearest holiday day (score > 0.2, capped at 180 days) | park-specific, see below |
| Hours | open, close, hours open | missing values use the park's median |
| Events | any event, plus six keyword groups (early entry, Halloween, Christmas, summer, festival, ticketed) | contribute ~0.3% of split gain: recurring events are already explained by date and last year |
| Park | age in months, hemisphere, latitude, longitude, company, country, tier, park id | tier is a curated judgement (`reference/park_tiers.csv`) |
| Last year | value at t-364 and mean of t-357/364/371 | same weekday, 51-53 weeks back, strictly past |
| Weather | same-day temperature, feels-like, rain, snow, wind, gusts, sun, code, threshold flags, anomalies against the park's own climatology, rain in the previous 3 days, wet days in the previous 7 | look-ahead weather is not used |
| Drift | mean of (actual - same weekday last year) over the trailing 28 days, using labels up to t-1, t-7 or t-14 | blank when labels are not that fresh |

### Holiday scores

For each park, `score(day) = sum over source regions of weight(park, region) x holiday_intensity(region, day)`, separately for national and school holidays, each in [0, 1].

1. **Calendars.** Public holidays and school holidays per region and date from OpenHolidays (where available), rule-derived calendars (mostly outside Europe, with medium/low confidence) and official term dates for Australia, New Zealand and sampled US districts. Overlapping rows combine as `1 - prod(1 - share)`.
2. **Market prior.** For each park, regions are weighted by population x a distance decay (scale by tier: 150, 600, 2500, 6000 km), an own-country boost and a crude affluence factor. All of these are assumptions (`reference/`).
3. **Weights.** A non-negative ridge regression on the residual of `log(crowd)` after removing weekday, seasonality, year effects, season-edge flags and non-seasonal events. The penalty pulls weights toward the prior; a nested time-series CV chooses how much data-driven signal to blend in (0 to 75%), separately for national and school. Block bootstrap gives intervals and stability. Parks with too little data or no out-of-fold gain keep the prior and are flagged `prior_only`.

Data-driven weights are blended in whenever the nested CV shows an out-of-fold gain above 0.001 R² over the prior. That is a low bar, so a park with almost no signal can still get some data blended in; check `oof_r2`, `school_strength`/`national_strength` and the per-row `confidence` and `stability` before reading much into one park's weights.

Holidays explain only a few percent of residual variance once seasonality is removed (median out-of-fold R² about 0.03; real calendars vs placebo-shifted ones: 0.029 vs about 0), so the scores are a useful feature, not a strong one on their own.

## Model

`GatedCrowdModel` (`src/crowdcast/models/gated.py`):

* **Full model**: LightGBM (600 trees, 127 leaves, learning rate 0.03) on every feature including park id and last-year values. Used when the park has at least `min_years` (1.0) of labelled history at the training cutoff.
* **Fallback model**: same learner without park id, last-year and drift features. Used for everything else, and every row is flagged `low_confidence`.
* **Horizons**: at training time each row draws an "as-of lag" (none, 1, 7 or 14 days) and blanks the drift columns it could not know; 25% of rows also lose all weather. At prediction time `lag=None` gives a long-horizon forecast, `lag=1/7/14` uses drift columns known at that lag. `drift_effect` (prediction with minus without drift) is returned so unusual shifts are visible.

Predictions are clipped to 0-100.

## Why the gate

Accuracy depends strongly on history (Spearman -0.54 between a park's years of history and its error). Parks with under half a year are barely better than guessing 50. A gate at one year keeps most parks and rows while removing most of the worst cases; stricter gates cost coverage for small gains. The fallback is not more accurate than the full model even for short-history parks, so the gate's value is **honest labelling**, not accuracy. See `experiments/03_history_gate.py`.

## Leakage controls

* Splits are by date; the test window is scored after refitting.
* Last-year features look back at least 357 days; drift at lag h only uses labels dated at or before t-h. Tests perturb future labels and assert the features do not move (`tests/test_features.py`, `tests/test_future.py`), and were mutation-checked against a deliberate off-by-one.
* Training ignores every row after the cutoff and every COVID row (tested by poisoning them).
* Weather look-ahead is opt-in and off; scoring on archived forecasts replaces same-day values with what a forecast issued days earlier said.

Known leak that is *not* removed: the rank normalisation of the target (above).

## Live forecasting

`crowdcast forecast` builds rows for each active park from the day after the last label to today + 15 days:

* holiday features from the same `DailyScores` object used in training;
* opening hours and event flags copied from the same weekday a year earlier (close-time error 32 minutes vs 70 with a recent median; event flag right on 80% of days vs 72% for all-zero on a test month);
* last-year and drift features from the labelled history, blank where labels are too old;
* weather: the Open-Meteo forecast API for recent and future days, the archive for older days, derived by the same code as training.

Rows beyond the forecast horizon are predicted without weather (test-year MAE about 17 vs 15.4 with it).
