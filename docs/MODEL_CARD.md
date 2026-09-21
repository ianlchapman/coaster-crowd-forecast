# Model card: gated crowd-level forecaster

## Intended use

Ranking and planning: "how busy will park P be on date D, relative to its own normal?" for parks that have history. Outputs are an index in points (0-100), not a headcount or wait time. Suitable for trip planning and as one input to staffing or marketing analysis. Not suitable as the sole basis for safety-critical or capacity-limit decisions.

## Inputs and output

* Input: park id, date, and the features in `docs/METHODOLOGY.md` (calendar, holidays, weather, opening hours, last-year level, optional recent drift).
* Output: `prediction` (0-100), `model` (`park` or `fallback`), `low_confidence`, `history_years`, and, when drift is used, `drift_effect`.

## Training data

Daily crowd index for about 140 parks worldwide, from mid-2014 to 2026-08-31, plus holiday calendars and Open-Meteo weather. The COVID window is excluded from training. Parks are unevenly represented: a park with several years of history dominates one added last year.

## Evaluation

Chronological back-test on 2025-09-01 to 2026-08-31 (models fitted to 2025-08-31). Full tables in [`RESULTS.md`](RESULTS.md).

* Long horizon, parks with at least a year of history: MAE 15.4, R² 0.52.
* With labels through yesterday: MAE 14.8; through 7 / 14 days ago: 15.0 / 15.0.
* Short-history parks (fallback model): MAE about 20; guessing 50 gives 25.
* Scored on archived weather forecasts instead of observed weather: about +0.1 MAE at 1 day ahead, +0.7 at 7 days ahead; with no weather at all, MAE about 17.

## Known limitations

1. **One test year.** Nothing here shows how errors vary across years; treat the third decimal as noise. Rebuilding the whole pipeline from raw inputs moved MAE from 15.44 to 15.40 because 1% of holiday scores changed by 0.0001.
2. **Rank-normalised target.** Each park's index is ranked over its full history, including the test period, so the test is slightly optimistic and levels are not comparable across parks or over time in absolute terms.
3. **Level shifts are missed.** Residuals are autocorrelated (lag-1 correlation about 0.5), and week-level shocks are about half of the error variance. Feb-Apr 2026 was under-predicted by about 4 points; year handling was ruled out, Easter and other moving holidays did not explain it, and recent-drift features remove about half of it when labels are current.
4. **Weather.** Training uses observed weather; live use gets forecasts (details above). Skill falls off with lead time and there is no weather beyond about 16 days.
5. **Future events and hours.** Not known ahead of time, so they are copied from the same weekday last year. New events, changed opening hours and closures are missed. `open_last_year` flags whether a park was open a year earlier, but predictions are produced for every date, including days a park will be closed.
6. **Curated assumptions.** Park tiers, region populations, affluence factors and rule-derived school calendars are judgement or approximation (`reference/`, `docs/DATA.md`). They shape a *prior* that the data can override, but errors do not vanish.
7. **Data provenance.** The crowd index comes from scraped queue times; measurement problems upstream (missing days, ride closures) are inherited.
8. **Loader lag.** Recent-drift gains need fresh labels; if the loader is behind, only long-horizon accuracy applies.
9. **Saved models are pickles.** `GatedCrowdModel.load` uses joblib, so only load files you created.

## Ethical considerations

No personal data is used. The main risk is over-trust in a number from a low-confidence park: those rows are flagged and should not be presented like the others.

## Reproducing

`crowdcast demo` and `pytest` need no data. Numbers above come from `crowdcast evaluate` and `experiments/` on the real data (see `docs/DATA.md`).
