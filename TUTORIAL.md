# Tutorial: building a crowd-level forecaster, and the traps on the way

This is the story of how this repository was built, written for software engineers and ML practitioners who are comfortable with pandas, gradient boosting and train/validation/test splits, and want to see the decisions, dead ends and checks that separate a notebook from something you would trust.

It follows the order the work actually happened. Each section says what we did, why, what the numbers said, and where to look in the code. Numbers come from the held-out year **2025-09-01 to 2026-08-31** and are reproduced by `crowdcast evaluate` and `experiments/` (see [docs/RESULTS.md](docs/RESULTS.md)) unless a section says otherwise.

## Why build on an existing crowd score?

This project did not start from a blank page. Crowd calendars already exist, and the one we used is genuinely impressive: it uses gradient-boosted regressors on the time of year, park opening times and weather, so it learns from years of past data, and its accuracy improves the longer a park has been on the site. For most days at most parks it is a very good guide, and we wanted to build on that work, not replace it.

What started this repository was a trip. We visited a park during the French school holidays, and the crowd score was well out: the park was packed. It was not a marginal miss.

The reason is easy to see once you look for it. A model built on time of year, opening hours and weather has no way to know that French schools are on holiday. The date looks like any other date, the hours are normal, the weather is fine, and school holidays in France (like those in other countries) do not fall on the same dates every year, so "last year around now" can be off by a week or more. The park was full because of who was free to visit, and nothing in the inputs described that.

That gave us the question this tutorial answers: **how do we add holidays, in the places a park's visitors come from, to a crowd score that is already good?** The answer became park-specific national and school holiday scores (section 3.2), a model that stays cautious where a park has little history (section 8), and an evaluation protocol that lets us say how much better, and where it is not (sections 1 and 2).

**Following along.** Everything runs on synthetic data with no network:

```bash
pip install -e ".[dev]"
crowdcast demo       # features -> back-test -> forecast on synthetic data
pytest               # the guarantees discussed below, as tests
```

The real crowd data is not included (see [NOTICE.md](NOTICE.md)), so real-data sections describe the results and point at the code.

---

## 0. The problem, and why it is harder than it looks

**Goal.** For a theme park and a date, predict `crowd_percent`, a 0-100 index of how busy the park is.

Three properties of the target shape everything that follows:

1. **It is rank-normalised per park.** 90 means "busier than 90% of that park's days". Every park's mean is about 50, so the park id says almost nothing about the *level*. It matters for the *shape* (which weekdays, seasons and holidays move that park).
2. **It is a panel with strong autocorrelation.** About 140 parks, about 12 years of daily data, heavy weekly and yearly seasonality. Neighbouring days are nearly the same row.
3. **The ranking includes the future.** Because each park's ranking uses its whole history, including the test period, test scores are slightly optimistic. We cannot fix that without recomputing the target, so we document it ([model card](docs/MODEL_CARD.md)).

Point 2 is the one that ruins naive projects, so we deal with it first.

## 1. Fix the evaluation protocol before touching a model

With autocorrelated data, a random split leaks: a random validation day sits between two training days from the same week. So everything is split by date, and the test year is scored once:

```python
# src/crowdcast/evaluation/splits.py
@dataclass(frozen=True)
class SplitDates:
    train_end: str = "2024-12-31"
    val_end: str = "2025-08-31"
    test_end: str = "2026-08-31"
```

| Split | Dates | Used for |
|---|---|---|
| train | up to 2024-12-31 | fitting |
| validation | 2025-01-01 to 2025-08-31 | choosing models and hyper-parameters |
| test | 2025-09-01 to 2026-08-31 | scoring once, after refitting on train + validation |

Two further decisions here:

* **Exclude the COVID window** (2020-03-01 to 2021-12-31) from training. Attendance then does not describe normal seasonality. On validation, excluding it was marginally better or equal, so we keep it excluded.
* **Keep the test set out of model selection.** Every hyper-parameter search touched validation only.

**Lesson.** Decide the protocol first and write it down. Every later result is only as honest as this step.

## 2. Baselines before models

Before a single tree, we scored dumb predictors on the test year:

| Baseline | Test MAE | R² |
|---|---|---|
| Same value every day (global mean) | 24.81 | -0.01 |
| Park x weekday average | 24.02 | 0.03 |
| Park x weekday x month average | 21.38 | 0.16 |
| Same weekday last year | 20.67 | 0.09 |

This table earns its keep in three ways. It tells you the *scale* of MAE (a model at 15 is not "85% accurate"). It shows that **"same weekday last year" is already a strong feature** (it beats the park x weekday average by 3 points), which tells you to engineer that carefully. And it sets the bar the model must clear: a gradient-boosted model that beats a lookup table by 1 point would not be worth its complexity.

## 3. Features that carried the signal

### 3.1 Last year, aligned by weekday

Theme-park busyness is very weekday-dependent, so "a year ago" must be a multiple of 7 days. We look back 364 days (52 weeks) and also average 357, 364 and 371 to smooth a bad day:

```python
# src/crowdcast/features/history.py
PRIOR_YEAR_LAGS = (357, 364, 371)  # 51, 52 and 53 weeks: same weekday, one year earlier


def prior_year_frame(labels, park_id, date):
    cols = {}
    for lag in PRIOR_YEAR_LAGS:
        key = pd.MultiIndex.from_arrays([park_id, date - pd.Timedelta(days=lag)])
        cols[lag] = labels.reindex(key).to_numpy()
    ...
```

The lags are all more than 350 days, so this is leakage-safe by construction. These two features plus the park id account for about half of the model's split gain.

The trade-off: a calendar-date lookup (`date - 1 year`) would break the weekday alignment, which we expected to matter more for a weekday-driven target, so we chose alignment. We did not test the date-exact alternative. The cost of our choice is that the 364-day offset drifts by one calendar day a year (two after a leap day), so fixed-date holidays and season boundaries can be off by a day, a weakness you would want to test if it mattered to you.

### 3.2 Holiday scores (the domain-heavy part)

A park's crowd depends on *which regions' holidays* feed it. A Spanish resort draws Spanish, French and British holidaymakers; a local park draws its own county. So for each park we build

```
national_score(park, day) = sum_r weight(park, r, national) * is_national_holiday(r, day)
school_score(park, day)   = sum_r weight(park, r, school)   * is_school_holiday(r, day)
```

Each score is in [0, 1]. The weights are the interesting bit:

1. A **prior** from geography (population x distance decay, with the decay scale set by how far the park's reach extends). This is assumption-heavy and lives in `reference/`.
2. A **per-park fit**: a non-negative ridge regression on the residual of `log(crowd)` after removing weekday, seasonality, year effects and non-seasonal events, with a penalty pulling the weights toward the prior. Nested time-series cross-validation decides how much data-driven signal to blend in (0 to 75%), separately for national and school.
3. **Honest reporting**: bootstrap intervals, a `stability` per weight, and `prior_only` flags for parks with too little data.

Two things worth knowing. Holidays explain only a few percent of the residual variance (median out-of-fold R² about 0.03; real calendars against calendars shifted by a placebo offset: 0.029 vs about 0), so the scores are a modest feature, not a strong one on their own. And the fit's acceptance bar is low (0.001 out-of-fold R² over the prior), so a park with almost no signal can still get some data blended in. That behaviour surfaced when a unit test on pure noise passed for two seeds and failed for one, and it is now documented. See `src/crowdcast/scoring/weights.py` and [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

### 3.3 The rest

Weekday, month, ISO week, sin/cos of day of year; opening hours; six keyword-group event flags; park age, hemisphere, coordinates, company, country, tier; the park id as a categorical. Events turned out nearly irrelevant (about 0.3% of split gain in total): recurring events are already explained by the date and by last year's value.

## 4. Choosing a model

We compared, chronologically, on validation and then on the test year:

| Model | Validation MAE | Test MAE |
|---|---|---|
| Ridge with park x weekday/month/holiday one-hots | 20.71 | 20.45 |
| Random forest | 18.16 | 17.63 |
| Extra trees | 17.90 | 17.27 |
| LightGBM (park id + last year), tuned | 17.87 (untuned) | 17.20 |
| LightGBM without park id | 18.40 | 17.90 |
| LightGBM without holiday features | 19.19 | n/a |

What to take from it:

* Tree ensembles beat everything simpler by about 3 points, and LightGBM, extra trees and a random forest are close to each other. **The model family is not where the gains are.**
* Ablations answer "was that feature worth building?". Holiday features are worth about 1.0 MAE on validation (19.19 vs 18.16); park identity about 0.5 to 0.7.
* Tuning was a six-point grid on validation; it picked 127 leaves and 50 minimum samples per leaf. Do not spend a week here.

The best no-weather setup was an average of extra trees and LightGBM (16.98). It stopped mattering once weather was added.

## 5. Adding weather

Weather comes from the free Open-Meteo APIs. Two engineering points before the result:

* **Rate limits.** The free tier is weighted by variables x days. Parks in the same 0.25 degree grid cell share one download (124 parks became about 100 requests), results are cached per park so runs resume, and the client waits out hourly limits but stops on the daily one. The retry logic is unit-tested against a fake transport (`tests/test_weather.py`).
* **Anomalies, not just values.** Alongside raw temperature and rain we give the model the anomaly against *that park's own climatology* (day-of-year mean over all years, smoothed over 15 days with wrap-around). A 22 °C day is warm in Manchester in April and ordinary in Orlando.

The ablation, LightGBM, all parks:

| Weather features | Test MAE | R² |
|---|---|---|
| None | 17.20 | 0.416 |
| Same-day | 15.87 | 0.494 |
| + previous 3 / 7 days | 15.83 | 0.495 |
| + tomorrow's weather | 15.92 | 0.491 |

Same-day weather is worth 1.3 MAE. Past-week context and look-ahead add nothing, so we dropped look-ahead: it would also not be known beyond the forecast horizon, which is the kind of "feature you cannot have at serving time" trap section 12 returns to.

## 6. Where is the remaining error?

At 15 to 16 MAE we stopped adding features and looked at residuals. These diagnostics were run interactively during the research phase and are recorded, not scripted, in [docs/RESULTS.md](docs/RESULTS.md) section 6:

* Residuals are autocorrelated (lag-1 correlation about 0.5).
* If you removed each park-week's average bias, the residual standard deviation would fall from about 21 to about 15.5: **week-level shocks are roughly half of the error variance**.
* Spring 2026 (Feb to Apr) was under-predicted by about 4 points, unevenly (US parks up to +6, southern-hemisphere parks -8).

A tempting explanation for the spring bias is the `year` feature: trees cannot extrapolate, so a new year falls in the "latest year" leaf. We tested it directly by dropping the feature and refitting: the bias moved from +3.7 to +3.2 and MAE did not change. Not the cause. Easter and other moving holidays (next section) did not explain it either. The model was tracking a real shift in demand that nothing in its inputs could see.

**Lesson.** A hypothesis you can test in a five-minute refit is cheaper than a plausible story. Also: when errors are *persistent*, the fix is information about the recent past, not a better function of the calendar (section 9).

## 7. What did not work (and why to write it down)

Two experiments, both reasonable, both negative:

* **Moving holidays.** Easter, Orthodox Easter, Ascension, Whit Monday, Chinese New Year, Eid, Diwali and Thanksgiving as signed distance to the nearest occurrence. Test MAE across variants: 15.82 to 15.89 against 15.86 without them. The fixed-date and holiday-score features already captured most of it.
* **Blending models trained on different data portions** (per-tier, per-region, recent-only, mature-only, bagged by park-year; simple average and NNLS stacking). Best blend 17.01 against 17.17 for the single model. The components' errors correlated at 0.92 to 0.99, so there was almost no diversity to exploit.

Negative results are useful: they stop the next person (or you in six months) re-running the same experiment, and they are the honest way to explain why the final model is a single LightGBM. They are in RESULTS.md, flagged as not reproducible from this repo.

## 8. Parks with little history: a gate that is about honesty

Accuracy depends heavily on how much history a park has: the Spearman correlation between a park's years of history and its error is -0.55. Parks with under half a year of data are barely better than guessing 50.

`GatedCrowdModel` routes each park to one of two models:

```python
# src/crowdcast/models/gated.py (simplified)
years = frame["park_id"].map(self.history_years).fillna(0).to_numpy()
gated = years >= self.config.min_years  # 1.0 years of labelled history at the cutoff
pred[gated] = full.predict(...)  # park id, last year, drift, weather
pred[~gated] = fallback.predict(...)  # no park id, no history features
low_confidence = ~gated
```

Sweeping the threshold (`experiments/03_history_gate.py`, park model applied to everyone):

| Min. years | Parks kept | MAE | Parks with MAE over 20 |
|---|---|---|---|
| 0 | 116 | 15.97 | 30 |
| 0.5 | 102 | 15.55 | 18 |
| **1** | **94** | **15.40** | **15** |
| 2 | 75 | 15.03 | 8 |
| 3 | 49 | 13.98 | 2 |

One year keeps 81% of parks and halves the number of bad ones. Going stricter buys little for the coverage it costs.

The surprise, which we measured and reported plainly: **the gate does not improve overall accuracy** (16.00 with the gate, 15.97 without), and the fallback model is no better than the full model even for the parks it serves. So why have it? Because it makes the *uncertainty visible*: each row now says whether to trust it. A feature that changes labelling rather than the metric is easy to under-value, and easy to over-claim; be explicit about which one it is.

## 9. Using the recent past: a drift feature

Since errors persist for weeks, give the model the recent error signal without giving it the answer. For each park we compute how far recent days ran above or below the same weekday last year, over a trailing 28 days:

```
drift_h(t) = mean over the 28 days ending at t-h of (actual - same weekday last year)
```

with `h` in (1, 7, 14). Leakage safety is the whole game here, so the definition is explicit: `drift_h` at date `t` uses only labels dated on or before `t - h`. That means a forecast made with labels known through `t - h` can compute it.

Results, parks with at least a year of history, one model:

| Labels known through | MAE | R² |
|---|---|---|
| none (long horizon) | 15.40 | 0.522 |
| t-14 | 15.04 | 0.543 |
| t-7 | 14.95 | 0.549 |
| t-1 | 14.79 | 0.561 |

It also removes about half of the spring bias. The gain exists only for near-term forecasts, so it should not be claimed for a "next quarter" forecast.

## 10. One model for every horizon

We wanted both long-horizon and next-day forecasts. The obvious design is two models. We measured what that costs: two independently trained models disagreed by more than 5 points on **42% of rows**, mostly noise from training rather than information from the drift input. A user would see two different numbers for the same day and no way to tell why.

The fix is to train **one** model on data where the extra inputs are randomly missing:

```python
# src/crowdcast/models/gated.py
as_of = rng.choice([0, *DRIFT_LAGS], len(train))  # 0 = no recent labels; else known through t - lag
for h in DRIFT_LAGS:
    x_full.loc[(as_of == 0) | (as_of > h), f"drift{h}"] = np.nan  # blank what that lag could not know
if cfg.wx_blank:
    blank = rng.random(len(train)) < cfg.wx_blank  # 25% of rows lose all weather
    x_full.loc[blank, WEATHER] = np.nan
```

At prediction time the caller says what is available (`lag=None` or 1/7/14; weather present or not). The same trees serve every case. With/without-drift predictions of the single model disagreed by more than 5 points on **17% of rows**, and that difference is now *explainable*: it is the drift input alone (returned as `drift_effect`).

Accuracy matched the two-model design at both horizons. Weather blanking costs about 0.07 MAE when weather is present and saves about 2 points when it is not (the dates beyond the 16-day forecast).

**Lesson.** "Missing at random during training" is a cheap way to get one model that degrades gracefully, instead of a zoo of models that must be kept consistent.

## 11. Score against what you will actually have

We trained on *observed* weather. Live, we will have *forecasts*. So we scored the test year on archived forecasts (Open-Meteo's previous-runs API: what the forecast said 1 and 7 days earlier), swapping the same-day weather columns and recomputing flags and anomalies:

| Weather at prediction time | Long horizon | Labels through t-1 |
|---|---|---|
| Observed | 15.40 | 14.79 |
| Forecast, 1 day ahead | 15.50 | 14.89 |
| Forecast, 7 days ahead | 16.08 | 15.45 |
| None | 16.94 | 16.27 |

Forecast noise costs about 0.1 MAE at one day and about 0.7 at seven, and it is still far better than no weather. A caveat we keep in the write-up: where the archived forecast lacked a variable, that row kept observed weather, so the 7-day figure is slightly flattering (19% of rows at 7 days).

**Lesson.** Whenever the deployed inputs differ from the training inputs, measure the gap. It is often small, but you want to know that, not assume it.

## 12. Serving: the future has no labels

A forecast is a set of feature rows for dates that have not happened. The hard requirement is **train/serve consistency**: the live path must build features the same way training did.

We did this by making both paths call the same functions:

* holiday features come from `DailyScores.holiday_features`, used for historical rows and future rows alike;
* last-year features come from `prior_year_frame`;
* weather features come from `derive_weather_features`, run over archive plus forecast so that anomalies use the same climatology.

Some inputs genuinely are unknown ahead of time, so they are approximated, and we measured each approximation instead of asserting it:

| Input | Approximation | Measured |
|---|---|---|
| Opening hours | same weekday last year (`date - 364`) | close-time error 32 min vs 70 min for a recent median |
| Event flags | copied from `date - 364`, else 0 | `has_event` right on 80% of days vs 72% for all-zero |
| Open or closed | `open_last_year` flag (not a filter) | 97% open when the flag is 1; only 61% when 0 |

And the builder itself is validated by *pretending*: `build_future_rows(..., asof=D)` acts as if labels stop at `D`, rebuilds a window whose real features are known, and compares them. The calendar, holiday, last-year, drift and weather columns match exactly, and the approximated ones match at the rates above (`tests/test_future.py`).

## 13. From research code to a repository

The research code worked but was a folder of scripts with hard-coded paths, global state and one-off analyses. Turning it into something reviewable took more effort than any modelling step, and it is where we spent care on **not changing the results**.

**Structure.** A `src/` package (`data`, `features`, `weather`, `scoring`, `parks`, `calendars`, `models`, `evaluation`), thin `pipeline.py` and `cli.py` glue, `experiments/` scripts that produce the published numbers, `reference/` for the small curated tables that are part of the method, `configs/model.yaml`. Pure functions take DataFrames; file I/O is confined to `pipeline.py`. That is what makes the tests fast and the future-row builder testable without a network.

**Prove equivalence, stage by stage.** Rewriting a working pipeline is exactly how silent regressions get in. So each ported stage was run against the old output on the real data before moving on:

| Stage | Compared against the old output | Result |
|---|---|---|
| Feature frame | 166,941 rows, every model column | identical |
| Weather features | 184,227 rows, 19 columns | identical up to the old file's rounding |
| Calendar matrices and sparse table | all region x date cells | max difference 0 |
| Park regions, enrichment, market prior | 343 regions, 141 parks, 7,857 market rows | identical |
| Holiday weights | 15,295 weight rows, strengths, intervals | identical (weights within 1e-5) |
| Back-test | 5 headline numbers | reproduced |

Two payoffs. The check caught a real bug in the port: the cold-weather threshold was written as `COLD_C + 5` (10 °C instead of 5 °C) and was only visible because the comparison showed `wx_cold` differing by 1.0. And it revealed the **noise floor**: the end-to-end MAE was 15.40 against 15.44 before, and the only cause was about 1% of holiday scores differing by 0.0001 because an old intermediate step rounded. LightGBM with 127 leaves and 600 trees is sensitive enough that this moves the score by about 0.04, and row-level forecasts from two builds differ by about 2 points on average. So we tell readers to treat differences below about 0.05 MAE as noise, and it suggests an easy improvement: average a few seeds.

**Tests that would fail if you broke the thing that matters.**

```python
# tests/test_features.py: perturb the future, assert the feature does not move
changed.loc[changed["date"] >= cut, "crowd_percent"] += 25
a, b = add_drift(base), add_drift(changed)
unaffected = a["date"] < cut + pd.Timedelta(days=lag)  # days whose window ends before the cut
np.testing.assert_allclose(a.loc[unaffected, f"drift{lag}"], b.loc[unaffected, f"drift{lag}"], equal_nan=True)
```

A test that cannot fail is decoration, so we checked: deliberately changing `roll.shift(h)` to `roll.shift(h - 1)` (an off-by-one that peeks a day ahead) makes the leakage tests fail at all three lags, and reverting makes them pass. Other tests pin behaviour that is easy to break silently: the COVID window and post-cutoff rows cannot influence a fit (poison them and assert predictions do not change), short-history and unseen parks route to the fallback, `lag=None` predictions equal predictions with drift removed, the weight fitter recovers a planted holiday signal and earns no out-of-fold skill on noise, and the retry logic behaves for hourly limits, daily limits, 5xx, empty bodies and client errors.

The test suite also earned its keep by finding bugs the port had introduced: an error message pointing users at a command that does not exist (`weather fetch` instead of `weather-fetch`), and a calendar merge that crashed when a source table was empty because pandas turned a column into `object` dtype.

**Be upfront about what is exempt.** The hand-curated calendar builders (term dates, holiday rules) are formatted but not lint-clean, and are excluded from the strict rules in `pyproject.toml` with a comment saying why: rewriting curated data tables for style risks changing the data. We re-ran them and confirmed the outputs are identical. That honesty is worth more than a green badge.

## 14. Data, licences and not embarrassing yourself later

The crowd data comes from a scraped public API. Before making anything public we read the actual terms rather than assuming, and that changed the repo:

* Queue-Times requires a "Powered by Queue-Times.com" credit and states nothing about redistribution, so raw and derived crowd data never enter the repo, and the tests and demo use a synthetic generator with the same schema.
* OpenHolidays' data is under **ODbL**, which is share-alike: the derived holiday tables cannot be published under the MIT licence.
* Open-Meteo's free API is for non-commercial use.
* Four coordinates we corrected came from OpenStreetMap, which needs its own credit.

`LICENSE` states that the MIT licence covers code only; [NOTICE.md](NOTICE.md) lists each source's terms. Do this at the start of a project, not the end.

## 15. The lessons, compressed

1. **Protocol first.** Chronological splits, one scored test, everything else on validation.
2. **Baselines set the scale.** "Same weekday last year" was hard to beat and told us what to engineer.
3. **Model family matters less than inputs.** Trees were close to each other; features and information (weather, recent history) moved the metric.
4. **Ablate every feature you build.** It tells you what to keep (holidays, weather) and what to drop (look-ahead, events, moving holidays).
5. **Look at residuals, then test hypotheses cheaply.** The `year` explanation for the spring bias took minutes to rule out.
6. **Make leakage impossible by definition,** then test it by perturbing the future and by mutation-checking the test.
7. **Random blanking beats model zoos** when you need graceful degradation.
8. **Measure the train/serve gap** (archived forecasts) and **approximate unknowns with measured proxies.**
9. **Prove equivalence when you refactor,** and treat the noise floor as a result.
10. **Say what a feature is for.** The history gate changes labelling, not accuracy; the docs say so.
11. **Read the data licences before publishing.**

## 16. Where to go next

Things this repo does not do, in the order we would tackle them:

* **Steady the row-level predictions** by averaging several LightGBM seeds (about 2 points of build-to-build variation per row).
* **Quantify uncertainty** (quantile or conformal intervals), so `low_confidence` becomes a number instead of a flag.
* **Predict closures.** Parks are predicted for every date, including days they will be closed.
* **Ingest real event schedules** instead of copying last year's flags.
* **Remove the target leak:** rank each park's index using only data available at the forecast date.
* **More than one test year,** and a rolling-origin evaluation to see how errors vary across years.
* **Monitor** the `drift_effect` and the gap between forecasts and outcomes once labels arrive.

For the details behind any section, see [docs/METHODOLOGY.md](docs/METHODOLOGY.md), [docs/RESULTS.md](docs/RESULTS.md), [docs/DATA.md](docs/DATA.md) and the [model card](docs/MODEL_CARD.md).
