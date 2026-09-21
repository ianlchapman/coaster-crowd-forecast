# Results

Test window: 2025-09-01 to 2026-08-31, 24,652 park-days over 116 parks. Models fitted on data up to 2025-08-31 (COVID window excluded), unless stated. MAE is mean absolute error in index points (lower is better). Every table below is produced by a script in this repo and was run on the real data (`crowdcast evaluate`, `experiments/`); the two exceptions are marked.

**How much to trust the third digit.** Re-running the whole pipeline from raw inputs (instead of reusing earlier intermediate files) moved the headline from 15.44 to 15.40 MAE, because about 1% of holiday scores differed by 0.0001 through a rounding step. Treat differences below about 0.05 MAE as noise. The test is a single year.

## 1. Model comparison (no weather) - `experiments/01_model_comparison.py`

| Model | Validation MAE | Test MAE | Test R² |
|---|---|---|---|
| Global mean | 23.52 | 24.81 | -0.01 |
| Park x weekday mean | 23.24 | 24.02 | 0.03 |
| Park x weekday x month mean | 21.52 | 21.38 | 0.16 |
| Same weekday last year | 22.77 | 20.67 | 0.09 |
| Ridge (park x weekday / month / holiday one-hots) | 20.71 | 20.45 | 0.24 |
| Random forest | 18.16 | 17.63 | 0.41 |
| Extra trees | 17.90 | 17.27 | 0.42 |
| LightGBM, park id, no last-year (tuned) | 18.16 (untuned) | 17.24 | 0.42 |
| LightGBM, park id + last-year (tuned) | 17.87 (untuned) | 17.20 | 0.42 |
| LightGBM without park id | 18.40 | 17.90 | 0.40 |
| LightGBM without holiday features | 19.19 | n/a | n/a |
| Extra trees + tuned LightGBM (average) | n/a | 16.98 | 0.44 |

* Tree ensembles beat everything simpler by about 3 to 4 points. Park identity is worth about 0.5 to 0.7 MAE (LightGBM with vs without it), and the holiday features about 1.0 on validation (19.19 without vs 18.16 with), so the holiday work pays for itself.
* Tuned parameters (chosen on validation): 127 leaves, 50 minimum samples per leaf.
* The average of Extra trees and LightGBM was the best no-weather setup, but adding weather made LightGBM alone better than the ensemble, so the deployed model is LightGBM only.

## 2. Weather - `experiments/02_weather_ablation.py`

LightGBM, all parks:

| Weather features | Validation MAE | Test MAE | Test R² |
|---|---|---|---|
| None | 17.77 | 17.20 | 0.416 |
| Same-day (temperature, rain, snow, wind, sun, flags, anomalies) | 16.40 | 15.87 | 0.494 |
| + previous 3 / 7 days | 16.39 | 15.83 | 0.495 |
| + tomorrow's weather (look-ahead) | 16.46 | 15.92 | 0.491 |

Same-day weather is worth about 1.3 MAE; past-week and look-ahead weather add nothing. Look-ahead is not used by the deployed model.

## 3. The deployed gated model - `crowdcast evaluate`

| Group | Rows | MAE | R² |
|---|---|---|---|
| All parks, gated system | 24,652 | 16.00 | 0.491 |
| All parks, full model with no gate | 24,652 | 15.97 | 0.491 |
| Parks with at least 1 year of history, park model (94 parks) | 21,650 | 15.40 | 0.522 |
| Same parks, fallback model would give | 21,650 | 15.79 | 0.515 |
| Short-history parks, fallback (22 parks) | 3,002 | 20.40 | 0.269 |
| Same parks, full model would give | 3,002 | 20.10 | 0.270 |
| Short-history parks, guess 50 | 3,002 | 25.11 | 0.00 |

The gate does not improve overall accuracy (16.00 vs 15.97); its value is that short-history parks are labelled low-confidence. The fallback is no better than the full model even where it is used.

Horizon (parks with at least 1 year of history), one model serving all of them:

| Labels known through | MAE | R² |
|---|---|---|
| none (long horizon) | 15.40 | 0.522 |
| t-14 | 15.04 | 0.543 |
| t-7 | 14.95 | 0.549 |
| t-1 | 14.79 | 0.561 |

Top features by share of split gain: park id 21%, mean of last year's values 16%, last year's same weekday 9%, school holiday 4.5%, weekday 4.2%, park age 3.7%.

## 4. History gate - `experiments/03_history_gate.py`

Park model applied to everyone (gate open), test year:

| Min. years of history | Parks kept | Rows kept | MAE | Parks with MAE over 20 |
|---|---|---|---|---|
| 0 | 116 (100%) | 100% | 15.97 | 30 |
| 0.5 | 102 (88%) | 95% | 15.55 | 18 |
| **1** | **94 (81%)** | **88%** | **15.40** | **15** |
| 2 | 75 (65%) | 78% | 15.03 | 8 |
| 3 | 49 (42%) | 58% | 13.98 | 2 |

Parks with under half a year of history (14 parks) have MAE 23 to 25, no better than guessing 50 (24 to 26); 7 of 116 parks are worse than guessing overall. Across parks, history length and error have Spearman correlation -0.55. A one-year gate keeps most parks and cuts the number of bad parks in half; stricter gates give small further gains at a large coverage cost.

## 5. Forecast weather instead of observed weather - `experiments/04_forecast_weather.py`

Scored with the weather that archived forecasts (Open-Meteo previous-runs API) said 1 and 7 days earlier. Parks with at least 1 year of history.

| Model, weather at prediction time | Long horizon | Labels through t-1 |
|---|---|---|
| Trained on observed weather only: observed | 15.33 | 14.70 |
| ...forecast 1 day ahead | 15.44 | 14.83 |
| ...forecast 7 days ahead | 16.02 | 15.39 |
| ...no weather | 18.88 | 18.23 |
| Deployed (25% of training rows weather-blanked): observed | 15.40 | 14.79 |
| ...forecast 1 day ahead | 15.50 | 14.89 |
| ...forecast 7 days ahead | 16.08 | 15.45 |
| ...no weather | 16.94 | 16.27 |

Weather-blanked training costs about 0.07 MAE with weather and saves about 2 points without it (dates past the 16-day forecast). The 7-day figure is a slight underestimate of the true loss: rows where the archived forecast is missing a variable keep observed weather (19% of rows at 7 days, under 1% at 1 day).

## 6. Earlier findings not reproduced by scripts here

These were measured during the research phase; the scripts are not part of this repository, so treat them as recorded results, not reproducible ones.

* **Moving holidays** (Easter, Orthodox Easter, Ascension, Whit Monday, Chinese New Year, Eid, Diwali, Thanksgiving as signed distance to the nearest date): test MAE 15.82 to 15.89 across the variants against 15.86 without them, i.e. no gain. Easter did not explain the Feb-Apr 2026 under-prediction either.
* **Blending models trained on different data portions** (per-tier, per-region, recent-only, mature-only, park-year bagging; simple average and NNLS stacking): 17.05 and 17.01 vs 17.17 for the single model. Component errors correlated at 0.92 to 0.99, so there was little to gain.
* **One model vs two for drift.** Two separately trained models (with and without drift) disagreed by more than 5 points on 42% of rows, mostly noise; one model with drift blanked at random in training disagreed on 17% of rows and matched both models' accuracy, which is why the deployed model is a single model.
* **Residual diagnostics.** Week-level shocks are about half of the residual variance; lag-1 residual correlation about 0.5; Feb-Apr 2026 was under-predicted by about 4 points and was not caused by the year feature (dropping it moved the bias from +3.7 to +3.2).
