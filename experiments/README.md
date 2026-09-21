# Experiments

Scripts that produce the numbers in `docs/RESULTS.md`. Each is a plain script built on the `crowdcast` package (no notebooks), prints
its table and writes a CSV under `data/processed/experiments/`. They need the real data (see `docs/DATA.md`), so they are not run
in CI.

| Script | Question | Needs |
|---|---|---|
| `01_model_comparison.py` | How do baselines, Ridge, forests and LightGBM compare on validation and test? | enhanced calendar |
| `02_weather_ablation.py` | Does weather help? Same-day vs past-week vs look-ahead. | + weather archive |
| `03_history_gate.py` | How does accuracy depend on a park's history, and where should the gate sit? | + weather archive |
| `04_forecast_weather.py` | How much accuracy is lost when using archived *forecasts* instead of observed weather? | + `crowdcast weather-previous-runs` |

Run from the repo root: `python experiments/01_model_comparison.py`.
