# Contributing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
make check        # ruff, mypy, pytest
```

Guidelines

* Tests use synthetic data only (`crowdcast.data.synthetic`); do not add real crowd data, weather dumps or trained models to the repo.
* Anything that builds features for live use must go through the same functions as training (`DailyScores.holiday_features`, `prior_year_frame`, `derive_weather_features`), and needs a leakage test: perturb the future, assert the feature does not move.
* Model changes should be judged on the chronological test protocol in `docs/METHODOLOGY.md`; report differences smaller than about 0.05 MAE as noise.
* `scripts/calendars/` holds hand-curated data builders; change data there only with a source and re-check the output tables.
