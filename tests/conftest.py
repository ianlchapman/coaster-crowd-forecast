"""Shared fixtures: a small synthetic dataset and a quickly-fitted model (no real data, no network)."""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

from crowdcast.config import Paths
from crowdcast.data.synthetic import make_synthetic
from crowdcast.demo import TIERS, SyntheticInputs, build_synthetic_inputs, synthetic_scores
from crowdcast.models.config import ModelConfig
from crowdcast.models.gated import GatedCrowdModel

warnings.filterwarnings("ignore", category=FutureWarning)

FAST = ModelConfig(lightgbm={"n_estimators": 60, "num_leaves": 15, "min_child_samples": 20, "verbose": -1, "n_jobs": 2})


@pytest.fixture(scope="session")
def inputs() -> SyntheticInputs:
    return build_synthetic_inputs(n_parks=4, start="2019-01-01", end="2023-12-31", seed=1)


@pytest.fixture(scope="session")
def fast_config() -> ModelConfig:
    return FAST


@pytest.fixture(scope="session")
def model(inputs: SyntheticInputs) -> GatedCrowdModel:
    return GatedCrowdModel(FAST).fit(inputs.frame, "2022-12-31")


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    paths = Paths(root)
    s = make_synthetic(n_parks=3, start="2019-01-01", end="2022-12-31", seed=2)
    paths.ensure(paths.raw_loader, paths.parks_dir, paths.weather_archive, paths.daily_scores.parent)
    s.crowd.to_csv(paths.crowd_calendar, index=False)
    s.parks.to_csv(paths.parks_csv, index=False)
    pd.DataFrame({"park_id": s.parks["id"], "country_iso2": s.parks["country"], "tier": [TIERS[i % 3] for i in range(3)],
                  "latitude": s.parks["latitude"], "longitude": s.parks["longitude"]}).to_csv(paths.parks_enriched, index=False)  # fmt: skip
    synthetic_scores(s).save(paths.daily_scores)
    for pid, g in s.weather.groupby("park_id"):
        g.to_csv(paths.weather_archive / f"park_{pid}.csv", index=False)
    return paths, s
