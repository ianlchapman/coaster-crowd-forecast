"""Model hyper-parameters and behaviour switches, loadable from ``configs/model.yaml``."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from crowdcast.config import REPO_ROOT

DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "model.yaml"


def _default_lightgbm() -> dict[str, Any]:
    return {
        "n_estimators": 600,
        "learning_rate": 0.03,
        "num_leaves": 127,
        "min_child_samples": 50,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "cat_smooth": 10,
        "n_jobs": 8,
        "verbose": -1,
    }


@dataclass(frozen=True)
class ModelConfig:
    min_years: float = 1.0
    wx_blank: float = 0.25
    exclude_covid: bool = True
    seed: int = 0
    lightgbm: dict[str, Any] = field(default_factory=_default_lightgbm)

    @classmethod
    def from_yaml(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> ModelConfig:
        raw = yaml.safe_load(Path(path).read_text())
        lgbm = {**_default_lightgbm(), **raw.pop("lightgbm", {})}
        return cls(**raw, lightgbm=lgbm)
