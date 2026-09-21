"""Models: the gated LightGBM predictor and the baselines it is compared against."""

from crowdcast.models.config import ModelConfig
from crowdcast.models.gated import GatedCrowdModel

__all__ = ["GatedCrowdModel", "ModelConfig"]
