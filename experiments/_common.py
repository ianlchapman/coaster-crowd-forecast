"""Shared helpers for the experiment scripts."""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import pandas as pd

from crowdcast.config import Paths
from crowdcast.pipeline import load_training_frame

warnings.filterwarnings("ignore")


def load_frame() -> pd.DataFrame:
    """Labelled feature frame from the real data (run the pipeline first, see docs/DATA.md)."""
    return load_training_frame(Paths())


def out_dir() -> Path:
    d = Paths().processed / "experiments"
    d.mkdir(parents=True, exist_ok=True)
    return d


def timed(label: str):  # noqa: ANN201
    class _T:
        def __enter__(self) -> None:
            self.t = time.time()

        def __exit__(self, *a: object) -> None:
            print(f"  [{label}: {time.time() - self.t:.0f}s]", flush=True)

    return _T()
