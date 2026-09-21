"""Chronological train / validation / test splits.

Crowd levels are strongly autocorrelated and seasonal, so random splits leak: a random validation day sits between two
training days of the same week. Everything here splits on date only.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SplitDates:
    """Inclusive end dates: train <= ``train_end`` < validation <= ``val_end`` < test <= ``test_end``."""

    train_end: str = "2024-12-31"
    val_end: str = "2025-08-31"
    test_end: str = "2026-08-31"

    def __post_init__(self) -> None:
        if not (pd.Timestamp(self.train_end) < pd.Timestamp(self.val_end) < pd.Timestamp(self.test_end)):
            raise ValueError("split dates must be strictly increasing")


def time_split(frame: pd.DataFrame, dates: SplitDates) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Boolean masks ``(train, validation, test)`` over ``frame['date']``."""
    d = frame["date"]
    train = d <= dates.train_end
    val = (d > dates.train_end) & (d <= dates.val_end)
    test = (d > dates.val_end) & (d <= dates.test_end)
    return train, val, test
