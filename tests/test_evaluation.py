import numpy as np
import pandas as pd
import pytest

from crowdcast.evaluation.metrics import regression_metrics
from crowdcast.evaluation.splits import SplitDates, time_split


def test_split_masks_are_disjoint_and_chronological():
    f = pd.DataFrame({"date": pd.date_range("2024-06-01", "2026-09-30")})
    tr, va, te = time_split(f, SplitDates())
    assert not (tr & va).any() and not (va & te).any() and not (tr & te).any()
    assert f.loc[tr, "date"].max() < f.loc[va, "date"].min() < f.loc[va, "date"].max() < f.loc[te, "date"].min()
    assert f.loc[te, "date"].max() == pd.Timestamp("2026-08-31")  # rows after test_end belong to no split


def test_split_dates_must_increase():
    with pytest.raises(ValueError):
        SplitDates("2025-01-01", "2024-01-01", "2026-01-01")


def test_metrics_known_values_and_clipping():
    y = np.array([10.0, 50.0, 90.0])
    m = regression_metrics(y, np.array([20.0, 50.0, 70.0]))
    assert m["MAE"] == pytest.approx(10.0)
    assert m["RMSE"] == pytest.approx(np.sqrt((100 + 0 + 400) / 3))
    assert m["rho"] == pytest.approx(1.0)
    # predictions are clipped to the 0-100 range before scoring
    assert regression_metrics(np.array([100.0, 0.0]), np.array([150.0, -20.0]))["MAE"] == 0
    perfect = regression_metrics(y, y)
    assert perfect["R2"] == 1.0
    per_park = regression_metrics(y, y + 10, park_id=np.array([1, 1, 2]))
    assert per_park["parkMAE_med"] == pytest.approx(10.0)
