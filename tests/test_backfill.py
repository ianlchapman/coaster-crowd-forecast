import pandas as pd

from crowdcast.evaluation import accuracy as acc
from crowdcast.evaluation.backfill import backfill_log


class _Fit:
    """Stands in for GatedCrowdModel: records the cutoff and lag it is asked for, predicts the lag."""

    calls: list[tuple[pd.Timestamp, int]] = []

    def __init__(self, config=None):
        self.cutoff = None

    def fit(self, frame, cutoff):
        self.cutoff = cutoff
        return self

    def predict(self, rows, lag=None):
        _Fit.calls.append((self.cutoff, lag))
        return pd.DataFrame(
            {
                "park_id": rows["park_id"].to_numpy(),
                "date": rows["date"].to_numpy(),
                "prediction": float(lag),
                "low_confidence": False,
            }
        )


def test_backfill_uses_only_labels_before_made_on_and_tags_rows(monkeypatch):
    import crowdcast.evaluation.backfill as bf

    monkeypatch.setattr(bf, "GatedCrowdModel", _Fit)
    _Fit.calls.clear()
    frame = pd.DataFrame({"park_id": 1, "date": pd.date_range("2026-01-01", periods=130)})
    out = backfill_log(frame, [pd.Timestamp("2026-02-01")], leads=(1, 7, 30, 90))
    assert set(out["source"]) == {acc.BACKTEST}
    assert list(out.columns) == acc.LOG_COLUMNS
    assert sorted(out["lead_days"]) == [1, 7, 30, 90]
    assert (out["date"] - out["made_on"]).dt.days.eq(out["lead_days"]).all()
    assert {c for c, _ in _Fit.calls} == {pd.Timestamp("2026-01-31")}  # fit through the day before made_on
    assert dict(zip(out["lead_days"], out["prediction"], strict=True)) == {
        1: 2.0,
        7: 8.0,
        30: 31.0,
        90: 91.0,
    }  # lag = lead + 1


def test_backfill_skips_dates_without_rows(monkeypatch):
    import crowdcast.evaluation.backfill as bf

    monkeypatch.setattr(bf, "GatedCrowdModel", _Fit)
    frame = pd.DataFrame({"park_id": 1, "date": pd.date_range("2026-01-01", periods=10)})
    out = backfill_log(frame, [pd.Timestamp("2026-01-05")], leads=(1, 7, 30))
    assert list(out["lead_days"]) == [1]


def test_backfill_refits_every_refit_days_and_never_peeks(monkeypatch):
    import crowdcast.evaluation.backfill as bf

    monkeypatch.setattr(bf, "GatedCrowdModel", _Fit)
    _Fit.calls.clear()
    frame = pd.DataFrame({"park_id": 1, "date": pd.date_range("2026-01-01", periods=60)})
    days = list(pd.date_range("2026-02-01", periods=10))
    out = backfill_log(frame, days, leads=(1,), refit_days=7)
    assert len(out) == 10
    assert sorted({c for c, _ in _Fit.calls}) == [pd.Timestamp("2026-01-31"), pd.Timestamp("2026-02-07")]
    assert all(
        c < d for (c, _), d in zip(_Fit.calls, days, strict=True)
    )  # every model was fit before the day it forecasts for
