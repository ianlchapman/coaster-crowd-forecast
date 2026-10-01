import pandas as pd

from crowdcast.cli import main
from crowdcast.evaluation import accuracy as acc


def _forecast(made_on: str, value: float, days: int = 100) -> pd.DataFrame:
    dates = pd.date_range(pd.Timestamp(made_on) + pd.Timedelta(days=1), periods=days)
    return pd.DataFrame({"park_id": 1, "date": dates, "prediction": value, "low_confidence": False})


def _crowd(start: str, days: int, value: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "park_id": 1,
            "date": pd.date_range(start, periods=days),
            "status": "open",
            "crowd_percent": value,
            "predicted": False,
        }
    )


def test_snapshot_keeps_only_requested_leads():
    snap = acc.snapshot(_forecast("2026-01-01", 50), "2026-01-01")
    assert sorted(snap["lead_days"]) == [1, 7, 14, 30, 90]
    assert snap["made_on"].eq(pd.Timestamp("2026-01-01")).all()


def test_append_log_replaces_same_day_and_keeps_others():
    a = acc.snapshot(_forecast("2026-01-01", 50), "2026-01-01")
    b = acc.snapshot(_forecast("2026-01-02", 60), "2026-01-02")
    log = acc.append_log(acc.append_log(None, a), b)
    log = acc.append_log(log, acc.snapshot(_forecast("2026-01-02", 70), "2026-01-02"))
    assert len(log) == 10
    assert log.loc[log["made_on"] == "2026-01-02", "prediction"].eq(70).all()


def test_actuals_exclude_future_predicted_and_closed_rows():
    crowd = _crowd("2026-01-01", 5, 40)
    crowd.loc[1, "predicted"] = True
    crowd.loc[2, "status"] = "closed"
    got = acc.actuals(crowd, "2026-01-05")
    assert list(got["date"].dt.day) == [1, 4]  # day 3 closed, day 2 predicted, day 5 not before asof


def test_score_and_summary_numbers():
    log = acc.snapshot(_forecast("2026-01-01", 50), "2026-01-01")
    observed = acc.actuals(_crowd("2026-01-01", 20, 40), "2026-01-20")
    detail = acc.score(log, observed, "2026-01-20")
    assert set(detail["lead_days"]) == {1, 7, 14}  # lead 30/90 dates are still in the future
    assert detail["error"].eq(10).all()
    summary = acc.summarise(detail, "2026-01-20")
    row = summary[(summary["window_days"] == 90) & (summary["lead_days"] == 7)].iloc[0]
    assert (row["accuracy_pct"], row["within_10_pct"], row["mae"], row["bias"]) == (90.0, 100.0, 10.0, 10.0)
    assert "90% accuracy, last 90 days" in str(acc.headline(summary))


def test_headline_none_when_nothing_scored():
    assert acc.headline(acc.summarise(pd.DataFrame(columns=acc.DETAIL_COLUMNS), "2026-01-20")) is None


def test_accuracy_command_end_to_end(data_dir, monkeypatch, tmp_path, capsys):
    paths, s = data_dir
    monkeypatch.setenv("CROWDCAST_DATA_DIR", str(paths.root))
    last = s.crowd["date"].max()
    fc = tmp_path / "fc.csv"
    _forecast(str(last - pd.Timedelta(days=20)), 50, 30).to_csv(fc, index=False)
    asof = str((last + pd.Timedelta(days=1)).date())
    assert main(["accuracy", "--forecast", str(fc), "--asof", str(last.date() - pd.Timedelta(days=20))]) == 0
    assert main(["accuracy", "--asof", asof]) == 0  # re-score only, using the existing log
    out = paths.accuracy_dir
    for name in (
        "predictions-log.csv",
        "accuracy-detail.csv",
        "accuracy-summary.csv",
        "accuracy-by-park.csv",
        "headline.txt",
    ):
        assert (out / name).exists()
    assert "accuracy" in capsys.readouterr().out
