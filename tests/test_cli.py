import pandas as pd
import pytest

from crowdcast.cli import build_parser, main
from crowdcast.models.config import ModelConfig


def test_demo_command_runs_end_to_end(capsys):
    assert main(["demo", "--parks", "4"]) == 0
    out = capsys.readouterr().out
    assert "Back-test on synthetic data" in out and "long horizon" in out and "Forecast of the days" in out


def test_missing_data_gives_a_helpful_error(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("CROWDCAST_DATA_DIR", str(tmp_path))
    assert main(["train"]) == 2
    assert "docs/DATA.md" in capsys.readouterr().err


def test_parser_requires_a_command_and_validates_lag():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
    with pytest.raises(SystemExit):
        build_parser().parse_args(["predict", "out.csv", "--lag", "3"])


FAST_YAML = "min_years: 1.0\nwx_blank: 0.25\nexclude_covid: true\nseed: 0\nlightgbm: {n_estimators: 30, num_leaves: 7, min_child_samples: 20, verbose: -1, n_jobs: 2}\n"


@pytest.fixture()
def on_disk(data_dir, monkeypatch):
    monkeypatch.setenv("CROWDCAST_DATA_DIR", str(data_dir[0].root))
    return data_dir[0]


def test_enhance_train_predict_and_evaluate_commands(on_disk, tmp_path, capsys):
    cfg = tmp_path / "fast.yaml"
    cfg.write_text(FAST_YAML)
    assert main(["enhance"]) == 0
    assert main(["train", "--cutoff", "2021-12-31", "--config", str(cfg)]) == 0
    assert on_disk.model_file.exists()
    out = tmp_path / "pred.csv"
    assert main(["predict", str(out), "--from", "2022-06-01", "--lag", "7"]) == 0
    pred = pd.read_csv(out, parse_dates=["date"])
    assert (
        pred["date"].min() >= pd.Timestamp("2022-06-01")
        and pred["prediction"].between(0, 100).all()
        and "drift_effect" in pred
    )
    capsys.readouterr()
    bt = tmp_path / "bt.csv"
    assert main(["evaluate", "--cutoff", "2021-12-31", "--test-end", "2022-12-31", "--out", str(bt)]) == 0
    text = capsys.readouterr().out
    assert "long horizon" in text and "Top features" in text and bt.exists()


def test_forecast_command_with_a_faked_weather_api(on_disk, tmp_path, monkeypatch, capsys):
    from crowdcast import pipeline
    from crowdcast.models.gated import GatedCrowdModel

    frame = pipeline.load_training_frame(on_disk)
    GatedCrowdModel(ModelConfig.from_yaml(_write(tmp_path / "f.yaml"))).fit(frame, "2021-12-31").save(
        on_disk.model_file
    )
    assert main(["train-status", "--cutoff", "2021-12-31", "--config", str(_write(tmp_path / "f2.yaml"))]) == 0
    weather = pd.read_csv(on_disk.weather_archive / "park_1.csv", parse_dates=["date"])
    everyone = pd.concat([weather.assign(park_id=p) for p in (1, 2, 3)])
    monkeypatch.setattr(
        pipeline, "fetch_forecast", lambda coords, cache, refresh=False: everyone[everyone["date"] >= "2022-11-01"]
    )
    out = tmp_path / "fc.csv"
    assert main(["forecast", str(out), "--to", "2023-01-05"]) == 0
    fc = pd.read_csv(out, parse_dates=["date"])
    assert fc["date"].max() == pd.Timestamp("2023-01-05") and {
        "weather", "days_ahead", "open_last_year", "is_open", "opens", "closes",
    } <= set(fc.columns)  # fmt: skip
    assert "weather:" in capsys.readouterr().out


def _write(path):
    path.write_text(FAST_YAML)
    return path
