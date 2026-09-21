import numpy as np
import pandas as pd
import pytest

from crowdcast.scoring import CalendarMatrices
from crowdcast.scoring.weights import (
    ParkFit,
    WeightConfig,
    WeightFitter,
    assemble,
    blend_with_prior,
    controls,
    fit_all,
    normalise,
    pick_lambda,
    ridge_nnls,
)


def test_ridge_nnls_is_non_negative_and_recovers_a_planted_signal():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 4))
    y = 2.0 * x[:, 0] - 3.0 * x[:, 1] + rng.normal(0, 0.1, 400)  # column 1 has a negative true effect
    u = ridge_nnls(y, x, lam=1e-3)
    assert (u >= 0).all()
    assert u[0] > 1.5 and u[1] == pytest.approx(0, abs=1e-6) and u[2] < 0.2


def test_blend_with_prior_sums_to_one_and_respects_the_cap():
    prior = np.array([0.7, 0.2, 0.1])
    data = np.array([0.0, 0.0, 5.0])  # data wants everything on the smallest market
    w = blend_with_prior(data, prior, rho=0.75, kappa=5.0)
    assert w.sum() == pytest.approx(1.0)
    assert w[2] <= 5.0 * prior[2] + 0.004 + 1e-9 + 0.05  # capped near kappa x prior (renormalisation may nudge it)
    assert blend_with_prior(data, prior, rho=0.0, kappa=5.0) == pytest.approx(prior)
    assert blend_with_prior(np.zeros(3), prior, rho=0.5, kappa=5.0) == pytest.approx(prior)  # no data signal -> prior


def test_normalise_handles_zero_vectors():
    assert normalise(np.zeros(3)) is None
    assert normalise(np.array([1.0, 3.0])).tolist() == [0.25, 0.75]


def test_pick_lambda_prefers_regularisation_on_pure_noise():
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=(300, 6)), rng.normal(size=300)
    lam, r2 = pick_lambda(y, x, np.logspace(-2, 4, 7))
    assert lam >= 1e2 and r2 <= 0.01  # noise: strongest penalty wins, no out-of-fold gain


def test_controls_have_full_rank_columns_and_include_weekday_and_season():
    u = pd.DataFrame({"park_id": 1, "date": pd.date_range("2019-01-01", periods=900)})
    events = pd.DataFrame({"park_id": [], "date": [], "event": []})
    events["event"] = events["event"].astype(str)
    x = controls(u, events, WeightConfig())
    assert x.shape[0] == 900 and np.linalg.matrix_rank(x) == x.shape[1]
    assert x.shape[1] >= 1 + 6 + 2 * 8  # intercept, weekdays, 8 harmonics


def _fitter(effect: float, seed: int = 0) -> WeightFitter:
    """One park fed by regions A (60%) and B (40%). Crowd rises on A's school holidays by ``effect`` (log points)."""
    dates = pd.date_range("2014-01-01", "2027-12-31")
    rng = np.random.default_rng(seed)
    n = len(dates)
    school = np.zeros((3, n), dtype=np.float32)
    for r, offset in enumerate((0, 40, 80)):  # staggered summer blocks per region
        for y in range(2014, 2028):
            start = pd.Timestamp(f"{y}-07-01") + pd.Timedelta(days=offset % 30)
            school[r, dates.get_indexer(pd.date_range(start, periods=45 + 5 * r))] = 1.0
    nat = np.zeros_like(school)
    matrices = CalendarMatrices(
        np.array(["A", "B", "C"]),
        dates,
        nat,
        school,
        np.arange(2014, 2028),
        np.ones((3, 14), bool),
        np.ones((3, 14), bool),
    )
    days = pd.date_range("2016-01-01", "2024-12-31")
    idx = dates.get_indexer(days)
    signal = effect * school[0, idx] + 0.2 * np.sin(2 * np.pi * days.dayofyear / 365.25) + rng.normal(0, 0.3, len(days))
    crowd_pct = np.clip(50 + 25 * signal, 1, 100)
    crowd = pd.DataFrame({"park_id": 1, "date": days, "status": "open", "predicted": False, "crowd_percent": crowd_pct})
    parks = pd.DataFrame({"park_id": [1], "country_iso2": ["GB"]})
    markets = pd.DataFrame(
        {"park_id": 1, "region_code": ["A", "B", "C"], "distance_km": [1, 2, 3], "prior_weight": [0.4, 0.4, 0.2]}
    )
    events = pd.DataFrame(
        {"park_id": pd.Series([], dtype=int), "date": pd.to_datetime([]), "event": pd.Series([], dtype=str)}
    )
    return WeightFitter(parks, markets, matrices, crowd, events, WeightConfig(bootstrap=5, workers=1))


def test_fitter_learns_the_region_that_actually_drives_the_park():
    W, S = fit_all(_fitter(effect=1.2))
    school = W[W["type"] == "school"].set_index("region_code")["weight"]
    assert school.sum() == pytest.approx(1.0, abs=1e-3)
    assert school["A"] > 0.4 + 0.05  # data moved weight onto A, above its 0.4 prior
    assert not W[W["type"] == "school"]["prior_only"].all()
    assert S.loc[0, "oof_r2"] > 0 and S.loc[0, "school_strength"] >= 0


@pytest.mark.parametrize("seed", [3, 4, 5])
def test_fitter_has_no_out_of_fold_skill_without_a_real_signal(seed: int):
    """On pure noise the fit must not claim skill; whenever it keeps the prior the weights must equal the prior exactly."""
    W, S = fit_all(_fitter(effect=0.0, seed=seed))
    assert S.loc[0, "oof_r2"] < 0.01
    school = W[W["type"] == "school"].set_index("region_code")["weight"]
    assert school.sum() == pytest.approx(1.0, abs=1e-3)
    if S.loc[0, "rho_s"] == 0:
        assert school.to_dict() == pytest.approx({"A": 0.4, "B": 0.4, "C": 0.2}, abs=1e-3)
        assert W[W["type"] == "school"]["prior_only"].all()


def test_a_real_signal_earns_a_large_out_of_fold_gain():
    _, S = fit_all(_fitter(effect=1.2))
    assert S.loc[0, "oof_r2"] > 0.05 and S.loc[0, "oof_r2"] > S.loc[0, "oof_r2_prior"] + 0.03


def test_park_with_too_little_data_is_prior_only():
    f = _fitter(effect=1.0)
    f.crowd = f.crowd.head(60)
    fit = f.fit_park(1)
    assert fit.info["reason"] == "too little data" and fit.weights["school"][1] is None


def test_assemble_drops_negligible_weights_and_renormalises():
    fit = ParkFit(
        1,
        ["A", "B"],
        {
            "national": (np.array([0.99995, 0.00005]), None, None, None, None),
            "school": (np.array([0.5, 0.5]), None, None, None, None),
        },
    )
    fit.info = {
        "park_id": 1,
        "std_n": np.nan,
        "std_s": np.nan,
        "beta_n": np.nan,
        "beta_s": np.nan,
        "bn_ci": None,
        "bs_ci": None,
    }
    W, S = assemble([fit])
    nat = W[W["type"] == "national"]
    assert nat["region_code"].tolist() == ["A"] and nat["weight"].iloc[0] == pytest.approx(1.0)
    assert len(S) == 1
