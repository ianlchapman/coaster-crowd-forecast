"""Estimate, per park, how much each source region's national and school holidays drive its busyness.

Per park:  ``y = controls + sum_r a_r * national_r(d) + sum_r b_r * school_r(d) + e``  with ``a, b >= 0``.

* ``y = log(max(crowd_percent, 1))``. Raw and rank-normal variants are refitted as a robustness check.
* Controls (unpenalised, partialled out by Frisch-Waugh): day of week, annual Fourier terms, year effects, season-edge flags and
  non-seasonal park events. Held out of the fit: the COVID window, predicted rows and non-open days.
* Holiday coefficients: non-negative ridge with a market-prior penalty ``lam * a_r^2 / prior_r`` (prior from
  ``parks.markets``). ``lam`` is chosen by blocked time-series CV; with no out-of-fold gain the prior is used (``prior_only``).
* Weights: data-driven ``a_r`` normalised per type and blended with the market prior. The blend ``rho`` (0 = prior only) is picked
  per type by nested time-series CV, capped at 0.75, and data can lift a market to at most ``kappa`` x its prior.
  Each type sums to 1. A block bootstrap (year-month blocks) gives intervals and stability.

Holidays explain only a few percent of residual variance once seasonality is removed (median out-of-fold R2 ~ 0.03), so most parks
lean on the prior; ``park_strength`` says how much each park actually learned from data.
"""

from __future__ import annotations

import logging
import re
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import norm, rankdata

from crowdcast.scoring.daily import CalendarMatrices

log = logging.getLogger(__name__)

COVID_START = "2020-03-01"
#: Where the COVID disruption ended by country (attendance recovered later in East Asia / Australia).
COVID_END_BY_COUNTRY = {
    "CN": "2023-02-28",
    "HK": "2023-02-28",
    "JP": "2022-12-31",
    "KR": "2022-12-31",
    "AU": "2022-12-31",
    "MY": "2022-12-31",
}
DEFAULT_COVID_END = "2021-12-31"
#: Events that are seasonal or promotional would absorb holiday effects, so they are not used as controls.
SKIP_EVENTS = re.compile(
    r"early entry|extra hours|extended|summer|christmas|holiday|easter|winter|spring|special ticketed|event in park|opening|closing"
    r"|noel|weihnacht|xmas|new year|valentine",
    re.I,
)
RHO_GRID = (0.0, 0.25, 0.5, 0.75)  # data share capped at 0.75 so every plausible market keeps some prior weight


@dataclass(frozen=True)
class WeightConfig:
    bootstrap: int = 40
    min_rows: int = 120
    school_mass: float = 0.6  # min share of a park's school prior with a school calendar for a day to be used
    school_min_rows: int = 150
    seasonal_harmonics: int = 8
    kappa: float = 5.0
    # validation only: shift every holiday calendar by N days (real vs placebo out-of-fold R2 is the sanity check)
    placebo_shift: int = 0
    workers: int = 6


#: ``(weights, ci_low, ci_high, stability, n_obs)``; the last four are ``None`` when the park falls back to the market prior.
WeightSet = tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]


@dataclass
class ParkFit:
    park_id: int
    regions: list[str]
    weights: dict[str, WeightSet]
    info: dict[str, object] = field(default_factory=dict)
    seconds: float = 0.0


# ---------------------------------------------------------------------------------------------------------------- maths
def controls(u: pd.DataFrame, events: pd.DataFrame, cfg: WeightConfig) -> np.ndarray:
    """Design matrix of nuisance controls for one park's (date-sorted) rows ``u``."""
    n = len(u)
    d = u["date"]
    cols: list[np.ndarray] = [np.ones(n)]
    cols += list(pd.get_dummies(d.dt.dayofweek).astype(float).to_numpy()[:, 1:].T)
    x = d.dt.dayofyear.to_numpy() / 365.25 * 2 * np.pi
    harmonics = cfg.seasonal_harmonics if n >= 800 else 4 if n >= 400 else 2
    for k in range(1, harmonics + 1):
        cols += [np.sin(k * x), np.cos(k * x)]
    years = d.dt.year.to_numpy()
    if n >= 300:
        cols += [(years == y).astype(float) for y in sorted(set(years))[1:]]
    gap = np.diff(
        d.to_numpy().astype("datetime64[D]").astype(int), prepend=-999, append=10**6
    )  # run boundaries: > 14 day gap
    first, last = np.zeros(n), np.zeros(n)
    for s in np.where(gap[:-1] > 14)[0]:
        first[s : s + 3] = 1
    for e in np.where(gap[1:] > 14)[0]:
        last[max(e - 2, 0) : e + 1] = 1
    cols += [first, last]
    e_park = events[events["park_id"] == u["park_id"].iloc[0]]
    e_park = e_park[~e_park["event"].str.contains(SKIP_EVENTS)]
    days_per_event = e_park.groupby("event")["date"].nunique()
    for name in days_per_event[days_per_event >= 15].index[:12]:
        flag = d.isin(set(e_park.loc[e_park["event"] == name, "date"])).astype(float).to_numpy()
        if 0 < flag.sum() < n:
            cols.append(flag)
    design = np.column_stack(cols)
    keep = [0] + [j for j in range(1, design.shape[1]) if design[:, j].std() > 0]
    return design[:, keep]


def ridge_nnls(y: np.ndarray, features: np.ndarray, lam: float) -> np.ndarray:
    """Non-negative ridge: min ||y - F u||^2 + lam ||u||^2 subject to u >= 0."""
    k = features.shape[1]
    a = np.vstack([features, np.sqrt(lam) * np.eye(k)])
    b = np.concatenate([y, np.zeros(k)])
    return nnls(a, b, maxiter=60 * k)[0]


def pick_lambda(y: np.ndarray, features: np.ndarray, lams: np.ndarray, folds_n: int = 6) -> tuple[float, float]:
    """Blocked time-series CV over ``lams``; returns ``(best lam, its out-of-fold R2)``."""
    n = len(y)
    folds = np.array_split(np.arange(n), folds_n)
    sst = (y**2).sum()
    r2 = {}
    for lam in lams:
        sse = 0.0
        for test in folds:
            train = np.setdiff1d(np.arange(n), test)
            u = ridge_nnls(y[train], features[train], lam)
            sse += ((y[test] - features[test] @ u) ** 2).sum()
        r2[lam] = 1 - sse / sst
    best = max(r2, key=lambda k: r2[k])
    return float(best), float(r2[best])


def normalise(a: np.ndarray) -> np.ndarray | None:
    s = a.sum()
    return a / s if s > 1e-9 else None


def blend_with_prior(a: np.ndarray, prior: np.ndarray, rho: float, kappa: float) -> np.ndarray:
    """``(1 - rho) * prior + rho * normalised(a)``, with each market capped at ``kappa`` x its prior (+ a small floor)."""
    w = normalise(a)
    if w is None:
        return prior
    w = (1 - rho) * prior + rho * w
    for _ in range(8):
        w = np.minimum(w, kappa * prior + 0.004)
        w = w / w.sum()
    return w


# ------------------------------------------------------------------------------------------------------------ the fitter
class WeightFitter:
    """Holds the shared inputs and fits one park at a time (picklable, so parks can be fitted in parallel)."""

    def __init__(
        self,
        parks: pd.DataFrame,
        markets: pd.DataFrame,
        matrices: CalendarMatrices,
        crowd: pd.DataFrame,
        events: pd.DataFrame,
        cfg: WeightConfig | None = None,
    ) -> None:
        self.cfg = cfg or WeightConfig()
        self.parks = parks
        self.markets = markets
        self.regions_index = {r: i for i, r in enumerate(matrices.regions)}
        self.dates = matrices.dates
        self.national, self.school = matrices.national, matrices.school
        if self.cfg.placebo_shift:
            self.national = np.roll(self.national, self.cfg.placebo_shift, axis=1)
            self.school = np.roll(self.school, self.cfg.placebo_shift, axis=1)
        self.years = list(matrices.years)
        self.school_covered = matrices.covered_school
        keep = (
            (crowd["status"] == "open")
            & ~crowd["predicted"].astype(str).str.lower().eq("true")
            & crowd["crowd_percent"].notna()
        )
        self.crowd = crowd[keep].copy()
        self.events = events

    # -- helpers
    def _park_rows(self, pid: int, country: str) -> pd.DataFrame:
        u = self.crowd[self.crowd["park_id"] == pid].sort_values("date")
        covid = (u["date"] >= COVID_START) & (u["date"] <= COVID_END_BY_COUNTRY.get(country, DEFAULT_COVID_END))
        return u[~covid].reset_index(drop=True)

    def _prior_only(
        self, pid: int, regions: list[str], prior: np.ndarray, school_ok: np.ndarray, n_days: int, reason: str
    ) -> ParkFit:
        weights: dict[str, WeightSet] = {}
        for kind, ok in (("national", np.ones(len(prior), bool)), ("school", school_ok)):
            w = prior * ok
            weights[kind] = (w / w.sum() if w.sum() > 0 else w, None, None, None, None)
        info = {
            "park_id": pid, "n_days": n_days, "oof_r2": np.nan, "oof_r2_prior": np.nan, "oof_r2_data": np.nan, "reason": reason,
            "school_used": False, "beta_n": np.nan, "beta_s": np.nan, "std_n": np.nan, "std_s": np.nan, "rho_n": 0.0, "rho_s": 0.0,
            "prior_nat": True, "prior_sch": True, "agree_raw": np.nan, "agree_rank": np.nan,
            "history_years": round(n_days / 365, 2), "bn_ci": None, "bs_ci": None,
        }  # fmt: skip
        return ParkFit(pid, regions, weights, info)

    # -- one park
    def fit_park(self, pid: int) -> ParkFit:  # noqa: C901 - one linear pipeline, split into steps by comments
        t0 = time.time()
        cfg = self.cfg
        park = self.parks[self.parks["park_id"] == pid].iloc[0]
        market = self.markets[self.markets["park_id"] == pid]
        ri = np.array([self.regions_index[r] for r in market["region_code"]])
        prior = market["prior_weight"].to_numpy()
        regions = list(market["region_code"])
        u = self._park_rows(pid, park["country_iso2"])
        school_ok = self.school_covered[ri].any(axis=1)

        def done(fit: ParkFit) -> ParkFit:
            fit.seconds = time.time() - t0
            return fit

        if len(u) < cfg.min_rows:
            return done(self._prior_only(pid, regions, prior, school_ok, len(u), "too little data"))
        day_idx = self.dates.get_indexer(pd.Index(u["date"]))
        years = u["date"].dt.year.to_numpy()
        school_mass = np.array(
            [(prior * self.school_covered[ri][:, self.years.index(y)]).sum() / prior.sum() for y in years]
        )
        use_school = (school_mass >= cfg.school_mass).sum() >= cfg.school_min_rows
        if use_school:
            u = u[school_mass >= cfg.school_mass].reset_index(drop=True)
            day_idx = self.dates.get_indexer(pd.Index(u["date"]))
        n = len(u)
        if n < cfg.min_rows:
            return done(self._prior_only(pid, regions, prior, school_ok, len(u), "too little usable data"))

        design = controls(u, self.events, cfg)
        nat_f = self.national[np.ix_(ri, day_idx)].T.astype(float)
        sch_f = self.school[np.ix_(ri, day_idx)].T.astype(float) if use_school else np.zeros((n, len(ri)))
        k = len(ri)
        p_nat = prior / prior.sum()
        p_sch = prior * school_ok
        p_sch = p_sch / p_sch.sum() if p_sch.sum() > 0 else p_sch
        f0 = np.hstack([nat_f, sch_f])
        sq = np.sqrt(np.concatenate([prior, prior]))
        live = f0.std(0) > 1e-9
        f = f0[:, live] * sq[live]
        if f.shape[1] == 0:
            return done(self._prior_only(pid, regions, prior, school_ok, n, "no holiday variation"))

        # targets: log level (main), raw level and rank-normal (robustness), each residualised on the controls
        y_log = np.log(np.maximum(u["crowd_percent"].to_numpy(), 1.0))
        y_raw = u["crowd_percent"].to_numpy() / 100.0
        y_rank = norm.ppf((rankdata(u["crowd_percent"].to_numpy()) - 0.5) / n)
        stacked = np.column_stack([y_log, y_raw, y_rank, f, f0])
        residual = stacked - design @ np.linalg.lstsq(design, stacked, rcond=None)[0]
        nl = f.shape[1]
        ys, fr, fu = residual[:, :3], residual[:, 3 : 3 + nl], residual[:, 3 + nl :]
        r_nat, r_sch = fu[:, :k], fu[:, k:]
        y0 = ys[:, 0]
        lams = np.mean((fr**2).sum(0)) * np.logspace(-3, 1.5, 10)
        lam, _ = pick_lambda(y0, fr, lams)

        def expand(uvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            a = np.zeros(2 * k)
            a[np.nonzero(live)[0]] = uvec * sq[live]
            return a[:k], a[k:]

        def blend(a: np.ndarray, pr: np.ndarray, rho: float) -> np.ndarray:
            return blend_with_prior(a, pr, rho, cfg.kappa)

        # nested CV: how much to trust data-driven weights vs the market prior (rho), separately per type
        rgrid = RHO_GRID if use_school else (0.0,)
        folds = np.array_split(np.arange(n), 6)
        sse = {(rn, rs): 0.0 for rn in RHO_GRID for rs in rgrid}
        for test in folds:
            train = np.setdiff1d(np.arange(n), test)
            a_nat, a_sch = expand(ridge_nnls(y0[train], fr[train], lam))
            for rn, rs in sse:
                score = np.column_stack(
                    [r_nat @ blend(a_nat, p_nat, rn), r_sch @ blend(a_sch, p_sch, rs) if use_school else np.zeros(n)]
                )
                beta = nnls(score[train], y0[train])[0]
                sse[(rn, rs)] += ((y0[test] - score[test] @ beta) ** 2).sum()
        sst = (y0**2).sum()
        r2s = {c: 1 - v / sst for c, v in sse.items()}
        r2_prior, r2_best = r2s[(0.0, 0.0)], max(r2s.values())
        if r2_best - r2_prior <= 0.001:  # data-driven weights add nothing beyond the prior
            rho_n, rho_s = 0.0, 0.0
        else:  # per type: smallest blend that keeps 75% of that type's own gain
            best_n, _ = max(r2s, key=lambda c: r2s[c])

            def smallest_rho(cands: list[tuple[float, float]], fixed_r2: float) -> float:
                gain = max(r for _, r in cands) - fixed_r2
                return 0.0 if gain <= 0.0005 else min(rho for rho, r in cands if r >= fixed_r2 + 0.75 * gain)

            rho_s = smallest_rho([(rs, r2s[(best_n, rs)]) for rs in rgrid], r2s[(best_n, 0.0)])
            rho_n = smallest_rho([(rn, r2s[(rn, rho_s)]) for rn in RHO_GRID], r2s[(0.0, rho_s)])
        r2 = r2s[(rho_n, rho_s)]

        a_nat, a_sch = expand(ridge_nnls(y0, fr, lam))
        w_nat = blend(a_nat, p_nat, rho_n)
        w_sch = blend(a_sch, p_sch, rho_s) if use_school else p_sch
        score_all = np.column_stack([r_nat @ w_nat, r_sch @ w_sch if use_school else np.zeros(n)])
        beta = nnls(score_all, y0)[0]
        if r2 <= 0:
            beta = np.zeros(2)
        std_effect = beta * score_all.std(0) / y0.std()

        # do the raw-level and rank-normal targets pick the same regions? (robustness)
        agree = []
        for j in (1, 2):
            lam_j, _ = pick_lambda(ys[:, j], fr, lams[::2])
            a2_nat, a2_sch = expand(ridge_nnls(ys[:, j], fr, lam_j))
            corr = [
                np.corrcoef(x, x2)[0, 1] for x, x2 in ((a_nat, a2_nat), (a_sch, a2_sch)) if x.std() > 0 and x2.std() > 0
            ]
            agree.append(np.mean(corr) if corr else np.nan)

        # block bootstrap over year-months for intervals and stability
        block = (u["date"].dt.year * 100 + u["date"].dt.month).to_numpy()
        blocks = np.unique(block)
        rng = np.random.default_rng(pid)
        wn_b, ws_b, bn_b, bs_b = [], [], [], []
        for _ in range(cfg.bootstrap):
            rows = np.concatenate([np.nonzero(block == b)[0] for b in rng.choice(blocks, len(blocks))])
            b_nat, b_sch = expand(ridge_nnls(y0[rows], fr[rows], lam))
            wn = blend(b_nat, p_nat, rho_n)
            ws = blend(b_sch, p_sch, rho_s) if use_school else p_sch
            sb = np.column_stack([r_nat[rows] @ wn, r_sch[rows] @ ws if use_school else np.zeros(len(rows))])
            bb = nnls(sb, y0[rows])[0]
            wn_b.append(wn)
            ws_b.append(ws)
            bn_b.append(bb[0])
            bs_b.append(bb[1])
        wn_arr, ws_arr = np.array(wn_b), np.array(ws_b)

        info = {
            "park_id": pid, "n_days": n, "oof_r2": r2, "oof_r2_prior": r2_prior, "oof_r2_data": r2_best,
            "reason": "" if r2 > 0 else "no out-of-fold gain", "school_used": use_school, "beta_n": beta[0], "beta_s": beta[1],
            "std_n": std_effect[0], "std_s": std_effect[1], "rho_n": rho_n, "rho_s": rho_s, "prior_nat": rho_n == 0,
            "prior_sch": rho_s == 0 or not use_school, "agree_raw": agree[0], "agree_rank": agree[1],
            "history_years": round(n / 365, 2), "bn_ci": np.percentile(bn_b, [5, 95]), "bs_ci": np.percentile(bs_b, [5, 95]),
        }  # fmt: skip
        weights: dict[str, WeightSet] = {}
        for kind, w, boot, feats, prior_flag in (
            ("national", w_nat, wn_arr, nat_f, info["prior_nat"]),
            ("school", w_sch, ws_arr, sch_f, info["prior_sch"]),
        ):
            if prior_flag:
                weights[kind] = (w, None, None, None, None)
                continue
            weights[kind] = (
                w,
                np.percentile(boot, 5, axis=0),
                np.percentile(boot, 95, axis=0),
                (boot > 1e-4).mean(0),
                (feats > 0.5).sum(0),
            )
        return done(ParkFit(pid, regions, weights, info))


# ------------------------------------------------------------------------------------------------- run + assemble
_FITTER: WeightFitter | None = None


def _init_worker(fitter: WeightFitter) -> None:
    global _FITTER
    _FITTER = fitter
    warnings.filterwarnings("ignore")


def _fit_one(pid: int) -> ParkFit:
    assert _FITTER is not None
    return _FITTER.fit_park(pid)


def fit_all(fitter: WeightFitter, park_ids: list[int] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit every park (in parallel) and return ``(park_region_weights, park_strength)``."""
    ids = park_ids if park_ids is not None else [int(p) for p in fitter.parks["park_id"]]
    t0 = time.time()
    if fitter.cfg.workers <= 1:
        _init_worker(fitter)
        fits = [_fit_one(p) for p in ids]
    else:
        with ProcessPoolExecutor(fitter.cfg.workers, initializer=_init_worker, initargs=(fitter,)) as pool:
            fits = list(pool.map(_fit_one, ids, chunksize=1))
    log.info("fitted %d parks in %.0fs (slowest %.0fs)", len(fits), time.time() - t0, max(f.seconds for f in fits))
    return assemble(fits)


def assemble(fits: list[ParkFit]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Weights table (one row per park x region x type) and per-park strength table."""
    rows: list[tuple[object, ...]] = []
    infos: list[dict[str, object]] = []
    for fit in fits:
        infos.append(fit.info)
        for kind, (w, lo, hi, stab, nobs) in fit.weights.items():
            keep = w >= 1e-4
            wn = w * keep
            wn = wn / wn.sum()
            for i, region in enumerate(fit.regions):
                if not keep[i]:
                    continue
                weight = round(float(wn[i]), 5)
                if lo is None or hi is None or stab is None or nobs is None:  # prior-only: nothing learned from data
                    rows.append((fit.park_id, region, kind, weight, np.nan, np.nan, np.nan, np.nan, "prior", True))
                    continue
                st, no = stab[i], nobs[i]
                conf = "high" if st >= 0.8 and no >= 30 else "medium" if st >= 0.5 and no >= 10 else "low"
                lo_i, hi_i = round(float(lo[i]), 5), round(float(hi[i]), 5)
                rows.append((fit.park_id, region, kind, weight, lo_i, hi_i, round(float(st), 3), no, conf, False))
    cols = [
        "park_id",
        "region_code",
        "type",
        "weight",
        "ci_low",
        "ci_high",
        "stability",
        "n_obs",
        "confidence",
        "prior_only",
    ]
    return pd.DataFrame(rows, columns=cols), strength_table(infos)


def strength_table(infos: list[dict[str, object]]) -> pd.DataFrame:
    """Per-park summary: standardised effect of each score (95th-percentile park = 1), uplifts and their intervals."""
    s = pd.DataFrame(infos)

    def scaled(col: str) -> pd.Series:
        v = s.loc[s[col].notna() & (s[col] > 0), col]
        p95 = np.percentile(v, 95) if len(v) else 1
        return (s[col] / p95).clip(0, 1)

    s["national_strength"], s["school_strength"] = scaled("std_n"), scaled("std_s")
    s["national_uplift_pct"] = ((np.exp(s["beta_n"]) - 1) * 100).round(1)
    s["school_uplift_pct"] = ((np.exp(s["beta_s"]) - 1) * 100).round(1)

    def interval(ci: object) -> str:
        return "" if ci is None else f"{(np.exp(ci[0]) - 1) * 100:.1f}..{(np.exp(ci[1]) - 1) * 100:.1f}"  # type: ignore[index]

    s["nat_uplift_ci"] = s["bn_ci"].map(interval)
    s["sch_uplift_ci"] = s["bs_ci"].map(interval)
    return s.drop(columns=["bn_ci", "bs_ci"])
