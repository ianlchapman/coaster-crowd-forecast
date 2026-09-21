"""Merge the four source calendar tables into one region x date holiday intensity matrix.

Sources (built by ``scripts/calendars/``): public holidays + European school holidays, rule-based non-European school
holidays, a European school gap-fill, and school holidays verified against official AU/NZ/US calendars (which supersede
the matching rule-based blocks).

Rules
-----
* Rows for a bare country code apply to every region of that country.
* Sub-region rows finer than the region table are dropped, except: Polish numeric codes are mapped to letters; two French
  codes fold into ``FR-GES`` (share x0.55); children of a country that is a single region (Swiss cantons, Portuguese districts ...)
  become an equal-weight fraction of that country.
* Overlapping rows on a region-date combine as ``1 - prod(1 - share)``, never a sum.
* Countries with no national rows at all get national holidays from the ``holidays`` package (medium confidence).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import holidays as holidays_pkg
import numpy as np
import pandas as pd
import pycountry

from crowdcast.scoring.daily import CalendarMatrices

log = logging.getLogger(__name__)

FIRST_DATE, LAST_DATE = "2014-01-01", "2027-12-31"
COLUMNS = ["region_code", "date", "type", "name", "share", "confidence"]
PL_NUMERIC_TO_LETTERS = {
    "02": "DS", "04": "KP", "06": "LU", "08": "LB", "10": "LD", "12": "MA", "14": "MZ", "16": "OP",
    "18": "PK", "20": "PD", "22": "PM", "24": "SL", "26": "SK", "28": "WN", "30": "WP", "32": "ZP",
}  # fmt: skip
FR_FOLDED = ("FR-57", "FR-6AE")  # Alsace-Moselle: folded into Grand Est
FR_FOLD_SHARE = 0.55
NO_SUNDAY_FILL = ("SA", "AE")  # weekend is Fri/Sat there, so Sunday holidays are real


@dataclass
class MergeResult:
    matrices: CalendarMatrices
    sparse: pd.DataFrame  # region_code, date, national, school (rows with any holiday)
    coverage: pd.DataFrame  # region_code, year, national_covered, school_covered
    dropped: dict[str, int]  # rows below the region level, by country


def load_sources(cal_dir: Path) -> pd.DataFrame:
    """Concatenate the source tables, letting verified rows replace the rule-based blocks they supersede."""
    read = lambda name: pd.read_csv(cal_dir / name, usecols=COLUMNS)  # noqa: E731
    eu, gapfill, rules, verified = (
        read("national_and_eu_school.csv"),
        read("school_gapfill_eu.csv"),
        read("school_non_eu.csv"),
        read("school_non_eu_verified.csv"),
    )
    replaces = pd.read_csv(cal_dir / "school_non_eu_replaces.csv")
    rules["year"] = rules["date"].str[:4].astype(int)
    key = (
        replaces[["region_code", "year", "old_name"]]
        .rename(columns={"old_name": "name"})
        .drop_duplicates()
        .assign(drop=True)
    )
    rules = rules.merge(key, on=["region_code", "year", "name"], how="left")
    log.info(
        "rule-based rows %d, superseded %d, verified rows added %d",
        len(rules),
        int(rules["drop"].fillna(False).sum()),
        len(verified),
    )
    rules = rules[rules["drop"].isna()].drop(columns=["year", "drop"])
    out = pd.concat([eu, gapfill, rules, verified], ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out["share"] = out["share"].astype(float)  # object dtype if a source table is empty
    return out


def fill_national(df: pd.DataFrame, countries_needed: set[str]) -> pd.DataFrame:
    """National holidays from the ``holidays`` package for countries that have no national rows in the sources."""
    have = {c.split("-")[0] for c in df.loc[df["type"] == "national", "region_code"]}
    rows: list[tuple[Any, ...]] = []
    for cc in sorted(countries_needed - have):
        try:
            found = holidays_pkg.country_holidays(cc, years=range(2014, 2028))
        except NotImplementedError:
            log.warning("no holidays package data for %s", cc)
            continue
        rows.extend(
            (cc, pd.Timestamp(d), "national", name, 1.0, "medium")
            for d, name in found.items()
            if pd.Timestamp(d).dayofweek != 6 or cc in NO_SUNDAY_FILL
        )
    log.info("national fill for %s: %d rows", sorted(countries_needed - have), len(rows))
    return pd.concat([df, pd.DataFrame(rows, columns=pd.Index(COLUMNS))], ignore_index=True)


def remap_subregions(df: pd.DataFrame) -> pd.DataFrame:
    def remap(code: str) -> str:
        return (
            "PL-" + PL_NUMERIC_TO_LETTERS[code[3:]]
            if code.startswith("PL-") and code[3:] in PL_NUMERIC_TO_LETTERS
            else code
        )

    df = df.copy()
    df["region_code"] = df["region_code"].map(remap)
    folded = df["region_code"].isin(FR_FOLDED)
    df.loc[folded, "share"] *= FR_FOLD_SHARE
    df.loc[folded, "region_code"] = "FR-GES"
    df["share"] = df["share"].clip(upper=1 - 1e-9)
    return df


def _combine_overlaps(df: pd.DataFrame) -> pd.DataFrame:
    """Combine rows on the same (region, date, type): 1 - prod(1 - share)."""
    df = df.assign(l=np.log1p(-df["share"]))
    g = df.groupby(["region_code", "date", "type"])["l"].sum().reset_index()
    g["s"] = 1 - np.exp(g["l"])
    return g


def _n_children(cc: str) -> int:
    try:
        return len([s for s in pycountry.subdivisions.get(country_code=cc) if s.parent_code is None]) or 1
    except (KeyError, LookupError):
        return 1


def merge_calendars(cal_dir: Path, regions: pd.DataFrame) -> MergeResult:
    """Build the region x date matrices, sparse table and per-year coverage flags."""
    codes = list(regions["region_code"])
    region_set, ridx = set(codes), {r: i for i, r in enumerate(codes)}
    country = dict(zip(regions["region_code"], regions["country"], strict=True))
    df = remap_subregions(fill_national(load_sources(cal_dir), {country[r] for r in codes}))
    g = _combine_overlaps(df)

    dates = pd.date_range(FIRST_DATE, LAST_DATE)
    didx = {d: i for i, d in enumerate(dates)}
    n_r, n_d = len(codes), len(dates)
    log_keep = {t: np.zeros((n_r, n_d)) for t in ("national", "school")}  # accumulated log(1 - s)
    child_frac = {t: np.zeros((n_r, n_d)) for t in ("national", "school")}
    by_country: dict[str, list[str]] = {}
    for r in codes:
        by_country.setdefault(country[r], []).append(r)
    covered: set[tuple[str, int, str]] = set()
    dropped: dict[str, int] = {}
    n_children: dict[str, int] = {}

    rows: Iterable[Any] = g.itertuples(index=False)  # dynamic namedtuples
    for row in rows:
        code, date, kind, s = row.region_code, row.date, row.type, row.s
        if date not in didx:
            continue
        j = didx[date]
        cc = code.split("-")[0]
        if code in region_set:
            targets = [ridx[code]]
        elif "-" not in code:
            targets = [ridx[r] for r in by_country.get(cc, [])]
            if not targets:
                dropped[cc] = dropped.get(cc, 0) + 1
                continue
        else:
            if (
                cc in region_set and country.get(cc) == cc
            ):  # the country is one region: children are equal fractions of it
                n = n_children.setdefault(cc, _n_children(cc))
                child_frac[kind][ridx[cc], j] += s / n
                covered.add((cc, date.year, kind))
            else:
                dropped[cc] = dropped.get(cc, 0) + 1
            continue
        covered.add((cc, date.year, kind))
        for i in targets:
            log_keep[kind][i, j] += np.log1p(-min(s, 1 - 1e-9))

    matrix = {t: 1 - np.exp(log_keep[t] + np.log1p(-np.minimum(child_frac[t], 1 - 1e-9))) for t in log_keep}
    years = sorted(set(dates.year))
    cov = {t: np.zeros((n_r, len(years)), dtype=bool) for t in log_keep}
    for cc, year, kind in covered:
        for r in by_country.get(cc, []):
            cov[kind][ridx[r], years.index(year)] = True
    for r in codes:  # one national row anywhere in a country's history counts as national coverage for all years
        if cov["national"][ridx[r]].any():
            cov["national"][ridx[r]] = True

    matrices = CalendarMatrices(
        np.array(codes),
        dates,
        matrix["national"].astype("float32"),
        matrix["school"].astype("float32"),
        np.array(years),
        cov["national"],
        cov["school"],
    )
    sparse = _sparse_table(matrix, codes, dates)
    coverage = pd.DataFrame(
        [
            (codes[i], years[k], bool(cov["national"][i, k]), bool(cov["school"][i, k]))
            for i in range(n_r)
            for k in range(len(years))
        ],
        columns=["region_code", "year", "national_covered", "school_covered"],
    )
    log.info("regions with no national coverage: %s", sorted(r for r in codes if not cov["national"][ridx[r]].any()))
    return MergeResult(matrices, sparse, coverage, dropped)


def _sparse_table(matrix: dict[str, np.ndarray], codes: list[str], dates: pd.DatetimeIndex) -> pd.DataFrame:
    parts = []
    for kind, m in matrix.items():
        ii, jj = np.nonzero(m > 1e-6)
        parts.append(
            pd.DataFrame(
                {"region_code": np.array(codes)[ii], "date": dates[jj], "type": kind, "value": m[ii, jj].round(4)}
            )
        )
    wide = (
        pd.concat(parts)
        .pivot_table(index=["region_code", "date"], columns="type", values="value", fill_value=0)
        .reset_index()
    )
    wide["date"] = wide["date"].dt.strftime("%Y-%m-%d")
    return wide
