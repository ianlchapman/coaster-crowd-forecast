"""Lightweight table contracts.

A :class:`TableSpec` states which columns a table must have, which values are allowed and which key must be unique.
:func:`validate` raises a single :class:`SchemaError` listing every problem, so a bad input fails early and readably.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import pandas as pd


class SchemaError(ValueError):
    """Raised when an input table does not match its contract."""


@dataclass(frozen=True)
class TableSpec:
    name: str
    required: Sequence[str]
    key: Sequence[str] = ()
    allowed: Mapping[str, Sequence[object]] = field(default_factory=dict)
    ranges: Mapping[str, tuple[float, float]] = field(default_factory=dict)


CROWD_CALENDAR = TableSpec(
    name="crowd-calendar",
    required=["park_id", "date", "status", "crowd_percent", "predicted", "opens", "closes", "events"],
    key=["park_id", "date"],
    allowed={"status": ["open", "closed", "unknown", "no_data"]},
    ranges={"crowd_percent": (0, 100)},
)
PARKS = TableSpec(
    name="parks",
    required=["id", "name", "company_name", "country", "latitude", "longitude", "first_year", "first_month"],
    key=["id"],
    ranges={"latitude": (-90, 90), "longitude": (-180, 180), "first_month": (1, 12)},
)
EVENTS = TableSpec(name="park-events", required=["park_id", "date", "symbol", "event"])


def validate(df: pd.DataFrame, spec: TableSpec) -> pd.DataFrame:
    """Return ``df`` unchanged if it satisfies ``spec``; raise :class:`SchemaError` describing every violation otherwise."""
    problems: list[str] = []
    missing = [c for c in spec.required if c not in df.columns]
    if missing:
        problems.append(f"missing columns: {missing}")
    present = [c for c in spec.key if c in df.columns]
    if present == list(spec.key) and spec.key and df.duplicated(list(spec.key)).any():
        problems.append(f"duplicate rows for key {list(spec.key)}: {int(df.duplicated(list(spec.key)).sum())}")
    for col, ok in spec.allowed.items():
        if col in df.columns:
            bad = sorted(set(df[col].dropna().unique()) - set(ok))
            if bad:
                problems.append(f"{col}: unexpected values {bad[:5]}")
    for col, (lo, hi) in spec.ranges.items():
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(v) and (v.min() < lo or v.max() > hi):
                problems.append(f"{col}: values outside [{lo}, {hi}] (min {v.min()}, max {v.max()})")
    if problems:
        raise SchemaError(f"{spec.name}: " + "; ".join(problems))
    return df
