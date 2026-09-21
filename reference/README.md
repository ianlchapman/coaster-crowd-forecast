# Reference tables

Small, hand-curated inputs that are part of the method. They are committed; everything else under `data/` is not.

| File | What it is | Provenance |
|---|---|---|
| `park_tiers.csv` | Market reach of each park: `local`, `regional`, `continental`, `intercontinental`, with a one-line reason and an optional decay-scale override (km). | Analyst judgement. Several tiers were set by the project owner (marked in the reason). This drives the prior over source markets, so treat it as an assumption, not a measurement. |
| `park_coordinate_fixes.csv` | Corrections for coordinates that are clearly wrong in the loader's `parks.csv`. | Checked against OpenStreetMap/Nominatim; each row says what was wrong. |
| `countries.csv` | Country population (millions), whether the country is split into admin-1 regions, and a crude 4-step "theme-park travel propensity" (`affluence`). | Population: rounded UN/World-Bank-style figures for 2024/25 entered by hand. `affluence`: judgement. Both are approximate. |
| `region_population.csv` | Admin-1 population (thousands) for the split countries. | Hand-entered approximate estimates (c. 2022-24), not from one official series; rescaled to the country totals in `countries.csv`. |

Why these are hand-entered: they only shape a *prior* over which source regions feed a park. The final weights blend that prior with
what the park's own crowd data says (see `docs/METHODOLOGY.md`), so modest errors here are damped, but they are not zero.
