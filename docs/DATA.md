# Data

Nothing under `data/` is committed (`.gitignore`). This page says what each input is, where it comes from and how to lay it out.

## Layout

```
data/
  raw/
    loader/    crowd-calendar.csv  parks.csv  park-events.csv      <- queue-times loader output (not redistributable)
    geo/       cities1000.txt  ne_admin1.geojson                    <- GeoNames, Natural Earth
  interim/
    calendars/ openholidays_cache/  national_and_eu_school.csv  school_*.csv  ...
    weather/   archive/park_<id>.csv  forecast/  previous_runs/
  processed/   parks/  holidays/  weather features  enhanced-crowd-calendar.csv  models/  experiments/
```

Set `CROWDCAST_DATA_DIR` to put the data directory elsewhere.

## Inputs from the loader (contract)

The three files are produced by a separate project that pulls queue times from queue-times.com and derives a daily crowd index. Their schemas are checked on load (`src/crowdcast/data/schema.py`); `crowdcast.data.synthetic` generates files with the same columns.

**`crowd-calendar.csv`**: one row per park-day.

| column | meaning |
|---|---|
| `park_id`, `date` | key (unique) |
| `status` | `open`, `closed`, `unknown`, `no_data` |
| `crowd_percent` | 0-100 rank-normalised crowd index; empty when unknown |
| `predicted` | `true` if the loader estimated the value rather than observing it (never used as a label) |
| `opens`, `closes` | `HH:MM` local time |
| `events` | free text of the day's events |

**`parks.csv`**: `id, name, company_name, country, latitude, longitude, first_year, first_month`. Some coordinates are wrong; `reference/park_coordinate_fixes.csv` corrects the ones found, and `parks-build` logs any park whose polygon country disagrees with the loader's.

**`park-events.csv`**: `park_id, date, symbol, event`, used to pick non-seasonal event controls when fitting holiday weights.

## External sources fetched by the code

| Source | Used for | Notes |
|---|---|---|
| Open-Meteo archive, forecast and previous-runs APIs | daily weather, forecasts, archived forecasts | Free tier, no key. Attribution required (CC BY 4.0 data); non-commercial use. Rate limits are handled (`weather/client.py`); requests are deduplicated per 0.25 degree grid cell. |
| OpenHolidays API | public and school holidays for 19 mostly European countries | Data listed under ODbL 1.0 (attribution and share-alike): do not redistribute the derived calendar tables except under ODbL. `scripts/calendars/10_fetch_openholidays.py` caches raw responses |
| `holidays` Python package | public-holiday fallback and lunar/Islamic anchor dates | MIT |
| GeoNames `cities1000` | places to spread each region's population over its geography | CC BY 4.0 |
| Natural Earth admin-1 (10m) | region polygons | public domain |
| Official term dates | AU/NZ and sampled US district school calendars, transcribed into `scripts/calendars/_au_nz_terms.py` and `_us_districts.py` | each block carries its source URL and confidence in the output tables |

Download `cities1000` from GeoNames (unzip to `cities1000.txt`) and a Natural Earth 10m admin-1 (states/provinces) GeoJSON, saved as `ne_admin1.geojson`, into `data/raw/geo/`. The GeoJSON must carry the properties `iso_a2`, `iso_3166_2`, `region_cod`, `region`, `geonunit` and `name`, which the region mapping reads.

## Why the crowd data is not included

It is derived from queue-times.com data. Their API page requires a "Powered by Queue-Times.com" credit linking to <https://queue-times.com/> and does not state redistribution rules, so check with them before publishing any derived table. See [`../NOTICE.md`](../NOTICE.md) for every source's terms. This repository therefore ships code, small curated reference tables and a synthetic generator only.

## Rebuilding from scratch

```bash
# 1. put the three loader files in data/raw/loader/ and the two geodata files in data/raw/geo/
crowdcast parks-build
python scripts/calendars/10_fetch_openholidays.py          # network, cached
python scripts/calendars/20_build_public_and_eu_school.py
python scripts/calendars/30_build_school_non_eu.py
python scripts/calendars/40_build_school_gapfill_eu.py
python scripts/calendars/50_build_school_verified.py
crowdcast calendars-merge
crowdcast holidays-fit && crowdcast holidays-score && crowdcast enhance
crowdcast weather-fetch
crowdcast train && crowdcast train-status && crowdcast evaluate
```

The calendar scripts are hand-curated data builders ported from the research phase. They are formatted but exempt from the strict lint rules on purpose (see `pyproject.toml`); their outputs were checked to be identical to the originals.

## Quality notes on the holiday calendars

* Most school calendars outside Europe, Australia and New Zealand are rule-derived (medium or low confidence). GB, Nordic and Turkish school calendars are rule-derived too.
* Mainland-European school data before about 2019 is missing; the scorer reports `school_coverage` below 1 there, and weight fitting only uses days where enough of a park's market is covered.
* Region populations and country affluence in `reference/` are approximate hand-entered figures.
