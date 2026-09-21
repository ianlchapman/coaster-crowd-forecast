# Third-party data notice

The code in this repository is MIT-licensed (see [LICENSE](LICENSE)). It ships **no third-party data**. When you run it you download, read and derive data from the sources below, and their terms apply to you. This page is a summary written for convenience, not legal advice; the providers' own terms are authoritative and can change, so check them before publishing anything.

## Sources and what they require

| Source | Used for | Terms | What to do |
|---|---|---|---|
| **Queue-Times.com** (via the separate loader) | crowd index, opening hours, events | Their API page requires that you display "Powered by Queue-Times.com" linking to <https://queue-times.com/> prominently in an app or service that uses the data. No other terms were found on the pages checked (September 2026); redistribution, storage and commercial-use rules are **not stated there**, so ask them or check for a fuller terms page before republishing any derived data. | Keep the credit below wherever you show or publish forecasts. Do not commit the raw or derived crowd tables (`.gitignore` already excludes `data/`). |
| **Open-Meteo** | historical weather, forecasts, archived forecasts | Data under **CC BY 4.0**. Their terms state: "You may only use the free API services for non-commercial purposes." Data is provided "without any warranty" and Open-Meteo "assumes no responsibility for any inaccuracies or omissions in the data". | Credit Open-Meteo (<https://open-meteo.com/>) where results are shown. For commercial use, use their paid service. |
| **OpenHolidays** | public and school holidays for 19 mostly European countries | The API software is AGPL-3.0. The **data** repository (`openpotato/openholidaysapi.data`) is listed under **ODbL 1.0**, which requires attribution and share-alike for derived databases you publicly use or distribute. | Credit OpenHolidays (<https://www.openholidaysapi.org/>). Treat the tables built from it (`national_and_eu_school.csv`, `school_gapfill_eu.csv`, and the merged calendar matrices) as ODbL-derived: do not redistribute them except under ODbL with attribution. This repo does not ship them. |
| **`holidays` Python package** | public-holiday fallback, lunar and Islamic anchor dates | MIT (a dependency; not redistributed here). | None beyond keeping its licence with the package. |
| **GeoNames** (`cities1000`) | places used to spread each region's population over its geography | **CC BY 4.0**. "You should give credit to GeoNames when using data or web services with a link or another reference to GeoNames." Provided "as is" without warranty or any representation of accuracy, timeliness or completeness. | Credit GeoNames (<https://www.geonames.org/>). |
| **Natural Earth** (admin-1 boundaries) | region polygons | Public domain. | Credit is appreciated, not required. |
| **OpenStreetMap / Nominatim** | four coordinate corrections in `reference/park_coordinate_fixes.csv` | ODbL 1.0. | Credit "© OpenStreetMap contributors". |
| **Official school term dates** | Australia, New Zealand and sampled US districts, transcribed into `scripts/calendars/_au_nz_terms.py` and `_us_districts.py` | Facts published by education authorities; each block records its source URL. | None, but they are transcriptions and can contain errors. |

## Credit line

Use this wherever forecasts or derived results are shown or published:

> Powered by [Queue-Times.com](https://queue-times.com/). Weather data by [Open-Meteo.com](https://open-meteo.com/) (CC BY 4.0). Holiday data includes [OpenHolidays](https://www.openholidaysapi.org/) (ODbL). Place data from [GeoNames](https://www.geonames.org/) (CC BY 4.0).

## Accuracy

* Forecasts are statistical estimates with typical errors of about 15 index points on a 0-100 scale (see [docs/MODEL_CARD.md](docs/MODEL_CARD.md)). They are not headcounts and not for safety or capacity decisions.
* Many school calendars outside Europe, Australia and New Zealand are **rule-derived** or approximated (medium or low confidence), and mainland-European school data before about 2019 is missing. Do not use them to plan real events without checking official sources.
* Weather values are third-party model output; forecasts get less accurate with lead time.

## No affiliation

This is an independent project. It is not affiliated with, endorsed by or sponsored by Queue-Times.com, Open-Meteo, OpenHolidays, GeoNames, or any theme park, operator or company mentioned. Names and trademarks belong to their owners.
