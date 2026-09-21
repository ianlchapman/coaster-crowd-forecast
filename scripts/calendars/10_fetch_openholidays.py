"""Download OpenHolidays API responses (public + school holidays, 2014-2027) into the interim cache.

Ported research script (hand-curated calendar data). Paths come from crowdcast.config.Paths; see docs/DATA.md.
"""

import json, os, time, urllib.request, urllib.error
from crowdcast.config import Paths

CACHE = str(Paths().interim / "calendars" / "openholidays_cache")
os.makedirs(CACHE, exist_ok=True)
BASE = "https://openholidaysapi.org"
OH = "AT BE BR CH CZ DE ES FR HU IE IT LU MX NL PL PT RO SE SK".split()
WIN = [
    ("2014-01-01", "2016-12-31"),
    ("2017-01-01", "2019-12-31"),
    ("2020-01-01", "2022-12-31"),
    ("2023-01-01", "2025-12-31"),
    ("2026-01-01", "2027-12-31"),
]


def get(path, name):
    f = os.path.join(CACHE, name + ".json")
    if os.path.exists(f):
        return
    url = BASE + path
    for i in range(4):
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers={"Accept": "application/json"}), timeout=60
            ) as r:
                data = r.read()
            open(f, "wb").write(data)
            time.sleep(0.4)
            return
        except Exception as e:
            print("retry", url, e)
            time.sleep(3 * (i + 1))
    print("FAILED", url)


if __name__ == "__main__":
    for c in OH:
        get(f"/Subdivisions?countryIsoCode={c}", f"sub_{c}")
        for ep, tag in (("PublicHolidays", "pub"), ("SchoolHolidays", "sch")):
            for a, b in WIN:
                get(f"/{ep}?countryIsoCode={c}&languageIsoCode=EN&validFrom={a}&validTo={b}", f"{tag}_{c}_{a[:4]}")
        print(c, flush=True)
