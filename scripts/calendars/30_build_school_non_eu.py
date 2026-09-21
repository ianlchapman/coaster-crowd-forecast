"""Rule-based school holidays for non-European countries -> school_non_eu.csv.

No network needed except the `holidays` package (used only for lunar / Islamic anchor dates: CNY, Eid, Diwali).
Ported research script (hand-curated rules); paths come from crowdcast.config.Paths.
"""

import csv, os
from datetime import date, timedelta
from dateutil.easter import easter
import holidays as H
from crowdcast.config import Paths

OUT = str(Paths().calendars_dir / "school_non_eu.csv")
LO, HI = date(2014, 1, 1), date(2027, 12, 31)
YEARS = range(2013, 2029)
D = timedelta
rows = {}  # (region,date,type,name) -> dict


def add(region, d, name, share, source, conf, notes):
    if d < LO or d > HI or share < 0.02:
        return
    k = (region, d, "school", name)
    if k in rows:
        rows[k]["share"] = min(1.0, rows[k]["share"] + share)
    else:
        rows[k] = dict(share=min(1.0, share), source=source, confidence=conf, notes=notes)


def rng(region, a, b, name, share, source, conf, notes):
    d = a
    while d <= b:
        add(region, d, name, share, source, conf, notes)
        d += D(1)


def mon_on_after(y, m, dd):
    d = date(y, m, dd)
    return d + D((0 - d.weekday()) % 7)


def sat_on_after(y, m, dd):
    d = date(y, m, dd)
    return d + D((5 - d.weekday()) % 7)


def sun_on_after(y, m, dd):
    d = date(y, m, dd)
    return d + D((6 - d.weekday()) % 7)


def fri_on_before(y, m, dd):
    d = date(y, m, dd)
    return d - D((d.weekday() - 4) % 7)


def labor_day(y):
    return mon_on_after(y, 9, 1)


def nth_weekday(y, m, wd, n):
    d = date(y, m, 1)
    d += D((wd - d.weekday()) % 7)
    return d + D(7 * (n - 1))


# ---------- lunar/islamic anchors from holidays pkg ----------
def first_named(country, y, subs):
    h = H.country_holidays(country, years=y)
    c = [d for d, n in h.items() if any(s in n for s in subs)]
    return min(c) if c else None


CNY = {y: first_named("CN", y, ["春节"]) for y in YEARS}
EID_F = {y: first_named("ID", y, ["Idul Fitri"]) for y in YEARS}
EID_A = {y: first_named("ID", y, ["Idul Adha"]) for y in YEARS}
DIWALI = {y: first_named("IN", y, ["Diwali"]) for y in YEARS}
for nm, dct in [("CNY", CNY), ("EID_F", EID_F), ("EID_A", EID_A), ("DIWALI", DIWALI)]:
    assert all(dct[y] for y in range(2014, 2028)), nm

# =================================================================== US
US_SRC = "Rule-derived from typical district calendars (state DOE guidance, NCES, major-district calendars, memory of typical patterns)"
# state: (last-day-of-school median 'MM-DD', first-day median 'MM-DD' or 'LD' = Labor Day)
US_SUMMER = {
    "AL": ("05-22", "08-06"),
    "AK": ("05-22", "08-20"),
    "AZ": ("05-22", "08-01"),
    "AR": ("05-24", "08-15"),
    "CA": ("06-08", "08-15"),
    "CO": ("05-29", "08-15"),
    "CT": ("06-18", "08-30"),
    "DE": ("06-15", "08-28"),
    "DC": ("06-15", "08-25"),
    "FL": ("05-30", "08-10"),
    "GA": ("05-24", "08-03"),
    "HI": ("05-25", "08-01"),
    "ID": ("05-28", "08-20"),
    "IL": ("06-01", "08-20"),
    "IN": ("05-25", "08-01"),
    "IA": ("05-25", "08-23"),
    "KS": ("05-20", "08-15"),
    "KY": ("05-25", "08-08"),
    "LA": ("05-22", "08-06"),
    "ME": ("06-15", "08-30"),
    "MD": ("06-15", "08-25"),
    "MA": ("06-20", "LD"),
    "MI": ("06-12", "LD<2023:08-28"),
    "MN": ("06-08", "LD"),
    "MS": ("05-22", "08-05"),
    "MO": ("05-22", "08-20"),
    "MT": ("06-05", "08-25"),
    "NE": ("05-25", "08-15"),
    "NV": ("05-30", "08-10"),
    "NH": ("06-15", "08-28"),
    "NJ": ("06-22", "LD"),
    "NM": ("05-25", "08-10"),
    "NY": ("06-26", "LD"),
    "NC": ("06-08", "08-25"),
    "ND": ("05-25", "08-25"),
    "OH": ("05-28", "08-20"),
    "OK": ("05-22", "08-10"),
    "OR": ("06-12", "08-28"),
    "PA": ("06-08", "08-25"),
    "RI": ("06-18", "08-28"),
    "SC": ("05-28", "08-15"),
    "SD": ("05-22", "08-20"),
    "TN": ("05-22", "08-03"),
    "TX": ("05-28", "08-12"),
    "UT": ("06-01", "08-20"),
    "VT": ("06-15", "08-28"),
    "VA": ("06-15", "08-25"),
    "WA": ("06-20", "08-30"),
    "WV": ("06-08", "08-15"),
    "WI": ("06-05", "LD"),
    "WY": ("05-25", "08-20"),
}
# spring break: anchors are "Monday on/after MMDD" or Easter-relative ("E-6" = Monday of Holy Week, "E+1" = Easter Monday)
US_SPRING = {
    "TX": {"0308": 0.55, "0315": 0.35, "E-6": 0.05},
    "OK": {"0315": 0.6, "0322": 0.25},
    "FL": {"0308": 0.3, "0315": 0.45, "0322": 0.15, "E-6": 0.1},
    "GA": {"0329": 0.15, "0405": 0.6, "0412": 0.1},
    "AL": {"0315": 0.25, "0322": 0.3, "0329": 0.3, "0405": 0.1},
    "TN": {"0322": 0.15, "0329": 0.25, "0405": 0.5},
    "SC": {"0329": 0.2, "0405": 0.5, "0412": 0.15},
    "NC": {"E-6": 0.35, "0405": 0.2, "E+1": 0.35},
    "KY": {"0329": 0.3, "0405": 0.5},
    "VA": {"0329": 0.4, "0405": 0.3, "E-6": 0.1},
    "LA": {"E-6": 0.35, "E+1": 0.2, "0315": 0.2},
    "MS": {"0315": 0.5, "0322": 0.3},
    "AR": {"0315": 0.55, "0322": 0.3},
    "NY": {"E-6": 0.35, "E+1": 0.3, "0412": 0.2},
    "MA": {"0415": 0.85},
    "ME": {"0415": 0.8},
    "NH": {"0415": 0.55, "E-6": 0.2},
    "VT": {"0329": 0.5, "0415": 0.3},
    "CT": {"E-6": 0.45, "E+1": 0.35},
    "RI": {"0415": 0.8},
    "NJ": {"E-6": 0.35, "E+1": 0.35, "0405": 0.1},
    "PA": {"E-6": 0.3, "E+1": 0.3, "0405": 0.1},
    "DE": {"E-6": 0.35, "E+1": 0.3},
    "MD": {"E-6": 0.35, "E+1": 0.35},
    "DC": {"0405": 0.5, "0412": 0.3},
    "WV": {"E-6": 0.3, "E+1": 0.3, "0329": 0.2},
    "OH": {"0322": 0.3, "0329": 0.3, "E+1": 0.2},
    "IN": {"0315": 0.2, "0322": 0.3, "0329": 0.3},
    "IL": {"0322": 0.5, "0329": 0.3},
    "MI": {"0322": 0.2, "0329": 0.35, "0405": 0.3},
    "WI": {"0322": 0.3, "0329": 0.3, "E-6": 0.2},
    "MN": {"0315": 0.4, "0322": 0.3, "E-6": 0.2},
    "IA": {"0315": 0.5, "0322": 0.3},
    "MO": {"0315": 0.35, "0322": 0.3, "0329": 0.2},
    "KS": {"0315": 0.5, "0322": 0.3},
    "NE": {"0315": 0.5, "0322": 0.3},
    "ND": {"0322": 0.3, "0329": 0.3},
    "SD": {"0315": 0.4, "0322": 0.3},
    "CA": {"0322": 0.15, "0329": 0.3, "0405": 0.3, "0415": 0.1},
    "WA": {"0329": 0.35, "0405": 0.35},
    "OR": {"0322": 0.45, "0329": 0.35},
    "ID": {"0315": 0.3, "0322": 0.3, "0329": 0.2},
    "NV": {"0322": 0.45, "0329": 0.25, "0405": 0.1},
    "AZ": {"0308": 0.35, "0315": 0.45},
    "NM": {"0322": 0.4, "0329": 0.3},
    "CO": {"0315": 0.4, "0322": 0.35, "0329": 0.15},
    "UT": {"0322": 0.4, "0329": 0.3},
    "MT": {"0322": 0.3, "0329": 0.3, "E-6": 0.2},
    "WY": {"0315": 0.4, "0322": 0.3},
    "AK": {"0315": 0.4, "0322": 0.3, "0329": 0.2},
    "HI": {"0315": 0.8},
}
assert set(US_SPRING) == set(US_SUMMER) and len(US_SUMMER) == 51


def md(y, s):
    return date(y, int(s[:2]), int(s[3:]))


def build_us():
    for st, (last, first) in US_SUMMER.items():
        reg = "US-" + st
        for y in range(2014, 2028):
            a = md(y, last)
            if first == "LD":
                b = labor_day(y)
            elif first.startswith("LD<2023"):
                b = labor_day(y) if y < 2023 else md(y, first.split(":")[1])
            else:
                b = md(y, first)
            w = 10
            note = (
                f"Median district last day {last} / first day {first} (weekends included); share ramps linearly (0.05-0.95) +-{w}d "
                "around each median, i.e. share of pupils/districts on break. Same MM-DD every year (Labor Day-tied where noted)."
            )
            d = a - D(w)
            while d <= b + D(w):
                s_start = min(0.95, max(0.05, 0.5 + (d - a).days / (2 * w)))
                s_end = min(0.95, max(0.05, 0.5 + (b - d).days / (2 * w)))
                add(reg, d, "Summer break", round(min(s_start, s_end), 2), US_SRC, "low", note)
                d += D(1)
            # spring break
            E = easter(y)
            for anc, sh in US_SPRING[st].items():
                if anc.startswith("E"):
                    m0 = E + D(int(anc[1:]))
                else:
                    m0 = mon_on_after(y, int(anc[:2]), int(anc[2:]))
                for i in range(7):
                    add(
                        reg,
                        m0 + D(i),
                        "Spring break",
                        sh,
                        US_SRC,
                        "low",
                        "Distribution across districts of the spring-break week (Mon-Sun); shares sum <1 as tail of districts/other weeks not modelled. "
                        f"Anchor {anc}: 'MMDD'=Monday on/after that date, 'E-6/E+1'=Holy-Week/Easter-Monday week.",
                    )
            # thanksgiving
            tg = nth_weekday(y, 11, 3, 4)
            nt = (
                "Share by day: Mon-Tue ~30% (full week off), Wed ~70%, Thu-Fri ~98%, weekend 1.0. Near-universal in US."
            )
            for off, sh in [(-3, 0.3), (-2, 0.3), (-1, 0.7), (0, 1.0), (1, 0.98), (2, 1.0), (3, 1.0)]:
                add(reg, tg + D(off), "Thanksgiving break", sh, US_SRC, "medium", nt)
            # winter
            wnt = "Core Dec 24-Jan 1 ~95%; Dec 19-23 and Jan 2-4 ~50% (district-dependent, many start Dec 21-23 and return Jan 2-6)."
            for dd in range(19, 32):
                dt = date(y, 12, dd)
                add(reg, dt, "Winter break", 0.5 if dd < 24 else 0.95, US_SRC, "medium", wnt)
            for dd in range(1, 5):
                dt = date(y + 1, 1, dd)
                add(reg, dt, "Winter break", 0.95 if dd == 1 else 0.5, US_SRC, "medium", wnt)


# ============================================================ skeleton (AU/NZ)
def term_skeleton(y, anchor, terms, hol_days):
    """Return list of holiday (start,end) blocks in year y. anchor=(m,d) Monday-on/after of term-1 start week."""
    t = mon_on_after(y, *anchor)
    blocks = []
    for i, wks in enumerate(terms[:3]):
        end = t + D(7 * wks) - D(3)  # Friday of last week (t is Monday)
        s = end + D(1)
        e = s + D(hol_days - 1)
        blocks.append((s, e))
        t = e + D(1)
        t = t + D((0 - t.weekday()) % 7)  # next Monday
    return blocks


AU_STATES = {
    # state: (anchor, terms, hol_days, Dec-end-anchor(day of Dec 'Friday on/before'), summer_end_offset)
    "NSW": ((1, 29), [10, 10, 10], 16, 20),
    "ACT": ((1, 29), [10, 10, 10], 16, 20),
    "SA": ((1, 27), [10, 10, 10], 16, 13),
    "WA": ((1, 29), [10, 10, 10], 16, 20),
    "TAS": ((1, 29), [10, 10, 10], 16, 20),
    "VIC": ((1, 26), [10, 10, 10], 16, 20),
    "QLD": ((1, 22), [10, 10, 10], 16, 13),
    "NT": ((1, 20), [10, 10, 10], 16, 13),
}


def build_au_nz():
    src = "Rule-derived: 10-week terms + 2-week breaks from a Term-1 start anchor, calibrated on my recollection of official 2023-2025 dept-of-education dates (NSW/VIC/QLD/SA/WA/TAS/ACT/NT); +-1 week error possible"
    names = ["Autumn/Easter school holidays", "Winter school holidays", "Spring school holidays"]
    for st, (anc, terms, hd, dec) in AU_STATES.items():
        reg = "AU-" + st
        note = f"State term-date model (anchor Term-1 Monday on/after {anc[0]:02d}-{anc[1]:02d}, 10-wk terms, {hd}d breaks incl. weekends); QLD/SA/NT summer starts ~Dec 13, others ~Dec 20. Not fetched from official pages."
        for y in range(2014, 2028):
            for (s, e), n in zip(term_skeleton(y, anc, terms, hd), names):
                rng(reg, s, e, n, 1.0, src, "medium", note)
            # summer: Friday on/before Dec X -> day after, until Term-1 start (Mon+2 next year)
        for y in range(2013, 2028):
            s = fri_on_before(y, 12, dec) + D(1)
            e = mon_on_after(y + 1, *anc) + D(1)
            rng(reg, s, e, "Summer school holidays", 1.0, src, "medium", note)
    # NZ: NSW-like skeleton
    src = "Rule-derived from NZ Ministry of Education term-date pattern (4 terms, ~2wk breaks); +-1 week possible"
    note = "NZ national term model: T1 Monday on/after Feb 3, 10-wk terms, 2-wk breaks; summer from ~Dec 20 to ~Feb 3."
    for y in range(2014, 2028):
        for (s, e), n in zip(term_skeleton(y, (2, 3), [10, 10, 10], 16), names):
            rng("NZ", s, e, n, 1.0, src, "medium", note)
    for y in range(2013, 2028):
        rng(
            "NZ",
            fri_on_before(y, 12, 20) + D(1),
            mon_on_after(y + 1, 2, 3) + D(1),
            "Summer school holidays",
            1.0,
            src,
            "medium",
            note,
        )


# ================================================================== CA
CA_PROV = ["AB", "BC", "MB", "NB", "NL", "NS", "ON", "PE", "QC", "SK"]


def build_ca():
    src = "Rule-derived from provincial ministry / major-board calendars (ON, BC, AB, QC checked from memory for 2018-2025)"
    for p in CA_PROV:
        reg = "CA-" + p
        for y in range(2014, 2028):
            # summer
            if p == "QC":
                a, b = date(y, 6, 21), date(y, 8, 25)
            elif p == "NB":
                a, b = sat_on_after(y, 6, 20), labor_day(y)
            elif p == "ON":
                a, b = sat_on_after(y, 6, 28), labor_day(y)
            else:
                a, b = sat_on_after(y, 6, 26), labor_day(y)
            rng(
                reg,
                a,
                b,
                "Summer break",
                1.0,
                src,
                "medium",
                "Summer break: last day rule (ON: Sat on/after Jun 28; QC Jun 21; others Sat on/after Jun 26) to Labor Day (QC: Aug 25). Not adjusted for pandemic-era changes.",
            )
            # spring/march break
            if p == "ON":
                s, n = mon_on_after(y, 3, 10), 5
            elif p == "BC":
                s, n = mon_on_after(y, 3, 11), 10
            elif p in ("AB", "MB", "SK"):
                s, n = mon_on_after(y, 3, 22), 5
            elif p == "QC":
                s, n = mon_on_after(y, 3, 1), 5
            elif p == "NS":
                s, n = mon_on_after(y, 3, 15), 5
            elif p == "NB":
                s, n = mon_on_after(y, 3, 10), 5
            else:
                s, n = mon_on_after(y, 3, 15), 5
            conf = "medium" if p in ("ON", "BC", "AB", "QC") else "low"
            length = 14 if p == "BC" else 7
            rng(
                reg,
                s - D(2),
                s + D(length - 1),
                "March break",
                1.0,
                src,
                conf,
                f"Spring/March break rule: Monday on/after that province's anchor date ({length // 7} wk incl. weekends). Board-level variation +-1 wk (esp. AB/SK/MB/NS/NL/PE).",
            )
            # winter
            rng(
                reg,
                sat_on_after(y, 12, 20),
                sun_on_after(y + 1, 1, 3),
                "Winter break",
                1.0,
                src,
                "medium",
                "Winter break: Sat on/after Dec 20 to Sun on/after Jan 3 (school resumes following Monday).",
            )


# ================================================================== JP
def build_jp():
    src = "Rule-derived from typical MEXT/board calendars (municipal boards set dates; no national statute)"
    for y in range(2014, 2028):
        rng(
            "JP",
            date(y, 3, 25),
            date(y, 4, 5),
            "Spring break",
            0.98,
            src,
            "medium",
            "Typical spring break Mar 25-Apr 5 (school year starts ~Apr 6-8). Golden Week (Apr 29-May 5) is national holiday, not included here.",
        )
        rng(
            "JP",
            date(y, 7, 21),
            date(y, 8, 31),
            "Summer vacation",
            0.97,
            src,
            "medium",
            "Typical 6-week summer Jul 21-Aug 31 (Tokyo/Kanto/Kansai; boards vary by a few days). Obon Aug 13-16 falls inside.",
        )
        rng("JP", date(y, 12, 25), date(y + 1, 1, 7), "Winter vacation", 0.97, src, "medium", "Typical Dec 25-Jan 7.")
        # Hokkaido: short summer, long winter
        rng(
            "JP-01",
            date(y, 3, 25),
            date(y, 4, 6),
            "Spring break",
            0.98,
            src,
            "low",
            "Hokkaido: cold-region calendar, spring break ~Mar 25-Apr 6.",
        )
        rng(
            "JP-01",
            date(y, 7, 26),
            date(y, 8, 17),
            "Summer vacation",
            0.95,
            src,
            "low",
            "Hokkaido: short summer (~Jul 26-Aug 17) offset by long winter break.",
        )
        rng(
            "JP-01",
            date(y, 12, 25),
            date(y + 1, 1, 15),
            "Winter vacation",
            0.95,
            src,
            "low",
            "Hokkaido: long winter break (~Dec 25-Jan 15).",
        )


# ================================================================== KR
def build_kr():
    src = "Rule-derived from Korean MOE typical semester structure (schools set exact dates, +-1 week)"
    for y in range(2014, 2028):
        rng(
            "KR",
            date(y, 7, 22),
            date(y, 8, 17),
            "Summer vacation",
            0.92,
            src,
            "medium",
            "Summer vacation ~Jul 22-Aug 17 (2nd semester starts ~Aug 18-25).",
        )
        rng(
            "KR",
            date(y, 12, 26),
            date(y + 1, 2, 28),
            "Winter vacation and year-end break",
            0.9,
            src,
            "medium",
            "Winter vacation from ~Dec 26 (varies Dec 23-Jan 8) to end Feb (incl. Feb spring/year-end break; new school year Mar 2). Kindergarten/some HS differ.",
        )


# ================================================================== CN / HK
def build_cn_hk():
    src = "Rule-derived from typical education-bureau calendars (Beijing/Shanghai/Guangdong; anchor CNY date from python-holidays CN)"
    for y in range(2014, 2028):
        c = CNY[y]
        rng(
            "CN",
            c - D(13),
            c + D(18),
            "Winter break (Spring Festival)",
            0.93,
            src,
            "medium",
            "~CNY-13d to CNY+18d; provinces vary (north longer, south shorter). National Spring Festival public holiday is in the national file.",
        )
        rng(
            "CN",
            date(y, 7, 10),
            date(y, 8, 31),
            "Summer break",
            0.93,
            src,
            "medium",
            "Typical Jul 10-Aug 31; Beijing/Shanghai start ~Jul 1-10, Guangdong ~Jul 15; school starts Sep 1.",
        )
        # HK
        src2 = "Rule-derived from HK EDB school calendar pattern (CNY/Easter anchors from python-holidays)"
        e = easter(y)
        rng(
            "HK",
            date(y, 7, 16),
            date(y, 8, 31),
            "Summer holiday",
            0.98,
            src2,
            "medium",
            "HK schools: summer from ~Jul 16 (post-exam) to Aug 31; new year Sep 1.",
        )
        rng(
            "HK",
            date(y, 12, 24),
            date(y + 1, 1, 1),
            "Christmas holiday",
            0.98,
            src2,
            "medium",
            "~Dec 24-Jan 1 (EDB: Christmas + New Year, ~Dec 24/25 to Jan 1).",
        )
        rng(
            "HK",
            c - D(1),
            c + D(5),
            "Lunar New Year holiday",
            0.98,
            src2,
            "medium",
            "CNY-1 to CNY+5 (EDB LNY holidays).",
        )
        rng(
            "HK",
            e - D(6),
            e + D(2),
            "Easter holiday",
            0.98,
            src2,
            "medium",
            "Holy Week Mon to Tue after Easter (EDB Easter holidays; 1 week+).",
        )


# ================================================================== IN / SG / MY / SA / AE / ID / TH / AR / BR / MX
def build_rest():
    # ---- IN
    src = "Rule-derived: CBSE/state board typical calendars; states differ hugely (CBSE ~ May-Jun; South ~Apr-May; North winter break)"
    for y in range(2014, 2028):
        rng(
            "IN",
            date(y, 4, 15),
            date(y, 4, 30),
            "Summer vacation (early states)",
            0.5,
            src,
            "low",
            "Southern/eastern states start mid-April; national share ~50%.",
        )
        rng(
            "IN",
            date(y, 5, 1),
            date(y, 6, 15),
            "Summer vacation",
            0.9,
            src,
            "low",
            "Core Indian summer break May 1-Jun 15 (share 0.9).",
        )
        rng(
            "IN",
            date(y, 6, 16),
            date(y, 6, 30),
            "Summer vacation (late)",
            0.5,
            src,
            "low",
            "Northern states re-open ~Jul 1; south reopens early Jun.",
        )
        dv = DIWALI[y]
        rng(
            "IN",
            dv - D(2),
            dv + D(3),
            "Diwali break",
            0.75,
            src,
            "low",
            "Diwali date from python-holidays IN; typical 4-6 day school break, varies by state.",
        )
        rng(
            "IN",
            date(y, 12, 25),
            date(y + 1, 1, 1),
            "Winter break (north India)",
            0.4,
            src,
            "low",
            "North India winter/Christmas break ~Dec 25-Jan 1 (share of pupils nationally ~40%).",
        )
    # ---- SG
    src = "Rule-derived from MOE Singapore school terms (4 terms, one-week Mar/Sep breaks, ~4-wk June break, year-end Nov-Dec)"
    for y in range(2014, 2028):
        s = sat_on_after(y, 3, 14)
        rng("SG", s, s + D(8), "March school holidays", 1.0, src, "medium", "Rule: Sat on/after Mar 14 + 9d. +-1 wk.")
        s = sat_on_after(y, 5, 25)
        rng(
            "SG",
            s,
            s + D(23),
            "June school holidays",
            1.0,
            src,
            "medium",
            "Rule: Sat on/after May 25, ~3.5 weeks; exact length varies by year (3-4 wks).",
        )
        s = sat_on_after(y, 9, 1)
        rng("SG", s, s + D(8), "September school holidays", 1.0, src, "medium", "Rule: Sat on/after Sep 1 + 9d.")
        rng(
            "SG",
            sat_on_after(y, 11, 14),
            date(y, 12, 31),
            "Year-end school holidays",
            1.0,
            src,
            "medium",
            "Rule: Sat on/after Nov 14 to Dec 31.",
        )
    # ---- MY
    src = "Rule-derived from KPM Malaysia school calendar patterns (Group A Fri-Sat states vs Group B differ by ~1 wk; not split)"
    for y in range(2014, 2028):
        rng(
            "MY",
            date(y, 3, 22),
            date(y, 3, 30),
            "Term-1 break",
            0.9,
            src,
            "low",
            "Approx Mar 22-30; in some years replaced by Raya-linked break.",
        )
        e = EID_F[y]
        rng(
            "MY",
            e - D(3),
            e + D(4),
            "Hari Raya break",
            0.75,
            src,
            "low",
            "Eid al-Fitr date (python-holidays ID) +-; schools typically get ~1 week around Raya, share <1 as it may already be in term break.",
        )
        rng(
            "MY",
            date(y, 5, 24),
            date(y, 6, 8),
            "Mid-year school holidays",
            0.95,
            src,
            "low",
            "Approx 2 weeks late May-early Jun.",
        )
        rng("MY", date(y, 8, 30), date(y, 9, 7), "Term-3 break", 0.9, src, "low", "Approx one week, Aug 30-Sep 7.")
        rng(
            "MY",
            date(y, 12, 13),
            date(y + 1, 1, 1),
            "Year-end school holidays",
            0.9,
            src,
            "low",
            "Long year-end break; 2019-2025 sessions varied (mid-Nov to mid-Dec start, Jan 2-11 restart); approximated Dec 13-Jan 1.",
        )
    # ---- SA
    src = "Rule-derived from Saudi MOE 3-term calendar pattern; Eid dates from python-holidays ID (Hijri), +-1 day vs KSA sighting"
    for y in range(2014, 2028):
        rng(
            "SA",
            date(y, 6, 15),
            date(y, 8, 24),
            "Summer break",
            0.95,
            src,
            "low",
            "Approx Jun 15-Aug 24 (older 2 semester calendars: from ~Jun-Jul); differs by year.",
        )
        e = EID_F[y]
        rng(
            "SA",
            e - D(8),
            e + D(10),
            "Eid al-Fitr break",
            0.9,
            src,
            "low",
            "~2.5 weeks around Eid al-Fitr (Ramadan end-of-term to post-Eid).",
        )
        a = EID_A[y]
        rng("SA", a - D(4), a + D(6), "Eid al-Adha break", 0.9, src, "low", "~10 days around Eid al-Adha.")
        rng(
            "SA",
            date(y, 12, 12),
            date(y, 12, 24),
            "Winter/term break",
            0.4,
            src,
            "low",
            "Low-confidence mid-winter term break (2023+ 3-term system); date/length varies by year.",
        )
    # ---- AE
    src = "Rule-derived from UAE MOE / KHDA calendars; Eid anchors from python-holidays ID; exact dates set yearly"
    for y in range(2014, 2028):
        rng(
            "AE",
            date(y, 7, 5),
            date(y, 8, 24),
            "Summer break",
            0.97,
            src,
            "medium",
            "Approx Jul 5-Aug 24 (many private schools Jul 1-Aug 25).",
        )
        rng(
            "AE",
            date(y, 12, 15),
            date(y + 1, 1, 2),
            "Winter break",
            0.97,
            src,
            "low",
            "Approx mid-Dec to Jan 2 (~2 weeks).",
        )
        e = EID_F[y]
        rng(
            "AE",
            e - D(4),
            e + D(4),
            "Spring break / Eid al-Fitr",
            0.85,
            src,
            "low",
            "Spring break is ~2 weeks in Mar/Apr, often adjacent to Eid al-Fitr; approximated Eid +-4 days.",
        )
    # ---- ID
    src = "Rule-derived from Kemendikbud national school calendar pattern; Eid dates from python-holidays ID"
    for y in range(2014, 2028):
        rng(
            "ID",
            date(y, 6, 22),
            date(y, 7, 13),
            "Mid-year school holidays",
            0.85,
            src,
            "low",
            "End of semester-2 break ~3 weeks late Jun-mid Jul (start varies by year with Ramadan).",
        )
        e = EID_F[y]
        rng("ID", e - D(7), e + D(7), "Lebaran break", 0.8, src, "low", "~1-2 weeks around Idul Fitri.")
        rng(
            "ID",
            date(y, 12, 20),
            date(y + 1, 1, 5),
            "Year-end school holidays",
            0.9,
            src,
            "low",
            "Semester-1 break ~Dec 20-Jan 5.",
        )
    # ---- TH
    src = "Rule-derived from Thai OBEC school calendar (2 terms: mid-May-Sep, Nov-Feb; long summer break Mar-mid May)"
    for y in range(2014, 2028):
        rng(
            "TH",
            date(y, 3, 1),
            date(y, 5, 14),
            "Summer break",
            0.9,
            src,
            "low",
            "Long summer break ~Mar 1-May 14 (varies by year and level).",
        )
        rng(
            "TH",
            date(y, 10, 1),
            date(y, 10, 31),
            "Mid-year break",
            0.8,
            src,
            "low",
            "~1 month break between terms in Oct (shortened in some years).",
        )
    # ---- AR
    src = "Rule-derived from Argentine provincial calendars (winter break staggered by province in 3 groups; summer Dec 20-Feb 28)"
    for y in range(2014, 2028):
        rng(
            "AR",
            date(y, 12, 20),
            date(y + 1, 2, 28),
            "Summer break",
            0.95,
            src,
            "low",
            "Classes end ~Dec 10-22, restart Feb 24-Mar 3.",
        )
        for anc, sh in [((7, 8), 0.3), ((7, 15), 0.35), ((7, 22), 0.35)]:
            m0 = mon_on_after(y, *anc)
            rng(
                "AR",
                m0 - D(2),
                m0 + D(11),
                "Winter break",
                sh,
                src,
                "low",
                "Two-week July break staggered across provinces in 3 groups (share of pupils per group); e.g. Buenos Aires, Cordoba, Mendoza differ year to year.",
            )
    # ---- BR
    src = "Rule-derived: typical Brazilian state (SP/RJ/MG) calendars; public/private vary; Carnival Easter-anchored"
    for y in range(2014, 2028):
        e = easter(y)
        rng("BR", e - D(51), e - D(46), "Carnival break", 0.9, src, "medium", "Sat-Ash Wednesday (Easter-51 to -46).")
        rng(
            "BR",
            date(y, 12, 20),
            date(y + 1, 2, 2),
            "Summer break",
            0.9,
            src,
            "low",
            "Classes end ~Dec 12-20, restart ~Feb 1-10; states vary.",
        )
        rng(
            "BR",
            date(y, 7, 7),
            date(y, 7, 20),
            "Winter break",
            0.65,
            src,
            "low",
            "Mid-year 2-week break; 3rd-4th Jul weeks in SP, varies by state (some Jul 1-31).",
        )
        rng(
            "BR",
            date(y, 7, 1),
            date(y, 7, 6),
            "Winter break (shoulder)",
            0.3,
            src,
            "low",
            "Shoulder of winter break in some states.",
        )
        rng(
            "BR",
            date(y, 7, 21),
            date(y, 7, 31),
            "Winter break (shoulder late)",
            0.3,
            src,
            "low",
            "Late-July winter break in some states.",
        )
    # ---- MX
    src = "Rule-derived from SEP (Mexico) national school calendar patterns; Easter from dateutil"
    for y in range(2014, 2028):
        e = easter(y)
        rng(
            "MX",
            date(y, 7, 16),
            date(y, 8, 24),
            "Summer break",
            0.98,
            src,
            "medium",
            "SEP school year ends ~Jul 15 (2024/25 Jul 15), restarts ~Aug 25-Sep 1; private schools differ ~1 wk.",
        )
        rng("MX", date(y, 12, 21), date(y + 1, 1, 6), "Winter break", 0.98, src, "medium", "~Dec 21-Jan 6.")
        rng(
            "MX",
            e - D(7),
            e + D(7),
            "Semana Santa / Pascua",
            0.98,
            src,
            "medium",
            "Two-week Holy Week + Easter week break (Sat E-7 to Sun E+7).",
        )


build_us()
build_au_nz()
build_ca()
build_jp()
build_kr()
build_cn_hk()
build_rest()

os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["region_code", "date", "type", "name", "share", "source", "confidence", "notes"])
    for k in sorted(rows, key=lambda k: (k[0], k[1], k[3])):
        r = rows[k]
        w.writerow(
            [k[0], k[1].isoformat(), "school", k[3], f"{r['share']:.2f}", r["source"], r["confidence"], r["notes"]]
        )
print(len(rows), "rows")
