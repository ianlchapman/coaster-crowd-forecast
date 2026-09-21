"""AU/NZ official term dates -> school-holiday blocks (same names as the rule-based blocks in 30_build_school_non_eu.py)."""

from datetime import date, timedelta
import csv, sys

MON = {m: i + 1 for i, m in enumerate("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split())}


def D(y, s):
    d, m = s.split()
    return date(y, MON[m], int(d))


def terms(y, *pairs):  # pairs like ('27 Jan','2 Apr')
    return [(D(y, a), D(y, b)) for a, b in pairs]


# student-day terms per state: {year: [(start,end)*4]}; None entries omitted
T = {}


def add(st, y, *p, conf, src, note=""):
    T.setdefault(st, {})[y] = dict(t=terms(y, *p), conf=conf, src=src, note=note)


VIC = "https://www.vic.gov.au/school-term-dates-and-holidays-victoria (Vic Dept of Education, official, incl. previous years)"
vic = {
    2014: ["28 Jan|4 Apr", "22 Apr|27 Jun", "14 Jul|19 Sep", "6 Oct|19 Dec"],
    2015: ["28 Jan|27 Mar", "13 Apr|26 Jun", "13 Jul|18 Sep", "5 Oct|18 Dec"],
    2016: ["27 Jan|24 Mar", "11 Apr|24 Jun", "11 Jul|16 Sep", "3 Oct|20 Dec"],
    2017: ["30 Jan|31 Mar", "18 Apr|30 Jun", "17 Jul|22 Sep", "9 Oct|22 Dec"],
    2018: ["30 Jan|29 Mar", "16 Apr|29 Jun", "16 Jul|21 Sep", "8 Oct|21 Dec"],
    2019: ["30 Jan|5 Apr", "23 Apr|28 Jun", "15 Jul|20 Sep", "7 Oct|20 Dec"],
    2020: ["29 Jan|24 Mar", "15 Apr|26 Jun", "13 Jul|18 Sep", "5 Oct|18 Dec"],
    2021: ["28 Jan|1 Apr", "19 Apr|25 Jun", "12 Jul|17 Sep", "4 Oct|17 Dec"],
    2022: ["31 Jan|8 Apr", "26 Apr|24 Jun", "11 Jul|16 Sep", "3 Oct|20 Dec"],
    2023: ["30 Jan|6 Apr", "24 Apr|23 Jun", "10 Jul|15 Sep", "2 Oct|20 Dec"],
    2024: ["30 Jan|28 Mar", "15 Apr|28 Jun", "15 Jul|20 Sep", "7 Oct|20 Dec"],
    2025: ["29 Jan|4 Apr", "22 Apr|4 Jul", "21 Jul|19 Sep", "6 Oct|19 Dec"],
    2026: ["28 Jan|2 Apr", "20 Apr|26 Jun", "13 Jul|18 Sep", "5 Oct|18 Dec"],
    2027: ["28 Jan|25 Mar", "12 Apr|25 Jun", "12 Jul|17 Sep", "4 Oct|17 Dec"],
}
for y, v in vic.items():
    add(
        "AU-VIC",
        y,
        *[tuple(x.split("|")) for x in v],
        conf="high",
        src=VIC,
        note="Vic official term dates; T1 start = students (govt schools)"
        if y >= 2018
        else "Vic official term dates; 2014-17 T1 date is teacher start, students start 0-2 days later; 2020 T1 ended early (COVID) and T2 partly remote (not modelled)",
    )
NSW = "https://education.nsw.gov.au/schooling/calendars (+ /future-and-past-nsw-term-and-vacation-dates), NSW Dept of Education official"
nsw = {
    2016: ["27 Jan|8 Apr", "26 Apr|1 Jul", "18 Jul|23 Sep", "10 Oct|20 Dec"],
    2017: ["27 Jan|7 Apr", "24 Apr|30 Jun", "17 Jul|22 Sep", "9 Oct|19 Dec"],
    2018: ["29 Jan|13 Apr", "30 Apr|6 Jul", "23 Jul|28 Sep", "15 Oct|21 Dec"],
    2019: ["29 Jan|12 Apr", "29 Apr|5 Jul", "22 Jul|27 Sep", "14 Oct|20 Dec"],
    2020: ["28 Jan|9 Apr", "27 Apr|3 Jul", "20 Jul|25 Sep", "12 Oct|18 Dec"],
    2021: ["27 Jan|1 Apr", "19 Apr|25 Jun", "12 Jul|17 Sep", "5 Oct|17 Dec"],
    2022: ["28 Jan|8 Apr", "26 Apr|1 Jul", "18 Jul|23 Sep", "10 Oct|20 Dec"],
    2023: ["31 Jan|6 Apr", "26 Apr|30 Jun", "18 Jul|22 Sep", "9 Oct|15 Dec"],
    2024: ["1 Feb|12 Apr", "30 Apr|5 Jul", "23 Jul|27 Sep", "14 Oct|18 Dec"],
    2025: ["6 Feb|11 Apr", "30 Apr|4 Jul", "22 Jul|26 Sep", "14 Oct|19 Dec"],
    2026: ["2 Feb|2 Apr", "22 Apr|3 Jul", "21 Jul|25 Sep", "13 Oct|17 Dec"],
    2027: ["3 Feb|9 Apr", "29 Apr|2 Jul", "20 Jul|24 Sep", "12 Oct|20 Dec"],
}
for y, v in nsw.items():
    hi = y >= 2023
    add(
        "AU-NSW",
        y,
        *[tuple(x.split("|")) for x in v],
        conf="high" if hi else "medium",
        src=NSW,
        note="NSW Eastern division; student-day boundaries incl. school development days"
        if hi
        else "NSW Eastern division; listed term starts are teacher/first days; students may return 0-3 days later (development days not itemised pre-2023)",
    )
QLD = "https://education.qld.gov.au/about-us/calendar/term-dates and /future-dates (Qld Dept of Education; past years via web.archive.org snapshots)"
qld = {
    2019: ["29 Jan|5 Apr", "23 Apr|28 Jun", "15 Jul|20 Sep", "8 Oct|13 Dec"],
    2021: ["27 Jan|1 Apr", "19 Apr|25 Jun", "12 Jul|17 Sep", "5 Oct|10 Dec"],
    2022: ["24 Jan|1 Apr", "19 Apr|24 Jun", "11 Jul|16 Sep", "4 Oct|9 Dec"],
    2023: ["23 Jan|31 Mar", "17 Apr|23 Jun", "10 Jul|15 Sep", "3 Oct|8 Dec"],
    2024: ["22 Jan|28 Mar", "15 Apr|21 Jun", "8 Jul|13 Sep", "30 Sep|13 Dec"],
    2025: ["28 Jan|4 Apr", "22 Apr|27 Jun", "14 Jul|19 Sep", "7 Oct|12 Dec"],
    2026: ["27 Jan|2 Apr", "20 Apr|26 Jun", "13 Jul|18 Sep", "6 Oct|11 Dec"],
    2027: ["27 Jan|25 Mar", "12 Apr|25 Jun", "12 Jul|17 Sep", "5 Oct|10 Dec"],
}
for y, v in qld.items():
    add(
        "AU-QLD",
        y,
        *[tuple(x.split("|")) for x in v],
        conf="high",
        src=QLD,
        note="Qld state school terms; 2022 T1 formal learning began 7 Feb (COVID) - not modelled",
    )
SA = "https://www.education.sa.gov.au/parents-and-families/term-dates-south-australian-state-schools (SA DfE; web.archive.org 2022 snapshot)"
sa = {
    2014: ["28 Jan|11 Apr", "28 Apr|4 Jul", "21 Jul|26 Sep", "13 Oct|12 Dec"],
    2015: ["27 Jan|10 Apr", "27 Apr|3 Jul", "20 Jul|25 Sep", "12 Oct|11 Dec"],
    2016: ["1 Feb|15 Apr", "2 May|8 Jul", "25 Jul|30 Sep", "17 Oct|16 Dec"],
    2017: ["30 Jan|13 Apr", "1 May|7 Jul", "24 Jul|29 Sep", "16 Oct|15 Dec"],
    2018: ["29 Jan|13 Apr", "30 Apr|6 Jul", "23 Jul|28 Sep", "15 Oct|14 Dec"],
    2019: ["29 Jan|12 Apr", "29 Apr|5 Jul", "22 Jul|27 Sep", "14 Oct|13 Dec"],
    2020: ["28 Jan|9 Apr", "27 Apr|3 Jul", "20 Jul|25 Sep", "12 Oct|11 Dec"],
    2021: ["27 Jan|9 Apr", "27 Apr|2 Jul", "19 Jul|24 Sep", "11 Oct|10 Dec"],
    2022: ["31 Jan|14 Apr", "2 May|8 Jul", "25 Jul|30 Sep", "17 Oct|16 Dec"],
    2023: ["30 Jan|14 Apr", "1 May|7 Jul", "24 Jul|29 Sep", "16 Oct|15 Dec"],
    2024: ["29 Jan|12 Apr", "29 Apr|5 Jul", "22 Jul|27 Sep", "14 Oct|13 Dec"],
    2025: ["28 Jan|11 Apr", "28 Apr|4 Jul", "21 Jul|26 Sep", "13 Oct|12 Dec"],
}
for y, v in sa.items():
    add("AU-SA", y, *[tuple(x.split("|")) for x in v], conf="high", src=SA, note="SA public school terms")
WA = "https://www.education.wa.edu.au/future-term-dates (WA Dept of Education, Gazette dates; past years via web.archive.org)"
wa = {
    2019: ["4 Feb|12 Apr", "29 Apr|5 Jul", "22 Jul|27 Sep", "14 Oct|19 Dec"],
    2020: ["3 Feb|9 Apr", "28 Apr|3 Jul", "20 Jul|25 Sep", "12 Oct|17 Dec"],
    2021: ["1 Feb|1 Apr", "19 Apr|2 Jul", "19 Jul|24 Sep", "11 Oct|16 Dec"],
    2022: ["31 Jan|8 Apr", "26 Apr|1 Jul", "18 Jul|23 Sep", "10 Oct|15 Dec"],
    2023: ["1 Feb|6 Apr", "24 Apr|30 Jun", "17 Jul|22 Sep", "9 Oct|14 Dec"],
    2024: ["31 Jan|28 Mar", "15 Apr|28 Jun", "15 Jul|20 Sep", "7 Oct|12 Dec"],
    2025: ["5 Feb|11 Apr", "28 Apr|4 Jul", "21 Jul|26 Sep", "13 Oct|18 Dec"],
    2026: ["2 Feb|2 Apr", "20 Apr|3 Jul", "20 Jul|25 Sep", "12 Oct|17 Dec"],
    2027: ["1 Feb|9 Apr", "26 Apr|2 Jul", "19 Jul|24 Sep", "11 Oct|16 Dec"],
}
for y, v in wa.items():
    add("AU-WA", y, *[tuple(x.split("|")) for x in v], conf="high", src=WA, note="WA public school terms")
TAS = "https://www.education.tas.gov.au/about-us/term-dates/ (Tas DECYP; past years via web.archive.org)"
tas = {
    2018: ["7 Feb|12 Apr", "30 Apr|6 Jul", "23 Jul|28 Sep", "15 Oct|20 Dec"],
    2020: ["5 Feb|9 Apr", "27 Apr|3 Jul", "20 Jul|25 Sep", "12 Oct|17 Dec"],
    2021: ["3 Feb|31 Mar", "21 Apr|2 Jul", "20 Jul|24 Sep", "11 Oct|16 Dec"],
    2022: ["9 Feb|13 Apr", "2 May|8 Jul", "26 Jul|30 Sep", "17 Oct|21 Dec"],
    2023: ["8 Feb|5 Apr", "26 Apr|7 Jul", "25 Jul|29 Sep", "16 Oct|21 Dec"],
    2025: ["6 Feb|11 Apr", "28 Apr|4 Jul", "21 Jul|26 Sep", "13 Oct|18 Dec"],
    2026: ["5 Feb|17 Apr", "4 May|10 Jul", "27 Jul|2 Oct", "19 Oct|18 Dec"],
    2027: ["4 Feb|9 Apr", "26 Apr|2 Jul", "19 Jul|24 Sep", "11 Oct|16 Dec"],
}
for y, v in tas.items():
    add(
        "AU-TAS",
        y,
        *[tuple(x.split("|")) for x in v],
        conf="high",
        src=TAS,
        note="Tas school (not college) student terms",
    )
add(
    "AU-ACT",
    2027,
    ("2 Feb", "9 Apr"),
    ("28 Apr", "2 Jul"),
    ("20 Jul", "24 Sep"),
    ("12 Oct", "17 Dec"),
    conf="medium",
    src="ACT Education Directorate 2026-30 term dates (act.gov.au PDF) as reported in web search summary; site blocked direct fetch",
    note="ACT continuing-student start 2 Feb",
)
add(
    "AU-NT",
    2026,
    ("28 Jan", "3 Apr"),
    ("14 Apr", "19 Jun"),
    ("14 Jul", "18 Sep"),
    ("6 Oct", "11 Dec"),
    conf="medium",
    src="NT Dept of Education term dates via nt.gov.au (blocked); values from web search summaries, end dates rounded to preceding Friday",
    note="NT public schools",
)
add(
    "AU-NT",
    2027,
    ("27 Jan", "2 Apr"),
    ("13 Apr", "18 Jun"),
    ("13 Jul", "17 Sep"),
    ("5 Oct", "10 Dec"),
    conf="medium",
    src="NT Dept of Education term dates via nt.gov.au (blocked); values from web search summaries",
    note="NT public schools; T1 start given as 28 Jan in source, T2 end 19 Jun (Sat)",
)

NAMES = ["Autumn/Easter school holidays", "Winter school holidays", "Spring school holidays"]


def rng(a, b):
    d = a
    while d <= b:
        yield d
        d += timedelta(1)


def au_blocks(st, y, rec):
    t = rec["t"]
    out = []
    for k in range(3):
        a = t[k][1] + timedelta(1)
        b = t[k + 1][0] - timedelta(1)
        out.append((NAMES[k], list(rng(a, b))))
    days = list(rng(date(y, 1, 1), t[0][0] - timedelta(1))) + list(rng(t[3][1] + timedelta(1), date(y, 12, 31)))
    out.append(("Summer school holidays", days))
    return out


NZSRC = "https://www.education.govt.nz/school/school-terms-and-holiday-dates/ (NZ MoE; 2020-23 via web.archive.org) and NZ Gazette 2024-sl4001 / 2025-sl5849 (2025-2027)"
nz = {
    2020: ("27 Jan", "7 Feb", "27 Mar", "15 Apr", "3 Jul", "20 Jul", "25 Sep", "12 Oct", "18 Dec"),
    2021: ("1 Feb", "9 Feb", "16 Apr", "3 May", "9 Jul", "26 Jul", "1 Oct", "18 Oct", "20 Dec"),
    2022: ("31 Jan", "8 Feb", "14 Apr", "2 May", "8 Jul", "25 Jul", "30 Sep", "17 Oct", "20 Dec"),
    2023: ("30 Jan", "7 Feb", "6 Apr", "24 Apr", "30 Jun", "17 Jul", "22 Sep", "9 Oct", "20 Dec"),
    2024: ("29 Jan", "7 Feb", "12 Apr", "29 Apr", "5 Jul", "22 Jul", "27 Sep", "14 Oct", "20 Dec"),
    2025: ("27 Jan", "7 Feb", "11 Apr", "28 Apr", "27 Jun", "14 Jul", "19 Sep", "6 Oct", "19 Dec"),
    2026: ("26 Jan", "9 Feb", "2 Apr", "20 Apr", "3 Jul", "20 Jul", "25 Sep", "12 Oct", "18 Dec"),
    2027: ("28 Jan", "3 Feb", "9 Apr", "27 Apr", "2 Jul", "19 Jul", "24 Sep", "11 Oct", "17 Dec"),
}


def nz_blocks(y, v):
    e, l, t1e, t2s, t2e, t3s, t3e, t4s, t4e = [D(y, x) for x in v]
    out = [
        ("Autumn/Easter school holidays", list(rng(t1e + timedelta(1), t2s - timedelta(1)))),
        ("Winter school holidays", list(rng(t2e + timedelta(1), t3s - timedelta(1)))),
        ("Spring school holidays", list(rng(t3e + timedelta(1), t4s - timedelta(1)))),
    ]
    summer = []
    # January: full break until earliest start; ramp over weekdays to latest start
    for d in rng(date(y, 1, 1), l - timedelta(1)):
        if d < e:
            summer.append((d, 1.0))
        else:
            wd = [x for x in rng(e, l) if x.weekday() < 5]
            if d.weekday() >= 5:
                summer.append((d, 1.0))
                continue
            k = wd.index(d) if d in wd else 0
            summer.append((d, max(0.05, round(1 - (k + 0.5) / len(wd), 2))))
    # December: schools close no later than t4e; earlier weekdays partial
    for d in rng(date(y, 12, 1), date(y, 12, 31)):
        if d > t4e:
            summer.append((d, 1.0))
        elif d > t4e - timedelta(4) and d.weekday() < 5:
            summer.append((d, 0.5))
    out.append(("Summer school holidays", summer))
    return out


def main(outrows, repl):
    for st, ys in T.items():
        for y, rec in sorted(ys.items()):
            for name, days in au_blocks(st, y, rec):
                if not days:
                    continue
                for d in days:
                    outrows.append((st, d.isoformat(), "school", name, 1.0, rec["src"], rec["conf"], rec["note"]))
                repl.append((st, y, name, name, "Replaced by official term-date derived block (" + rec["conf"] + ")"))
    for y, v in nz.items():
        conf = "high" if y >= 2023 else "high"
        note = "NZ state schools. Terms 2-3 fixed by regulation; T1 start (window) and T4 end are school-set: Jan shows share ramp across the official start window; Dec share 0.5 on the 3 weekdays before the latest permitted closing date"
        for name, days in nz_blocks(y, v):
            for item in days:
                d, s = item if isinstance(item, tuple) else (item, 1.0)
                outrows.append(
                    (
                        "NZ",
                        d.isoformat(),
                        "school",
                        name,
                        s,
                        NZSRC,
                        "high" if not name.startswith("Summer") else "medium",
                        note if name.startswith("Summer") else "NZ MoE fixed term dates",
                    )
                )
            repl.append(("NZ", y, name, name, "Replaced by MoE term dates"))
