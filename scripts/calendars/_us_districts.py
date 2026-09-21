"""US data-driven school-break blocks from published large-district calendars (2024-25 / 2025-26 / some 2026-27),
weighted per state, extended to neighbouring years by weekday-aligned shifts. Sample sizes are small -> 'medium' at best."""

from datetime import date, timedelta


def d(s):
    y, m, dd = s.split("-")
    return date(int(y), int(m), int(dd))


# name: (state, weight(enrollment k, approx), lasts{yr:date}, firsts, springs{yr:(s,e)}, tgs, winters{yr_of_dec:(s,e)}, source)
DIST = {
    "Broward CPS": (
        "FL",
        240,
        {2026: "2026-06-03"},
        {2025: "2025-08-11"},
        {2026: ("2026-03-16", "2026-03-20")},
        {2025: ("2025-11-24", "2025-11-28")},
        {2025: ("2025-12-22", "2026-01-02")},
        "browardschools.com 2025-26 calendar (via nbcmiami/floridaschools summaries)",
    ),
    "Hillsborough CPS": (
        "FL",
        220,
        {2026: "2026-05-29"},
        {2025: "2025-08-11"},
        {2026: ("2026-03-16", "2026-03-20")},
        {2025: ("2025-11-24", "2025-11-28")},
        {2025: ("2025-12-22", "2026-01-02")},
        "hillsboroughschools.org 25-26 Student Academic Calendar",
    ),
    "Miami-Dade CPS": (
        "FL",
        330,
        {2025: "2025-06-05", 2026: "2026-06-04"},
        {2024: "2024-08-15"},
        {2025: ("2025-03-24", "2025-03-28"), 2026: ("2026-03-20", "2026-03-27")},
        {2024: ("2024-11-25", "2024-11-29")},
        {2024: ("2024-12-23", "2025-01-03")},
        "dadeschools.net 2024-25 / 2025-26 calendars",
    ),
    "Orange CPS": ("FL", 200, {}, {2026: "2026-08-11"}, {}, {}, {}, "ocps.net 2026-27 calendar"),
    "Houston ISD": (
        "TX",
        180,
        {2025: "2025-06-04", 2026: "2026-06-04"},
        {2024: "2024-08-12", 2025: "2025-08-12"},
        {2025: ("2025-03-10", "2025-03-14"), 2026: ("2026-03-09", "2026-03-13")},
        {2024: ("2024-11-25", "2024-11-29")},
        {2024: ("2024-12-23", "2025-01-03"), 2025: ("2025-12-22", "2026-01-06")},
        "houstonisd.org calendars 2024-25, 2025-26",
    ),
    "Dallas ISD": (
        "TX",
        140,
        {2025: "2025-05-23", 2026: "2026-05-22"},
        {2024: "2024-08-12", 2025: "2025-08-12"},
        {2026: ("2026-03-16", "2026-03-20")},
        {},
        {2024: ("2024-12-23", "2025-01-03"), 2025: ("2025-12-22", "2026-01-02")},
        "dallasisd.org calendars 2024-25, 2025-26",
    ),
    "LAUSD": (
        "CA",
        400,
        {2026: "2026-06-10"},
        {2025: "2025-08-14"},
        {2026: ("2026-03-30", "2026-04-03")},
        {2025: ("2025-11-24", "2025-11-28")},
        {2025: ("2025-12-22", "2026-01-09")},
        "lausd.org 2025-26 instructional calendar",
    ),
    "San Diego USD": (
        "CA",
        95,
        {2026: "2026-05-27"},
        {2025: "2025-08-11"},
        {2026: ("2026-03-09", "2026-03-13")},
        {2025: ("2025-11-24", "2025-11-28")},
        {2025: ("2025-12-22", "2026-01-02")},
        "sandiegounified.org 2025-26 academic calendar",
    ),
    "Cobb County SD": (
        "GA",
        110,
        {2026: "2026-05-20"},
        {2025: "2025-08-04"},
        {2026: ("2026-04-06", "2026-04-10")},
        {2025: ("2025-11-24", "2025-11-28")},
        {2025: ("2025-12-22", "2026-01-05")},
        "cobbk12.org 2025-26 calendar",
    ),
    "Gwinnett CPS": (
        "GA",
        180,
        {2026: "2026-05-20"},
        {2024: "2024-08-05", 2025: "2025-08-04"},
        {},
        {2024: ("2024-11-25", "2024-11-29")},
        {},
        "gcpsk12.net 2024-25 / 2025-26 calendars",
    ),
    "Atlanta PS": (
        "GA",
        50,
        {2025: "2025-05-29"},
        {2024: "2024-08-01"},
        {2025: ("2025-04-07", "2025-04-11")},
        {},
        {2024: ("2024-12-23", "2025-01-06")},
        "atlantapublicschools.us 2024-25 calendar",
    ),
    "Wake County": (
        "NC",
        160,
        {2026: "2026-06-11"},
        {2025: "2025-08-25"},
        {2026: ("2026-03-30", "2026-04-06")},
        {2025: ("2025-11-26", "2025-11-28")},
        {2025: ("2025-12-22", "2026-01-02")},
        "wcpss.net 2025-26 calendar",
    ),
    "Charlotte-Mecklenburg": (
        "NC",
        140,
        {2026: "2026-06-10"},
        {2025: "2025-08-25"},
        {2026: ("2026-04-03", "2026-04-10")},
        {2025: ("2025-11-26", "2025-11-28")},
        {2025: ("2025-12-22", "2026-01-02")},
        "cms.k12.nc.us 2025-26 calendar",
    ),
    "Fairfax County": (
        "VA",
        180,
        {2026: "2026-06-17"},
        {2025: "2025-08-18"},
        {2026: ("2026-03-30", "2026-04-03")},
        {},
        {2024: ("2024-12-23", "2025-01-05"), 2025: ("2025-12-22", "2026-01-02")},
        "fcps.edu 2024-25 / 2025-26 standard calendars",
    ),
    "Virginia Beach": (
        "VA",
        65,
        {2026: "2026-06-12"},
        {2025: "2025-08-25"},
        {2026: ("2026-04-06", "2026-04-10")},
        {},
        {2025: ("2025-12-22", "2026-01-02")},
        "vbschools.com 2025-26 calendar",
    ),
}
# single-district samples (state NOT replaced; used only to assess B): NYC, CPS, Philly, Columbus, Detroit
SINGLE = {
    "NY": "NYC DOE: first 2025-09-04, last 2026-06-26 (2025: 06-26); spring 2025 04-14..18; winter 2024-25 12-23..01-01",
    "IL": "Chicago PS: first 2024-08-26 / 2025-08-18; last 2025-06-12 / 2026-06-04; spring 2026 03-23..27; winter 12-22..01-02",
    "PA": "Philadelphia SD 2025-26: first 08-25, last 2026-06-12; spring 03-30..04-05; winter 12-24..01-05",
    "OH": "Columbus CS 2025-26: first 07-28, last 06-03; spring 04-03..04-10; winter 12-22..01-02",
    "MI": "Detroit PSCD 2025-26: first 08-25, last 06-05; spring 03-30..04-02; winter 12-22..01-02",
}


def christmas_anchor(y):
    c = date(y, 12, 25)
    return c - timedelta(c.weekday())


def thanksgiving(y):
    d0 = date(y, 11, 1)
    first_thu = d0 + timedelta((3 - d0.weekday()) % 7)
    return first_thu + timedelta(21)


def nearest_md(base, y):
    tgt = date(y, base.month, base.day)
    k = round((tgt - base).days / 7)
    return base + timedelta(7 * k)


def ext(s, e):
    if s.weekday() == 0:
        s -= timedelta(2)
    if e.weekday() == 4:
        e += timedelta(2)
    return s, e


def rng(a, b):
    x = a
    while x <= b:
        yield x
        x += timedelta(1)


def get_point(dct, Y):
    if Y in dct:
        return d(dct[Y]), "direct"
    if not dct:
        return None, None
    b = min(dct, key=lambda k: abs(k - Y))
    return nearest_md(d(dct[b]), Y), "derived"


def get_tg(dct, Y):
    if Y in dct:
        s, e = map(d, dct[Y])
        return ext(s, e), "direct"
    if not dct:
        return None, None
    b = min(dct, key=lambda k: abs(k - Y))
    s, e = map(d, dct[b])
    T0 = thanksgiving(b)
    T1 = thanksgiving(Y)
    return ext(T1 + (s - T0), T1 + (e - T0)), "derived"


def get_win(dct, Ydec):
    if Ydec in dct:
        s, e = map(d, dct[Ydec])
        return ext(s, e), "direct"
    if not dct:
        return None, None
    b = min(dct, key=lambda k: abs(k - Ydec))
    s, e = map(d, dct[b])
    A0 = christmas_anchor(b)
    A1 = christmas_anchor(Ydec)
    return ext(A1 + (s - A0), A1 + (e - A0)), "derived"


def wshare(items, test):
    tot = sum(w for w, _ in items)
    return sum(w for w, x in items if test(x)) / tot if tot else None


def build(rows, repl):
    states = {}
    for n, v in DIST.items():
        states.setdefault(v[0], []).append(n)
    for st, names in states.items():
        if len(names) < 2:
            continue
        rc = "US-" + st
        for Y in (2024, 2025, 2026):
            # Thanksgiving
            items = []
            kinds = set()
            srcs = []
            for n in names:
                v = DIST[n]
                r, k = get_tg(v[5], Y)
                if r:
                    items.append((v[1], r))
                    kinds.add(k)
                    srcs.append(n)
            if len(items) >= 2:
                for dd in rng(date(Y, 11, 15), date(Y, 12, 3)):
                    s = wshare(items, lambda r: r[0] <= dd <= r[1])
                    if s:
                        rows.append(
                            (
                                rc,
                                dd.isoformat(),
                                "school",
                                "Thanksgiving break",
                                round(s, 2),
                                "Large-district calendars: " + "; ".join(DIST[n][7] for n in srcs),
                                "medium",
                                f"Weighted (enrollment) share of sampled districts on Thanksgiving break; districts: {', '.join(srcs)}; "
                                + (
                                    "direct from published year"
                                    if kinds == {"direct"}
                                    else "some districts weekday-aligned from adjacent published year (Thanksgiving-anchored)"
                                ),
                            )
                        )
                repl.append(
                    (
                        rc,
                        Y,
                        "Thanksgiving break",
                        "Thanksgiving break",
                        "Data-driven from published district calendars (B: Wed-Sun full share)",
                    )
                )
            # Spring
            items = [(DIST[n][1], DIST[n][4][Y]) for n in names if Y in DIST[n][4]]
            items = [(w, ext(d(a), d(b))) for w, (a, b) in items]
            if len(items) >= 2:
                lo = min(r[0] for _, r in items)
                hi = max(r[1] for _, r in items)
                for dd in rng(lo, hi):
                    s = wshare(items, lambda r: r[0] <= dd <= r[1])
                    if s:
                        rows.append(
                            (
                                rc,
                                dd.isoformat(),
                                "school",
                                "Spring break",
                                round(s, 2),
                                "Large-district calendars: " + "; ".join(DIST[n][7] for n in names if Y in DIST[n][4]),
                                "medium",
                                f"Weighted share of {len(items)} sampled districts (published dates); other districts (tail) not modelled so shares sum below 1",
                            )
                        )
                repl.append(
                    (rc, Y, "Spring break", "Spring break", "Published district spring breaks (B: distribution guess)")
                )
            # Summer
            st_items = []
            en_items = []
            kinds = set()
            for n in names:
                v = DIST[n]
                p, k = get_point(v[2], Y)
                if p:
                    st_items.append((v[1], p))
                    kinds.add(k)
                p, k = get_point(v[3], Y)
                if p:
                    en_items.append((v[1], p))
                    kinds.add(k)
            if len(st_items) >= 2 and len(en_items) >= 2:
                for dd in rng(date(Y, 5, 10), date(Y, 9, 12)):
                    if dd <= date(Y, 7, 15):
                        s = wshare(st_items, lambda p: dd > p)
                    else:
                        s = wshare(en_items, lambda p: dd < p)
                    if s:
                        rows.append(
                            (
                                rc,
                                dd.isoformat(),
                                "school",
                                "Summer break",
                                round(s, 2),
                                "Large-district calendars (last/first day of school)",
                                "medium",
                                f"Weighted share of sampled districts out of session (last-day sample n={len(st_items)}, first-day n={len(en_items)}); "
                                + (
                                    "all published for this year"
                                    if kinds == {"direct"}
                                    else "some last/first days weekday-aligned (+-3d) from adjacent published year"
                                ),
                            )
                        )
                repl.append(
                    (
                        rc,
                        Y,
                        "Summer break",
                        "Summer break",
                        "Data-driven from published last/first days (B: fixed-date median rule)",
                    )
                )
        # Winter: year Y block = Jan part (winter starting Y-1) + Dec part (Y)
        for Y in (2025, 2026):
            jan = []
            dec = []
            kinds = set()
            for n in names:
                v = DIST[n]
                r, k = get_win(v[6], Y - 1)
                if r:
                    jan.append((v[1], r))
                    kinds.add(k)
                r, k = get_win(v[6], Y)
                if r:
                    dec.append((v[1], r))
                    kinds.add(k)
            if len(jan) >= 2 and len(dec) >= 2:
                for dd in rng(date(Y, 1, 1), date(Y, 1, 10)):
                    s = wshare(jan, lambda r: r[0] <= dd <= r[1])
                    if s:
                        rows.append(
                            (
                                rc,
                                dd.isoformat(),
                                "school",
                                "Winter break",
                                round(s, 2),
                                "Large-district calendars",
                                "medium",
                                "Weighted share of sampled districts on winter break; "
                                + (
                                    "direct"
                                    if kinds == {"direct"}
                                    else "derived from adjacent-year break by Christmas-week alignment"
                                ),
                            )
                        )
                for dd in rng(date(Y, 12, 12), date(Y, 12, 31)):
                    s = wshare(dec, lambda r: r[0] <= dd <= r[1])
                    if s:
                        rows.append(
                            (
                                rc,
                                dd.isoformat(),
                                "school",
                                "Winter break",
                                round(s, 2),
                                "Large-district calendars",
                                "medium",
                                "Weighted share of sampled districts on winter break; "
                                + (
                                    "direct"
                                    if kinds == {"direct"}
                                    else "derived from adjacent-year break by Christmas-week alignment"
                                ),
                            )
                        )
                repl.append(
                    (
                        rc,
                        Y,
                        "Winter break",
                        "Winter break",
                        "Data-driven from published district winter breaks (B: 0.5 edges Dec19-23/Jan2-4)",
                    )
                )
