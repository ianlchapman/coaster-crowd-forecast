"""Rule-derived school-holiday gap fill (GB, Nordics, TR, RU, IE, ES/IT 2026H2-2027) -> school_gapfill_eu.csv.

Ported research script (hand-curated rules); paths come from crowdcast.config.Paths.
"""

import csv, collections, datetime as dt, sys
from crowdcast.config import Paths

D = dt.date
OUT = str(Paths().calendars_dir) + "/"
LO, HI = D(2014, 1, 1), D(2027, 12, 31)
rows = {}  # (region,date) -> dict


def easter(y):
    a = y % 19
    b = y // 100
    c = y % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    mo = (h + l - 7 * m + 114) // 31
    da = (h + l - 7 * m + 114) % 31 + 1
    return D(y, mo, da)


def mon_in(y, m, d0, d1):
    for d in range(d0, d1 + 1):
        x = D(y, m, d)
        if x.weekday() == 0:
            return x


def fri_in(y, m, d0, d1):
    for d in range(d0, d1 + 1):
        x = D(y, m, d)
        if x.weekday() == 4:
            return x


def last_fri_le(y, m, d):
    x = D(y, m, d)
    while x.weekday() != 4:
        x -= dt.timedelta(1)
    return x


def isomon(y, w):
    return D.fromisocalendar(y, w, 1)


def dr(a, b):
    x = a
    while x <= b:
        yield x
        x += dt.timedelta(1)


class Cal:
    def __init__(s, region, source):
        s.r = region
        s.src = source
        s.w = {}

    def add(s, day, share, name, conf, note):
        if day.weekday() >= 5 or share <= 0:
            return
        c = s.w.setdefault(day, [])
        c.append((share, name, conf, note))

    def rng(s, a, b, share, name, conf, note):
        for x in dr(a, b):
            s.add(x, share, name, conf, note)

    def week(s, mon, share, name, conf, note):
        s.rng(mon, mon + dt.timedelta(4), share, name, conf, note)

    def emit(s):
        wd = {}
        for day, c in s.w.items():
            tot = min(1.0, sum(x[0] for x in c))
            best = max(c, key=lambda x: x[0])
            conf = best[2] if tot >= 0.5 else "low"
            notes = "; ".join(dict.fromkeys(x[3] for x in c))
            wd[day] = (round(tot, 2), best[1], conf, notes)
        out = dict(wd)
        for day in list(wd) + []:
            pass
        # weekends: derived from adjacent Fri/Mon
        cand = set()
        for day in wd:
            for k in range(1, 4):
                cand.add(day + dt.timedelta(k))
                cand.add(day - dt.timedelta(k))
        for x in cand:
            if x.weekday() < 5 or x in out:
                continue
            fri = x - dt.timedelta(x.weekday() - 4) if x.weekday() >= 5 else None
            fri = x - dt.timedelta(x.weekday() - 4)
            mon = fri + dt.timedelta(3)
            a, b = wd.get(fri), wd.get(mon)
            cs = [t for t in (a, b) if t]
            if not cs:
                continue
            t = max(cs, key=lambda t: t[0])
            out[x] = (t[0], t[1], t[2], t[3] + "; weekend inherits adjacent Fri/Mon share")
        res = []
        for day, (sh, name, conf, notes) in out.items():
            if LO <= day <= HI and sh >= 0.05:
                res.append((s.r, day, name, sh, s.src, conf, notes))
        return res


ALL = []


def finish(cal):
    ALL.extend(cal.emit())


# ---------------- GB England / Wales -----------------
ENG_SRC = "gov.uk DfE term-date guidance + typical LA calendars (Kent, Lancashire, Birmingham, Hampshire, Essex, Surrey, Leeds, Manchester patterns); rule-derived, not scraped per LA"


def eng_like(region, conf_main, wales=False):
    c = Cal(
        region,
        ENG_SRC
        if not wales
        else "Welsh LA calendars (Cardiff, Swansea, Gwynedd pattern) assumed to follow English pattern; rule-derived",
    )
    base = lambda t: "Rule: " + t
    for y in range(2013, 2028):
        # Christmas (start in Dec y)
        F1 = last_fri_le(y, 12, 21)
        F2 = last_fri_le(y, 12, 22)
        n = "Christmas holidays"
        nt = base(
            "break from Sat after last Fri<=21 Dec; if Fri 22 Dec exists, its weekdays 0.5 (some LAs finish that week)"
        )
        c.rng(F1 + dt.timedelta(3), D(y, 12, 31), 1, n, conf_main, nt)
        if F2 > F1:
            c.rng(F1 + dt.timedelta(1), F2, 0.5, n, "low", nt)
        # Jan return (Jan of y+1)
        j = D(y + 1, 1, 1)
        ramp = {1: 1, 2: 0.9, 3: 0.8, 4: 0.5, 5: 0.4, 6: 0.2, 7: 0.1, 8: 0.05}
        for d, sh in ramp.items():
            c.add(
                D(y + 1, 1, d),
                sh,
                n,
                conf_main if sh >= 0.8 else "low",
                base("Jan return ramp by calendar day (schools reopen 2-8 Jan; share = pupils still off)"),
            )
        # Summer
        E = fri_in(y, 7, 17, 23)
        ns = "Summer holidays"
        nts = base("term ends Fri in 17-23 Jul (weekdays after) ; return ramp 1-7 Sep")
        c.rng(E + dt.timedelta(3), D(y, 8, 31), 1, ns, conf_main, nts)
        if E.day <= 19:
            c.add(E + dt.timedelta(3), 0.5, ns, "low", nts)
            c.add(E + dt.timedelta(4), 0.4, ns, "low", nts)  # duplicates cap at 1 via sum
        for d, sh in {1: 0.95, 2: 0.9, 3: 0.8, 4: 0.6, 5: 0.3, 6: 0.15, 7: 0.05}.items():
            c.add(D(y, 9, d), sh, ns, "low" if sh < 0.8 else conf_main, nts)
        # Easter
        Ea = easter(y)
        w_after = 1 if Ea <= D(y, 4, 6) else (0.75 if Ea <= D(y, 4, 10) else (0.4 if Ea <= D(y, 4, 13) else 0))
        ne = "Easter holidays"
        nte = base(
            "two windows: straddling (Mon Easter-6 to Fri Easter+5, weight by Easter date) or before-Easter (Mon Easter-13 to Easter Mon)"
        )
        if w_after > 0:
            c.rng(Ea - dt.timedelta(6), Ea + dt.timedelta(5), w_after, ne, conf_main if w_after >= 0.75 else "low", nte)
        if w_after < 1:
            c.rng(
                Ea - dt.timedelta(13), Ea + dt.timedelta(1), 1 - w_after, ne, conf_main if w_after == 0 else "low", nte
            )
        # Feb half-term
        m = mon_in(y, 2, 12, 18)
        nf = "February half-term"
        ntf = base("Mon in 12-18 Feb (0.8), adjacent week (0.2)")
        c.week(m, 0.8, nf, conf_main, ntf)
        c.week(m + dt.timedelta(7 if m.day <= 14 else -7), 0.2, nf, "low", ntf)
        # May half-term
        lm = D(y, 5, 31)
        while lm.weekday() != 0:
            lm -= dt.timedelta(1)
        c.week(lm, 0.9, "May half-term", conf_main, base("week of last Mon of May (spring bank hol week)"))
        c.week(lm - dt.timedelta(7), 0.1, "May half-term", "low", base("some LAs/Whit week earlier"))
        # Oct half-term
        m = mon_in(y, 10, 22, 28)
        no = "October half-term"
        nto = base("week of Mon in 22-28 Oct (0.8); following week 0.2 if Mon<=24 Oct")
        c.week(m, 0.8, no, conf_main, nto)
        c.week(m + dt.timedelta(7), 0.2, no, "low", nto) if m.day <= 24 else c.week(
            m - dt.timedelta(7), 0.2, no, "low", nto
        )
    finish(c)


eng_like("GB-ENG", "medium")
eng_like("GB-WLS", "low", True)

# ---------------- Scotland -----------------
c = Cal("GB-SCT", "Edinburgh/Glasgow/Aberdeen/Highland council calendar patterns; rule-derived, not scraped")
for y in range(2013, 2028):
    b = lambda t: "Rule: " + t
    E = fri_in(y, 6, 26, 32 - 1) if False else None
    d = D(y, 6, 26)
    while d.weekday() != 4:
        d += dt.timedelta(1)
    n = "Summer holidays"
    nt = b("term ends Fri 26 Jun-2 Jul; pupils back 12-20 Aug (ramp)")
    c.rng(d + dt.timedelta(3), D(y, 8, 9), 1, n, "medium", nt)
    c.add(d - dt.timedelta(1), 0.3, n, "low", nt)
    for dd, sh in {
        10: 1,
        11: 0.95,
        12: 0.8,
        13: 0.6,
        14: 0.5,
        15: 0.4,
        16: 0.3,
        17: 0.25,
        18: 0.2,
        19: 0.1,
        20: 0.05,
    }.items():
        c.add(D(y, 8, dd), sh, n, "low" if sh < 0.8 else "medium", nt)
    # Oct
    m = mon_in(y, 10, 12, 18)
    no = "October holidays"
    nto = b("week of Mon 12-18 Oct 0.8; second week 0.35 (some councils take 2 weeks)")
    c.week(m, 0.8, no, "medium", nto)
    c.week(m + dt.timedelta(7), 0.35, no, "low", nto)
    # Xmas
    F = last_fri_le(y, 12, 23)
    n = "Christmas holidays"
    nt = b("break from Sat after last Fri<=23 Dec; return ramp 4-9 Jan")
    c.rng(F + dt.timedelta(3), D(y, 12, 31), 1, n, "medium", nt)
    for dd, sh in {1: 1, 2: 1, 3: 1, 4: 0.9, 5: 0.8, 6: 0.55, 7: 0.35, 8: 0.2, 9: 0.1}.items():
        c.add(D(y + 1, 1, dd), sh, n, "low" if sh < 0.8 else "medium", nt)
    # Feb midterm
    m = mon_in(y, 2, 10, 16)
    c.add(m, 0.6, "February mid-term", "low", b("Mon (+Tue 0.4) in 10-16 Feb; councils differ"))
    c.add(m + dt.timedelta(1), 0.4, "February mid-term", "low", b("Mon (+Tue 0.4) in 10-16 Feb; councils differ"))
    # Easter
    P = mon_in(y, 4, 4, 10)
    ne = "Easter holidays"
    nte = b("two weeks from Mon 4-10 Apr (0.6) or one week earlier (0.3)")
    c.rng(P, P + dt.timedelta(11), 0.6, ne, "low", nte)
    c.rng(P - dt.timedelta(7), P + dt.timedelta(4), 0.3, ne, "low", nte)
finish(c)

# ---------------- Northern Ireland -----------------
c = Cal("GB-NIR", "NI Department of Education statutory pattern (summer Jul-Aug) + typical NI calendars; rule-derived")
for y in range(2013, 2028):
    b = lambda t: "Rule: " + t
    n = "Summer holidays"
    nt = b("NI schools close ~30 Jun, reopen ~1-3 Sep (statutory Jul-Aug)")
    c.rng(D(y, 7, 1), D(y, 8, 31), 1, n, "medium", nt)
    c.add(D(y, 6, 30), 0.6, n, "low", nt)
    c.add(D(y, 6, 29), 0.3, n, "low", nt)
    for dd, sh in {1: 0.8, 2: 0.5, 3: 0.2}.items():
        c.add(D(y, 9, dd), sh, n, "low", nt)
    x = D(y, 10, 31)
    m = x - dt.timedelta(x.weekday())
    c.week(
        m,
        0.5,
        "Halloween break",
        "low",
        b("week containing 31 Oct at 0.5 (NI Halloween break is 1-2 days; week approximated)"),
    )
    F = last_fri_le(y, 12, 22)
    n = "Christmas holidays"
    nt = b("break from Sat after last Fri<=22 Dec; return ramp 3-8 Jan")
    c.rng(F + dt.timedelta(3), D(y, 12, 31), 1, n, "medium", nt)
    for dd, sh in {1: 1, 2: 1, 3: 0.85, 4: 0.6, 5: 0.4, 6: 0.2, 7: 0.1}.items():
        c.add(D(y + 1, 1, dd), sh, n, "low" if sh < 0.8 else "medium", nt)
    m = mon_in(y, 2, 12, 21)
    c.add(m, 0.7, "February mid-term", "low", b("Mon+Tue in 12-21 Feb"))
    c.add(m + dt.timedelta(1), 0.7, "February mid-term", "low", b("Mon+Tue in 12-21 Feb"))
    Ea = easter(y)
    ne = "Easter holidays"
    nte = b("Thu before Easter to Tue after Easter (weekdays), approx")
    for k in (-3, -2, 1, 2):
        c.add(Ea + dt.timedelta(k), 0.8, ne, "low", nte)
finish(c)


# ---------------- Nordics -----------------
def xmas(c, y, ramp_pre, jan, src, conf="medium"):
    for dd, sh in ramp_pre.items():
        c.add(D(y, 12, dd), sh, "Christmas holidays", conf if sh >= 0.8 else "low", src)
    for dd, sh in jan.items():
        c.add(D(y + 1, 1, dd), sh, "Christmas holidays", conf if sh >= 0.8 else "low", src)


b = lambda t: "Rule: " + t
# DK
c = Cal(
    "DK",
    "Danish Ministry of Children and Education norms (summer wk27-32, autumn wk42, winter wk7); municipal calendars vary +-days; rule-derived",
)
for y in range(2013, 2028):
    F = isomon(y, 26) + dt.timedelta(4)
    n = "Summer holidays"
    nt = b("Sat after Fri of ISO wk26 through Sun before Mon of ISO wk33 (weeks 27-32); return ramp Mon-Wed wk33")
    c.rng(F + dt.timedelta(3), isomon(y, 33) - dt.timedelta(1), 1, n, "medium", nt)
    m = isomon(y, 33)
    for k, sh in enumerate((0.6, 0.35, 0.1)):
        c.add(m + dt.timedelta(k), sh, n, "low", nt)
    c.week(isomon(y, 42), 1, "Autumn holidays", "medium", b("ISO week 42"))
    c.week(isomon(y, 7), 1, "Winter holidays", "medium", b("ISO week 7 (uge 7)"))
    Ea = easter(y)
    c.rng(
        Ea - dt.timedelta(6),
        Ea + dt.timedelta(1),
        1,
        "Easter holidays",
        "medium",
        b("Mon of Holy Week through Easter Monday"),
    )
    xmas(c, y, {21: 0.4, 22: 0.8}, {}, b("Christmas Dec 23-Jan 1 full; ramp 21-22 Dec and 2-3 Jan"))
    c.rng(
        D(y, 12, 23),
        D(y, 12, 31),
        1,
        "Christmas holidays",
        "medium",
        b("Christmas Dec 23-Jan 1 full; ramp 21-22 Dec and 2-3 Jan"),
    )
    c.rng(
        D(y + 1, 1, 1),
        D(y + 1, 1, 1),
        1,
        "Christmas holidays",
        "medium",
        b("Christmas Dec 23-Jan 1 full; ramp 21-22 Dec and 2-3 Jan"),
    )
    c.add(D(y + 1, 1, 2), 0.5, "Christmas holidays", "low", b("ramp"))
    c.add(D(y + 1, 1, 3), 0.1, "Christmas holidays", "low", b("ramp"))
finish(c)
# NO
c = Cal(
    "NO",
    "Utdanningsdirektoratet/municipal patterns (Oslo, Bergen, Trondheim); county-staggered weeks approximated; rule-derived",
)
for y in range(2013, 2028):
    F = fri_in(y, 6, 18, 24)
    n = "Summer holidays"
    nt = b("last day Fri 18-24 Jun (Thu 0.5); back Mon 15-21 Aug (Mon 0.5, Tue 0.3, Wed 0.15)")
    c.rng(F + dt.timedelta(3), D(y, 8, 31), 1, n, "medium", nt)
    c.add(F, 0.4, n, "low", nt) if False else None
    c.add(F - dt.timedelta(1), 0.5, n, "low", nt)
    M = mon_in(y, 8, 15, 21)
    for dd in range(1, M.day):
        pass
    # remove full-share days from M onward by overriding: do explicit windows instead
    c.w = {k: v for k, v in c.w.items() if not (k >= M and k.year == y and k.month >= 8 and v and v[0][1] == n)}
    c.rng(F + dt.timedelta(3), M - dt.timedelta(1), 1, n, "medium", nt)
    for k, sh in enumerate((0.5, 0.3, 0.15)):
        c.add(M + dt.timedelta(k), sh, n, "low", nt)
    c.week(
        isomon(y, 8), 0.65, "Winter holidays", "low", b("vinterferie: ISO wk8 majority (0.65), wk9 (0.35) by county")
    )
    c.week(
        isomon(y, 9), 0.35, "Winter holidays", "low", b("vinterferie: ISO wk8 majority (0.65), wk9 (0.35) by county")
    )
    c.week(isomon(y, 40), 0.75, "Autumn holidays", "low", b("høstferie ISO wk40 (0.75) / wk41 (0.25)"))
    c.week(isomon(y, 41), 0.25, "Autumn holidays", "low", b("høstferie ISO wk40 (0.75) / wk41 (0.25)"))
    Ea = easter(y)
    c.rng(
        Ea - dt.timedelta(6),
        Ea + dt.timedelta(1),
        0.9,
        "Easter holidays",
        "medium",
        b("Mon of Holy Week through Easter Monday (some schools start Wed)"),
    )
    t = b("Christmas Dec 21-Jan 1 full; ramp 19-20 Dec and 2-3 Jan")
    c.rng(D(y, 12, 21), D(y, 12, 31), 1, "Christmas holidays", "medium", t)
    c.add(D(y, 12, 20), 0.6, "Christmas holidays", "low", t)
    c.add(D(y, 12, 19), 0.2, "Christmas holidays", "low", t)
    c.add(D(y + 1, 1, 1), 1, "Christmas holidays", "medium", t)
    c.add(D(y + 1, 1, 2), 0.6, "Christmas holidays", "low", t)
    c.add(D(y + 1, 1, 3), 0.1, "Christmas holidays", "low", t)
finish(c)
# SE
c = Cal(
    "SE",
    "Swedish municipal calendars (Stockholm, Goteborg, Malmo pattern); sportlov staggered wk7-10 (sv.wikipedia); rule-derived",
)
for y in range(2013, 2028):
    F = fri_in(y, 6, 8, 14)
    n = "Summer holidays"
    nt = b("term ends Fri 8-14 Jun; back ~Wed of ISO wk34 area 15-20 Aug (ramp)")
    R = isomon(y, 34) + dt.timedelta(2)
    c.rng(F + dt.timedelta(3), R - dt.timedelta(4), 1, n, "medium", nt)
    for k, sh in ((-3, 0.75), (-2, 0.5), (-1, 0.3)):
        c.add(R + dt.timedelta(k), sh, n, "low", nt)
    c.add(R, 0.1, n, "low", nt)
    for w in (7, 8, 9, 10):
        c.week(
            isomon(y, w),
            0.25,
            "Sportlov",
            "low",
            b("sportlov: county-staggered ISO wk7-10; equal 0.25 weights assumed (rotation not looked up)"),
        )
    c.week(isomon(y, 44), 0.9, "Autumn holidays", "low", b("hostlov ISO wk44 majority (some municipalities wk43/45)"))
    Ea = easter(y)
    nt2 = b("paasklov: only some municipalities; Holy Week 0.4 (Mon-Thu), week after Easter 0.15")
    c.rng(Ea - dt.timedelta(6), Ea - dt.timedelta(3), 0.4, "Easter holidays", "low", nt2)
    c.rng(Ea + dt.timedelta(2), Ea + dt.timedelta(5), 0.15, "Easter holidays", "low", nt2)
    # Christmas: Sat after Fri 18-22 Dec through 7 Jan (return first weekday >= 7 Jan)
    F = last_fri_le(y, 12, 22)
    t = b("jullov: from Sat after Fri 18-22 Dec until first weekday on/after 7 Jan (Epiphany 6 Jan)")
    c.rng(F + dt.timedelta(3), D(y, 12, 31), 1, "Christmas holidays", "medium", t)
    r = D(y + 1, 1, 7)
    while r.weekday() >= 5:
        r += dt.timedelta(1)
    c.rng(D(y + 1, 1, 1), r - dt.timedelta(1), 1, "Christmas holidays", "medium", t)
    c.add(r, 0.1, "Christmas holidays", "low", t)
finish(c)
# FI
c = Cal(
    "FI",
    "Opetushallitus/municipal calendars (Helsinki, Espoo, Tampere, Oulu pattern); hiihtoloma regional wk8/9/10 per fi.wikipedia; rule-derived",
)
for y in range(2013, 2028):
    n = "Summer holidays"
    nt = b("term ends Sat 1-7 Jun; autumn term starts Mon-Thu 6-13 Aug (ramp)")
    e = D(y, 6, 1)
    while e.weekday() != 5:
        e += dt.timedelta(1)
    M = mon_in(y, 8, 6, 12)
    c.rng(e - dt.timedelta(1), M - dt.timedelta(3), 1, n, "medium", nt) if False else None
    c.rng(e, M - dt.timedelta(1), 1, n, "medium", nt)
    for k, sh in enumerate((0.5, 0.3, 0.15, 0.05)):
        c.add(M + dt.timedelta(k), sh, n, "low", nt)
    for w, sh in ((8, 0.45), (9, 0.3), (10, 0.25)):
        c.week(
            isomon(y, w),
            sh,
            "Winter holidays (hiihtoloma)",
            "low",
            b(
                "regional staggering: wk8 S/W coast (Uusimaa, V-Suomi, Satakunta) 0.45, wk9 central/SE 0.3, wk10 N/E 0.25 (population shares approximate; rotation across years not applied)"
            ),
        )
    c.week(isomon(y, 42), 0.75, "Autumn holidays", "low", b("syysloma ISO wk42 majority (0.75), wk43 (0.25)"))
    c.week(isomon(y, 43), 0.25, "Autumn holidays", "low", b("syysloma ISO wk42 majority (0.75), wk43 (0.25)"))
    t = b("Christmas: Dec 21 - Jan 6 (Epiphany); return ramp 7-9 Jan; last day ~Dec 20")
    c.rng(D(y, 12, 21), D(y, 12, 31), 1, "Christmas holidays", "medium", t)
    c.add(D(y, 12, 20), 0.5, "Christmas holidays", "low", t)
    c.rng(D(y + 1, 1, 1), D(y + 1, 1, 6), 1, "Christmas holidays", "medium", t)
    for dd, sh in {7: 0.6, 8: 0.25, 9: 0.1}.items():
        c.add(D(y + 1, 1, dd), sh, "Christmas holidays", "low", t)
finish(c)

# ---------------- TR -----------------
c = Cal(
    "TR",
    "MEB (Turkish Ministry of National Education) academic calendars; dates recalled/rule-derived, not verified against archives",
)
SUM = {
    2014: ((6, 20), (9, 8)),
    2015: ((6, 12), (9, 14)),
    2016: ((6, 24), (9, 19)),
    2017: ((6, 16), (9, 11)),
    2018: ((6, 15), (9, 17)),
    2019: ((6, 21), (9, 9)),
    2020: ((6, 30), (8, 31)),
    2021: ((6, 18), (9, 6)),
    2022: ((6, 17), (9, 12)),
    2023: ((6, 16), (9, 11)),
    2024: ((6, 14), (9, 9)),
    2025: ((6, 20), (9, 8)),
    2026: ((6, 26), (9, 7)),
    2027: ((6, 25), (9, 6)),
}
for y, (e, s) in SUM.items():
    c.rng(
        D(y, *e) + dt.timedelta(1),
        D(y, *s) - dt.timedelta(1),
        1,
        "Summer holidays",
        "low",
        "MEB calendar: last day/first day of school; low: recalled, unverified"
        + (" (2026-27 planned/approx)" if y >= 2026 else ""),
    )
for y in range(2014, 2028):
    m = mon_in(y, 1, 17, 23)
    c.rng(
        m,
        m + dt.timedelta(11),
        1,
        "Semester break",
        "low",
        b("two-week mid-year break, Mon 17-23 Jan (unverified per year)"),
    )
    if y >= 2017:
        m = mon_in(y, 11, 10, 16)
        c.week(
            m,
            1,
            "Autumn break (ara tatil)",
            "low",
            b("one-week Nov break since 2017, Mon 10-16 Nov (2017: approximated)"),
        )
finish(c)

# ---------------- RU -----------------
c = Cal(
    "RU",
    "Federal education law (summer 1 Jun-31 Aug, school year starts 1 Sep) + typical regional quarter breaks (Moscow pattern); rule-derived",
)
for y in range(2014, 2028):
    n = "Summer holidays"
    nt = b("federal: 1 Jun-31 Aug; last-bell week 26-31 May ramp 0.6")
    c.rng(D(y, 6, 1), D(y, 8, 31), 1, n, "medium", nt)
    c.rng(D(y, 5, 26), D(y, 5, 31), 0.6, n, "low", nt)
    m = mon_in(y, 10, 24, 30)
    c.week(m, 0.8, "Autumn holidays", "low", b("1 week Mon 24-30 Oct; regional quarters differ"))
    m = mon_in(y, 3, 23, 29)
    c.week(m, 0.75, "Spring holidays", "low", b("1 week Mon 23-29 Mar; regional quarters differ"))
    t = b("winter: 29 Dec-11 Jan (Moscow pattern) 0.9; 26-28 Dec 0.5")
    c.rng(D(y, 12, 29), D(y, 12, 31), 0.9, "Winter holidays", "medium", t)
    c.rng(D(y, 12, 26), D(y, 12, 28), 0.5, "Winter holidays", "low", t)
    c.rng(D(y + 1, 1, 1), D(y + 1, 1, 10), 0.9, "Winter holidays", "medium", t)
finish(c)

# ---------------- Existing school data for overlap / IE / ES / IT -----------------
exist = set()
ex_rows = collections.defaultdict(list)
for r in csv.DictReader(open(OUT + "national_and_eu_school.csv")):
    if r["type"] == "school":
        exist.add((r["region_code"], r["date"]))
        ex_rows[r["region_code"]].append(r)
# IE
c = Cal(
    "IE",
    "Irish Dept of Education circulars (Easter/Christmas/mid-term/summer rules) + patterns in the OpenHolidays 2019-2026 data; rule-derived",
)
for y in range(2014, 2028):
    Ea = easter(y)
    t = b("Fri Easter-9 to Sun Easter+7 (matches 2020-2026 official pattern)")
    c.rng(Ea - dt.timedelta(9), Ea + dt.timedelta(7), 1, "Easter", "medium", t)
    m = mon_in(y, 2, 15, 21) if y < 2027 else D(2027, 2, 15)
    c.week(
        m,
        1,
        "February mid-term break",
        "low" if y < 2020 else "low",
        b("Mon 15-21 Feb (2027: shifted 364d from 2026 official)"),
    )
    m = mon_in(y, 10, 25, 31)
    c.week(m, 1, "October mid-term break", "low", b("Mon 25-31 Oct (matches 2020-2025 pattern)"))
    F = D(y, 12, 22)
    t = b("Christmas: ~22 Dec to ~4 Jan (2019-2025 official spans 20 Dec-7 Jan)")
    c.rng(D(y, 12, 22), D(y, 12, 31), 1, "Christmas", "low", t)
    c.rng(D(y + 1, 1, 1), D(y + 1, 1, 4), 1, "Christmas", "low", t)
    t = b(
        "IE primary closes 30 Jun, secondary earlier (~1 Jun exams) -> Jun 0.5; Jul 1-Aug 24 1.0; 25-31 Aug 0.5 (schools reopen late Aug)"
    )
    for x in dr(D(y, 6, 1), D(y, 6, 30)):
        c.add(x, 0.5, "Summer holidays", "low", t)
    c.rng(D(y, 7, 1), D(y, 8, 24), 1, "Summer holidays", "medium", t)
    for x in dr(D(y, 8, 25), D(y, 8, 31)):
        c.add(x, 0.5, "Summer holidays", "low", t)
finish(c)


# ES / IT 2026H2-2027: transform OpenHolidays rows from the previous cycle
def shift(rec, y_to):
    d = D.fromisoformat(rec["date"])
    nm = rec["name"].lower()
    y_from = d.year
    if "easter" in nm or "holy" in nm:
        return d + (easter(y_to) - easter(y_from)), "Easter-relative shift"
    if any(k in nm for k in ("christmas", "summer")):
        try:
            return d.replace(year=y_to), "same calendar date"
        except ValueError:
            return None, None
    k = y_to - y_from
    return d + dt.timedelta(364 * k + (7 if False else 0)), "weekday-preserving 364d shift"


trans = []
for reg in ex_rows:
    if reg[:2] not in ("ES", "IT"):
        continue
    by = collections.defaultdict(list)
    for r in ex_rows[reg]:
        by[r["date"][:4]].append(r)
    for y_to in (2026, 2027):
        for r in ex_rows[reg]:
            y_from = int(r["date"][:4])
            src_year = 2025 if y_to == 2026 else 2026
            nm = r["name"].lower()
            # source: Dec + Jun-Sep from 2025 (full year), Jan/Feb/Mar/Apr/Oct-Nov etc from prior year
            if y_from != y_to - 1:
                continue
            d2, how = shift(r, y_to)
            if d2 is None or d2.year != y_to and not (("christmas" in nm) and d2.year == y_to):
                continue
            trans.append(
                (
                    reg,
                    d2,
                    r["name"],
                    float(r["share"]),
                    "Transformed from OpenHolidays " + str(y_from) + " rows (" + how + ") of OpenHolidays school data",
                    "low",
                    "Extrapolated: "
                    + how
                    + " from "
                    + r["date"]
                    + "; NOT official; OpenHolidays SchoolHolidays lacks this period",
                )
            )
# keep only transformations that fill missing dates and do not spill
seen = set()
ES_IT = []
for t in trans:
    if (t[0], t[1].isoformat()) in exist or (t[0], t[1]) in seen:
        continue
    if t[1] < D(2026, 1, 1) or t[1] > HI:
        continue
    seen.add((t[0], t[1]))
    ES_IT.append(t)
ALL.extend(ES_IT)

# ---------------- write -----------------
final = {}
for r in ALL:
    reg, day, name, sh, src, conf, notes = r
    key = (reg, day.isoformat)
    k = (reg, day.isoformat() if hasattr(day, "isoformat") else day)
    if k in exist:
        continue
    if k in final:
        if reg[:2] in ("ES", "IT"):
            continue
        raise SystemExit("dup " + str(k))
    final[k] = (reg, k[1], "school", name, sh, src, conf, notes)
with open(OUT + "school_gapfill_eu.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow("region_code,date,type,name,share,source,confidence,notes".split(","))
    for k in sorted(final):
        w.writerow(final[k])
cnt = collections.Counter((v[0], v[1][:4], v[6]) for v in final.values())
print(len(final))
