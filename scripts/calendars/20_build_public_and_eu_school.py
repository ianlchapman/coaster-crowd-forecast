"""National holidays (all countries) and European school holidays -> national_and_eu_school.csv.

OpenHolidays responses (10_fetch_openholidays.py) where available (high confidence), python-holidays otherwise (medium).

Ported research script (hand-curated calendar data). Paths come from crowdcast.config.Paths; see docs/DATA.md.
"""

import json, glob, os, csv, collections, datetime as dt, sys
import holidays, pycountry
from crowdcast.config import Paths

_P = Paths()
CACHE = str(_P.interim / "calendars" / "openholidays_cache")
OUT = str(_P.calendars_dir)
D = str(_P.calendars_dir / "reports")
os.makedirs(D, exist_ok=True)
ISO = {s.code for s in pycountry.subdivisions}
ISO_C = {c.alpha_2 for c in pycountry.countries}
Y0, Y1 = 2014, 2027
ALL = "US FR CN DE GB JP BE NL KR ES AU DK CA IT SE HK AT BR PL MY MX SA IE CH NO FI CZ SK PT LU IN AE SG ID TH NZ AR UY RU TR RO HU".split()
OH = "AT BE BR CH CZ DE ES FR HU IE IT LU MX NL PL PT RO SE SK".split()
MAP = {
    "FR": {
        "AR": "ARA",
        "BF": "BFC",
        "BT": "BRE",
        "CO": "COR",
        "CV": "CVL",
        "GE": "GES",
        "HF": "HDF",
        "IF": "IDF",
        "NA": "NAQ",
        "NO": "NOR",
        "OC": "OCC",
        "PC": "PAC",
        "PL": "PDL",
        "GP": "GP",
        "MQ": "MQ",
        "GY": "GF",
        "RU": "RE",
        "YT": "YT",
        "BL": "BL",
        "MF": "MF",
        "SP": "PM",
    },
    "PL": dict(
        DS="02",
        KP="04",
        LB="08",
        LD="10",
        LU="06",
        MA="12",
        MZ="14",
        OP="16",
        PD="20",
        PK="18",
        PM="22",
        SK="26",
        SL="24",
        WN="28",
        WP="30",
        ZP="32",
    ),
    "PT": {"AC": "20", "MA": "30"},
    "CZ": {"PR": "10"},
    "RO": {"BU": "B"},
    "AT": {"BL": "1", "KÄ": "2", "NÖ": "3", "OÖ": "4", "SB": "5", "SM": "6", "TI": "7", "VA": "8", "WI": "9"},
    "IT": dict(
        AB="65",
        BA="77",
        CL="78",
        CM="72",
        ER="45",
        FV="36",
        LA="62",
        LI="42",
        LO="25",
        MA="57",
        MO="67",
        PI="21",
        PU="75",
        SA="88",
        SI="82",
        TO="52",
        TR="32",
        UM="55",
        VA="23",
        VE="34",
    ),
}


def valid(c):
    return c in ISO_C or c in ISO


def mapc(code):
    """OH code -> (iso 2-part region, depth) or None"""
    p = code.split("-")
    if p[0] not in MAP and len(p) == 2:
        return code if valid(code) else None
    m = MAP.get(p[0], {})
    if len(p) >= 2:
        h = f"{p[0]}-{m.get(p[1], p[1])}"
        return h if valid(h) else None
    return None


def dr(a, b):
    a = dt.date.fromisoformat(a)
    b = dt.date.fromisoformat(b)
    while a <= b:
        yield a
        a += dt.timedelta(1)


def load(tag, c):
    out = []
    for f in sorted(glob.glob(f"{CACHE}/{tag}_{c}_*.json")):
        out += json.load(open(f))
    return out


def ename(r):
    for n in r["name"]:
        if n["language"] == "EN":
            return n["text"]
    return r["name"][0]["text"]


def nchildren(c, parent):
    try:
        subs = json.load(open(f"{CACHE}/sub_{c}.json"))
    except Exception:
        return 0
    for s in subs:
        if s["code"] == parent:
            return len(s.get("children", []))
    return 0


rows = {}  # (region,date,type,name) -> dict


def add(region, d, typ, name, share, src, conf, notes=""):
    if not (Y0 <= d.year <= Y1):
        return
    if "(estimated)" in name and conf == "medium":
        conf = "low"
        notes = (notes + "; " if notes else "") + "Astronomically estimated lunar date (+/-1 day)"
    k = (region, d.isoformat(), typ, name)
    if k in rows:
        r = rows[k]
        r["share"] = min(1.0, round(r["share"] + share, 4))
        return
    rows[k] = dict(
        region_code=region,
        date=d.isoformat(),
        type=typ,
        name=name,
        share=round(share, 4),
        source=src,
        confidence=conf,
        notes=notes,
    )


# ---------- OH national ----------
OHN = {}  # c -> {(region,date)}  ; OH years
OH_YEARS = {}
oh_pub_rows = collections.defaultdict(list)
for c in OH:
    recs = load("pub", c)
    yrs = set()
    S = set()
    hassub = False
    for r in recs:
        if r.get("temporalScope") != "FullDay":
            continue
        typ = r["type"]
        share = 1.0
        notes = ""
        if typ == "Optional":
            if c == "BR" and ename(r) in ("Carnival", "Corpus Christi", "Ash Wednesday"):
                share = {"Carnival": 0.7, "Corpus Christi": 0.5, "Ash Wednesday": 0.3}[ename(r)]
                notes = "Ponto facultativo (optional holiday, widely observed); share is a judgement estimate"
            else:
                continue
        elif typ not in ("Public", "Bank"):
            continue
        if typ == "Bank":
            notes = "Bank holiday (banks/administration closed)"
            share = 1.0
        for d in dr(r["startDate"], r["endDate"]):
            yrs.add(d.year)
            if d.weekday() == 6:
                continue  # Sunday holidays dropped (already non-working); consistent with holidays-pkg handling
            regs = []
            if r.get("nationwide", False) or not r.get("subdivisions"):
                regs = [(c, share, notes)]
            else:
                for s in r["subdivisions"]:
                    code = s["code"]
                    if code.count("-") == 1:
                        m = mapc(code)
                        if m:
                            regs.append((m, share, notes))
                            hassub = True
                    elif c == "FR" and code.startswith("FR-GE-"):
                        regs.append(
                            (
                                {"BR": "FR-6AE", "HR": "FR-6AE", "MO": "FR-57"}.get(code[-2:], "FR-57"),
                                share,
                                "Alsace-Moselle local law (Bas-Rhin/Haut-Rhin = FR-6AE, Moselle = FR-57)",
                            )
                        )
            for reg, sh, nt in regs:
                if reg == "FR-GES" and any(x[0] == "FR-GES" for x in [(k[0],) for k in [(reg,)]]):
                    pass
                key = (reg, d.isoformat(), "national", ename(r))
                if key in rows:
                    continue
                add(
                    reg,
                    d,
                    "national",
                    ename(r),
                    sh,
                    "OpenHolidays API (openholidaysapi.org/PublicHolidays)",
                    "high" if "estimate" not in nt else "medium",
                    nt,
                )
                S.add((reg, d.isoformat()))
    OHN[c] = S
    OH_YEARS[c] = {y for y in yrs if Y0 <= y <= Y1}
    OH_YEARS[c + "_hassub"] = hassub

# ---------- holidays pkg ----------
TERR = {"US": {"US-AS", "US-GU", "US-MP", "US-PR", "US-VI", "US-UM"}, "FR": {"FR-NC", "FR-PF", "FR-TF", "FR-WF"}}
FRREMAP = {"FR-971": "FR-GP", "FR-972": "FR-MQ", "FR-973": "FR-GF", "FR-974": "FR-RE", "FR-976": "FR-YT"}
NO_SUNDAY_FILTER = {"SA", "AE"}
PK = {}  # c -> {year: {region: {date: name}}}
pkg_err = {}
for c in ALL:
    try:
        probe = holidays.country_holidays(c, years=2024)
    except Exception as e:
        pkg_err[c] = str(e)
        continue
    lang = "en_US" if "en_US" in getattr(probe, "supported_languages", ()) else None
    subs = [f"{c}-{s}" for s in probe.subdivisions if f"{c}-{s}" in ISO]
    subs = [s for s in subs if s not in TERR.get(c, ())] if False else subs
    res = {}
    for y in range(Y0, Y1 + 1):
        try:

            def get(sub):
                kw = dict(years=y, observed=True)
                if lang:
                    kw["language"] = lang
                if sub:
                    kw["subdiv"] = sub.split("-", 1)[1]
                return {
                    d: n
                    for d, n in holidays.country_holidays(c, **kw).items()
                    if c in NO_SUNDAY_FILTER or d.weekday() != 6
                }

            if not subs:
                res[y] = {c: get(None)}
            else:
                per = {s: get(s) for s in subs}
                core = [s for s in subs if s not in TERR.get(c, ())]
                if c == "US":
                    cn = get(None)  # federal set is country-wide by definition
                else:
                    common = set.intersection(*[set(per[s]) for s in core])
                    cn = {}
                    for d in common:
                        cn[d] = collections.Counter(per[s][d] for s in core).most_common(1)[0][0]
                r = {c: cn}
                for s in subs:
                    ex = {d: n for d, n in per[s].items() if d not in cn}
                    if ex:
                        r[FRREMAP.get(s, s)] = ex
                res[y] = r
        except Exception as e:
            pkg_err[(c, y)] = str(e)
    PK[c] = res

# ---------- merge national ----------
disagree = []
for c in ALL:
    for y in range(Y0, Y1 + 1):
        pk = PK.get(c, {}).get(y)
        ohy = c in OH and y in OH_YEARS[c]
        if ohy and pk:
            # compare
            ohset = {(r, d) for (r, d) in OHN[c] if d.startswith(str(y))}
            pkset = {(r, dt.date.isoformat(d)) for r, m in pk.items() for d in m}
            pk_country = {x for x in pkset if x[0] == c}
            oh_country = {x for x in ohset if x[0] == c}
            oh_sub = {x for x in ohset if x[0] != c}
            pk_sub = {x for x in pkset if x[0] != c}
            for x in sorted(oh_country - pk_country):
                disagree.append((c, y, "country", "OH only", x[0], x[1]))
            for x in sorted(pk_country - oh_country):
                disagree.append((c, y, "country", "holidays-pkg only", x[0], x[1]))
            if OH_YEARS[c + "_hassub"]:
                for x in sorted(oh_sub - pk_sub):
                    disagree.append((c, y, "subdivision", "OH only", x[0], x[1]))
                for x in sorted(pk_sub - oh_sub):
                    disagree.append((c, y, "subdivision", "holidays-pkg only", x[0], x[1]))
        if ohy and pk:
            # supplement subdivision-level from pkg for regions where OH lists nothing that year
            ohregs = {r for (r, d) in OHN[c] if d.startswith(str(y))}
            for reg, m in pk.items():
                if reg == c or reg in ohregs:
                    continue
                for d, n in m.items():
                    if (c, d.isoformat()) in OHN[c]:
                        continue
                    add(
                        reg,
                        d,
                        "national",
                        n,
                        1.0,
                        "python-holidays 0.104 (subdivision supplement)",
                        "medium",
                        "Subdivision-level holiday from holidays package; OpenHolidays lists no rows for this region",
                    )
        if not ohy and pk:
            hi = c in OH
            for reg, m in pk.items():
                for d, n in m.items():
                    notes = "python-holidays observed/substituted dates"
                    if c == "CN" and y == 2027:
                        notes += "; 2027 State Council schedule not yet published, nominal only"
                    add(
                        reg,
                        d,
                        "national",
                        n,
                        1.0,
                        "python-holidays 0.104 (github.com/vacanza/python-holidays)",
                        "medium",
                        notes,
                    )

# ---------- OH school ----------
SCH_C = []
school_cov = collections.defaultdict(set)
for c in OH:
    if c == "BR":
        continue  # non-European: handled by 30_build_school_non_eu.py
    recs = load("sch", c)
    if not recs:
        continue
    SCH_C.append(c)
    for r in recs:
        if r["type"] != "School" or r.get("temporalScope") != "FullDay":
            continue
        name = ename(r)
        targets = []  # (region, share, notes, conf)
        if r.get("nationwide") and not r.get("subdivisions") and not r.get("groups"):
            targets.append((c, 1.0, "", "high"))
        subs = r.get("subdivisions", [])
        groups = r.get("groups", [])
        deeper = collections.defaultdict(set)
        for s in subs:
            code = s["code"]
            p = code.split("-")
            if c == "FR" and len(p) >= 2 and p[1].startswith("Z"):
                continue
            if len(p) == 2:
                m = mapc(code)
                if m:
                    targets.append((m, 1.0, "", "high"))
            elif len(p) == 3:
                deeper[f"{p[0]}-{p[1]}"].add(code)
        for parent, kids in deeper.items():
            m = mapc(parent)
            tot = nchildren(c, parent)
            if not m or not tot:
                continue
            sh = min(1.0, len(kids) / tot)
            if sh >= 0.999:
                targets.append((m, 1.0, "", "high"))
            else:
                targets.append(
                    (
                        m,
                        sh,
                        f"Subregion-level record covering {len(kids)}/{tot} child units of the region (unweighted by population)",
                        "medium",
                    )
                )
        if c == "BE":
            for g in groups:
                if g["code"] == "BE-NL":
                    targets += [("BE-VLG", 1.0, "Flemish community calendar", "high")]
                elif g["code"] == "BE-FR":
                    targets += [
                        ("BE-WAL", 1.0, "French community calendar", "high"),
                        (
                            "BE-BRU",
                            1.0,
                            "French community calendar; Brussels Dutch-language schools follow the Flemish calendar (share overstated for BE-BRU)",
                            "medium",
                        ),
                    ]
                elif g["code"] == "BE-DE":
                    targets += [("BE-WAL", 0.0, "", "high")]  # German community folded in; ignored
            targets = [t for t in targets if t[1] > 0]
        for reg, sh, nt, conf in targets:
            for d in dr(r["startDate"], r["endDate"]):
                add(reg, d, "school", name, sh, "OpenHolidays API (openholidaysapi.org/SchoolHolidays)", conf, nt)
                school_cov[c].add(d.year)

# ---------- write ----------
os.makedirs(OUT, exist_ok=True)
with open(OUT + "/national_and_eu_school.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames="region_code,date,type,name,share,source,confidence,notes".split(","))
    w.writeheader()
    for k in sorted(rows):
        w.writerow(rows[k])
with open(D + "/disagreements.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow("country,year,level,where,region,date".split(","))
    w.writerows(disagree)
json.dump(
    {
        "errs": {str(k): v for k, v in pkg_err.items()},
        "oh_years": {k: sorted(v) if isinstance(v, set) else v for k, v in OH_YEARS.items()},
        "school_years": {c: sorted(v) for c, v in school_cov.items()},
    },
    open(D + "/build_meta.json", "w"),
    indent=1,
)
print(len(rows), "rows", len(disagree), "disagreements", pkg_err)
