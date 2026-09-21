"""School holidays checked against official AU/NZ term dates and US district calendars -> school_non_eu_verified.csv.

Also writes school_non_eu_replaces.csv: the (region, year, name) blocks of school_non_eu.csv that these rows supersede.
Ported research script; paths come from crowdcast.config.Paths.
"""

import sys, csv, os, statistics as S

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _au_nz_terms as au_nz, _us_districts as us, pandas as pd
from datetime import date
from crowdcast.config import Paths

rows = []
repl = []
au_nz.main(rows, repl)
us.build(rows, repl)
base = str(Paths().calendars_dir) + "/"
os.makedirs(base + "reports", exist_ok=True)
df = pd.DataFrame(rows, columns="region_code,date,type,name,share,source,confidence,notes".split(","))
assert not df.duplicated(["region_code", "date", "type", "name"]).any(), df[
    df.duplicated(["region_code", "date", "type", "name"], keep=False)
].head()
b = pd.read_csv(base + "school_non_eu.csv")
b["year"] = b.date.str[:4].astype(int)
# every replaced block must exist in B
bk = set(zip(b.region_code, b.year, b.name))
R = pd.DataFrame(repl, columns=["region_code", "year", "old_name", "new_name", "reason"]).drop_duplicates(
    ["region_code", "year", "old_name"]
)
missing = R[[(r, y, n) not in bk for r, y, n in zip(R.region_code, R.year, R.old_name)]]
print("replaced blocks not in B (dropped from replaces? kept as ADD):", len(missing))
# rows for blocks not in B are ADDs -> fine; keep in R only those in B
R = R[[(r, y, n) in bk for r, y, n in zip(R.region_code, R.year, R.old_name)]]
# also ensure new block's year matches row-year set: split rows by year of date
df["year"] = df.date.str[:4].astype(int)
# rows falling outside the replaced (region,year,name) set would be unreplaced ADDs -> check
keys = set(zip(R.region_code, R.year, R.old_name))
add = df[[(r, y, n) not in keys for r, y, n in zip(df.region_code, df.year, df.name)]]
print(
    "ADD rows outside replaced blocks:",
    len(add),
    add.groupby(["region_code", "year", "name"]).size().head(20).to_dict(),
)
df.drop(columns="year").to_csv(base + "school_non_eu_verified.csv", index=False)
R.to_csv(base + "school_non_eu_replaces.csv", index=False)
print(len(df), len(R))


# accuracy: compare B (>=0.5 share dates) to E for AU/NZ mid-year blocks
def span(x, thr=0.5):
    x = x[x.share >= thr]
    return (x.date.min(), x.date.max(), len(x)) if len(x) else None


stats = {}
for (rc, y, n), g in df.groupby(["region_code", "year", "name"]) if False else []:
    pass
df["year"] = df.date.str[:4].astype(int)
out = []
for (rc, y, n), g in df.groupby(["region_code", "year", "name"]):
    if rc.startswith("AU") or rc == "NZ":
        if n.startswith("Summer"):
            continue
        bb = b[(b.region_code == rc) & (b.year == y) & (b.name == n)]
        if len(bb) == 0:
            continue
        e0, e1 = g.date.min(), g.date.max()
        b0, b1 = bb.date.min(), bb.date.max()
        dd = lambda a, c: (date.fromisoformat(a) - date.fromisoformat(c)).days
        ov = len(set(g.date) & set(bb.date))
        un = len(set(g.date) | set(bb.date))
        out.append((rc, y, n, dd(b0, e0), dd(b1, e1), ov / un))
o = pd.DataFrame(out, columns=["rc", "y", "n", "start_err", "end_err", "jacc"])
o["abs_start"] = o.start_err.abs()
o["abs_end"] = o.end_err.abs()
print(o.groupby("rc")[["abs_start", "abs_end", "jacc"]].mean().round(2))
print(o.groupby("rc").apply(lambda x: (x.jacc >= 0.99).mean()).round(2))
o18 = o[(o.y >= 2018) & (o.y <= 2026)]
print(o18.groupby("rc")[["abs_start", "abs_end", "jacc"]].mean().round(2))
print("overall abs start/end", o18.abs_start.mean(), o18.abs_end.mean(), "median jacc", o18.jacc.median())
o.to_csv(base + "reports/au_nz_rules_vs_official.csv", index=False)
# US accuracy: B vs E for shares on replaced blocks
u = df[df.region_code.str.startswith("US-")]
res = []
for (rc, y, n), g in u.groupby(["region_code", "year", "name"]):
    bb = b[(b.region_code == rc) & (b.year == y) & (b.name == n)].set_index("date").share
    ee = g.set_index("date").share
    idx = sorted(set(bb.index) | set(ee.index))
    res.append(
        (
            rc,
            y,
            n,
            sum(abs(ee.get(i, 0) - bb.get(i, 0)) for i in idx),
            len(idx),
            (ee[ee >= 0.5].index.min() if (ee >= 0.5).any() else None),
            (bb[bb >= 0.5].index.min() if (bb >= 0.5).any() else None),
            (ee[ee >= 0.5].index.max() if (ee >= 0.5).any() else None),
            (bb[bb >= 0.5].index.max() if (bb >= 0.5).any() else None),
        )
    )
r = pd.DataFrame(res, columns=["rc", "y", "n", "sumabsdiff", "ndays", "E50first", "B50first", "E50last", "B50last"])
r.to_csv(base + "reports/us_rules_vs_districts.csv", index=False)
pd.set_option("display.width", 250)
print(r[r.y.isin([2025])].to_string())
