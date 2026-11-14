"""Median chlorophyll by distance from a product's own land mask (run manually; reads files, no network).

    uv run python scripts/adjacency_table.py <dir> [min_valid_cells]

<dir> holds one ERDDAP griddap .csv per scene of one box (d_YYYY-MM-DD.csv: time, altitude, latitude,
longitude, chlor_a), fetched as docs/coastwatch/v13/research/coastal-adjacency.md says. "Land" is a
cell with no value in any scene of the directory; distance is in the sensor's own cells (1 = touches
land, a diagonal touch included). Prints the table that note records, per sensor and coast."""
import sys, glob, numpy as np
from scipy.ndimage import distance_transform_edt
d = sys.argv[1]; minv = int(sys.argv[2]) if len(sys.argv) > 2 else 0
days = {}
for f in sorted(glob.glob(d + "/d_*.csv")):
    if not open(f, "rb").read(5) == b"time,": continue  # an error page saved in place of a scene
    a = np.genfromtxt(f, delimiter=",", skip_header=2, usecols=(2, 3, 4))
    if a.ndim != 2 or not len(a): continue
    lats = np.unique(a[:, 0])[::-1]; lons = np.unique(a[:, 1])
    g = a[:, 2].reshape(len(lats), len(lons)); g[g <= 0] = np.nan
    days[f[-14:-4]] = g
stack = np.stack(list(days.values()))
ever = np.isfinite(stack).any(0)
dist = np.floor(distance_transform_edt(ever) + 1e-9)  # 1 = touches a never-valid cell (4-neighbour distance 1; a diagonal touch is 1.41 -> 1)
minv = minv or int(0.25 * ever.sum())
bins = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 20), (21, 400)]
rows = []; over = []
for name, g in days.items():
    ok = np.isfinite(g)
    if ok.sum() < minv: continue
    rows.append([np.nanmedian(g[ok & (dist >= a) & (dist <= b)]) if (ok & (dist >= a) & (dist <= b)).sum() >= 15 else np.nan for a, b in bins])
    r1 = g[ok & (dist == 1)]; r2 = g[ok & (dist == 2)]; r3 = g[ok & (dist == 3)]
    over.append([(x > 10).mean() if x.size >= 15 else np.nan for x in (r1, r2, r3)] + [r1.size])
rows = np.array(rows); over = np.array(over)
print(f"{d}: grid {stack.shape[1]}x{stack.shape[2]}, {len(days)} scenes, {len(rows)} clear days (>= {minv} valid cells), ever-valid {ever.mean():.2f}")
print("cells from land   " + "  ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in bins))
print("median mg/m3      " + "  ".join(f"{x:.2f}" for x in np.nanmedian(rows, 0)))
print("min..max of days  " + "  ".join(f"{np.nanmin(rows[:, i]):.1f}..{np.nanmax(rows[:, i]):.1f}" for i in range(len(bins))))
print("share > 10 mg/m3 (median of days): ring1 %.0f%%  ring2 %.0f%%  ring3 %.0f%%; ring-1 cells per day ~%d" % (100 * np.nanmedian(over[:, 0]), 100 * np.nanmedian(over[:, 1]), 100 * np.nanmedian(over[:, 2]), np.nanmedian(over[:, 3])))
print("ratio ring1/ring3 %.1f, ring2/ring3 %.1f, ring3/(4-5) %.1f" % tuple(np.nanmedian(rows[:, i] / rows[:, j]) for i, j in ((0, 2), (1, 2), (2, 3))))
