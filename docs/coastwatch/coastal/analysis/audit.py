# /// script
# dependencies = ["netCDF4", "numpy", "scipy", "matplotlib"]
# ///
"""Nearshore coverage audit of C-HARM v3.1 (and OLCI 300 m chlorophyll) along California.

Reference geography: ETOPO 2022 15-arc-second (~460 m) elevation. Coastal water = elevation < 0
connected to the open ocean (excludes the Salton Sea and other inland basins; the Sacramento-San
Joaquin Delta east of Carquinez Strait is excluded explicitly). Every 15" water pixel is assigned to
the C-HARM 0.03 deg cell that contains its centre (the pipeline's own nearest-cell rule) and classified.
Distances are to the nearest non-water ETOPO pixel (mainland or island), in km.

Nothing here modifies or re-publishes data; source values are only read.
"""
import gzip, json, sys
from pathlib import Path

import numpy as np
import netCDF4
from scipy import ndimage

HERE = Path(__file__).parent
ETOPO = HERE.parent / "etopo" / "ca15s.nc"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

LAT0, DLAT, LON0, DLON, W, H, NODATA = 31.3, 0.03, -127.5, 0.03, 351, 391, 65535
BANDS = [(0, 1), (1, 3), (3, 5), (5, 10), (10, 25)]
REGIONS = [  # (name, lat_min, lat_max) along the coast; SF Bay interior reported separately
    ("North Coast", 40.0, 42.0),
    ("Mendocino/Sonoma", 38.3, 40.0),
    ("San Francisco Bay area", 37.1, 38.3),
    ("Monterey Bay", 36.5, 37.1),
    ("Central Coast", 34.45, 36.5),
    ("Southern California", 32.53, 34.45),
]


def load_etopo():
    d = netCDF4.Dataset(ETOPO)
    lat = d["latitude"][:].astype(np.float64)
    lon = d["longitude"][:].astype(np.float64)
    z = np.asarray(d["z"][:], dtype=np.float32)
    if lat[0] > lat[-1]:
        lat, z = lat[::-1], z[::-1]
    water = z < 0
    lab, _ = ndimage.label(water)
    # the open ocean is the component touching the western edge
    ocean_ids = set(np.unique(lab[:, 0])) - {0}
    ocean = np.isin(lab, list(ocean_ids))
    LON, LAT = np.meshgrid(lon, lat)
    delta = (LON > -122.05) & (LAT > 37.6) & (LAT < 38.6)  # Suisun Bay, the Delta and its channels
    ocean &= ~delta
    return lat, lon, ocean


def distance_km(lat, lon, ocean):
    """Distance from each ocean pixel to the nearest non-ocean pixel, per latitude slab with its own
    east-west scale (error < 1 % within a 1-degree slab)."""
    dist = np.full(ocean.shape, np.nan, dtype=np.float32)
    dy = abs(lat[1] - lat[0]) * 111.32
    pad = 60  # pixels of context (~28 km) above and below each slab
    for s in range(0, lat.size, 240):
        e = min(lat.size, s + 240)
        a, b = max(0, s - pad), min(lat.size, e + pad)
        dx = abs(lon[1] - lon[0]) * 111.32 * np.cos(np.radians(lat[(s + e) // 2]))
        dt = ndimage.distance_transform_edt(ocean[a:b], sampling=(dy, dx))
        dist[s:e] = dt[s - a : s - a + (e - s)]
    dist[~ocean] = np.nan
    return dist


def region_index(lat, lon):
    LON, LAT = np.meshgrid(lon, lat)
    reg = np.full(LAT.shape, -1, dtype=np.int8)
    for i, (_, lo, hi) in enumerate(REGIONS):
        reg[(LAT >= lo) & (LAT < hi)] = i
    sfbay = (LON > -122.52) & (LAT > 37.4) & (LAT < 38.25)  # inside the Golden Gate
    return reg, sfbay


def cell_of(lat, lon):
    LON, LAT = np.meshgrid(lon, lat)
    i = np.floor((LAT - LAT0) / DLAT + 0.5).astype(int)
    j = np.floor((LON - LON0) / DLON + 0.5).astype(int)
    ok = (i >= 0) & (i < H) & (j >= 0) & (j < W)
    return np.clip(i, 0, H - 1), np.clip(j, 0, W - 1), ok


def read_published(path):
    a = np.frombuffer(gzip.decompress(Path(path).read_bytes()), dtype="<u2").reshape(H, W)
    return a != NODATA


def read_erddap(path):
    d = netCDF4.Dataset(path)
    out = {}
    lat = d["latitude"][:]
    lon = d["longitude"][:]
    lon = np.where(lon > 180, lon - 360, lon)
    for v in ("pseudo_nitzschia", "particulate_domoic", "cellular_domoic"):
        x = np.ma.filled(d[v][0].astype(np.float64), np.nan)
        if lat[0] > lat[-1]:
            x = x[::-1]
        valid = np.isfinite(x)
        # place on the published grid (same 0.03 deg lattice)
        g = np.zeros((H, W), dtype=bool)
        la = np.sort(lat)
        ii = np.floor((la - LAT0) / DLAT + 0.5).astype(int)
        jj = np.floor((lon - LON0) / DLON + 0.5).astype(int)
        mi, mj = (ii >= 0) & (ii < H), (jj >= 0) & (jj < W)
        g[np.ix_(ii[mi], jj[mj])] = valid[np.ix_(mi, mj)]
        out[v] = g
    return out


def water_fraction(ci, cj, cok, ocean):
    tot = np.zeros((H, W)); wat = np.zeros((H, W))
    np.add.at(tot, (ci[cok], cj[cok]), 1)
    np.add.at(wat, (ci[cok], cj[cok]), ocean[cok])
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tot > 0, wat / tot, np.nan)


CLASSES = ["covered (bloom and toxin values)", "toxin mask (bloom value only)", "shoreline cell (mostly land in 3 km cell)", "ocean cell without any value", "toxin value without bloom value"]


def classify(pn, pda, wf_cell, ci, cj):
    PN, PDA = pn[ci, cj], pda[ci, cj]
    WF = wf_cell[ci, cj]
    cls = np.full(ci.shape, 3, dtype=np.int8)
    cls[PN & PDA] = 0
    cls[PN & ~PDA] = 1
    none = ~PN & ~PDA
    cls[none & (WF < 0.5)] = 2
    cls[~PN & PDA] = 4
    return cls


def table(cls, dist, reg, sfbay, mask):
    rows = []
    for ri, (name, _, _) in enumerate(REGIONS + [("Inside San Francisco Bay", 0, 0), ("All California", 0, 0)]):
        if ri < len(REGIONS):
            sel = mask & (reg == ri) & ~sfbay
        elif ri == len(REGIONS):
            sel = mask & sfbay
        else:
            sel = mask & (reg >= 0)
        for lo, hi in BANDS + [(0, 10)]:
            s = sel & (dist >= lo) & (dist < hi)
            n = int(s.sum())
            if n == 0:
                continue
            c = np.bincount(cls[s], minlength=len(CLASSES)) / n
            rows.append({"region": name, "band_km": f"{lo}-{hi}", "area_km2": round(n * PIX_KM2, 1), **{CLASSES[k]: round(float(c[k]) * 100, 1) for k in range(len(CLASSES))},
                         "with_bloom_value_%": round(float(c[0] + c[1] + 0) * 100, 1), "with_toxin_value_%": round(float(c[0] + c[4]) * 100, 1)})
    return rows


if __name__ == "__main__":
    lat, lon, ocean = load_etopo()
    dist = distance_km(lat, lon, ocean)
    reg, sfbay = region_index(lat, lon)
    ci, cj, cok = cell_of(lat, lon)
    PIX_KM2 = (abs(lat[1] - lat[0]) * 111.32) * (abs(lon[1] - lon[0]) * 111.32 * np.cos(np.radians(37.0)))
    wf = water_fraction(ci, cj, cok, ocean)
    mask = ocean & cok & (dist < 25) & (reg >= 0)
    np.savez_compressed(OUT / "geo.npz", lat=lat, lon=lon, dist=np.nan_to_num(dist, nan=-1), reg=reg, sfbay=sfbay, ocean=ocean)

    cur = HERE / "cur"
    pn, pda, cda = (read_published(cur / f"lead0_{v}.u16.gz") for v in ("pseudo_nitzschia", "particulate_domoic", "cellular_domoic"))
    cls = classify(pn, pda, wf, ci, cj)
    res = {"pix_km2": PIX_KM2, "current": table(cls, dist, reg, sfbay, mask)}
    # pDA vs cDA mask agreement on the published grid
    res["pda_vs_cda_cells_differ"] = int((pda != cda).sum())

    # domain: offshore cells (25-200 km) without values, inside the region latitudes
    off = ocean & cok & (dist >= 25) & (dist < 200) & (reg >= 0)
    res["offshore_25_200km_without_bloom_value_%"] = round(float((~pn[ci[off], cj[off]]).mean() * 100), 2)

    # parity: published grid vs ERDDAP for the same valid day
    par = HERE / "charm_hist" / "2026-10-10.nc"
    if par.exists():
        e = read_erddap(par)
        res["parity_2026-10-10"] = {v: int((e[v] != p).sum()) for v, p in (("pseudo_nitzschia", pn), ("particulate_domoic", pda), ("cellular_domoic", cda))}

    # through time: ERDDAP nowcasts, nearshore 0-10 km
    hist = []
    for f in sorted((HERE / "charm_hist").glob("20*.nc")):
        try:
            e = read_erddap(f)
        except Exception as ex:  # noqa: BLE001
            hist.append({"date": f.stem, "error": str(ex)[:80]}); continue
        c2 = classify(e["pseudo_nitzschia"], e["particulate_domoic"], wf, ci, cj)
        row = {"date": f.stem}
        for lo, hi in [(0, 1), (0, 3), (0, 5), (0, 10)]:
            s = mask & (dist >= lo) & (dist < hi)
            row[f"bloom_{lo}-{hi}"] = round(float(np.isin(c2[s], [0, 1]).mean() * 100), 1)
            row[f"toxin_{lo}-{hi}"] = round(float(np.isin(c2[s], [0, 4]).mean() * 100), 1)
        s = mask & (dist < 10)
        for ri, (name, _, _) in enumerate(REGIONS):
            ss = s & (reg == ri) & ~sfbay
            row[f"toxin_0-10_{name}"] = round(float(np.isin(c2[ss], [0, 4]).mean() * 100), 1)
        row["offshore_bloom_%"] = round(float(e["pseudo_nitzschia"][ci[off], cj[off]].mean() * 100), 1)
        hist.append(row)
    res["history"] = hist
    (OUT / "charm_audit.json").write_text(json.dumps(res, indent=1))

    # coverage map, current nowcast
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    img = np.full(ocean.shape, np.nan)
    show = ocean & cok & (dist < 25) & (reg >= 0)
    img[show] = cls[show]
    land = ~ocean
    fig, ax = plt.subplots(figsize=(7.5, 10), dpi=150)
    ax.imshow(np.where(land, 1, np.nan), origin="lower", extent=[lon[0], lon[-1], lat[0], lat[-1]], cmap=ListedColormap(["#d9d4c7"]), interpolation="nearest")
    cmap = ListedColormap(["#5b8fb9", "#e3a33b", "#8c8c8c", "#c0392b", "#7d3c98"])
    ax.imshow(img, origin="lower", extent=[lon[0], lon[-1], lat[0], lat[-1]], cmap=cmap, vmin=-0.5, vmax=4.5, interpolation="nearest")
    for _, lo, hi in REGIONS:
        ax.axhline(lo, color="#333", lw=0.4, ls=":")
    ax.set_xlim(-124.8, -117.0); ax.set_ylim(32.4, 42.1); ax.set_aspect(1 / np.cos(np.radians(37)))
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=cmap(k), label=CLASSES[k]) for k in range(4)], loc="lower left", fontsize=7, frameon=True)
    ax.set_title("C-HARM v3.1 nowcast valid 2026-10-10: coastal water within 25 km of shore", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "coverage_map.png"); plt.close(fig)
    for name, box in {"monterey": (-122.3, -121.7, 36.45, 37.05), "sfbay": (-122.9, -121.9, 37.35, 38.25), "humboldt": (-124.5, -123.8, 40.4, 41.2), "socal": (-119.0, -117.1, 32.6, 34.2)}.items():
        fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
        ax.imshow(np.where(land, 1, np.nan), origin="lower", extent=[lon[0], lon[-1], lat[0], lat[-1]], cmap=ListedColormap(["#d9d4c7"]), interpolation="nearest")
        ax.imshow(img, origin="lower", extent=[lon[0], lon[-1], lat[0], lat[-1]], cmap=cmap, vmin=-0.5, vmax=4.5, interpolation="nearest")
        ax.set_xlim(box[0], box[1]); ax.set_ylim(box[2], box[3]); ax.set_aspect(1 / np.cos(np.radians((box[2] + box[3]) / 2)))
        ax.set_title(f"{name}: C-HARM nowcast 2026-10-10", fontsize=8)
        fig.tight_layout(); fig.savefig(OUT / f"coverage_{name}.png"); plt.close(fig)
    print(json.dumps({k: v for k, v in res.items() if k not in ("current", "history")}, indent=1))
