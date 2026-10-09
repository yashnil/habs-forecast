"""Render real California layers for the forecast-upgrade map lab.

Every overlay is reprojected to Web Mercator by nearest-neighbour sampling, so each screen
pixel shows one real source cell. The one exception is `charm_display`, an explicitly
labelled display interpolation between C-HARM cell centres that never extends beyond the
footprint of valid cells (see forecast-upgrade/04-map-experiments.md).

usage: python render_lab.py <scratch-dir> <lab-dir>
  scratch-dir holds samples/ (fetch_samples.py), state/ (statewide C-HARM) and wcofs/ (fetch_wcofs.py)
"""
from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from PIL import Image

R = 6378137.0
# C-HARM display classes, cw-probability-classes-v1 (design reset rev. 2). Display steps, not risk levels.
P_CLASSES = ["#3a385b", "#4c436a", "#5f4e79", "#735986", "#886492", "#9c709c", "#af7ea4", "#c28cab", "#d39cb3", "#e5abbc"]
# Chlorophyll-a, log10 0.1-30 mg/m3. Observation family: deep blue-teal to pale lime; avoids the
# violet model ramp, the amber reserved for official notices and the teal accent of station dots.
CHL_ANCHORS = ["#102a4c", "#14466e", "#126483", "#147f85", "#2a9a7c", "#5bb267", "#9bc851", "#d6dd5e", "#f3ef9c"]
CHL_RANGE = (0.1, 30.0)


def hex2rgb(h):
    return [int(h[i : i + 2], 16) for i in (1, 3, 5)]


def chl_rgb(v):
    t = (np.log10(np.clip(np.nan_to_num(v, nan=CHL_RANGE[0]), *CHL_RANGE)) - math.log10(CHL_RANGE[0])) / (math.log10(CHL_RANGE[1]) - math.log10(CHL_RANGE[0]))
    a = np.array([hex2rgb(c) for c in CHL_ANCHORS], float)
    x = t * (len(a) - 1)
    i = np.clip(np.floor(x).astype(int), 0, len(a) - 2)
    f = (x - i)[..., None]
    return (a[i] * (1 - f) + a[i + 1] * f).astype(np.uint8)


def prob_rgb(v):
    pal = np.array([hex2rgb(c) for c in P_CLASSES], np.uint8)
    return pal[np.clip((np.nan_to_num(v) * 10).astype(int), 0, 9)]


def merc_y(lat):
    return R * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))


def lat_of(y):
    return np.degrees(2 * np.arctan(np.exp(y / R)) - np.pi / 2)


def to_mercator(values, lats, lons, colour, upsample=2):
    """Nearest-neighbour reprojection of a regular lat/lon grid (cell centres) to Mercator."""
    lats = np.asarray(lats, float); lons = np.asarray(lons, float)
    if lats[0] > lats[-1]:
        lats = lats[::-1]; values = values[::-1]
    dlat = (lats[-1] - lats[0]) / (len(lats) - 1); dlon = (lons[-1] - lons[0]) / (len(lons) - 1)
    west, east = lons[0] - dlon / 2, lons[-1] + dlon / 2
    south, north = lats[0] - dlat / 2, lats[-1] + dlat / 2
    width = len(lons) * upsample
    yt, yb = merc_y(north), merc_y(south)
    height = int(round(width * (yt - yb) / (R * math.radians(east - west))))
    plat = lat_of(yt - (np.arange(height) + 0.5) / height * (yt - yb))
    plon = west + (np.arange(width) + 0.5) / width * (east - west)
    rows = np.clip(np.floor((plat - south) / dlat).astype(int), 0, len(lats) - 1)
    cols = np.clip(np.floor((plon - west) / dlon).astype(int), 0, len(lons) - 1)
    v = values[rows[:, None], cols[None, :]]
    ok = np.isfinite(v)
    rgba = np.zeros(v.shape + (4,), np.uint8)
    rgba[..., :3] = colour(np.where(ok, v, np.nan))
    rgba[..., 3] = np.where(ok, 255, 0)
    corners = [[west, north], [east, north], [east, south], [west, south]]
    return Image.fromarray(rgba, "RGBA"), corners


def display_interpolate(values, factor=4):
    """Bilinear interpolation between valid cell centres, for display only.

    Normalised convolution: invalid cells carry no weight, and the output is masked to the
    nearest-neighbour footprint of valid cells, so nothing is drawn where the source has no value.
    """
    from scipy.ndimage import zoom

    ok = np.isfinite(values)
    num = zoom(np.where(ok, values, 0.0), factor, order=1, mode="nearest", grid_mode=True)
    den = zoom(ok.astype(float), factor, order=1, mode="nearest", grid_mode=True)
    out = np.where(den > 1e-6, num / np.maximum(den, 1e-6), np.nan)
    footprint = np.repeat(np.repeat(ok, factor, 0), factor, 1)
    return np.where(footprint, out, np.nan)


def main(scratch: Path, lab: Path):
    out = lab / "layers"
    out.mkdir(parents=True, exist_ok=True)
    meta = {"layers": [], "currents": {}}

    def add(layer_id, img, corners, **kw):
        p = out / f"{layer_id}.png"
        img.save(p, optimize=True)
        meta["layers"].append({"id": layer_id, "url": f"layers/{p.name}", "corners": corners, "bytes": p.stat().st_size, **kw})

    # --- C-HARM statewide, native and display-interpolated, all leads ---
    for lead in range(4):
        with xr.open_dataset(scratch / "state" / f"charm_lead{lead}.nc") as d:
            v = d.particulate_domoic.values[0].astype(float)
            lats, lons = d.latitude.values, d.longitude.values - 360
            valid = str(d.time.values[0])[:10]
        img, c = to_mercator(v, lats, lons, prob_rgb, upsample=4)
        add(f"charm_native_lead{lead}", img, c, kind="model", product="C-HARM v3.1 particulate DA > 500 ng/L", native="0.03° (~3 km)", date=valid, lead=lead, render="nearest")
        vi = display_interpolate(v, 4)
        dl, dn = (lats[1] - lats[0]) / 4, (lons[1] - lons[0]) / 4
        il = lats[0] - (lats[1] - lats[0]) / 2 + dl / 2 + np.arange(vi.shape[0]) * dl
        io = lons[0] - (lons[1] - lons[0]) / 2 + dn / 2 + np.arange(vi.shape[1]) * dn
        img, c = to_mercator(vi, il, io, prob_rgb, upsample=1)
        add(f"charm_display_lead{lead}", img, c, kind="model", product="C-HARM v3.1 particulate DA > 500 ng/L", native="0.03° (~3 km)", date=valid, lead=lead, render="display interpolation between 3 km cell centres")

    # --- satellite chlorophyll: every downloaded day, per region ---
    var_of = {"olci300": "chlor_a", "viirs750": "chla", "viirs_n20": "chlor_a", "dineof": "chlor_a"}
    native = {"olci300_s3a_CI": "0.0025° (~250 m)", "olci300_s3a_DI": "0.0025° (~250 m)", "viirs750_1day": "0.0075° (~750 m)", "viirs_n20_4km": "0.0375° (~4 km)", "dineof_2km_sq": "0.0208° (~2 km), gap-filled"}
    product = {"olci300_s3a_CI": "Sentinel-3A OLCI chlorophyll-a, NOAA CoastWatch NRT", "olci300_s3a_DI": "Sentinel-3A OLCI chlorophyll-a, NOAA CoastWatch NRT", "viirs750_1day": "VIIRS chlorophyll-a, 1-day composite, CoastWatch West Coast", "viirs_n20_4km": "NOAA-20 VIIRS chlorophyll-a, NOAA CoastWatch NRT", "dineof_2km_sq": "VIIRS + OLCI chlorophyll-a, DINEOF gap-filled, science quality"}
    for reg in ["monterey", "north_coast", "socal_bight"]:
        for f in sorted(glob.glob(str(scratch / "samples" / reg / "*.nc"))):
            name = Path(f).stem
            lid, day = name[:-11], name[-10:]
            if lid not in native:
                continue
            with xr.open_dataset(f) as d:
                var = [x for x in d.data_vars][0]
                v = d[var].values.squeeze().astype(float)
                lats, lons = d.latitude.values, d.longitude.values
            ok = np.isfinite(v)
            img, c = to_mercator(v, lats, lons, chl_rgb, upsample=2 if "olci" in lid else 4)
            add(f"{reg}_{lid}_{day}", img, c, kind="observation", region=reg, product=product[lid], native=native[lid], date=day,
                valid_cells=int(ok.sum()), cells=int(v.size), median=float(np.nanmedian(v)) if ok.any() else None, render="nearest")

    # --- WCOFS surface currents: arrows (thinned) and 72 h surface drift from seeds ---
    files = sorted(glob.glob(str(scratch / "wcofs" / "wcofs_*_f*.nc")))
    steps = []
    for f in files:
        with xr.open_dataset(f) as d:
            steps.append((np.datetime64(d.attrs["valid_time"][:19]), d.u_eastward.values[0], d.v_northward.values[0], d.lat.values, d.lon.values))
    t0 = steps[0][0]
    hrs = np.array([(s[0] - t0) / np.timedelta64(1, "h") for s in steps])
    lat, lon = steps[0][3], steps[0][4]

    def arrows(stride, scale_deg, bbox):
        feats = []
        # one forecast hour for the arrows: about +24 h from the run (valid time shown in the lab)
        _, u, v, _, _ = steps[int(np.argmin(np.abs(hrs - 21)))]
        for i in range(0, len(lat), stride):
            for j in range(0, len(lon), stride):
                if not (bbox[1] <= lat[i] <= bbox[3] and bbox[0] <= lon[j] <= bbox[2]):
                    continue
                uu, vv = u[i, j], v[i, j]
                if not (np.isfinite(uu) and np.isfinite(vv)):
                    continue
                sp = float(np.hypot(uu, vv))
                dx, dy = uu / sp * scale_deg * min(1, sp / 0.3) / math.cos(math.radians(lat[i])), vv / sp * scale_deg * min(1, sp / 0.3)
                tip = [lon[j] + dx, lat[i] + dy]
                ang = math.atan2(dy, dx)
                h = scale_deg * 0.35
                l1 = [tip[0] - h * math.cos(ang - 0.5), tip[1] - h * math.sin(ang - 0.5)]
                l2 = [tip[0] - h * math.cos(ang + 0.5), tip[1] - h * math.sin(ang + 0.5)]
                feats.append({"type": "Feature", "properties": {"speed": round(sp, 3)}, "geometry": {"type": "MultiLineString", "coordinates": [[[float(lon[j]), float(lat[i])], [float(tip[0]), float(tip[1])]], [[float(l1[0]), float(l1[1])], [float(tip[0]), float(tip[1])], [float(l2[0]), float(l2[1])]]]}})
        return {"type": "FeatureCollection", "features": feats}

    def field_at(k, y, x):
        i = np.clip(np.searchsorted(lat, y) - 1, 0, len(lat) - 2); j = np.clip(np.searchsorted(lon, x) - 1, 0, len(lon) - 2)
        fy, fx = (y - lat[i]) / (lat[i + 1] - lat[i]), (x - lon[j]) / (lon[j + 1] - lon[j])
        out = []
        for arr in (steps[k][1], steps[k][2]):
            q = arr[i : i + 2, j : j + 2]
            if not np.all(np.isfinite(q)):
                return None
            out.append(q[0, 0] * (1 - fy) * (1 - fx) + q[0, 1] * (1 - fy) * fx + q[1, 0] * fy * (1 - fx) + q[1, 1] * fy * fx)
        return out

    def drift(seeds, hours=72, dt=0.5):
        feats = []
        for (x, y) in seeds:
            path = [[x, y]]; t = hrs[0]
            while t < min(hours + hrs[0], hrs[-1]):
                k = int(np.clip(np.searchsorted(hrs, t, side="right") - 1, 0, len(hrs) - 2))
                w = (t - hrs[k]) / (hrs[k + 1] - hrs[k])
                a, b = field_at(k, y, x), field_at(k + 1, y, x)
                if a is None or b is None:
                    break  # reached land or the model edge: stop, never extrapolate
                uu, vv = a[0] * (1 - w) + b[0] * w, a[1] * (1 - w) + b[1] * w
                y += vv * dt * 3600 / 111320.0
                x += uu * dt * 3600 / (111320.0 * math.cos(math.radians(y)))
                t += dt
                path.append([round(float(x), 5), round(float(y), 5)])
            feats.append({"type": "Feature", "properties": {"hours": round(float(t - hrs[0]), 1)}, "geometry": {"type": "LineString", "coordinates": path}})
        return {"type": "FeatureCollection", "features": feats}

    seeds = [(x, y) for y in np.arange(36.6, 37.0, 0.08) for x in np.arange(-122.3, -121.85, 0.08)]
    meta["currents"] = {
        "run": str(t0), "hours": hrs.tolist(), "native": "WCOFS 0.04° regular-grid output (~4 km; model grid ~4 km curvilinear)",
        "arrows_state": "currents_state.geojson", "arrows_monterey": "currents_monterey.geojson", "drift_monterey": "drift_monterey.geojson",
    }
    (lab / "currents_state.geojson").write_text(json.dumps(arrows(6, 0.16, (-126.5, 32, -117, 42.6))))
    (lab / "currents_monterey.geojson").write_text(json.dumps(arrows(1, 0.03, (-122.7, 36.3, -121.7, 37.3))))
    (lab / "drift_monterey.geojson").write_text(json.dumps(drift(seeds)))
    (lab / "layers.json").write_text(json.dumps(meta, indent=1))
    print(f"{len(meta['layers'])} layers, {sum(l['bytes'] for l in meta['layers']) / 1e6:.1f} MB")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
