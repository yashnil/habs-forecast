"""`cwp derive`: derived chlorophyll layers in v1/derived/ (never in manifest.json).

- chl_estimate_latest: chlorophyll estimated where the latest clear view (Sentinel-3 OLCI 300 m) has
  no observation: clouds, the nearshore strip and bays (derived/estimate.py);
- chl_outlook_8d: the mean chlorophyll of the next 8 days, climatology + AR(1) (derived/outlook.py).

The released web app reads only manifest.json and rejects unknown manifest fields, so these layers
are listed in their own index, v1/derived/index.json, which only clients that know it read. The
background state the estimate carries between runs is listed in v1/derived/state.json.

Grids use the manifest's satellite grid format (chunked uint16 little-endian + gzip; log10 values
for chlorophyll; plain values for the uncertainty grids, RMSE in log10(mg m-3)). Observed pixels
are never written into the estimate grid.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from datetime import date
from pathlib import Path

import numpy as np

from ..context import RunContext
from ..models import GridChunks, LayerArtifact, Manifest, ValueGrid
from ..process import grid as gridcodec
from ..process.tiles import write_chunks
from ..publish.files import write_json
from ..sources import satellite
from . import estimate as E
from . import outlook as O

DERIVED = "derived"
INDEX = "index.json"
STATE = "state.json"
VALIDATION = satellite.REPO / "data" / "reference" / "chl_estimate_validation.json"
CHARM_PIER_VALIDATION = satellite.REPO / "data" / "reference" / "charm_nearshore_validation.json"
LOG_RANGE = satellite.LOG_RANGE
UNC_RANGE = (0.0, 3.0)  # log10 RMSE
WEIGHT_RANGE = (0.0, 32.0)
STATE_VERSION = f"estimate-state-1|tau={E.TAU_DAYS}|{satellite.PROCESSING_VERSION}"
CHUNK = satellite.CHUNK
OLCI_LATEST = f"{satellite.OLCI.key}_chl_latest"


def _codec(lo: float, hi: float):
    return lambda block: gridcodec.encode(block, lo, hi)[0]


def _grid(base: str, present: list[str], width: int, height: int, lat_first: float, lat_step: float, lon_first: float, lon_step: float, lo: float, hi: float, log: bool) -> dict:
    scale = (hi - lo) / gridcodec.MAX_CODE
    tmpl = f"{base}/{{row}}_{{col}}.u16.gz"
    g = ValueGrid(
        url=tmpl, width=width, height=height, lat_first=lat_first, lat_step=lat_step, lon_first=lon_first, lon_step=lon_step,
        scale_factor=scale, add_offset=lo, max_quantization_error=scale / 2, transform="log10" if log else "none",
        chunks=GridChunks(rows=CHUNK, cols=CHUNK, url_template=tmpl, present=present),
    )
    return g.model_dump(mode="json")


def write_grid(out: Path, base: str, values: np.ndarray, geo: tuple[float, float, float, float], lo: float, hi: float, log: bool) -> tuple[dict, int]:
    present, nbytes = write_chunks(_codec(lo, hi), values, CHUNK, CHUNK, out / base)
    h, w = values.shape
    return _grid(base, present, w, h, geo[0], geo[1], geo[2], geo[3], lo, hi, log), nbytes


def read_grid(out: Path, g: dict) -> np.ndarray:
    """Values as stored (log10 for a log10 grid: no 10** here)."""
    vg = ValueGrid.model_validate(g)
    arr = np.full((vg.height, vg.width), np.nan)
    assert vg.chunks is not None
    for key in vg.chunks.present:
        r, c = (int(x) for x in key.split("_"))
        h = min(vg.chunks.rows, vg.height - r * vg.chunks.rows)
        w = min(vg.chunks.cols, vg.width - c * vg.chunks.cols)
        blob = (out / vg.chunks.url_template.format(row=r, col=c)).read_bytes()
        arr[r * vg.chunks.rows : r * vg.chunks.rows + h, c * vg.chunks.cols : c * vg.chunks.cols + w] = gridcodec.decode(blob, w, h, vg.scale_factor, vg.add_offset)
    return arr


def _read_json(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _target_of(lyr: LayerArtifact) -> satellite.Target:
    g = lyr.grid
    assert g is not None
    return satellite.Target(g.lat_first, g.lon_first, g.lon_step, g.height, g.width)


def _log10(v: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.isfinite(v) & (v > 0), np.log10(np.where(v > 0, v, 1.0)), np.nan)


def _h(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:10]


# ---------------------------------------------------------------- background state
def advance_state(ctx: RunContext, target: satellite.Target, day_layers: list[LayerArtifact], notes: list[str]) -> tuple[np.ndarray, np.ndarray, str | None, dict | None, int]:
    """Background (log10) and weight grids through the newest published day: the previous state
    plus every published day newer than it. Returns (b, w, through, state json, bytes)."""
    out = ctx.out_dir
    prev = _read_json(out / DERIVED / STATE)
    geo = (target.lat_first, -target.step, target.lon_first, target.step)
    b = np.full((target.height, target.width), np.nan)
    w = np.zeros((target.height, target.width))
    through: date | None = None
    if prev and prev.get("version") == STATE_VERSION and prev.get("shape") == [target.height, target.width] and prev.get("lat_first") == target.lat_first and prev.get("lon_first") == target.lon_first:
        try:
            b = read_grid(out, prev["background"])
            w = np.nan_to_num(read_grid(out, prev["weight"]))
            through = date.fromisoformat(prev["through"])
        except Exception as e:  # a damaged state restarts from the published days
            notes.append(f"estimate: previous background state unreadable ({e}); rebuilt from the published days")
            b = np.full((target.height, target.width), np.nan)
            w = np.zeros((target.height, target.width))
            through = None
    elif prev:
        notes.append("estimate: previous background state made by other processing; rebuilt from the published days")
    days = sorted((lyr for lyr in day_layers if lyr.grid is not None), key=lambda lyr: lyr.time.observed_date or "")
    added = []
    for lyr in days:
        d = date.fromisoformat(lyr.time.observed_date or "")
        if through is not None and d <= through:
            continue
        vals = _log10(satellite.load_published(out, lyr.grid))
        b, w = E.update_background(b, w, float((d - through).days) if through else 0.0, vals)
        through = d
        added.append(d.isoformat())
    if through is None:
        return b, w, None, None, 0
    notes.append(f"estimate: background through {through} ({len(added)} new day(s): {', '.join(added) or 'none'})")
    if not added and prev and prev.get("version") == STATE_VERSION:
        return b, w, through.isoformat(), prev, 0
    base = f"{DERIVED}/state-{through.isoformat()}-{_h(STATE_VERSION, through.isoformat(), ctx.run_id)}"
    bg, n1 = write_grid(out, f"{base}/background", np.where(w > 1e-3, b, np.nan), geo, *LOG_RANGE, log=True)
    wg, n2 = write_grid(out, f"{base}/weight", np.where(w > 1e-3, np.minimum(w, WEIGHT_RANGE[1]), np.nan), geo, *WEIGHT_RANGE, log=False)
    state = {
        "version": STATE_VERSION,
        "description": "Background state of the chlorophyll estimate (derived/estimate.py): per-pixel exponentially weighted mean of valid log10 chlorophyll (e-folding "
        f"{E.TAU_DAYS:g} days) and its effective number of views. Pipeline input, not a product.",
        "through": through.isoformat(),
        "days_added_this_run": added,
        "shape": [target.height, target.width],
        "lat_first": target.lat_first,
        "lon_first": target.lon_first,
        "background": bg,
        "weight": wg,
    }
    return b, w, through.isoformat(), state, n1 + n2


# ---------------------------------------------------------------- geometry (cached per grid)
_GEOM: dict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}


def geometry(target: satellite.Target) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(fill area = water in the satellite scope, distance to shore km, bay mask, water)."""
    key = (target.lat_first, target.lon_first, target.step, target.height, target.width)
    if key not in _GEOM:
        lat_mid = target.lat_first - target.step * target.height / 2
        water = satellite.marine_water(target)
        area = water & satellite.coastal_mask(target)
        _GEOM.clear()
        _GEOM[key] = (area, E.distance_km(water, target.step, lat_mid), E.bay_mask(water, target.step, lat_mid), water)
    return _GEOM[key]


@__import__("functools").lru_cache(maxsize=1)
def load_validation(path: str = str(VALIDATION)) -> dict:
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------- layers
def estimate_layer(ctx: RunContext, latest: LayerArtifact, b: np.ndarray, w: np.ndarray, through: str, notes: list[str]) -> tuple[dict, int, dict]:
    out = ctx.out_dir
    target = _target_of(latest)
    area, dist, bay, _ = geometry(target)
    val = load_validation()
    lat_mid = target.lat_first - target.step * target.height / 2
    dy, dx = E.cell_km(target.step, lat_mid)
    obs = _log10(satellite.load_published(out, latest.grid))
    est, code = E.estimate(obs, b, w, area, dist, bay, dy, dx, val["nearshore_offset_log10"])
    est = np.clip(est, LOG_RANGE[0], LOG_RANGE[1])
    unc_table = np.array([math.nan] + [val["uncertainty_by_class"][k]["rmse_log10"] for k in E.CLASS_KEYS])
    unc = unc_table[code]
    assert not np.isfinite(est[np.isfinite(obs)]).any(), "an observed pixel was overwritten"
    geo = (target.lat_first, -target.step, target.lon_first, target.step)
    issued = ctx.now.date().isoformat()
    base = f"{DERIVED}/estimate-{issued}-{_h(latest.grid.url if latest.grid else '', through, ctx.run_id)}"
    g, n1 = write_grid(out, f"{base}/values", est, geo, *LOG_RANGE, log=True)
    ug, n2 = write_grid(out, f"{base}/uncertainty", unc, geo, *UNC_RANGE, log=False)
    stats = coverage_stats(obs, est, area, dist, bay, target)
    s = val["summary"]
    comp = latest.composite
    layer = {
        "id": "chl_estimate_latest",
        "kind": "chl_estimate",
        "label": "Chlorophyll, estimated where clouds or the coast hide it",
        "issued": issued,
        "based_on_days": [d.date for d in comp.days if d.pixels_used > 0] if comp else [latest.time.observed_date],
        "observed_layer": OLCI_LATEST,
        "units": "mg m-3",
        "method": (
            "Estimated, not observed. Filled only where the Sentinel-3 OLCI 300 m latest clear view has no value (clouds, fog, the nearshore strip, bays). "
            f"Each pixel's recent typical value (its own valid views, weighted toward the last {E.TAU_DAYS:g} days) plus today's departure from it, "
            f"spread from the observed pixels nearby on a {E.ANOMALY_L_KM:g} km scale (optimal interpolation). Pixels the satellite rarely sees, such as "
            f"the outermost ~300 m next to the shore, borrow the typical value of water within about {E.BACKGROUND_L_KM:g} km, raised by a measured shoreward "
            f"offset (x{10 ** val['nearshore_offset_log10']['open']:.1f} open coast, x{10 ** val['nearshore_offset_log10']['bay']:.1f} bays). Pixels without enough recent nearby data stay empty."
        ),
        "basis": f"Latest clear view {comp.oldest_observed_date if comp else ''}..{latest.time.observed_date} and every published OLCI day through {through} (background memory, e-folding {E.TAU_DAYS:g} days).",
        "validation": {
            "metric": "RMSE log10(mg m-3)",
            "by_band": {"0-0.6 km": s["clouds"]["0-0.6 km"]["estimate"]["rmse_log10"], "0.6-3 km": s["clouds"]["0.6-3 km"]["estimate"]["rmse_log10"],
                        "over 3 km": s["clouds"]["over 3 km"]["estimate"]["rmse_log10"], "bays": s["clouds"]["bays"]["estimate"]["rmse_log10"]},
            "baseline": "older view: the pixel's last valid observation 8-30 days old (the app itself shows nothing in these pixels)",
            "baseline_by_band": {"0-0.6 km": s["clouds"]["0-0.6 km"]["older_view"]["rmse_log10"], "0.6-3 km": s["clouds"]["0.6-3 km"]["older_view"]["rmse_log10"],
                                 "over 3 km": s["clouds"]["over 3 km"]["older_view"]["rmse_log10"], "bays": s["clouds"]["bays"]["older_view"]["rmse_log10"]},
            "never_observed_strip": {"0-0.3 km open coast": s["never_observed_0.3"]["0-0.6 km"]["estimate"]["rmse_log10"], "0-0.3 km bays": s["never_observed_0.3"]["bays"]["estimate"]["rmse_log10"],
                                     "0-0.6 km open coast": s["never_observed_0.6"]["0-0.6 km"]["estimate"]["rmse_log10"], "0-0.6 km bays": s["never_observed_0.6"]["bays"]["estimate"]["rmse_log10"]},
            "n": int(sum(v["estimate"]["n"] for sc in s.values() for v in sc.values())),
            "test_period": val["data"],
            "wins": f"Beats the older view in {val['estimate_beats_older_view']}.",
            "note": "Held-out: real observations hidden in realistic cloud shapes for a whole 7-day window (or, for the strip, on every day) and estimated causally. "
            "An error of 0.3 log10 is a factor of 2; 0.5 a factor of 3.2; 0.6 a factor of 4. Details: data/reference/chl_estimate_validation.json.",
        },
        "caveats": [
            "Estimated, not observed: never read it as a satellite measurement. Observed pixels are in the observed layer and are not repeated here.",
            "Chlorophyll is algae biomass, not toxin. It does not identify Pseudo-nitzschia or say whether shellfish or fish are safe.",
            "Next to the shore and in bays the error is larger (typically a factor of 2.5-4). In the outermost ~300 m strip the satellite almost never sees, the "
            "error was measured on the nearest pixels it does see; the strip itself cannot be checked against satellite data.",
            "Turbid bays and river plumes: the satellite chlorophyll algorithm reads sediment and dissolved matter as chlorophyll there, so estimates inherit that bias.",
            f"The background needs recent views: after a long cloudy spell (no view for several weeks) pixels stay empty rather than guessed. Coverage this run: {stats['strip_0_0.6km_filled_pct']}% of the 0-0.6 km strip gaps and {stats['bay_filled_pct']}% of bay gaps were filled.",
        ],
        "uncertainty_units": "RMSE log10(mg m-3) measured for the pixel's class (distance band, open coast or bay, own or borrowed background)",
        "grid": g,
        "uncertainty_grid": ug,
    }
    return layer, n1 + n2, stats


def coverage_stats(obs: np.ndarray, est: np.ndarray, area: np.ndarray, dist: np.ndarray, bay: np.ndarray, target: satellite.Target) -> dict:
    gap = area & ~np.isfinite(obs)
    filled = gap & np.isfinite(est)

    def pct(sel: np.ndarray) -> float:
        n = int((gap & sel).sum())
        return round(100.0 * int((filled & sel).sum()) / n, 1) if n else 100.0

    def obs_pct(sel: np.ndarray) -> float:
        n = int((area & sel).sum())
        return round(100.0 * int((area & sel & np.isfinite(obs)).sum()) / n, 1) if n else 0.0

    strip = (dist <= 0.6) & ~bay
    strip3 = (dist > 0.6) & (dist <= 3) & ~bay
    return {
        "water_cells": int(area.sum()),
        "observed_pct": obs_pct(np.ones_like(area)),
        "filled_pct_of_gaps": pct(np.ones_like(area)),
        "strip_0_0.6km_observed_pct": obs_pct(strip),
        "strip_0_0.6km_filled_pct": pct(strip),
        "strip_0.6_3km_observed_pct": obs_pct(strip3),
        "strip_0.6_3km_filled_pct": pct(strip3),
        "bay_observed_pct": obs_pct(bay),
        "bay_filled_pct": pct(bay),
        "shown_pct": round(100.0 * int((area & (np.isfinite(obs) | np.isfinite(est))).sum()) / max(1, int(area.sum())), 1),
        "shown_strip_0_0.6km_pct": round(100.0 * int((area & strip & (np.isfinite(obs) | np.isfinite(est))).sum()) / max(1, int((area & strip).sum())), 1),
        "shown_bay_pct": round(100.0 * int((area & bay & (np.isfinite(obs) | np.isfinite(est))).sum()) / max(1, int((area & bay).sum())), 1),
    }


def outlook_layer(ctx: RunContext, latest: LayerArtifact | None, notes: list[str]) -> tuple[dict, int, dict]:
    out = ctx.out_dir
    ref = O.load_reference()
    start, modis, urls = O.fetch_latest(ctx, ref)
    age = (ctx.now.date() - start).days
    if age > O.MAX_PREDICTOR_AGE_DAYS:
        raise ValueError(f"newest MODIS-Aqua 8-day composite starts {start} ({age} days ago): too old for an 8-day outlook")
    r = O.outlook(modis, start, ref)
    g = ref["grid"]
    geo = (g["lat_first"], g["lat_step"], g["lon_first"], g["lon_step"])
    issued = ctx.now.date()
    base = f"{DERIVED}/outlook-{issued.isoformat()}-{_h(start.isoformat(), issued.isoformat(), ctx.run_id)}"
    vg, n1 = write_grid(out, f"{base}/values", np.clip(O.to_grid(r["forecast_log10"], ref), *LOG_RANGE), geo, *LOG_RANGE, log=True)
    ug, n2 = write_grid(out, f"{base}/uncertainty", O.to_grid(r["uncertainty_log10"], ref), geo, *UNC_RANGE, log=False)
    sk = ref["skill_8d"]
    by_lead = ref["skill_by_lead"]
    p0, p1 = r["predictor_window"]
    bay_cov = strip_cov = None
    if latest is not None and latest.grid is not None:
        # does the 4 km footprint reach the bays? (share of in-scope bay water under a covered cell)
        target = _target_of(latest)
        area, _, bay, _ = geometry(target)
        fp = np.zeros((g["height"], g["width"]), bool)
        fp[ref["_px"][:, 0], ref["_px"][:, 1]] = True
        rr = np.round((target.lats() - g["lat_first"]) / g["lat_step"]).astype(int)
        cc = np.round((target.lons() - g["lon_first"]) / g["lon_step"]).astype(int)
        okr, okc = (rr >= 0) & (rr < g["height"]), (cc >= 0) & (cc < g["width"])
        under = np.zeros(area.shape, bool)
        under[np.ix_(okr, okc)] = fp[np.ix_(rr[okr], cc[okc])]
        bay_cov = round(100.0 * int((under & bay & area).sum()) / max(1, int((bay & area).sum())), 1)
        strip_cov = round(100.0 * int((under & area & ~bay).sum()) / max(1, int((area & ~bay).sum())), 1)
    late = (issued - r["valid_from"]).days
    layer = {
        "id": "chl_outlook_8d",
        "kind": "chl_outlook",
        "label": "Chlorophyll outlook, next 8 days",
        "issued": issued.isoformat(),
        "valid_from": r["valid_from"].isoformat(),
        "valid_to": r["valid_to"].isoformat(),
        "units": "mg m-3",
        "native_resolution_m": 4000,
        "method": (
            "Forecast of the MEAN chlorophyll over the 8 days, not a day-by-day map and not toxin. Each 4 km cell's typical chlorophyll for this 8-day period "
            "of the year (MODIS-Aqua climatology 2003-2021, smoothed over neighbouring periods), plus the cell's departure from its typical value in the newest "
            f"MODIS-Aqua 8-day composite ({p0.isoformat()}..{p1.isoformat()}), damped by the cell's AR(1) coefficient (about {ref['ar1_phi_by_lag_days']['8']:.2f} on average; "
            "fitted on 2003-2015). Cells the composite has no value for (cloud) get the climatology alone."
        ),
        "basis": (
            f"MODIS-Aqua 8-day composite {p0.isoformat()}..{p1.isoformat()} (NASA L3 4 km R2022 NRT, NOAA CoastWatch ERDDAP {O.MODIS_DATASET}), values in "
            f"{r['cells_with_anomaly']} of {r['cells']} cells; climatology and AR(1) from data/reference/chl_outlook_reference.json (MODIS-Aqua 8-day 4 km, 2003-01..2021-06)."
        ),
        "request_urls": urls,
        "skill": {
            "metric": "RMSE ln(mg m-3)",
            "model": sk["model"],
            "climatology": sk["climatology"],
            "persistence": sk["persistence"],
            "test_period": f"{sk['test_period']}; fitted on 2003-2015, variant chosen on 2016-2018",
            "n": sk["n"],
            "note": (
                f"Held-out MODIS-Aqua 8-day composites ({sk['n_composites']}); n = cells x composites. Climatology minus model RMSE: 95% CI {sk['climatology_minus_model_95ci']}; "
                f"persistence minus model: {sk['persistence_minus_model_95ci']} (block bootstrap). An RMSE of 0.76 ln is a typical error of a factor of about 2. "
                f"The test forecast the next composite exactly as this outlook does. At 16 days the same method scores {by_lead['16d']['clim+AR1']} "
                f"(climatology {by_lead['16d']['climatology']}); beyond that it is barely better than the climatology, so no longer outlook is offered."
            ),
        },
        "caveats": [
            "Outlook of the 8-day average, not a day-by-day forecast. Chlorophyll is algae biomass, not toxin.",
            "Typical error: a factor of about 2 either way. Read it as higher or lower than usual for the season, not as a precise value.",
            "Coverage: the 4 km cells of the research grid along the open coast (roughly the first 25-30 km offshore). Bays and harbours are not covered"
            + (f" ({bay_cov}% of bay water lies under a covered cell)" if bay_cov is not None else "")
            + ": there is no outlook for San Francisco, San Pablo, Humboldt, Tomales or San Diego Bay or Elkhorn Slough.",
            "4 km cells: the nearshore strip inside a cell can differ a lot from the cell's average.",
            "The climatology ends in mid-2021 and does not know about later changes. MODIS-Aqua is near the end of its mission; when its data stop, this outlook stops.",
        ]
        + ([f"This outlook's period began {late} day(s) before it was issued (satellite processing delay)."] if late > 0 else []),
        "coverage": {"cells": r["cells"], "cells_with_live_anomaly": r["cells_with_anomaly"], "open_coast_water_under_cells_pct": strip_cov, "bay_water_under_cells_pct": bay_cov},
        "uncertainty_units": "expected RMSE log10(mg m-3) per cell (held-out 2016-2021; the climatology's error where the composite has no value)",
        "grid": vg,
        "uncertainty_grid": ug,
    }
    return layer, n1 + n2, {"bay_under_cells_pct": bay_cov, "cells_with_anomaly": r["cells_with_anomaly"], "predictor": p0.isoformat(), "mean_anomaly_ln": r["mean_anomaly_ln"]}


def sst_layer(ctx: RunContext) -> tuple[dict, int, dict]:
    from . import sst as S

    domain = ctx.options.get("satellite_domain", satellite.CALIFORNIA)
    day, target, vals, urls = S.fetch_latest(ctx, domain)
    if not np.isfinite(vals).any():
        raise ValueError(f"MUR SST {day}: no value inside the coastal scope")
    geo = (target.lat_first, -target.step, target.lon_first, target.step)
    base = f"{DERIVED}/sst-{day.isoformat()}-{_h(urls[-1], ctx.run_id)}"
    g, n = write_grid(ctx.out_dir, f"{base}/values", vals, geo, *S.RANGE_C, log=False)
    f = vals[np.isfinite(vals)]
    layer = {
        "id": "sst_latest",
        "kind": "sst",
        "label": "Sea surface temperature",
        "units": "degC",
        "observed_dates": [day.isoformat()],
        "product_class": "analysis",
        "source": "JPL MUR SST v4.1 (Multi-scale Ultra-high Resolution analysis, GHRSST L4), via NOAA CoastWatch West Coast ERDDAP jplMURSST41",
        "source_url": f"{S.SERVER}/griddap/{S.DATASET}.html",
        "request_urls": urls,
        "resolution_km": 1.0,
        "method": "Daily gap-free analysis blending several satellite sensors (infrared and microwave) with in-situ data; published as received (quantized to 0.001 degC), clipped to the coastal ocean and connected bays.",
        "caveats": [
            "An analysis, not a direct observation: where clouds or fog hide the sea, MUR interpolates from other sensors and nearby days.",
            "The grid is 0.01 degrees (about 1 km), but under clouds the effective detail is coarser (microwave data are about 25 km); near shore and in bays values lean on interpolation.",
            "Surface skin/foundation temperature of the open water, not the temperature at depth where fish may hold.",
            f"Analysis day {day.isoformat()} (about one day behind real time).",
        ],
        "stats": {"cells": int(f.size), "min_c": round(float(f.min()), 2), "max_c": round(float(f.max()), 2)},
        "grid": g,
    }
    return layer, n, {"day": day.isoformat(), "cells": int(f.size)}


def charm_extension_entry(path: Path = CHARM_PIER_VALIDATION) -> dict | None:
    """Validation of showing the nearest valued C-HARM cell's nowcast at nearshore points (piers),
    from pipeline/scripts/validate_charm_nearshore.py. Not a layer: the readout's skill statement."""
    v = _read_json(path)
    if not v or v.get("status") != "validated":
        return None
    keys = ("metric", "model_variable", "n", "n_events", "n_stations", "dates", "base_rate", "auc", "auc_95ci", "brier_model", "brier_climatology", "bss", "bss_95ci",
            "mean_model_probability", "auc_climatology", "median_distance_km", "verdict", "verdict_sentence")
    skill = {k: {kk: ev.get(kk) for kk in keys if kk in ev} for k, ev in v["events"].items()}
    any_skill = any(ev.get("verdict") == "skill" for ev in v["events"].values())
    return {
        "id": "charm_nearshore_extension",
        "kind": "validation",
        "label": "Nearest C-HARM cell shown at the shore: checked against pier samples",
        "skill": skill,
        "has_demonstrated_skill": any_skill,
        "readout": "Model context from the nearest C-HARM cell offshore; it has shown skill at the piers for: "
        + ", ".join(k for k, ev in v["events"].items() if ev.get("verdict") == "skill")
        if any_skill
        else "Context only: the nearest C-HARM cell offshore has no demonstrated skill at the shore (checked against CalHABMAP pier samples).",
        "note": (
            f"Same-day C-HARM v3.1 nowcast of the nearest cell with a value vs CalHABMAP pier water samples, {v['period']['charm_record'][0]}..{v['period']['charm_record'][1]}. "
            "For particulate domoic acid above 500 ng/L the nearest-cell probability ranked samples worse than chance and was far less accurate than each pier's "
            "own month-by-month history; for Pseudo-nitzschia above 10,000 cells/L it ranked them only slightly better than chance and its probabilities were "
            "far too high. A model probability for water offshore is not a measurement at the pier and says nothing about toxin in seafood. "
            "Details: data/reference/charm_nearshore_validation.json."
        ),
        "caveats": v.get("caveats", []),
    }


# ---------------------------------------------------------------- run
def referenced_dirs(doc: dict | None) -> set[str]:
    """Top-level folders under derived/ an index or state document points to."""
    out: set[str] = set()

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k in ("url", "url_template") and isinstance(v, str) and v.startswith(f"{DERIVED}/"):
                    out.add(v.split("/")[1])
                else:
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(doc)
    return out


def prune(out: Path, keep_docs: list[dict | None]) -> None:
    keep = set().union(*(referenced_dirs(d) for d in keep_docs))
    base = out / DERIVED
    if not base.exists():
        return
    for p in base.iterdir():
        if p.is_dir() and p.name not in keep:
            shutil.rmtree(p)


def run(ctx: RunContext, manifest: Manifest | None = None) -> dict:
    """Build the derived layers from the published OLCI layers in ctx.out_dir. Never raises for
    a missing input: the index then lists the layers that could be built (maybe none)."""
    out = ctx.out_dir
    notes: list[str] = []
    if manifest is None:
        manifest = Manifest.model_validate_json((out / "manifest.json").read_text())
    prev_index = _read_json(out / DERIVED / INDEX)
    prev_state = _read_json(out / DERIVED / STATE)
    latest = next((lyr for lyr in manifest.layers if lyr.layer_id == OLCI_LATEST and lyr.grid is not None), None)
    day_layers = [lyr for lyr in manifest.layers if lyr.layer_id.startswith(f"{satellite.OLCI.key}_chl_2")]
    layers: list[dict] = []
    stats: dict = {}
    nbytes = 0
    state = prev_state
    if latest is None:
        notes.append("no Sentinel-3 OLCI latest clear view in the manifest: no estimate this run")
    else:
        newest = date.fromisoformat(latest.time.observed_date or "")
        if (ctx.now.date() - newest).days > satellite.WINDOW_DAYS:
            notes.append(f"OLCI latest clear view is from {newest}, older than {satellite.WINDOW_DAYS} days: no estimate this run")
        else:
            target = _target_of(latest)
            b, w, through, state, n = advance_state(ctx, target, day_layers, notes)
            nbytes += n
            if through is not None:
                try:
                    lyr, n, st = estimate_layer(ctx, latest, b, w, through, notes)
                    layers.append(lyr)
                    nbytes += n
                    stats["estimate"] = st
                except Exception as e:
                    notes.append(f"estimate not built: {e}")
    try:
        lyr, n, st = outlook_layer(ctx, latest, notes)
        layers.append(lyr)
        nbytes += n
        stats["outlook"] = st
    except Exception as e:
        notes.append(f"outlook not built: {e}")
    try:
        lyr, n, st = sst_layer(ctx)
        layers.append(lyr)
        nbytes += n
        stats["sst"] = st
    except Exception as e:
        notes.append(f"sea surface temperature not built: {e}")
    ext = charm_extension_entry()
    if ext:
        layers.append(ext)
    index = {
        "version": 1,
        "generated_at": ctx.now_iso,
        "pipeline_run_id": ctx.run_id,
        "pipeline_version": ctx.pipeline_version,
        "description": "Derived chlorophyll layers (estimates and outlooks). Not observations; each states its method, validation or skill and uncertainty.",
        "layers": layers,
        "notes": notes,
    }
    write_json(out / DERIVED / INDEX, index)
    if state is not None:
        write_json(out / DERIVED / STATE, state)
    prune(out, [index, state, prev_index, prev_state])
    return {"layers": [lyr["id"] for lyr in layers], "bytes_written": nbytes, "notes": notes, "stats": stats}


def summary_line(r: dict) -> str:
    return json.dumps({k: r[k] for k in ("layers", "bytes_written", "stats", "notes")}, indent=2)


