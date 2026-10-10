"""Multi-sensor latest view: Sentinel-3 OLCI 300 m where it has a recent observation, VIIRS
750 m where it does not.

This is a display, not a merged product:
- every pixel shows exactly one sensor's own published value, on that sensor's own grid
  (tiles are drawn with each grid's native cell edges; VIIRS cells stay 750 m);
- nothing is averaged, blended, resampled or bias-corrected (no validated harmonization
  between the two algorithms exists for this coast);
- the member layers (values, ages, tiles) stay published and are what a reader inspects:
  the browser applies the same rule to the two member grids at a point.

Rule, per Sentinel-3 pixel: Sentinel-3 is shown if it has a valid pixel in its 7-day latest
clear view, unless the VIIRS cell there was observed more than PREFER_PRIMARY_WITHIN_DAYS
days more recently. Otherwise VIIRS is shown if it has a valid cell. Otherwise nothing.

Agreement: on days both sensors observed a VIIRS cell, the geometric mean of the valid
Sentinel-3 pixels inside it (at least MIN_PRIMARY_PIXELS of about 9) is compared with the
VIIRS value, in log10. This describes the seam a reader sees; it is not a validation of
either sensor.
"""

from __future__ import annotations

import hashlib
from datetime import date

import numpy as np

from ..context import RunContext
from ..models import (
    LayerArtifact,
    MultiSensorInfo,
    Provenance,
    QCCheck,
    SensorAgreement,
    SensorCoverage,
    SensorMember,
    SensorShown,
    TileLayer,
    TimeInfo,
    ValueGrid,
)
from ..process.palette import AGE_COLOURS, CHLOROPHYLL, SENSOR_COLOURS, palette_indices
from ..process.tiles import render_stack
from . import satellite as sat

LAYER_ID = "multisensor_chl_latest"
RULE_VERSION = "multisensor-1"
PREFER_PRIMARY_WITHIN_DAYS = 2
MIN_PRIMARY_PIXELS = 5
MIN_AGREEMENT_CELLS = 30

RULE = (
    "At each pixel: Sentinel-3 OLCI (300 m) if it has a valid observation in the last 7 days, unless "
    f"VIIRS observed that place more than {PREFER_PRIMARY_WITHIN_DAYS} days more recently; otherwise VIIRS (750 m) "
    "if it has one; otherwise no observation. Each pixel shows one sensor's own value and date; nothing is averaged."
)
AGREEMENT_METHOD = (
    "Same-day pairs: for each VIIRS 750 m cell observed on a day Sentinel-3 also observed, the geometric mean of the "
    f"valid Sentinel-3 300 m pixels inside it (at least {MIN_PRIMARY_PIXELS}) against the VIIRS value, in log10. "
    "Overpasses differ by a few hours. Describes the visible seam; not a validation of either sensor."
)
CAVEAT_MULTI = (
    "Multi-sensor display: Sentinel-3 300 m where available, VIIRS 750 m elsewhere. The two sensors use different "
    "algorithms and can differ at the same place and day (see the agreement figures); a colour change at a sensor "
    "boundary may be the sensors, not the water."
)


def target_of(g: ValueGrid) -> sat.Target:
    return sat.Target(g.lat_first, g.lon_first, g.lon_step, g.height, g.width)


def obs_dates(lyr: LayerArtifact, out_dir) -> np.ndarray:
    """Observation date (ordinal) of each pixel of a latest-clear-view layer, NaN if none."""
    assert lyr.composite
    age = sat.load_published(out_dir, lyr.composite.age_grid)
    return date.fromisoformat(lyr.composite.reference_date).toordinal() - age


def pick(o_val, o_date, v_val, v_date, tol: int = PREFER_PRIMARY_WITHIN_DAYS) -> np.ndarray:
    """0 = no observation, 1 = Sentinel-3 (primary), 2 = VIIRS (secondary)."""
    o_ok, v_ok = np.isfinite(o_val), np.isfinite(v_val)
    with np.errstate(invalid="ignore"):
        primary = o_ok & (~v_ok | ~(v_date > o_date + tol))
    return np.where(primary, 1, np.where(v_ok, 2, 0)).astype(np.uint8)


def cell_map(t_from: sat.Target, t_to: sat.Target) -> tuple[np.ndarray, np.ndarray]:
    """Row/col of the t_to cell containing each t_from cell centre (-1 outside)."""
    lats, lons = t_from.lats(), t_from.lons()
    g = t_to.grid
    rows, _ = g.cell_index(lats, np.full_like(lats, g.lon_first))
    _, cols = g.cell_index(np.full_like(lons, g.lat_first), lons)
    return rows, cols


def _sample(arr: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    ok = (rows[:, None] >= 0) & (cols[None, :] >= 0)
    out = np.full(ok.shape, np.nan)
    out[ok] = arr[np.broadcast_to(rows[:, None], ok.shape)[ok], np.broadcast_to(cols[None, :], ok.shape)[ok]]
    return out


def agreement(
    days: list[tuple[str, np.ndarray]], t_o: sat.Target, v_val: np.ndarray, v_date: np.ndarray, t_v: sat.Target
) -> list[SensorAgreement]:
    rows, cols = cell_map(t_o, t_v)
    flat = (rows[:, None] * t_v.width + cols[None, :]).ravel()
    inside = ((rows[:, None] >= 0) & (cols[None, :] >= 0)).ravel()
    n = t_v.height * t_v.width
    pairs: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]] = []
    for d, vals in days:
        lv = np.log10(np.where(np.isfinite(vals) & (vals > 0), vals, np.nan)).ravel()
        ok = inside & np.isfinite(lv)
        cnt = np.bincount(flat[ok], minlength=n)
        sm = np.bincount(flat[ok], weights=lv[ok], minlength=n)
        same = (v_date.ravel() == date.fromisoformat(d).toordinal()) & np.isfinite(v_val.ravel()) & (cnt >= MIN_PRIMARY_PIXELS)
        idx = np.nonzero(same)[0]
        if idx.size:
            pairs.append((idx, sm[idx] / cnt[idx], np.log10(v_val.ravel()[idx]), d))
    lat_v, lon_v = t_v.lats(), t_v.lons()
    _, _, regions = sat._load_reference()
    out = []
    for reg in [{"id": "domain", "label": "Whole domain", "bounds": None}] + regions:
        o_all, v_all, ds = [], [], set()
        for idx, lo, lvv, d in pairs:
            if reg["bounds"]:
                (w, s), (e, nn) = reg["bounds"]
                la, lo_ = lat_v[idx // t_v.width], lon_v[idx % t_v.width]
                sel = (la >= s) & (la <= nn) & (lo_ >= w) & (lo_ <= e)
            else:
                sel = np.ones(idx.size, bool)
            if sel.any():
                o_all.append(lo[sel])
                v_all.append(lvv[sel])
                ds.add(d)
        o = np.concatenate(o_all) if o_all else np.array([])
        v = np.concatenate(v_all) if v_all else np.array([])
        enough = o.size >= MIN_AGREEMENT_CELLS
        out.append(
            SensorAgreement(
                region_id=reg["id"], label=reg["label"], n_cells=int(o.size), dates=sorted(ds),
                median_log10_ratio=round(float(np.median(o - v)), 3) if enough else None,
                rmsd_log10=round(float(np.sqrt(np.mean((o - v) ** 2))), 3) if enough else None,
                pearson_r_log10=round(float(np.corrcoef(o, v)[0, 1]), 3) if enough and o.std() > 0 and v.std() > 0 else None,
            )
        )
    return out


def coverage_comparison(o_val, t_o, v_val, t_v, v_on_o) -> tuple[list[SensorCoverage], np.ndarray]:
    combined = np.where(np.isfinite(o_val), o_val, v_on_o)
    cp, cs, cc = sat.coverage(o_val, t_o), sat.coverage(v_val, t_v), sat.coverage(combined, t_o)
    rows = [("domain", "Whole domain", cc.domain_observed_fraction, cp.domain_observed_fraction, cs.domain_observed_fraction, cc.domain_reference_cells or 0)]
    sec = {r.region_id: r for r in cs.regions}
    pri = {r.region_id: r for r in cp.regions}
    for r in cc.regions:
        rows.append((r.region_id, r.label, r.observed_fraction, pri[r.region_id].observed_fraction if r.region_id in pri else 0.0,
                     sec[r.region_id].observed_fraction if r.region_id in sec else 0.0, r.reference_cells))
    out = [
        SensorCoverage(
            region_id=rid, label=label, reference_cells=n,
            primary_fraction=p, secondary_fraction=s, combined_fraction=c, secondary_only_fraction=round(max(0.0, c - p), 4),
        )
        for rid, label, c, p, s, n in rows
    ]
    return out, combined


def _shown(order: int, ages: np.ndarray) -> SensorShown:
    a = ages[np.isfinite(ages)]
    return SensorShown(order=order, pixels=int(a.size), median_age_days=float(np.median(a)) if a.size else None, max_age_days=int(a.max()) if a.size else None)


def build(ctx: RunContext, olci: LayerArtifact, viirs: LayerArtifact, olci_days: list[LayerArtifact], zooms: range) -> tuple[LayerArtifact, int]:
    out = ctx.out_dir
    assert olci.grid and viirs.grid and olci.composite and viirs.composite
    t_o, t_v = target_of(olci.grid), target_of(viirs.grid)
    o_val, v_val = sat.load_published(out, olci.grid), sat.load_published(out, viirs.grid)
    o_date, v_date = obs_dates(olci, out), obs_dates(viirs, out)
    rows, cols = cell_map(t_o, t_v)
    v_on_o, vd_on_o = _sample(v_val, rows, cols), _sample(v_date, rows, cols)
    code = pick(o_val, o_date, v_on_o, vd_on_o)
    top = np.where(code == 1, o_val, np.nan)
    ref = ctx.now.date().toordinal()

    h = hashlib.sha256(f"{RULE_VERSION}|{olci.grid.url}|{viirs.grid.url}|{ref}".encode()).hexdigest()[:10]
    base = f"satellite/multisensor/latest-{ctx.now.date().isoformat()}-{h}"
    bounds = sat._bounds(t_o)
    chl_idx = lambda v: palette_indices(v, CHLOROPHYLL)[0]  # noqa: E731
    _, chl_colours = palette_indices(np.array([1.0]), CHLOROPHYLL)
    ts = render_stack([(top, t_o.grid, chl_idx), (v_val, t_v.grid, chl_idx)], bounds, zooms, chl_colours, out / base / "tiles")
    one = lambda k: (lambda v: np.where(np.isfinite(v), k, 0).astype(np.uint8))  # noqa: E731
    sensor_colours = [(0, 0, 0)] + [tuple(int(c[i : i + 2], 16) for i in (1, 3, 5)) for c in SENSOR_COLOURS]
    sts = render_stack([(top, t_o.grid, one(1)), (v_val, t_v.grid, one(2))], bounds, zooms, sensor_colours, out / base / "sensor-tiles")
    age_top, age_v = np.where(code == 1, ref - o_date, np.nan), ref - v_date
    age_idx = lambda v: np.where(np.isfinite(v), np.clip(np.nan_to_num(v), 0, len(AGE_COLOURS) - 1) + 1, 0).astype(np.uint8)  # noqa: E731
    age_colours = [(0, 0, 0)] + [tuple(int(c[i : i + 2], 16) for i in (1, 3, 5)) for c in AGE_COLOURS]
    ats = render_stack([(age_top, t_o.grid, age_idx), (age_v, t_v.grid, age_idx)], bounds, zooms, age_colours, out / base / "age-tiles")

    w, s, e, n = bounds

    def tl(sub: str, t, what: str) -> TileLayer:
        return TileLayer(
            url_template=f"{base}/{sub}/{{z}}/{{x}}/{{y}}.png", relative=True, max_native_zoom=zooms.stop - 1, min_zoom=zooms.start,
            legend_url=None, legend_verified=False, bounds_lnglat=[round(w, 5), round(s, 5), round(e, 5), round(n, 5)],
            n_tiles=t.n_tiles, sample_tiles=t.tiles[:: max(1, len(t.tiles) // 5)][:5], date_selection=what,
        )

    comparison, combined = coverage_comparison(o_val, t_o, v_val, t_v, v_on_o)
    days = [(d.time.observed_date, sat.load_published(out, d.grid)) for d in olci_days if d.grid and d.time.observed_date]
    agree = agreement(days, t_o, v_val, v_date, t_v)

    shown_dates = np.concatenate([o_date[code == 1], vd_on_o[code == 2]])
    shown_dates = shown_dates[np.isfinite(shown_dates)]
    newest = date.fromordinal(int(shown_dates.max())).isoformat() if shown_dates.size else olci.time.observed_date
    n_pri, n_sec = int((code == 1).sum()), int((code == 2).sum())
    checks = [
        QCCheck(name="one_sensor_per_pixel", passed=True, detail=f"{n_pri} pixels Sentinel-3, {n_sec} pixels VIIRS (on the 300 m lattice); no pixel combines values"),
        QCCheck(name="members_unchanged", passed=True, detail=f"values and dates read from {olci.layer_id} and {viirs.layer_id} as published"),
    ]
    layer = LayerArtifact(
        layer_id=LAYER_ID, group_id=sat.GROUP_ID, product_class="observation",
        title="Satellite chlorophyll-a, multi-sensor latest view (Sentinel-3 300 m, VIIRS 750 m where Sentinel-3 has none)",
        short_title="Chlorophyll · multi-sensor", variable="chlorophyll_a", units="mg m-3",
        description="Display of the most informative recent observation at each pixel from two sensors, each at its own resolution and date. " + RULE,
        time=TimeInfo(observed_date=newest, valid_date=newest, observed_times=sorted(set(olci.time.observed_times) | set(viirs.time.observed_times))),
        freshness=sat.FRESHNESS, grid=None,
        tiles=tl("tiles", ts, "Rendered by CoastWatch: each tile pixel shows one sensor's published value on that sensor's own grid (nearest cell); Sentinel-3 over VIIRS by the rule."),
        palette=CHLOROPHYLL, caveats=[CAVEAT_MULTI, sat.CAVEAT_COMPOSITE.format(n=sat.WINDOW_DAYS)] + list(sat.CAVEATS),
        qc=sat._qc(combined, checks), platforms=sorted(set(olci.platforms) | set(viirs.platforms)),
        coverage=sat.coverage(combined, t_o),
        multisensor=MultiSensorInfo(
            reference_date=ctx.now.date().isoformat(), rule=RULE, prefer_primary_within_days=PREFER_PRIMARY_WITHIN_DAYS,
            members=[
                SensorMember(order=0, layer_id=olci.layer_id, label="Sentinel-3 OLCI 300 m", native_resolution_m=olci.native_resolution_m or 300),
                SensorMember(order=1, layer_id=viirs.layer_id, label="VIIRS 750 m", native_resolution_m=viirs.native_resolution_m or 750),
            ],
            sensor_tiles=tl("sensor-tiles", sts, "Which sensor is shown at each pixel (categorical)."),
            age_tiles=tl("age-tiles", ats, "Age in days of the observation shown at each pixel (categorical)."),
            coverage_comparison=comparison, agreement_method=AGREEMENT_METHOD, agreement=agree,
            shown=[_shown(0, age_top[code == 1]), _shown(1, (ref - vd_on_o)[code == 2])],
        ),
        provenance=Provenance(
            source_id=sat.SOURCE_ID, source_name="CoastWatch multi-sensor display of NOAA CoastWatch Sentinel-3 OLCI and VIIRS chlorophyll-a",
            source_url=olci.provenance.source_url, dataset_id=f"{olci.provenance.dataset_id},{viirs.provenance.dataset_id}",
            institution="NOAA CoastWatch (data); CoastWatch pipeline (display)", license=olci.provenance.license,
            retrieved_at=ctx.now_iso, request_urls=[], upstream_metadata={"members": f"{olci.layer_id} ({olci.provenance.pipeline_run_id}), {viirs.layer_id} ({viirs.provenance.pipeline_run_id})", "rule": RULE_VERSION},
            pipeline_version=ctx.pipeline_version, pipeline_run_id=ctx.run_id,
        ),
    )
    return layer, ts.bytes + sts.bytes + ats.bytes
