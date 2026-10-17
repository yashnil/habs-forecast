import { describe, expect, it } from "vitest";
import type { LayerArtifact, Manifest, ValueGrid } from "@/generated/schema";
import type { CurrentField } from "@/lib/currents";
import { currentsHourly } from "@/lib/currents";
import { satelliteDays } from "@/lib/layers";
import {
  footprintOf,
  footprintOutline,
  multiSampler,
  obsArea,
  obsPoint,
  obsTransect,
  oceanCoverage,
  radarLayerFor,
  radarSampler,
  rangeIn,
  sameDayPairs,
  satLayerFor,
  satSampler,
  summarise,
  type Member,
  type S3Day,
} from "@/v4/research/obsGrid";
import { obsAreaCsv, obsPointCsv, obsTransectCsv } from "@/v4/research/exports";
import manifestJson from "../fixture-data/v1/manifest.json";

/**
 * Research over observation grids: exact pixels with their own dates, gaps kept as gaps (never filled,
 * never joined), counts of observed against listed pixels, the multi-sensor rule pixel by pixel, the
 * same-day inter-sensor pairs only where both sensors published that place and date, and exports that
 * say what an empty value means. Synthetic grids unless the fixture is named.
 */
const manifest = manifestJson as unknown as Manifest;
const N = 65535;
const RUN = { pipeline_run_id: "run-1", pipeline_version: "v9", generated_at: "2026-10-08T18:00:00Z" };

/** A w × h lattice with 1° cells centred on integer lat/lon (row 0 at `lat0`, rows going south when `south`). */
function lattice(w: number, h: number, lat0 = 0, lon0 = 0, step = 1, south = false): ValueGrid {
  return { url: "", encoding: "uint16le+gzip", width: w, height: h, lat_first: lat0, lat_step: south ? -step : step, lon_first: lon0, lon_step: step, scale_factor: 0.001, add_offset: 0, nodata: N, max_quantization_error: 0.0005, transform: "none", chunks: { rows: 512, cols: 512, url_template: "", present: ["0_0"] } } as unknown as ValueGrid;
}

/** values row-major (null = no observation); ages (days before `ref`) for a latest clear view. */
function member(id: string, g: ValueGrid, values: (number | null)[], ages: number[] | null, ref = "2026-10-08", extra: Partial<LayerArtifact> = {}): Member {
  const codes = values.map((v) => (v == null ? N : Math.round(v / 0.001)));
  const layer = {
    layer_id: id,
    group_id: "satellite_chlorophyll_hr",
    title: id,
    short_title: id,
    units: "mg m-3",
    platforms: id.startsWith("viirs") ? ["VIIRS", "NOAA-21"] : ["Sentinel-3A"],
    native_resolution_m: id.startsWith("viirs") ? 750 : 300,
    grid: g,
    time: { observed_date: ref },
    composite: ages ? { reference_date: ref, window_days: 7, age_grid: { ...g, nodata: N }, days: [], age_histogram: [], newest_observed_date: ref, oldest_observed_date: ref } : null,
    provenance: { dataset_id: `${id}-ds`, license: `${id} licence`, institution: "NOAA CoastWatch", source_url: "https://example.test" },
    qc: { checks: [], n_cells: 0, n_valid: 0, valid_fraction: 0 },
    ...extra,
  } as unknown as LayerArtifact;
  const at = (a: number[]) => (r: number, c: number) => (r < 0 || c < 0 || r >= g.height || c >= g.width ? N : a[r * g.width + c]);
  return { layer, code: at(codes), age: ages ? at(ages.map((a, i) => (values[i] == null ? N : a))) : null };
}

describe("satellite pixels: exact, dated, gaps kept", () => {
  // 4 × 1 line of pixels: observed 2 days ago, cloud, observed 5 days ago, observed 2 days ago
  const g = lattice(4, 1);
  const m = member("olci300_chl_latest", g, [1.5, null, 0.4, 2], [2, 0, 5, 2]);
  const s = satSampler(m);

  it("each pixel carries its own observation date from the age grid", () => {
    expect(obsPoint(s, 0, 0)).toMatchObject({ value: 1.5, date: "2026-10-06" });
    expect(obsPoint(s, 0, 2)).toMatchObject({ value: 0.4, date: "2026-10-03" });
    // the cloud pixel: no value, no date, and nothing taken from its neighbours
    expect(obsPoint(s, 0, 1)).toMatchObject({ value: null, date: null, native: null });
    expect(obsPoint(s, 0, 9)).toBeNull();
  });

  it("an area lists every pixel and counts the gaps", () => {
    const cells = obsArea(s, [-0.5, -0.5, 3.5, 0.5]);
    expect(cells.map((c) => c.value)).toEqual([1.5, null, 0.4, 2]);
    const sum = summarise(cells);
    expect(sum).toMatchObject({ inGrid: 4, observed: 3, min: 0.4, median: 1.5, max: 2 });
    expect(sum.dates).toEqual([
      { key: "2026-10-06", n: 2 },
      { key: "2026-10-03", n: 1 },
    ]);
  });

  it("a transect keeps the cloud pixel as a gap and lies flat per pixel", () => {
    const cells = obsTransect(s, { lat: 0, lon: -0.4 }, { lat: 0, lon: 3.4 });
    expect(cells.map((c) => c.value)).toEqual([1.5, null, 0.4, 2]);
    expect(cells.map((c) => c.date)).toEqual(["2026-10-06", null, "2026-10-03", "2026-10-06"]);
    expect(summarise(cells)).toMatchObject({ inGrid: 4, observed: 3 });
    // beyond the grid: one entry outside the domain, not a pixel
    const out = obsTransect(s, { lat: 0, lon: 2 }, { lat: 0, lon: 6 });
    expect(out.at(-1)).toMatchObject({ inDomain: false, value: null });
    expect(summarise(out).inGrid).toBe(2);
  });

  it("a single-day layer dates every pixel with its overpass day", () => {
    const d = satSampler(member("olci300_chl_2026-10-05", g, [1, null, 1, 1], null, "2026-10-05"));
    expect(obsArea(d, [-0.5, -0.5, 3.5, 0.5]).map((c) => c.date)).toEqual(["2026-10-05", null, "2026-10-05", "2026-10-05"]);
  });

  it("a footprint is the outline of the observed pixels only", () => {
    // two adjacent observed pixels: 6 unit edges around them, none between them
    expect(footprintOutline(g, [{ row: 0, col: 2 }, { row: 0, col: 3 }]).coordinates).toHaveLength(6);
  });

  it("lattice ranges work for south-going rows", () => {
    expect(rangeIn(37.2, -0.0025, 321, 37.0, 37.1)).toEqual([40, 80]);
  });
});

describe("multi-sensor: one sensor per pixel, by the published rule", () => {
  // primary 300 m-like lattice: 2 × 2 cells of 1°; secondary: one 2° cell covering them all
  const pg = lattice(2, 2);
  const sg = lattice(1, 1, 0.5, 0.5, 2);
  const layer = { layer_id: "multisensor_chl_latest", group_id: "satellite_chlorophyll_hr", multisensor: { prefer_primary_within_days: 2 } } as unknown as LayerArtifact;

  it("Sentinel-3 where it observed, VIIRS where it did not or is more than 2 days newer", () => {
    // primary: [obs 2 d ago, none, obs 6 d ago, obs 4 d ago]; secondary: one value 1 d ago
    const p = member("olci300_chl_latest", pg, [1, null, 3, 4], [2, 0, 6, 4]);
    const s = member("viirs750_chl_latest", sg, [9], [1]);
    const ms = multiSampler(layer, p, s);
    const cells = obsArea(ms, [-0.5, -0.5, 1.5, 1.5]);
    expect(cells.map((c) => [c.value, c.sensor, c.date])).toEqual([
      [1, 1, "2026-10-06"], // Sentinel-3; VIIRS only 1 day newer
      [9, 2, "2026-10-07"], // no Sentinel-3
      [9, 2, "2026-10-07"], // VIIRS 5 days newer
      [9, 2, "2026-10-07"], // VIIRS 3 days newer
    ]);
    const sum = summarise(cells);
    expect(sum.sensors).toEqual([
      { key: 1, n: 1 },
      { key: 2, n: 3 },
    ]);
    // the three VIIRS display pixels come from one native VIIRS pixel
    expect(sum.nativeSecondary).toBe(1);
    // nothing averaged: every value is one sensor's own
    expect(cells.every((c) => c.value === 1 || c.value === 9)).toBe(true);
    // the footprint of a VIIRS value is its native 2° pixel, of a Sentinel-3 value its own pixel
    expect(footprintOf(ms, cells[1])).toMatchObject({ g: sg, row: 0, col: 0, lat: 0.5, lon: 0.5 });
    expect(footprintOf(ms, cells[0])).toMatchObject({ g: pg, row: 0, col: 0 });
  });
});

describe("inter-sensor consistency: only same place, same date", () => {
  // VIIRS: 2 cells of 1°; Sentinel-3 day grid: 10 × 5 pixels of 0.2° (5 × 5 inside each VIIRS cell)
  const vg = lattice(2, 1);
  const s3g = lattice(10, 5, -0.4, -0.4, 0.2);

  it("pairs a VIIRS cell with the geometric mean of that day's Sentinel-3 pixels inside it", () => {
    const viirs = satSampler(member("viirs750_chl_latest", vg, [1, 2], [3, 4])); // dated Oct 5, Oct 4
    const s3vals = (r: number, c: number) => (c < 5 ? 2 : r === 0 ? 1 : null); // cell 1: only 5 pixels in one row
    const day: S3Day = { g: s3g, value: s3vals, from: "day" };
    const res = sameDayPairs(viirs, (d) => (d === "2026-10-05" ? day : d === "2026-10-04" ? { ...day, value: (r, c) => (c >= 5 && r === 0 ? 1 : null) } : null), [-0.5, -0.5, 1.5, 0.5]);
    expect(res.viirsCells).toBe(2);
    expect(res.pairs).toHaveLength(2);
    expect(res.pairs[0]).toMatchObject({ date: "2026-10-05", viirs: 1, nS3: 25 });
    expect(res.pairs[0].log10Ratio).toBeCloseTo(Math.log10(2), 9);
    expect(res.pairs[1]).toMatchObject({ date: "2026-10-04", viirs: 2, nS3: 5 });
    expect(res.medianLog10).toBeCloseTo((Math.log10(2) + Math.log10(0.5)) / 2, 9);
  });

  it("no pair without a Sentinel-3 observation that day, or with too few pixels", () => {
    const viirs = satSampler(member("viirs750_chl_latest", vg, [1, 2], [3, 3]));
    const none = sameDayPairs(viirs, () => null, [-0.5, -0.5, 1.5, 0.5]);
    expect(none).toMatchObject({ viirsCells: 2, sameDayCandidates: 0, noS3Days: ["2026-10-05"], pairs: [], medianLog10: null });
    const few = sameDayPairs(viirs, () => ({ g: s3g, value: (r, c) => (r === 0 && c < 4 ? 1 : null), from: "day" }), [-0.5, -0.5, 1.5, 0.5]);
    expect(few).toMatchObject({ sameDayCandidates: 2, pairs: [], medianLog10: null });
  });
});

describe("radar cells", () => {
  const g = lattice(3, 1);
  const f: CurrentField = { g, u: new Float32Array([0.3, NaN, 0]), v: new Float32Array([0.4, NaN, -0.2]) };
  const layer = { layer_id: "hfr2km_currents_20261008T12Z", group_id: "currents", title: "HF radar", time: { valid_time: "2026-10-08T12:00:00Z" }, platforms: ["HF radar (HFRNet)"], provenance: { dataset_id: "ucsdHfrW2", license: "Public domain" } } as unknown as LayerArtifact;
  const s = radarSampler(layer, f);

  it("speed and direction of each valid cell, with the hour; an empty cell stays empty", () => {
    expect(obsPoint(s, 0, 0)).toMatchObject({ date: "2026-10-08T12:00:00Z" });
    expect(obsPoint(s, 0, 0)!.value).toBeCloseTo(0.5, 6);
    expect(obsPoint(s, 0, 0)!.dir).toBeCloseTo(36.87, 1);
    expect(obsPoint(s, 0, 1)).toMatchObject({ value: null, date: null });
    expect(obsPoint(s, 0, 2)!.dir).toBeCloseTo(180, 5);
    expect(summarise(obsArea(s, [-0.5, -0.5, 2.5, 0.5]))).toMatchObject({ inGrid: 3, observed: 2 });
  });

  it("the radar transect export says an empty value is no data, not calm water", () => {
    const csv = obsTransectCsv(s, { lat: 0, lon: 0 }, { lat: 0, lon: 2 }, obsTransect(s, { lat: 0, lon: 0 }, { lat: 0, lon: 2 }), RUN);
    expect(csv).toContain("# Product class: observation (HF-radar surface currents), not a forecast");
    expect(csv).toContain("not calm water");
    const rows = csv.trim().split("\n").filter((l) => !l.startsWith("#"));
    expect(rows[0]).toBe("km,row,col,lat,lon,observed_hour_utc,u_ms,v_ms,speed_ms,direction_toward_deg,in_domain");
    expect(rows[2]).toMatch(/^[\d.]+,0,1,0,1,,,,,,yes$/);
  });
});

describe("coverage by the pipeline's rule: C-HARM ocean cells with an observed pixel", () => {
  it("counts only reference cells with a value, observed when any pixel inside has one", () => {
    // reference: 2 ocean cells of 1° and one land cell (nodata)
    const ref = { g: lattice(3, 1), codes: new Uint16Array([1, 1, N]) };
    // pixels of 0.5°: the first ocean cell has one observed pixel, the second none
    const pg = lattice(6, 2, -0.25, -0.25, 0.5);
    const vals: (number | null)[] = [null, 1, null, null, 1, 1, null, null, null, null, 1, 1];
    const s = satSampler(member("olci300_chl_2026-10-05", pg, vals, null, "2026-10-05"));
    expect(oceanCoverage(ref, [-0.5, -0.5, 2.5, 0.5], s)).toEqual({ cells: 2, observed: 1 });
  });
});

describe("exports carry their provenance and say what an empty value is", () => {
  const g = lattice(2, 1);
  const s = satSampler(member("olci300_chl_latest", g, [1.25, null], [2, 0]));
  it("area: observation, units, sensor, licence, run, and empty = no observation", () => {
    const csv = obsAreaCsv(s, [-0.5, -0.5, 1.5, 0.5], obsArea(s, [-0.5, -0.5, 1.5, 0.5]), RUN);
    expect(csv).toContain("# Product class: observation (satellite chlorophyll-a), not a forecast");
    expect(csv).toContain("units: mg/m³");
    expect(csv).toContain("Sensor / platform: Sentinel-3 300 m");
    expect(csv).toContain("olci300_chl_latest-ds");
    expect(csv).toContain("# Licence: olci300_chl_latest licence");
    expect(csv).toContain("# Pipeline run: run-1 (version v9)");
    expect(csv).toContain("Empty value = no observation");
    expect(csv).toContain("not low chlorophyll");
    expect(csv).toContain("not a toxin measurement");
    const rows = csv.trim().split("\n").filter((l) => !l.startsWith("#"));
    expect(rows).toEqual(["row,col,lat,lon,observed_date,chlorophyll_mg_m3", "0,0,0,0,2026-10-06,1.25", "0,1,0,1,,"]);
  });
  it("point: the pixel and the same pixel on each published day", () => {
    const csv = obsPointCsv(s, obsPoint(s, 0, 0)!, [{ label: "olci300_chl_2026-10-03", px: null }], RUN);
    const rows = csv.trim().split("\n").filter((l) => !l.startsWith("#"));
    expect(rows).toEqual(["layer,observed_date,chlorophyll_mg_m3", "selected,2026-10-06,1.25", "olci300_chl_2026-10-03,,"]);
  });
});

describe("fixture: only published days and hours are offered", () => {
  it("a day published without a grid is never chosen as a layer", () => {
    const empty = satelliteDays(manifest, "olci300").filter((d) => !d.grid);
    expect(empty.length).toBeGreaterThan(0);
    for (const d of empty) expect(satLayerFor(manifest, { product: "olci300", day: d.time.observed_date! })).toBeNull();
    expect(satLayerFor(manifest, { product: "olci300", day: "2026-10-05" })?.layer_id).toBe("olci300_chl_2026-10-05");
    expect(satLayerFor(manifest, { product: "multi", day: null })?.layer_id).toBe("multisensor_chl_latest");
  });
  it("radar: the newest hour by default, any published hour, or the 24-hour mean", () => {
    const hours = currentsHourly(manifest);
    expect(radarLayerFor(manifest, { hour: null, mean: false })?.layer_id).toBe(hours.at(-1)!.layer_id);
    expect(radarLayerFor(manifest, { hour: hours[0].time.valid_time!, mean: false })?.layer_id).toBe(hours[0].layer_id);
    expect(radarLayerFor(manifest, { hour: null, mean: true })?.layer_id).toBe("hfr2km_currents_mean24h");
  });
});
