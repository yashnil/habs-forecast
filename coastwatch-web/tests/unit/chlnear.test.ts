import { readFileSync } from "node:fs";
import path from "node:path";
import { gzipSync } from "node:zlib";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { LayerArtifact, ValueGrid } from "@/generated/schema";
import { CHL_NEAR_COPY } from "@/content/copy";
import { readChlNear, waterProbe, type ChlNear, type WaterProbe } from "@/lib/chlNear";
import { decodeWaterFallback } from "@/lib/waterFallback";

/**
 * Chlorophyll near a place (lib/chlNear): sea pixels only, shoreline pixels and isolated extremes
 * set aside (never deleted, never capped), a neighbourhood median when the point has no usable
 * pixel. The Crescent City case reproduces the 2026-10-12 OLCI readout of 310 mg/m³ from a land
 * point (docs: the investigation in the PR).
 */

afterEach(() => vi.unstubAllGlobals());

const S = 6 / 65534;
const N = 65535;
const code = (v: number) => Math.round((Math.log10(v) + 3) / S);
const STEP = 0.0025;

type Spec = { lat0: number; lon0: number; h: number; w: number; value: (lat: number, lon: number) => number | null; age?: (lat: number, lon: number) => number };

let seq = 0;
/** A one-chunk OLCI-like layer (0.0025°, log10 codes, latest-clear-view composite with ages), served by a stubbed fetch. */
function layerOf(s: Spec): { base: string; layer: LayerArtifact } {
  const base = `https://t${++seq}.test/v1`;
  const vals: number[] = [];
  const ages: number[] = [];
  for (let r = 0; r < s.h; r++)
    for (let c = 0; c < s.w; c++) {
      const lat = s.lat0 - r * STEP;
      const lon = s.lon0 + c * STEP;
      const v = s.value(+lat.toFixed(5), +lon.toFixed(5));
      vals.push(v == null ? N : code(v));
      ages.push(v == null ? N : (s.age?.(lat, lon) ?? 2));
    }
  const grid: ValueGrid = {
    url: "g/{row}_{col}.u16.gz",
    encoding: "uint16le+gzip",
    width: s.w,
    height: s.h,
    lat_first: s.lat0,
    lat_step: -STEP,
    lon_first: s.lon0,
    lon_step: STEP,
    scale_factor: S,
    add_offset: -3,
    nodata: N,
    max_quantization_error: S / 2,
    transform: "log10",
    chunks: { rows: s.h, cols: s.w, url_template: "g/{row}_{col}.u16.gz", present: ["0_0"] },
  };
  const age: ValueGrid = { ...grid, url: "a/{row}_{col}.u16.gz", scale_factor: 1, add_offset: 0, transform: "none", max_quantization_error: 0.5, chunks: { ...grid.chunks!, url_template: "a/{row}_{col}.u16.gz" } };
  const files: Record<string, number[]> = { [`${base}/g/0_0.u16.gz`]: vals, [`${base}/a/0_0.u16.gz`]: ages };
  const prev = globalThis.fetch;
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) => {
      const body = files[url];
      if (body) {
        const b = Buffer.alloc(body.length * 2);
        body.forEach((c, i) => b.writeUInt16LE(c, i * 2));
        const gz = gzipSync(b);
        return new Response(gz.buffer.slice(gz.byteOffset, gz.byteOffset + gz.byteLength) as ArrayBuffer);
      }
      return prev ? prev(url) : new Response("", { status: 404 });
    }),
  );
  const layer = {
    layer_id: "olci300_chl_latest",
    native_resolution_m: 300,
    grid,
    time: {},
    composite: { reference_date: "2026-10-14", window_days: 7, age_grid: age },
  } as unknown as LayerArtifact;
  return { base, layer };
}

/** Sea west of `shoreLon`, land east of it. */
function straightShore(shoreLon: number): WaterProbe {
  const mPerDegLon = (lat: number) => 111_320 * Math.cos((lat * Math.PI) / 180);
  return {
    isWater: (_lat, lon) => lon < shoreLon,
    nearLand: (lat, lon, m) => lon + m / mPerDegLon(lat) >= shoreLon,
  };
}
const allSea: WaterProbe = { isWater: () => true, nearLand: () => false };

/** Deterministic "ordinary water" values in [lo, hi]. */
const ordinary = (lat: number, lon: number, lo = 1, hi = 9) => lo + (((Math.abs(Math.round(lat * 4000) * 31 + Math.round(lon * 4000) * 17) % 97) / 96) * (hi - lo));

const headline = (n: ChlNear) => (n.kind === "at" || n.kind === "median" ? n.value : null);

// ------------------------------------------------------------------ Crescent City, 2026-10-12

/** OLCI 2026-10-12 around Battery Point (the investigation's table 3a; "-" is no value). */
const CC_TABLE: Record<string, (number | null)[]> = {
  "41.74875": [67.2, null, null, null, null, null, null, null, null],
  "41.74625": [6.4, 6.4, 8.4, 9.2, 309.9, null, null, null, 5.9],
  "41.74375": [5.4, 5.1, 5.1, 6.2, 309.9, null, null, null, 3.6],
  "41.74125": [2.5, 1.8, 2.6, 2.6, 6.7, 11.4, 11.7, 11.7, 7.5],
  "41.73875": [1.0, 0.9, 1.1, 4.6, 4.6, 7.1, 8.0, 8.9, 8.9],
  "41.73625": [0.7, 0.7, 0.7, 2.8, 2.4, 2.4, 2.8, 5.0, 3.1],
};
const CC_LONS = [-124.21625, -124.21375, -124.21125, -124.20875, -124.20625, -124.20375, -124.20125, -124.19875, -124.19625];
const PORT = { lat: 41.74901, lon: -124.20398 }; // CDFW port point, on land

const bundled = waterProbe(decodeWaterFallback(JSON.parse(readFileSync(path.resolve(__dirname, "../../public/basemap/water-fallback.json"), "utf8"))));

/** The 2026-10-12 field: the table where it has cells; elsewhere sea pixels read ordinary 1–9 and land has no value. */
function crescentCity() {
  return layerOf({
    lat0: 41.77625,
    lon0: -124.25125,
    h: 25,
    w: 37,
    value: (lat, lon) => {
      const row = CC_TABLE[lat.toFixed(5)];
      const k = CC_LONS.findIndex((x) => Math.abs(x - lon) < 1e-6);
      if (row && k >= 0) return row[k];
      return bundled.isWater(lat, lon) ? ordinary(lat, lon) : null;
    },
  });
}

describe("Crescent City, 2026-10-12: the 310 mg/m³ shoreline pixel", () => {
  it("the bundled water puts the port point on land and the 309.9 pixel within one OLCI pixel of the shore", () => {
    expect(bundled.isWater(PORT.lat, PORT.lon)).toBe(false);
    expect(bundled.isWater(41.74625, -124.20625)).toBe(true);
    expect(bundled.nearLand(41.74625, -124.20625, 300)).toBe(true);
    // open water 2 km west of the harbour is not near land
    expect(bundled.nearLand(41.74, -124.24, 300)).toBe(false);
  });

  it("headlines the median of clear water pixels, not the nearest pixel, and says the 310 is not used", async () => {
    const { base, layer } = crescentCity();
    const r = await readChlNear(base, layer, PORT.lat, PORT.lon, { radiusKm: 1.5, water: bundled });
    expect(r.kind).toBe("median");
    if (r.kind !== "median") return;
    expect(r.water).toBe(true);
    expect(r.atPoint).toBe("none");
    expect(r.value).toBeGreaterThan(2);
    expect(r.value).toBeLessThan(12);
    expect(r.max).toBeLessThan(100); // the 309.9 is not among the pixels used
    expect(r.n).toBeGreaterThanOrEqual(5);
    expect(r.date).toBe("2026-10-12");
    expect(r.nearest.km).toBeLessThan(1.5);
    expect(r.aside).toMatchObject({ reason: "shore" });
    expect(r.aside!.value).toBeCloseTo(309.9, 0);
    const line = CHL_NEAR_COPY.aside(r.aside!);
    expect(line).toContain("One pixel right against the shore read 310 mg/m³");
    expect(line).toContain("it is not used here");
    expect(line).not.toMatch(/toxi|forecast|!/i);
    expect(CHL_NEAR_COPY.basis({ n: r.n, radiusKm: r.radiusKm, min: r.min, max: r.max, nearestKm: r.nearest.km })).toMatch(/^Median of \d+ clear water pixels within 1\.5 km · range [\d.]+–[\d.]+ · nearest [\d.]+ km away$/);
    // two cells of one native pixel: no "range 8.4–8.4"
    expect(CHL_NEAR_COPY.basis({ n: 2, radiusKm: 1.5, min: 8.41, max: 8.42, nearestKm: 1.4 })).toBe("Median of 2 clear water pixels within 1.5 km · nearest 1.4 km away");
  });

  it("the My Coast 5 km reading gives the same kind of answer", async () => {
    const { base, layer } = crescentCity();
    const r = await readChlNear(base, layer, PORT.lat, PORT.lon, { radiusKm: 5, water: bundled });
    expect(r.kind).toBe("median");
    expect(headline(r)).toBeLessThan(12);
    expect(r.kind === "median" && r.aside?.value).toBeCloseTo(309.9, 0);
  });

  it("without the water geometry, Q1 and Q2 are skipped and recorded, and the isolated-extreme rule still sets the 310 aside", async () => {
    const { base, layer } = crescentCity();
    const r = await readChlNear(base, layer, PORT.lat, PORT.lon, { radiusKm: 1.5, water: null });
    const withWater = await readChlNear(base, layer, PORT.lat, PORT.lon, { radiusKm: 1.5, water: bundled });
    expect(r.water).toBe(false);
    expect(r.kind).toBe("median");
    if (r.kind !== "median") return;
    expect(r.value).toBeLessThan(12);
    expect(r.aside).toMatchObject({ reason: "spike" });
    expect(r.aside!.value).toBeCloseTo(309.9, 0);
    // pixels the water rules would have excluded are now used, not hidden
    expect(withWater.kind === "median" && r.n > withWater.n).toBe(true);
  });
});

// ------------------------------------------------------------------ synthetic cases

const LAT0 = 37.0;
const LON0 = -122.5;
const centre = { lat: LAT0 - 20 * STEP, lon: LON0 + 20 * STEP };

describe("open-water readings", () => {
  it("a point on clear open water reads its own pixel, with its date", async () => {
    const { base, layer } = layerOf({ lat0: LAT0, lon0: LON0, h: 41, w: 41, value: (lat, lon) => ordinary(lat, lon, 2, 4), age: () => 3 });
    const r = await readChlNear(base, layer, centre.lat, centre.lon, { radiusKm: 1.5, water: allSea });
    expect(r.kind).toBe("at");
    expect(r.atPoint).toBe("used");
    expect(headline(r)).toBeCloseTo(ordinary(+centre.lat.toFixed(5), +centre.lon.toFixed(5), 2, 4), 2);
    expect(r.kind === "at" && r.date).toBe("2026-10-11");
    expect(r.kind === "at" && r.aside).toBeNull();
  });

  it("a genuine bloom patch (many contiguous 30–60 pixels) is kept: high headline, nothing set aside", async () => {
    const bloom = (lat: number, lon: number) => Math.hypot(lat - centre.lat, lon - centre.lon) < 0.02;
    const { base, layer } = layerOf({ lat0: LAT0, lon0: LON0, h: 41, w: 41, value: (lat, lon) => (bloom(lat, lon) ? ordinary(lat, lon, 30, 60) : ordinary(lat, lon, 1, 3)) });
    const r = await readChlNear(base, layer, centre.lat, centre.lon, { radiusKm: 1.5, water: allSea });
    expect(r.kind).toBe("at");
    expect(headline(r)).toBeGreaterThanOrEqual(30);
    expect(r.kind === "at" && r.aside).toBeNull();
    // under a cloud gap at the point, the neighbourhood median still shows the bloom
    const gap = layerOf({ lat0: LAT0, lon0: LON0, h: 41, w: 41, value: (lat, lon) => (Math.hypot(lat - centre.lat, lon - centre.lon) < 0.003 ? null : bloom(lat, lon) ? ordinary(lat, lon, 30, 60) : ordinary(lat, lon, 1, 3)) });
    const g = await readChlNear(gap.base, gap.layer, centre.lat, centre.lon, { radiusKm: 1.5, water: allSea });
    expect(g.kind).toBe("median");
    expect(headline(g)).toBeGreaterThanOrEqual(30);
    expect(g.kind === "median" && g.aside).toBeNull();
  });

  it("no fixed ceiling: a coherent patch above 300 mg/m³ is reported as it is", async () => {
    const { base, layer } = layerOf({ lat0: LAT0, lon0: LON0, h: 41, w: 41, value: (lat, lon) => (Math.hypot(lat - centre.lat, lon - centre.lon) < 0.01 ? ordinary(lat, lon, 300, 470) : ordinary(lat, lon, 2, 5)) });
    const r = await readChlNear(base, layer, centre.lat, centre.lon, { radiusKm: 1.5, water: allSea });
    expect(r.kind).toBe("at");
    expect(headline(r)).toBeGreaterThan(300);
  });

  it("an isolated spike offshore is set aside; the headline is the water around it", async () => {
    const { base, layer } = layerOf({ lat0: LAT0, lon0: LON0, h: 41, w: 41, value: (lat, lon) => (Math.abs(lat - centre.lat) < 1e-6 && Math.abs(lon - centre.lon) < 1e-6 ? 150 : ordinary(lat, lon, 1, 3)) });
    const r = await readChlNear(base, layer, centre.lat, centre.lon, { radiusKm: 1.5, water: allSea });
    expect(r.kind).toBe("median");
    expect(r.atPoint).toBe("set-aside");
    if (r.kind !== "median") return;
    expect(r.value).toBeLessThan(3);
    expect(r.aside).toMatchObject({ reason: "spike" });
    expect(r.aside!.value).toBeCloseTo(150, 0);
    expect(CHL_NEAR_COPY.aside(r.aside!)).toContain("One pixel nearby read 150 mg/m³, more than ten times the water around it");
  });

  it("an extreme with fewer than 5 neighbours cannot be judged isolated and is kept", async () => {
    const few = new Set([0, 1, 2, 3].map((i) => i));
    const { base, layer } = layerOf({
      lat0: LAT0,
      lon0: LON0,
      h: 41,
      w: 41,
      value: (lat, lon) => {
        const dr = Math.round((centre.lat - lat) / STEP);
        const dc = Math.round((lon - centre.lon) / STEP);
        if (dr === 0 && dc === 0) return 150;
        if (dr === 0 && few.has(dc - 1)) return 2; // 4 neighbours east of it
        return null;
      },
    });
    const r = await readChlNear(base, layer, centre.lat, centre.lon, { radiusKm: 1.5, water: allSea });
    expect(r.kind).toBe("at");
    expect(headline(r)).toBeCloseTo(150, 0);
  });

  it("all cloud: no value, the none state", async () => {
    const { base, layer } = layerOf({ lat0: LAT0, lon0: LON0, h: 41, w: 41, value: () => null });
    const r = await readChlNear(base, layer, centre.lat, centre.lon, { radiusKm: 1.5, water: allSea });
    expect(r.kind).toBe("none");
    expect(r.atPoint).toBe("none");
  });

  it("only shoreline pixels nearby: no number, the flagged state", async () => {
    // sea west of the point's column + 1; the only pixels with a value lie about 110 m from the shore
    const shore = centre.lon + 1.5 * STEP;
    const { base, layer } = layerOf({ lat0: LAT0, lon0: LON0, h: 41, w: 41, value: (lat, lon) => (lon < shore && lon > shore - STEP ? 80 : null) });
    const r = await readChlNear(base, layer, centre.lat, centre.lon, { radiusKm: 1.5, water: straightShore(shore) });
    expect(r.kind).toBe("flagged");
    expect(headline(r)).toBeNull();
    expect(CHL_NEAR_COPY.flagged("Crescent City", 1.5)).toContain("not treated as a measurement");
  });

  it("pixels on land are never used, even when they are the only ones at the point", async () => {
    const shore = centre.lon - 4 * STEP; // the point is ~0.9 km inland
    const { base, layer } = layerOf({ lat0: LAT0, lon0: LON0, h: 41, w: 41, value: (lat, lon) => (lon >= shore ? 250 : ordinary(lat, lon, 2, 4)) });
    const r = await readChlNear(base, layer, centre.lat, centre.lon, { radiusKm: 1.5, water: straightShore(shore) });
    expect(r.atPoint).toBe("set-aside");
    expect(r.kind).toBe("median");
    expect(headline(r)).toBeLessThan(4);
    // land pixels are excluded, not "set aside" with a line: nothing to explain about them
    expect(r.kind === "median" && r.aside).toBeNull();
  });
});
