import { readFileSync } from "node:fs";
import path from "node:path";
import { VectorTile } from "@mapbox/vector-tile";
import { PbfReader } from "pbf";
import { afterEach, describe, expect, it, vi } from "vitest";
import { CA_BOUNDS } from "@/lib/basemap";
import { decodeWaterFallback, rasterWater, WATER_FALLBACK_URL, type WaterGeometry } from "@/lib/waterFallback";

/** The bundled water: the fallback for the basemap's own water when its tiles cannot be read, so
 *  data rasters still stay off land (lib/waterFallback, lib/waterMask). */
const Q = 2 ** 26;
const file = JSON.parse(readFileSync(path.resolve(__dirname, "../../public/basemap/water-fallback.json"), "utf8"));
const bundled = decodeWaterFallback(file);

/** A fallback file from rings in normalised Mercator (0..1, y down). */
function geometry(polygons: number[][][][], domain: [number, number, number, number] = [-180, -85.06, 180, 85.06]) {
  const enc = (ring: number[][]) => {
    const q = ring.map(([x, y]) => [Math.round(x * Q), Math.round(y * Q)]);
    return q.flatMap(([x, y], i) => (i ? [x - q[i - 1][0], y - q[i - 1][1]] : [x, y]));
  };
  return { quant: Q, domain_lnglat: domain, polygons: polygons.map((p) => p.map(enc)) };
}
const square = (x0: number, y0: number, x1: number, y1: number) => [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];

/** Tile z/x/y and pixel (column, row) of a point, 512 px tiles. */
function at(lon: number, lat: number, z: number) {
  const k = 2 ** z;
  const fx = ((lon + 180) / 360) * k;
  const fy = (0.5 - Math.log(Math.tan(Math.PI / 4 + (lat * Math.PI) / 360)) / (2 * Math.PI)) * k;
  return { x: Math.floor(fx), y: Math.floor(fy), c: Math.floor((fx % 1) * 512), r: Math.floor((fy % 1) * 512) };
}

describe("rasterising the bundled water", () => {
  it("fills inside, leaves outside, and anti-aliases a shore inside a pixel", () => {
    // west half of the world plus a quarter pixel (8 px tile at zoom 0: 4.25 px)
    const g = decodeWaterFallback(geometry([[square(0, 0, 4.25 / 8, 1)]]));
    const m = rasterWater(g, 0, 0, 0, 8);
    for (let r = 0; r < 8; r++) expect([...m.slice(r * 8, r * 8 + 8)]).toEqual([255, 255, 255, 255, 64, 0, 0, 0]);
  });
  it("keeps holes (islands) as land", () => {
    const g = decodeWaterFallback(geometry([[square(0, 0, 1, 1), square(0.25, 0.25, 0.75, 0.75)]]));
    const m = rasterWater(g, 0, 0, 0, 8);
    expect(m[3 * 8 + 3]).toBe(0);
    expect(m[3 * 8 + 0]).toBe(255);
    expect(m[0]).toBe(255);
  });
  it("places the shore in the right pixel of a deeper tile, including rings that only pass beside it", () => {
    // water east of x = 0.5 + 100.5 px at zoom 1 (1024 px world); tile 1/1/0 covers x 0.5..1
    const shore = 0.5 + 100.5 / 1024;
    const g = decodeWaterFallback(geometry([[square(shore, 0, 1, 1)], [square(0.01, 0.01, 0.02, 0.02)]]));
    const m = rasterWater(g, 1, 1, 0, 512);
    expect(m[10 * 512 + 99]).toBe(0);
    expect(m[10 * 512 + 100]).toBe(128);
    expect(m[10 * 512 + 101]).toBe(255);
    expect(rasterWater(g, 1, 0, 1, 512).every((v) => v === 0)).toBe(true);
  });
  it("leaves data unclipped (water) outside its domain, where it has no coastline", () => {
    // domain: the western hemisphere only; no water in it
    const g = decodeWaterFallback(geometry([], [-180, -85, 0, 85]));
    const m = rasterWater(g, 0, 0, 0, 8);
    expect([...m.slice(0, 8)]).toEqual([0, 0, 0, 0, 255, 255, 255, 255]);
  });
});

describe("the bundled water file", () => {
  it("covers the map's whole pannable area and is ocean-only OpenMapTiles water", () => {
    expect(file.domain_lnglat).toEqual([CA_BOUNDS[0][0], CA_BOUNDS[0][1], CA_BOUNDS[1][0], CA_BOUNDS[1][1]]);
    expect(file.source).toMatch(/OpenMapTiles/);
    expect(file.method).toMatch(/Ocean class only/);
  });
  it("is water at sea, land inland, and keeps small islands", () => {
    const water = (lon: number, lat: number) => {
      const t = at(lon, lat, 12);
      return rasterWater(bundled, 12, t.x, t.y, 512)[t.r * 512 + t.c];
    };
    expect(water(-122.0, 36.8)).toBe(255); // Monterey Bay
    expect(water(-122.6, 37.75)).toBe(255); // off the Golden Gate
    expect(water(-121.65, 36.68)).toBe(0); // Salinas
    expect(water(-122.41, 37.77)).toBe(0); // San Francisco
    expect(water(-123.0016, 37.6989)).toBe(0); // Southeast Farallon Island
    expect(water(-118.45, 33.4)).toBe(0); // Santa Catalina Island
  });
  it("agrees with the basemap's own ocean in a recorded tile (Moss Landing, zoom 12) to within 2 px of the shore", () => {
    // OpenMapTiles tile 12/662/1597 as served by OpenFreeMap (planet 2026-10-04): its ocean
    // polygons, rasterised the same way
    const vt = new VectorTile(new PbfReader(new Uint8Array(readFileSync(path.resolve(__dirname, "../fixture-data/basemap/omt-12-662-1597.pbf")))));
    const lyr = vt.layers.water;
    const rings: number[][][] = [];
    for (let i = 0; i < lyr.length; i++) {
      const f = lyr.feature(i);
      if (f.type === 3 && f.properties.class === "ocean") for (const r of f.loadGeometry()) rings.push(r.map((p) => [(662 + p.x / lyr.extent) / 4096, (1597 + p.y / lyr.extent) / 4096]));
    }
    expect(rings.length).toBeGreaterThan(0);
    const basemap = rasterWater(decodeWaterFallback(geometry(rings.map((r) => [r]))), 12, 662, 1597, 512);
    const ours = rasterWater(bundled, 12, 662, 1597, 512);
    const S = 512;
    const wet = (m: Uint8ClampedArray, i: number) => m[i] >= 128;
    let water = 0, differ = 0, farFromShore = 0;
    for (let r = 0; r < S; r++)
      for (let c = 0; c < S; c++) {
        const i = r * S + c;
        if (wet(basemap, i)) water++;
        if (wet(basemap, i) === wet(ours, i)) continue;
        differ++;
        // a disagreement must sit at the basemap's shore: a pixel partly water, or within 2 px of a
        // pixel of the other kind
        let near = basemap[i] > 0 && basemap[i] < 255;
        for (let dr = -2; dr <= 2 && !near; dr++)
          for (let dc = -2; dc <= 2 && !near; dc++) {
            const rr = r + dr, cc = c + dc;
            if (rr >= 0 && rr < S && cc >= 0 && cc < S && wet(basemap, rr * S + cc) !== wet(basemap, i)) near = true;
          }
        if (!near) farFromShore++;
      }
    expect(water).toBeGreaterThan(S * S * 0.2);
    expect(farFromShore).toBe(0);
    expect(differ / (S * S)).toBeLessThan(0.005);
  });
});

describe("choosing the water for a data tile", () => {
  const tiny = geometry([[square(0, 0, 0.5, 1)]]);
  const fallbackOnly = (omt: (u: string) => Promise<Response>, fallbackOk = true) =>
    vi.fn((u: string) => (u === WATER_FALLBACK_URL ? (fallbackOk ? Promise.resolve(new Response(JSON.stringify(tiny))) : Promise.reject(new Error("offline"))) : omt(u)));
  const expected = (g: WaterGeometry) => rasterWater(g, 0, 0, 0, 8);
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.useRealTimers();
    vi.resetModules();
  });

  it("uses the bundled water when the basemap's tile server is unreachable", async () => {
    vi.stubGlobal("OffscreenCanvas", class {});
    const fetch = fallbackOnly(() => Promise.reject(new TypeError("Failed to fetch")));
    vi.stubGlobal("fetch", fetch);
    const { waterMask } = await import("@/lib/waterMask");
    expect(await waterMask(0, 0, 0, 8)).toEqual(expected(decodeWaterFallback(tiny)));
    expect(fetch.mock.calls.map((c) => c[0])).toContain("https://tiles.openfreemap.org/planet");
  });
  it("uses the bundled water when the basemap's tiles are slow, after BASEMAP_WAIT_MS", async () => {
    vi.useFakeTimers();
    vi.stubGlobal("OffscreenCanvas", class {});
    vi.stubGlobal("fetch", fallbackOnly(() => new Promise<Response>(() => {})));
    const { waterMask, BASEMAP_WAIT_MS } = await import("@/lib/waterMask");
    let done = false;
    const p = waterMask(0, 0, 0, 8).then((m) => ((done = true), m));
    await vi.advanceTimersByTimeAsync(BASEMAP_WAIT_MS - 100);
    expect(done).toBe(false);
    await vi.advanceTimersByTimeAsync(200);
    expect(await p).toEqual(expected(decodeWaterFallback(tiny)));
  });
  it("uses the bundled water where the browser cannot read vector tiles (no OffscreenCanvas)", async () => {
    vi.stubGlobal("fetch", fallbackOnly(() => Promise.reject(new Error("not asked"))));
    const { waterMask, basemapWater } = await import("@/lib/waterMask");
    expect(await basemapWater(0, 0, 0, 8)).toBeNull(); // never an all-land mask
    expect(await waterMask(0, 0, 0, 8)).toEqual(expected(decodeWaterFallback(tiny)));
  });
  it("draws data unclipped only if neither the basemap nor the bundled water can be read", async () => {
    vi.stubGlobal("OffscreenCanvas", class {});
    vi.stubGlobal("fetch", fallbackOnly(() => Promise.reject(new TypeError("Failed to fetch")), false));
    const { waterMask } = await import("@/lib/waterMask");
    expect(await waterMask(0, 0, 0, 8)).toBeNull();
  });
});
