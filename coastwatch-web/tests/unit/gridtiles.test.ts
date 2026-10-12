import { readFileSync } from "node:fs";
import path from "node:path";
import { gunzipSync, inflateSync } from "node:zlib";
import { describe, expect, it } from "vitest";
import type { LayerArtifact, Manifest } from "@/generated/schema";
import { codeColours, GRID_TILE_SIZE, renderGridTile, supportsGridTiles } from "@/lib/gridTiles";
import manifestJson from "../fixture-data/v1/manifest.json";

/** C-HARM grid tiles: the same colours as the published image at every cell, cell edges within
 *  half a tile pixel of the true boundary, and no value stays transparent. Fixture data. */
const FIX = path.resolve(__dirname, "../fixture-data/v1");
const m = manifestJson as unknown as Manifest;
const layer = m.layers.find((l) => l.layer_id === "charm_particulate_domoic_lead1") as LayerArtifact;
const g = layer.grid!;
const codes = new Uint16Array(new Uint8Array(gunzipSync(readFileSync(path.join(FIX, g.url)))).buffer);
const lut = codeColours(g, layer.palette!);
const R = 6378137;
const mercY = (lat: number) => R * Math.log(Math.tan(Math.PI / 4 + (lat * Math.PI) / 360));

/** Minimal PNG decoder (8-bit RGBA or RGB, non-interlaced): enough for the pipeline's images. */
function decodePng(buf: Buffer): { w: number; h: number; px: Uint8Array } {
  let i = 8;
  let w = 0, h = 0, ct = 0;
  const idat: Buffer[] = [];
  while (i < buf.length) {
    const len = buf.readUInt32BE(i);
    const type = buf.toString("ascii", i + 4, i + 8);
    const data = buf.subarray(i + 8, i + 8 + len);
    if (type === "IHDR") {
      w = data.readUInt32BE(0);
      h = data.readUInt32BE(4);
      ct = data[9];
      expect(data[8]).toBe(8);
      expect(data[12]).toBe(0);
    } else if (type === "IDAT") idat.push(data);
    i += 12 + len;
  }
  const bpp = ct === 6 ? 4 : ct === 2 ? 3 : 0;
  expect(bpp).toBeGreaterThan(0);
  const raw = inflateSync(Buffer.concat(idat));
  const stride = w * bpp;
  const out = new Uint8Array(w * h * 4);
  const prev = new Uint8Array(stride);
  const cur = new Uint8Array(stride);
  for (let y = 0; y < h; y++) {
    const f = raw[y * (stride + 1)];
    for (let x = 0; x < stride; x++) {
      const v = raw[y * (stride + 1) + 1 + x];
      const a = x >= bpp ? cur[x - bpp] : 0, b = prev[x], c = x >= bpp ? prev[x - bpp] : 0;
      const p = a + b - c, pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
      cur[x] = (v + [0, a, b, (a + b) >> 1, pa <= pb && pa <= pc ? a : pb <= pc ? b : c][f]) & 255;
    }
    for (let x = 0; x < w; x++) {
      out.set([cur[x * bpp], cur[x * bpp + 1], cur[x * bpp + 2], bpp === 4 ? cur[x * bpp + 3] : 255], (y * w + x) * 4);
    }
    prev.set(cur);
  }
  return { w, h, px: out };
}

/** The tile and pixel (at zoom z) containing a point. */
function tilePixel(lat: number, lon: number, z: number, size = GRID_TILE_SIZE) {
  const n = 2 ** z;
  const fx = ((lon + 180) / 360) * n;
  const fy = ((1 - Math.log(Math.tan(Math.PI / 4 + (lat * Math.PI) / 360)) / Math.PI) / 2) * n;
  return { x: Math.floor(fx), y: Math.floor(fy), px: Math.floor((fx % 1) * size), py: Math.floor((fy % 1) * size), fy };
}

describe("grid tiles: colours", () => {
  it("are drawn only for linear step palettes on plain grids", () => {
    expect(supportsGridTiles(layer)).toBe(true);
    expect(supportsGridTiles({ ...layer, palette: { ...layer.palette!, interpolation: "linear" } })).toBe(false);
    expect(supportsGridTiles({ ...layer, grid: null })).toBe(false);
  });

  it("follow the pipeline's step rule for every code, and no-data is transparent", () => {
    const stops = layer.palette!.stops;
    const hex = (u: number) => `#${[u & 255, (u >>> 8) & 255, (u >>> 16) & 255].map((v) => v.toString(16).padStart(2, "0")).join("")}`;
    for (let code = 0; code < 65535; code += 97) {
      const v = Math.min(1, Math.max(0, code * g.scale_factor + g.add_offset));
      // numpy searchsorted(side="right") - 1
      const idx = Math.max(0, stops.filter((s) => s.value <= v).length - 1);
      expect(hex(lut[code])).toBe(stops[idx].color);
      expect(lut[code] >>> 24).toBe(255);
    }
    expect(lut[g.nodata ?? 65535]).toBe(0);
  });

  it("equal the published image at the centre of every cell", () => {
    const img = decodePng(readFileSync(path.join(FIX, layer.image!.url)));
    const [w, s, e, n] = layer.image!.bounds_lnglat;
    const z = 9;
    const tiles = new Map<string, Uint8ClampedArray>();
    let compared = 0;
    for (let r = 0; r < g.height; r++)
      for (let c = 0; c < g.width; c++) {
        const lat = g.lat_first + r * g.lat_step;
        const lon = g.lon_first + c * g.lon_step;
        const ix = Math.floor(((lon - w) / (e - w)) * img.w);
        const iy = Math.floor(((mercY(n) - mercY(lat)) / (mercY(n) - mercY(s))) * img.h);
        const ref = img.px.subarray((iy * img.w + ix) * 4, (iy * img.w + ix) * 4 + 4);
        const t = tilePixel(lat, lon, z);
        const key = `${t.x}/${t.y}`;
        if (!tiles.has(key)) tiles.set(key, renderGridTile(g, codes, lut, z, t.x, t.y));
        const o = (t.py * GRID_TILE_SIZE + t.px) * 4;
        const got = tiles.get(key)!.subarray(o, o + 4);
        if (ref[3] === 0) expect(got[3]).toBe(0);
        else expect([...got]).toEqual([...ref]);
        compared++;
      }
    expect(compared).toBe(g.width * g.height);
  });
});

describe("grid tiles: cell edges", () => {
  it("each north-south edge lies within half a tile pixel of the true cell boundary (zoom 12)", () => {
    const z = 12;
    const cache = new Map<string, Uint8ClampedArray>();
    const tileAt = (x: number, y: number) => cache.get(`${x}/${y}`) ?? (cache.set(`${x}/${y}`, renderGridTile(g, codes, lut, z, x, y)), cache.get(`${x}/${y}`)!);
    let checked = 0;
    for (let c = 0; c < g.width; c++)
    for (let r = 1; r < g.height; r++) {
      const a = codes[(r - 1) * g.width + c], b = codes[r * g.width + c];
      if (lut[a] === lut[b]) continue; // no visible edge between cells of one colour class
      const latB = g.lat_first + (r - 0.5) * g.lat_step; // true boundary between rows r-1 and r
      const lon = g.lon_first + c * g.lon_step;
      const t = tilePixel(latB, lon, z);
      const edgeY = (t.fy % 1) * GRID_TILE_SIZE; // exact boundary, in tile pixels
      const tile = tileAt(t.x, t.y);
      const col = (py: number) => new Uint32Array(tile.buffer)[py * GRID_TILE_SIZE + t.px];
      const north = lut[g.lat_step > 0 ? b : a], south = lut[g.lat_step > 0 ? a : b];
      // the drawn edge: first pixel row (from the top) showing the southern cell's colour
      const lo = Math.max(0, Math.floor(edgeY) - 3), hi = Math.min(GRID_TILE_SIZE - 1, Math.ceil(edgeY) + 3);
      let drawn = -1;
      for (let py = lo; py <= hi; py++) if (col(py) === south && (py === lo || col(py - 1) === north)) drawn = py;
      if (drawn < 0) continue; // the edge falls on a tile seam
      expect(Math.abs(drawn - edgeY)).toBeLessThanOrEqual(0.5 + 1e-6);
      checked++;
    }
    expect(checked).toBeGreaterThan(5);
  });

  it("each east-west edge lies within half a tile pixel too", () => {
    const z = 12;
    const cache = new Map<string, Uint32Array>();
    const tileAt = (x: number, y: number) => cache.get(`${x}/${y}`) ?? (cache.set(`${x}/${y}`, new Uint32Array(renderGridTile(g, codes, lut, z, x, y).buffer)), cache.get(`${x}/${y}`)!);
    let checked = 0;
    for (let r = 0; r < g.height; r++)
    for (let c = 1; c < g.width; c++) {
      const a = codes[r * g.width + c - 1], b = codes[r * g.width + c];
      if (lut[a] === lut[b]) continue;
      const lonB = g.lon_first + (c - 0.5) * g.lon_step;
      const lat = g.lat_first + r * g.lat_step;
      const t = tilePixel(lat, lonB, z);
      const n = 2 ** z;
      const edgeX = ((((lonB + 180) / 360) * n) % 1) * GRID_TILE_SIZE;
      const tile = tileAt(t.x, t.y);
      const at = (px: number) => tile[t.py * GRID_TILE_SIZE + px];
      const lo = Math.max(1, Math.floor(edgeX) - 3), hi = Math.min(GRID_TILE_SIZE - 1, Math.ceil(edgeX) + 3);
      let drawn = -1;
      for (let px = lo; px <= hi; px++) if (at(px) === lut[b] && at(px - 1) === lut[a]) drawn = px;
      if (drawn < 0) continue;
      expect(Math.abs(drawn - edgeX)).toBeLessThanOrEqual(0.5 + 1e-6);
      checked++;
    }
    expect(checked).toBeGreaterThan(5);
  });

  it("a tile wholly outside the grid is fully transparent", () => {
    const t = tilePixel(50, -150, 6);
    expect(renderGridTile(g, codes, lut, 6, t.x, t.y).every((v) => v === 0)).toBe(true);
  });
});

describe("ocean-only clipping", () => {
  it("keeps colour, scales opacity by water coverage, and never makes a transparent pixel visible", async () => {
    const { clipToWater } = await import("@/lib/waterMask");
    const px = new Uint8ClampedArray([10, 20, 30, 255, 10, 20, 30, 255, 10, 20, 30, 255, 0, 0, 0, 0]);
    clipToWater(px, new Uint8ClampedArray([255, 128, 0, 255]));
    expect([...px]).toEqual([10, 20, 30, 255, 10, 20, 30, 128, 10, 20, 30, 0, 0, 0, 0, 0]);
  });
  // where the basemap tile cannot be read: tests/unit/waterfallback.test.ts
});
