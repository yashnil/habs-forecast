import { readFileSync } from "node:fs";
import path from "node:path";
import { gunzipSync } from "node:zlib";
import { describe, expect, it } from "vitest";
import type { Manifest } from "@/generated/schema";
import { decodeValue, type FieldGrid } from "@/nx/chart/layers/grid";
import { ADJACENT_KEY, ADJACENT_WORDS, appliesAdjacency, isLandAdjacent, loadAdjacency, resetAdjacency, type AdjacencyMask } from "@/nx/explore/adjacency";
import { buildExploreModel } from "@/nx/explore/model";
import { noValueWords, OUTSIDE_KEY } from "@/nx/explore/words";

const root = path.resolve(__dirname, "../../../..");
const FIX = path.join(root, "tests/fixture-data/v1");
const manifest = JSON.parse(readFileSync(path.join(FIX, "manifest.json"), "utf8")) as Manifest;
const mask = JSON.parse(readFileSync(path.join(root, "public/reference/land-adjacency-v1.json"), "utf8")) as AdjacencyMask;
const layer = (id: string) => manifest.layers.find((l) => l.layer_id === id)!;

/** The fixture's OLCI latest view, decoded (one 512 × 512 chunk holds it). */
function latest() {
  const g = layer("olci300_chl_latest").grid as unknown as FieldGrid & { chunks: { url_template: string } };
  const raw = gunzipSync(readFileSync(path.join(FIX, g.chunks.url_template.replace("{row}", "0").replace("{col}", "0"))));
  const codes = new Uint16Array(raw.buffer, raw.byteOffset, raw.byteLength / 2);
  const value = (r: number, c: number) => decodeValue(g, codes[r * g.width + c]);
  const at = (r: number, c: number) => ({ lat: g.lat_first + r * g.lat_step, lon: g.lon_first + c * g.lon_step });
  return { g, value, at };
}
/** The top display class of the chlorophyll ramp: the last of eight log steps over 0.05–50 mg/m³. */
const TOP = 50 / 1000 ** (1 / 8);

describe("the left-out cells (a static reference on the OLCI lattice)", () => {
  const m: AdjacencyMask = { lat_first: 37, lon_first: -122, step: 0.0025, height: 3, width: 12, rows_rle: [[[2, 2], [8, 1]], [], [[0, 12]]] };
  it("finds a cell by the nearest lattice cell, inside a run or not, and nothing outside the lattice", () => {
    const at = (r: number, c: number) => isLandAdjacent(m, 37 - r * 0.0025, -122 + c * 0.0025);
    expect([0, 1, 2, 3, 4, 7, 8, 9].map((c) => at(0, c))).toEqual([false, false, true, true, false, false, true, false]);
    expect(at(1, 5)).toBe(false);
    expect(at(2, 0) && at(2, 11)).toBe(true);
    expect(isLandAdjacent(m, 37 - 0.001, -122 + 2 * 0.0025 + 0.001)).toBe(true); // anywhere inside the cell
    expect(at(-1, 2) || at(3, 2) || at(0, -1) || at(0, 12)).toBe(false);
  });
  it("is on the lattice the OLCI layers are published on", () => {
    const g = layer("olci300_chl_latest").grid!;
    expect(mask.step).toBe(g.lon_step);
    expect(Math.abs(((mask.lat_first - g.lat_first) / mask.step) % 1)).toBeLessThan(1e-6);
    expect(Math.abs(((g.lon_first - mask.lon_first) / mask.step) % 1)).toBeLessThan(1e-6);
    expect(mask.rows_rle.length).toBe(mask.height);
  });
  it("is read once, and a failure to read it is null (the reading then says the weaker true thing, no view)", async () => {
    resetAdjacency();
    let n = 0;
    const ok = (async () => (n++, { ok: true, json: async () => m })) as unknown as typeof fetch;
    expect(await loadAdjacency(ok)).toEqual(m);
    expect(await loadAdjacency(ok)).toEqual(m);
    expect(n).toBe(1);
    resetAdjacency();
    expect(await loadAdjacency((async () => ({ ok: false })) as unknown as typeof fetch)).toBeNull();
    resetAdjacency();
    expect(await loadAdjacency((async () => { throw new Error("offline"); }) as unknown as typeof fetch)).toBeNull();
    resetAdjacency();
  });
});

describe("which frames say it", () => {
  it("the layer's own QC decides: OLCI layers made with the rule, never VIIRS, never a layer made before it", () => {
    expect(appliesAdjacency(layer("olci300_chl_latest"))).toBe(true);
    expect(appliesAdjacency(layer("olci300_chl_2026-10-05"))).toBe(true);
    expect(appliesAdjacency(layer("viirs750_chl_latest"))).toBe(false);
    expect(appliesAdjacency({ qc: { checks: [{ name: "coastal_scope" }] } })).toBe(false); // satellite-processing-4, as staging holds
    expect(appliesAdjacency(null)).toBe(false);
  });
  it("Explore's frames carry it", () => {
    const model = buildExploreModel({ manifest, derived: null, stations: [], now: new Date("2026-10-13T06:00:00Z"), baseUrl: "/data/fixture/v1" });
    const chl = model.products.find((p) => p.id === "chl")!;
    expect(chl.frames.filter((f) => f.track === "olci").every((f) => f.besideLandLeftOut === true)).toBe(true);
    expect(chl.frames.filter((f) => f.track === "viirs").some((f) => f.besideLandLeftOut)).toBe(false);
  });
  it("its words are its own: not cloud's, not a model's footprint", () => {
    expect(ADJACENT_WORDS).toBe("Left out: beside land, where the satellite reads land and water together. No value is not a low value.");
    expect(ADJACENT_WORDS).not.toMatch(/no view|cloud|outside/i);
    expect(ADJACENT_KEY).not.toBe(OUTSIDE_KEY);
    expect(noValueWords({ id: "charm", kind: "model" }, "shoreward")).not.toContain("Left out");
  });
});

describe("the fixture's latest view: no dark fringe along the shore", () => {
  const { g, value, at } = latest();
  const cells: { r: number; c: number; v: number | null; out: boolean }[] = [];
  for (let r = 0; r < g.height; r++) for (let c = 0; c < g.width; c++) cells.push({ r, c, v: value(r, c), out: isLandAdjacent(mask, at(r, c).lat, at(r, c).lon) });
  const out = (r: number, c: number) => r >= 0 && r < g.height && c >= 0 && c < g.width && cells[r * g.width + c].out;
  const valued = cells.filter((x) => x.v !== null);

  it("no left-out cell carries a value", () => {
    expect(cells.filter((x) => x.out).length).toBeGreaterThan(1500);
    expect(valued.filter((x) => x.out)).toEqual([]);
  });
  it("the cells now nearest the shore are not the top chlorophyll class (before the rule: median 39 mg/m³, 68 % in the top class)", () => {
    // valued cells that touch the left-out band: the shore-most water the layer now draws
    const shoreMost = valued.filter((x) => [-1, 0, 1].some((dr) => [-1, 0, 1].some((dc) => out(x.r + dr, x.c + dc))));
    expect(shoreMost.length).toBeGreaterThan(300);
    const vs = shoreMost.map((x) => x.v as number).sort((a, b) => a - b);
    const median = vs[vs.length >> 1];
    const top = vs.filter((v) => v >= TOP).length / vs.length;
    expect(median).toBeLessThan(12);
    expect(top).toBeLessThan(0.25);
    // and the layer as a whole: the top class is under 1 % of its cells (it was 2.7 %, almost all of it on the shore)
    expect(valued.filter((x) => (x.v as number) >= TOP).length / valued.length).toBeLessThan(0.01);
  });
});
