import { describe, expect, it } from "vitest";
import type { FieldGrid } from "@/nx/chart/layers/grid";
import { coverageEdges, decodeValue, gridBounds, persistence } from "@/nx/chart/layers/grid";
import { decimate } from "@/nx/chart/layers/FieldLayer";
import { compassWord, nearestCurrent, nearestStation, pixelDate, readPoint, researchHref, runs, transect, RESEARCH_VAR } from "@/nx/explore/read";

// a 4 x 5 log10 grid, north row first, 0.01 degree cells
const g: FieldGrid = { width: 5, height: 4, lat_first: 36.9, lat_step: -0.01, lon_first: -122.1, lon_step: 0.01, scale_factor: 0.001, add_offset: -1, nodata: 65535, transform: "log10" };
const N = 65535;
// code 1000 -> log10 0 -> 1 mg/m3; 2000 -> 10; 0 -> 0.1
const codes = Uint16Array.from([1000, 1000, N, 2000, 2000, 0, 1000, N, N, 1000, 1000, 1000, 1000, 1000, 1000, N, N, N, N, N]);

describe("reading the tapped cell (T19, T20)", () => {
  it("is the cell under the point, decoded, and null where there is no value", () => {
    expect(readPoint(g, codes, 36.9, -122.1)?.value).toBeCloseTo(1, 6);
    expect(readPoint(g, codes, 36.9, -122.06)?.value).toBeCloseTo(10, 4);
    expect(readPoint(g, codes, 36.9, -122.08)?.value).toBeNull();
    expect(readPoint(g, codes, 36.87, -122.1)?.value).toBeNull();
    expect(readPoint(g, codes, 40, -122.1)).toBeNull();
  });
  it("decodes linear grids as is and the nodata code as null, never zero", () => {
    expect(decodeValue({ ...g, transform: "none" }, 2000)).toBeCloseTo(1, 6);
    expect(decodeValue(g, N)).toBeNull();
  });
  it("a latest view's pixel carries its own date", () => {
    expect(pixelDate("2026-10-08", 2)).toBe("2026-10-06");
  });
});

describe("the transect along the latitude", () => {
  it("has one point per cell, west to east, and breaks at every gap", () => {
    const t = transect(g, codes, 36.9, -122.08, 5);
    expect(t.map((p) => p.value === null)).toEqual([false, false, true, false, false]);
    expect(t[0].km).toBeLessThan(0);
    expect(t[t.length - 1].km).toBeGreaterThan(0);
    expect(runs(t).map((r) => r.length)).toEqual([2, 2]);
  });
});

describe("the nearest measured current", () => {
  const cg: FieldGrid = { ...g, transform: "none" };
  const u = new Float32Array(20).fill(NaN), v = new Float32Array(20).fill(NaN);
  u[6] = 0.5;
  v[6] = 0;
  it("is the nearest measured cell, its speed and the direction the water moves toward", () => {
    const c = nearestCurrent(cg, u, v, 36.89, -122.09)!;
    expect(c.speed).toBeCloseTo(0.5, 6);
    expect(c.dirTo).toBeCloseTo(90, 6);
    expect(compassWord(c.dirTo)).toBe("east");
    expect(c.km).toBeLessThan(0.1);
  });
  it("says none beyond its reach rather than borrowing a far cell", () => {
    expect(nearestCurrent(cg, u, v, 36.5, -122.09, 6)).toBeNull();
  });
});

describe("the link to Research (station, from, to, var)", () => {
  it("names the station, a 60-day window ending on the reading's date, the variable and the port", () => {
    const href = researchHref({ station: "HABs-SantaCruzWharf", to: "2026-10-06", product: "chl", place: "port-592" });
    const q = new URL(href, "http://x").searchParams;
    expect(href.startsWith("/research?")).toBe(true);
    expect(q.get("station")).toBe("HABs-SantaCruzWharf");
    expect(q.get("from")).toBe("2026-08-07");
    expect(q.get("to")).toBe("2026-10-06");
    expect(q.get("var")).toBe("chl_extracted");
    expect(q.get("place")).toBe("port-592");
    expect(RESEARCH_VAR.sst).toBe("temp");
    expect(RESEARCH_VAR.charm).toBe("pDA");
  });
  it("picks the nearest station", () => {
    const s = nearestStation([{ id: "a", name: "A", lat: 36.6, lon: -121.9, last: null }, { id: "b", name: "B", lat: 36.96, lon: -122.02, last: null }], 36.9, -122.0);
    expect(s?.id).toBe("b");
  });
});

describe("grid geometry and quality", () => {
  it("bounds reach half a cell past the outer centres", () => {
    const [w, s, e, n] = gridBounds(g);
    expect(w).toBeCloseTo(-122.105, 6);
    expect(e).toBeCloseTo(-122.055, 6);
    expect(n).toBeCloseTo(36.905, 6);
    expect(s).toBeCloseTo(36.865, 6);
  });
  it("the coverage edge outlines observed cells only", () => {
    const one = coverageEdges({ ...g }, (k) => k === 6);
    expect(one).toHaveLength(4);
    const pair = coverageEdges({ ...g }, (k) => k === 6 || k === 7);
    expect(pair).toHaveLength(6);
  });
  it("persistence is each cell's share of hours measured", () => {
    const p = persistence([Uint8Array.from([1, 0, 1]), Uint8Array.from([1, 0, 0])], 3);
    expect(Array.from(p)).toEqual([1, 0, 0.5]);
  });
  it("decimation keeps every n-th cell, never an average", () => {
    const d = decimate(Uint16Array.from([1, 2, 3, 4, 5, 6, 7, 8, 9]), 3, 3, 2);
    expect([d.w, d.h, ...Array.from(d.codes)]).toEqual([2, 2, 1, 3, 7, 9]);
  });
});

describe("isotherm lettering", () => {
  it("letters whole even °F where a row crosses them, never across a cell without a value", async () => {
    const { isothermLabels } = await import("@/nx/explore/read");
    // °C grid, 1 row of 4 cells across a gap: 13.0, 14.5 | gap | 15.0 (55.4, 58.1, -, 59 °F)
    const sg: FieldGrid = { width: 4, height: 3, lat_first: 37, lat_step: -0.01, lon_first: -122, lon_step: 0.01, scale_factor: 0.1, add_offset: 0, nodata: 65535, transform: "none" };
    const row = [130, 145, 65535, 150];
    const c = Uint16Array.from([...row, ...row, ...row]);
    const labels = isothermLabels(sg, c, 2, 1);
    expect(labels.map((l) => l.text)).toEqual(["56°"]);
    expect(labels.every((l) => l.lon > -122 && l.lon < -121.99)).toBe(true);
  });
});
