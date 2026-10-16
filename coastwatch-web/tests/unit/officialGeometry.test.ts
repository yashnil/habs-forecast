import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import type { OfficialDataset } from "@/generated/official";
import { clipLimitsToWater, clipLineToWater, labelledGeometry, overWater } from "@/lib/officialGeometry";
import { decodeWaterFallback } from "@/lib/waterFallback";

const FIX = path.resolve(__dirname, "../fixture-data/v1");
const m = JSON.parse(readFileSync(path.join(FIX, "manifest.json"), "utf8")) as { official_url: string };
const official: OfficialDataset = JSON.parse(readFileSync(path.join(FIX, m.official_url), "utf8"));

describe("official geometry labels", () => {
  it("names each latitude limit by the notices it bounds, keeps the pipeline's label, changes no geometry", () => {
    const g = labelledGeometry(official)!;
    const limits = g.features.filter((f) => (f.properties as { kind?: string }).kind === "lat_limit");
    expect(limits.length).toBeGreaterThan(0);
    for (const f of limits) {
      const p = f.properties as { label: string; pipeline_label: string };
      expect(p.label).toMatch(/^[A-Z].* notices? · (north|south) limit \d+°\d+\.\d+′ N$|^Agency notice · /);
      expect(p.pipeline_label).toMatch(/· (north|south) limit$/);
    }
    const anchovy = limits.find((f) => (f.properties as { record_ids: string[] }).record_ids.some((id) => /anchovy/.test(id)));
    if (anchovy) expect((anchovy.properties as { label: string }).label).toMatch(/^Northern anchovy notices · (north|south) limit/);
    // geometry untouched
    const raw = official.geometry as unknown as GeoJSON.FeatureCollection;
    expect(g.features.map((f) => f.geometry)).toEqual(raw.features.map((f) => f.geometry));
    expect(g.features.length).toBe(raw.features.length);
  });

  it("returns null without a dataset", () => {
    expect(labelledGeometry(null)).toBeNull();
  });
});

// ---------------------------------------------------------------- limits clipped to water

/** A fallback file from rings in (lon, lat); the first ring of a polygon is its outer ring, the rest holes. */
const Q = 2 ** 26;
const merc = (lon: number, lat: number) => [(lon + 180) / 360, 0.5 - Math.log(Math.tan(Math.PI / 4 + (lat * Math.PI) / 360)) / (2 * Math.PI)];
function water(polygons: number[][][][], domain: [number, number, number, number] = [0, 0, 1, 1]) {
  const enc = (ring: number[][]) => {
    const q = ring.map(([lon, lat]) => merc(lon, lat)).map(([x, y]) => [Math.round(x * Q), Math.round(y * Q)]);
    return q.flatMap(([x, y], i) => (i ? [x - q[i - 1][0], y - q[i - 1][1]] : [x, y]));
  };
  return decodeWaterFallback({ quant: Q, domain_merc_q: domain.map((v) => Math.round(v * Q)) as [number, number, number, number], split_zoom: 8, polygons: polygons.map((p) => p.map(enc)) });
}
const box = (w: number, s: number, e: number, n: number) => [[w, s], [e, s], [e, n], [w, n]];

describe("official limits over water only", () => {
  // the sea west of -122.0 between 36 and 37 N, with an island (a hole) at -122.3..-122.2
  const sea = water([[box(-123, 36, -122, 37), box(-122.3, 36.4, -122.2, 36.6)]]);

  it("tells water from land and from an island inside the water", () => {
    expect(overWater(sea, -122.5, 36.5)).toBe(true);
    expect(overWater(sea, -121.9, 36.5)).toBe(false); // the shore side
    expect(overWater(sea, -122.25, 36.5)).toBe(false); // the island
    expect(overWater(sea, -122.25, 36.7)).toBe(true); // north of the island
    // outside the geometry's domain there is no coastline: water (unknown), as rasterWater treats it;
    // inside the domain, where no polygon is, it is land
    const west = water([[box(-123, 36, -122, 37)]], [0, 0, merc(-121.8, 36)[0], 1]);
    expect(overWater(west, -121.5, 36.5)).toBe(true); // east of the domain
    expect(overWater(west, -121.9, 36.5)).toBe(false); // in the domain, no water polygon
    expect(overWater(west, -124, 36.5)).toBe(false);
  });

  it("keeps the parts of a line over water, splitting at the island and stopping at the shore", () => {
    const parts = clipLineToWater([[-122.6, 36.5], [-121.8, 36.5]], sea);
    expect(parts).toHaveLength(2);
    // the first part runs from the line's own start to the island's west shore
    expect(parts[0][0]).toEqual([-122.6, 36.5]);
    expect(parts[0].at(-1)![0]).toBeCloseTo(-122.3, 2);
    // the second from the island's east shore to the coast at -122.0, never onto land
    expect(parts[1][0][0]).toBeCloseTo(-122.2, 2);
    expect(parts[1].at(-1)![0]).toBeCloseTo(-122.0, 2);
    for (const part of parts) for (const [lon] of part) expect(lon).toBeLessThanOrEqual(-122.0);
    // samples every ~150 m: a 0.2° run at 36.5 N (about 17.9 km) holds about 120 points
    expect(parts[1].length).toBeGreaterThan(100);
    expect(parts[1].length).toBeLessThan(140);
    // a line wholly on land yields nothing; one wholly at sea is kept whole
    expect(clipLineToWater([[-121.9, 36.5], [-121.5, 36.5]], sea)).toEqual([]);
    expect(clipLineToWater([[-122.9, 36.8], [-122.5, 36.8]], sea)).toHaveLength(1);
  });

  it("clips only the latitude limits, keeps every property and the feature count, and never invents a line on land", () => {
    const fc: GeoJSON.FeatureCollection = {
      type: "FeatureCollection",
      features: [
        { type: "Feature", properties: { kind: "lat_limit", label: "Anchovy notices · south limit 36°30′ N", pipeline_label: "36°30′ N · south limit", record_ids: ["a"] }, geometry: { type: "LineString", coordinates: [[-122.6, 36.5], [-121.8, 36.5]] } },
        { type: "Feature", properties: { kind: "lat_limit", label: "On land", pipeline_label: "x" }, geometry: { type: "LineString", coordinates: [[-121.9, 36.9], [-121.5, 36.9]] } },
        { type: "Feature", properties: { kind: "county", label: "Monterey" }, geometry: { type: "Polygon", coordinates: [[[-121.9, 36.5], [-121.5, 36.5], [-121.5, 36.9], [-121.9, 36.9], [-121.9, 36.5]]] } },
      ],
    };
    const out = clipLimitsToWater(fc, sea);
    expect(out.features).toHaveLength(3);
    expect(out.features[0].properties).toEqual(fc.features[0].properties);
    expect(out.features[0].geometry.type).toBe("MultiLineString");
    expect((out.features[0].geometry as GeoJSON.MultiLineString).coordinates).toHaveLength(2);
    expect((out.features[1].geometry as GeoJSON.MultiLineString).coordinates).toEqual([]);
    expect(out.features[2]).toBe(fc.features[2]);
    expect(fc.features[0].geometry.type).toBe("LineString"); // the input is not changed
  });

  it("with the bundled water, every published limit keeps a part at sea and loses nothing offshore", () => {
    const bundled = decodeWaterFallback(JSON.parse(readFileSync(path.resolve(__dirname, "../../public/basemap/water-fallback.json"), "utf8")));
    const g = clipLimitsToWater(labelledGeometry(official)!, bundled);
    const limits = g.features.filter((f) => (f.properties as { kind?: string }).kind === "lat_limit");
    expect(limits.length).toBeGreaterThan(0);
    const raw = (official.geometry as unknown as GeoJSON.FeatureCollection).features.filter((f) => (f.properties as { kind?: string }).kind === "lat_limit");
    limits.forEach((f, i) => {
      const parts = (f.geometry as GeoJSON.MultiLineString).coordinates;
      expect(parts.length).toBeGreaterThan(0);
      // the offshore start of the pipeline's line is kept exactly; the clipped line is never longer
      const [x0, y0] = (raw[i].geometry as GeoJSON.LineString).coordinates[0];
      expect(parts[0][0]).toEqual([+x0.toFixed(5), +y0.toFixed(5)]);
      const span = Math.max(...parts.flatMap((p) => p.map(([x]) => x))) - Math.min(...parts.flatMap((p) => p.map(([x]) => x)));
      const rawSpan = Math.abs((raw[i].geometry as GeoJSON.LineString).coordinates.at(-1)![0] - x0);
      expect(span).toBeLessThanOrEqual(rawSpan + 1e-5);
    });
  });
});
