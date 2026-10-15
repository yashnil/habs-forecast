import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import type { Manifest } from "@/generated/schema";
import { cellAt, decodeGrid, sample, valueOf } from "@/lib/grid";
import { charmLayer, charmLayers } from "@/lib/layers";
import manifestJson from "../fixture-data/v1/manifest.json";
import report from "../fixture-data/v1/verification/charm-points.json";

const FIX = path.resolve(__dirname, "../fixture-data/v1");
const manifest = manifestJson as unknown as Manifest;

async function codesFor(url: string) {
  const buf = readFileSync(path.join(FIX, url));
  return decodeGrid(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
}

describe("value grid decoding (browser code vs pipeline)", () => {
  it("decodes to the same values the pipeline verified against the source", async () => {
    const rows = (report.rows as Array<{ layer_id: string; status: string; lat: number; lon: number; grid_value: number | null }>).filter(
      (r) => r.status === "pass",
    );
    expect(rows.length).toBeGreaterThanOrEqual(50);
    for (const r of rows) {
      const layer = charmLayers(manifest).find((l) => l.layer_id === r.layer_id)!;
      const codes = await codesFor(layer.grid!.url);
      const c = cellAt(layer.grid!, r.lat, r.lon)!;
      const v = valueOf(layer.grid!, codes, c);
      if (r.grid_value === null) expect(v).toBeNull();
      else expect(v!).toBeCloseTo(r.grid_value, 5);
    }
  });

  it("returns null outside the grid", () => {
    const g = charmLayer(manifest, "pseudo_nitzschia", 0)!.grid!;
    expect(cellAt(g, 45, -122)).toBeNull();
    expect(cellAt(g, 37, -130)).toBeNull();
  });

  it("near shore without a toxin value, reports the nearest cell and its distance", async () => {
    const layer = charmLayer(manifest, "particulate_domoic", 1)!;
    const codes = await codesFor(layer.grid!.url);
    const wharf = sample(layer.grid!, codes, 36.604, -121.889);
    expect(wharf.kind).toBe("value");
    if (wharf.kind === "value") {
      expect(wharf.nearest).toBe(true);
      expect(wharf.distanceKm).toBeGreaterThan(2);
      expect(wharf.distanceKm).toBeLessThan(15);
    }
    const offshore = sample(layer.grid!, codes, 36.79, -121.95);
    expect(offshore.kind === "value" && !offshore.nearest).toBe(true);
  });

  it("never invents a value deep inland", async () => {
    const layer = charmLayer(manifest, "particulate_domoic", 1)!;
    const codes = await codesFor(layer.grid!.url);
    expect(sample(layer.grid!, codes, 37.4, -121.7).kind).toBe("none");
  });
});

describe("raster georeferencing", () => {
  it("image corners are the grid's cell edges, in MapLibre order", () => {
    for (const l of charmLayers(manifest)) {
      const g = l.grid!;
      const [tl, tr, br, bl] = l.image!.corners_lnglat;
      const west = g.lon_first - g.lon_step / 2;
      const east = g.lon_first + g.lon_step * (g.width - 0.5);
      const latEdges = [g.lat_first - g.lat_step / 2, g.lat_first + g.lat_step * (g.height - 0.5)];
      const north = Math.max(...latEdges);
      const south = Math.min(...latEdges);
      expect(tl).toEqual([expect.closeTo(west, 6), expect.closeTo(north, 6)]);
      expect(tr).toEqual([expect.closeTo(east, 6), expect.closeTo(north, 6)]);
      expect(br).toEqual([expect.closeTo(east, 6), expect.closeTo(south, 6)]);
      expect(bl).toEqual([expect.closeTo(west, 6), expect.closeTo(south, 6)]);
      expect(l.image!.crs).toBe("EPSG:3857");
    }
  });
});

describe("cell labels at harbour zoom (carto T2)", async () => {
  const { cellLabels } = await import("@/lib/grid");
  const g = { width: 4, height: 3, lat_first: 36.0, lat_step: 0.03, lon_first: -122.0, lon_step: 0.03, scale_factor: 1 / 65534, add_offset: 0, nodata: 65535, transform: "none" } as unknown as import("@/generated/schema").ValueGrid;
  const codes = new Uint16Array([65535, 32767, 65535, 6553, 65535, 65535, 52427, 65535, 65535, 65535, 65535, 65535]);
  it("labels only valued cells inside the bounds, nearest the centre first, with a legible colour flag", () => {
    const fc = cellLabels(g, codes, [-122.02, 35.98, -121.9, 36.05]);
    expect(fc.features.map((f) => f.properties!.text).sort()).toEqual(["10", "50", "80"]);
    const f80 = fc.features.find((f) => f.properties!.text === "80")!;
    expect(f80.geometry.coordinates).toEqual([-121.94, 36.03]);
    expect(f80.properties!.dark).toBe(true);
    expect(fc.features.find((f) => f.properties!.text === "10")!.properties!.dark).toBe(false);
    // a bounds that excludes a cell's centre excludes its label; a max caps the count nearest the centre
    expect(cellLabels(g, codes, [-122.0, 35.99, -121.95, 36.01]).features.map((f) => f.properties!.text)).toEqual(["50"]);
    expect(cellLabels(g, codes, [-122.02, 35.98, -121.9, 36.05], 1).features).toHaveLength(1);
  });
});
