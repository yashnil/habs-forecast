import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import type { OfficialDataset } from "@/generated/official";
import { labelledGeometry } from "@/lib/officialGeometry";

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
