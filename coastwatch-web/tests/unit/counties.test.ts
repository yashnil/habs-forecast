import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { COUNTIES } from "@/content/copy";

const fc = JSON.parse(readFileSync(path.resolve(__dirname, "../../public/basemap/ca-counties.json"), "utf8"));
const labels = fc.features.filter((f: { properties: { kind: string } }) => f.properties.kind === "label");
const borders = fc.features.filter((f: { properties: { kind: string } }) => f.properties.kind === "border");

// The 58 counties (California State Association of Counties list)
const NAMES = "Alameda Alpine Amador Butte Calaveras Colusa Contra_Costa Del_Norte El_Dorado Fresno Glenn Humboldt Imperial Inyo Kern Kings Lake Lassen Los_Angeles Madera Marin Mariposa Mendocino Merced Modoc Mono Monterey Napa Nevada Orange Placer Plumas Riverside Sacramento San_Benito San_Bernardino San_Diego San_Francisco San_Joaquin San_Luis_Obispo San_Mateo Santa_Barbara Santa_Clara Santa_Cruz Shasta Sierra Siskiyou Solano Sonoma Stanislaus Sutter Tehama Trinity Tulare Tuolumne Ventura Yolo Yuba"
  .split(" ")
  .map((n) => n.replace(/_/g, " "));
const OCEAN = ["Del Norte", "Humboldt", "Mendocino", "Sonoma", "Marin", "San Francisco", "San Mateo", "Santa Cruz", "Monterey", "San Luis Obispo", "Santa Barbara", "Ventura", "Los Angeles", "Orange", "San Diego"];

describe("county reference layer (public/basemap/ca-counties.json)", () => {
  it("has exactly the 58 counties, each once, with its FIPS code", () => {
    expect(labels.map((f: { properties: { name: string } }) => f.properties.name).sort()).toEqual([...NAMES].sort());
    const fips = labels.map((f: { properties: { fips: string } }) => f.properties.fips);
    expect(new Set(fips).size).toBe(58);
    for (const f of fips) expect(f).toMatch(/^06\d{3}$/);
  });

  it("marks the 15 ocean-facing counties, and Bay counties separately", () => {
    const by = (k: string) => labels.filter((f: { properties: { coast: string } }) => f.properties.coast === k).map((f: { properties: { name: string } }) => f.properties.name).sort();
    expect(by("ocean")).toEqual([...OCEAN].sort());
    expect(by("bay")).toEqual(["Alameda", "Contra Costa", "Santa Clara", "Solano"]);
  });

  it("keeps labels in California and the borders as lines only (no polygons, no offshore areas)", () => {
    for (const f of labels) {
      const [lon, lat] = f.geometry.coordinates;
      expect(lon).toBeGreaterThan(-124.5);
      expect(lon).toBeLessThan(-114);
      expect(lat).toBeGreaterThan(32.5);
      expect(lat).toBeLessThan(42.1);
    }
    expect(borders).toHaveLength(1);
    expect(["LineString", "MultiLineString"]).toContain(borders[0].geometry.type);
    expect(fc.features.some((f: { geometry: { type: string } }) => /Polygon/.test(f.geometry.type))).toBe(false);
  });

  it("records its source, licence and version, and says what county lines are not", () => {
    expect(fc.source.url).toBe("https://gis.data.ca.gov/datasets/CDT::california-county-boundaries-and-identifiers");
    expect(fc.source.license).toMatch(/Public domain/);
    expect(fc.source.sha256).toMatch(/^[0-9a-f]{64}$/);
    expect(COUNTIES.note).toMatch(/do not define fishery closures, crab fishing zones, state waters/);
    expect(COUNTIES.attribution).toMatch(/CA Dept\. of Technology/);
  });
});
