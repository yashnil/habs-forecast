import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { SPECIES } from "@/content/species";
import { buildTodayModel, type PortTodayInputs, type StatusRow } from "@/pilot/facts";
import { COAST_LINES } from "@/v4/coast/coastIsobaths";
import { fisheryForTrip, tripStatusLine } from "@/v4/coast/tripStatus";

const dungeness = SPECIES.find((s) => s.id === "dungeness")!;
const row = (over: Partial<StatusRow>): StatusRow => ({ fishery: "dungeness_commercial", label: "Dungeness crab (commercial)", level: "delayed", severity: "red", word: "Delayed", scope: null, headline: "Commercial season delayed.", cause: null, causeWord: null, area: null, next_review: null, source: { name: "CDFW", url: null }, checked_at: "2026-10-12", verified: false, records: [], notices: [], stale: false, seasonal: false, conflict: null, fact: {} as StatusRow["fact"], ...over });

describe("the trip's status reads Today's model", () => {
  it("picks Today's fishery for the trip's species and activity", () => {
    expect(fisheryForTrip(dungeness, "commercial")).toBe("dungeness_commercial");
    expect(fisheryForTrip(dungeness, "recreational")).toBe("dungeness_recreational");
  });

  it("says the row's own word and headline, not a second composition", () => {
    const line = tripStatusLine([row({})], dungeness, "commercial");
    expect(line.kind).toBe("row");
    if (line.kind === "row") {
      expect(line.word).toBe("Delayed");
      expect(line.sentence).toBe("Commercial season delayed.");
    }
  });

  it("agrees with buildTodayModel for a curated commercial record", () => {
    const inputs: PortTodayInputs = {
      place: { place_id: "port-550", label: "Monterey", lat: 36.6, lon: -121.9 },
      now: "2026-10-13T06:00:00Z",
      dataGeneratedAt: "2026-10-12T00:00:00Z",
      selected: ["dungeness_commercial"],
      fisheries: { generated_at: "2026-10-12T00:00:00Z", records: [{ id: "x", fishery: "dungeness_commercial", label: "Dungeness crab (commercial)", level: "delayed", cause: "whale_entanglement", headline: "Commercial season delayed", detail: "Detail", area_label: "Central", applies_to_places: [], next_review: null, source_name: "CDFW", source_url: "https://wildlife.ca.gov/x", checked_at: "2026-10-12", review_state: "not_reviewed" }] } as never,
      biotoxin: null,
      notices: [],
      charm: null,
      outlook: null,
      conditions: null,
    };
    const model = buildTodayModel(inputs).status;
    const line = tripStatusLine(model, dungeness, "commercial");
    expect(line.kind).toBe("row");
    if (line.kind === "row") {
      expect(line.word).toBe(model[0].word);
      expect(line.sentence).toBe(model[0].headline);
    }
  });

  it("none on record never reads without 'not a promise it is open'", () => {
    const line = tripStatusLine([row({ level: "no_restriction_on_record", word: "None on record", headline: "No closure or restriction on record here." })], dungeness, "commercial");
    expect(line.kind === "row" && line.sentence).toBe("Not a promise it is open.");
  });

  it("a disputed or conflicting notice reads as unclear, awaiting a person's review", () => {
    const line = tripStatusLine([row({})], dungeness, "commercial", true);
    expect(line.kind === "row" && line.word).toBe("Status unclear");
    expect(line.kind === "row" && line.sentence).toContain("awaiting a person's review");
  });
});

describe("the landing's coast is real geometry", () => {
  it("every point is a vertex of public/basemap/isobaths.geojson, at its own depth", () => {
    const fc = JSON.parse(readFileSync(path.join(process.cwd(), "public/basemap/isobaths.geojson"), "utf8")) as { features: { properties: { depth_m: number }; geometry: { coordinates: [number, number][] } }[] };
    for (const line of COAST_LINES) {
      const source = new Set(fc.features.filter((f) => f.properties.depth_m === line.depth).flatMap((f) => f.geometry.coordinates.map((c) => c.join(","))));
      expect(line.points.length).toBeGreaterThan(20);
      for (const p of line.points) expect(source.has(p.join(",")), `${line.depth}: ${p}`).toBe(true);
    }
  });
});
