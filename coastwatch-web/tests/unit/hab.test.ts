import { describe, expect, it } from "vitest";
import type { ValueGrid } from "@/generated/schema";
import type { StationLite } from "@/lib/data";
import { ageWords, fmtMeasured, measuredEvidence, neighbourhood, ringPolygon, stationsAround } from "@/lib/hab";

const NOW = new Date("2026-10-13T19:00:00Z");
const HERE = { lat: 36.6, lon: -121.89 };

function station(id: string, lat: number, lon: number, samples: Record<string, [string, number | null][]>): StationLite {
  return {
    station_id: id,
    name: id,
    lat,
    lon,
    status: "active",
    source_url: "https://example.org",
    latest: Object.entries(samples).flatMap(([variable, s]) => {
      const last = [...s].reverse().find((x) => x[1] != null);
      return last ? [{ variable, label: variable, units: "", kind: "measured", value: last[1], date: last[0], qualifier: null, n_last_365d: s.length }] : [];
    }),
    recent: samples,
  };
}

// the nearest station has only old samples; one 41 km away sampled 13 days ago; one 300 km away yesterday
const wharf = station("near-old", 36.605, -121.89, { pDA: [["2022-08-31", 0.1]], pn_seriata: [["2025-08-27", 27100]] });
const cruz = station("cruz-recent", 36.96, -122.02, { pDA: [["2026-09-30", 0.202]], pn_seriata: [["2026-09-30", 4900]], pn_delicatissima: [["2026-09-30", 0]] });
const far = station("far-new", 34.0, -120.0, { pDA: [["2026-10-12", 1.5]] });

describe("measured evidence", () => {
  it("prefers a recent sample within 60 km over a nearer old one, and keeps its age and distance", () => {
    const e = measuredEvidence([wharf, cruz, far], HERE.lat, HERE.lon, "toxin", NOW)!;
    expect(e.station.station_id).toBe("cruz-recent");
    expect(e.ageDays).toBe(13);
    expect(e.near).toBe(true);
    expect(e.km).toBeGreaterThan(35);
    expect(e.values).toEqual([{ variable: "pDA", value: 0.202 }]);
  });

  it("falls back to the nearest old sample when nothing nearby is recent, never to a far recent one", () => {
    const e = measuredEvidence([wharf, far], HERE.lat, HERE.lon, "toxin", NOW)!;
    expect(e.station.station_id).toBe("near-old");
    expect(e.ageDays).toBeGreaterThan(1000);
  });

  it("names a far station only as not near", () => {
    const e = measuredEvidence([far], HERE.lat, HERE.lon, "toxin", NOW)!;
    expect(e.near).toBe(false);
  });

  it("reports both Pseudo-nitzschia size classes from the same sample, zeros included", () => {
    const e = measuredEvidence([cruz], HERE.lat, HERE.lon, "cells", NOW)!;
    expect(e.values).toEqual([
      { variable: "pn_seriata", value: 4900 },
      { variable: "pn_delicatissima", value: 0 },
    ]);
  });

  it("returns nothing when no station measures it", () => {
    expect(measuredEvidence([], HERE.lat, HERE.lon, "toxin", NOW)).toBeNull();
    expect(measuredEvidence(null, HERE.lat, HERE.lon, "chl", NOW)).toBeNull();
  });

  it("lists stations around the waters, nearest first, within the radius", () => {
    const a = stationsAround([far, cruz, wharf], HERE.lat, HERE.lon, NOW, 80);
    expect(a.map((x) => x.station.station_id)).toEqual(["near-old", "cruz-recent"]);
  });
});

// a 10 × 10 grid of 0.03° cells; value code 65535 = no value (land or the model's nearshore gap)
function grid(fill: (r: number, c: number) => number | null): { g: ValueGrid; codes: Uint16Array } {
  const g = { add_offset: 0, scale_factor: 0.01, height: 10, width: 10, lat_first: 36.45, lat_step: 0.03, lon_first: -122.05, lon_step: 0.03, max_quantization_error: 0.005, nodata: 65535, url: "x" } as ValueGrid;
  const codes = new Uint16Array(100);
  for (let r = 0; r < 10; r++) for (let c = 0; c < 10; c++) {
    const v = fill(r, c);
    codes[r * 10 + c] = v == null ? 65535 : Math.round(v * 100);
  }
  return { g, codes };
}

describe("model neighbourhood", () => {
  it("uses the waters' own cell when it has a value", () => {
    const { g, codes } = grid(() => 0.5);
    const n = neighbourhood(g, codes, 36.6, -121.9);
    expect(n.here).toBeCloseTo(0.5);
    expect(n.n).toBeGreaterThan(10);
  });

  it("when the waters' cell has no value, summarises only real cells within the radius and never fills the gap", () => {
    // columns 0–4 offshore (values rising westward), 5–9 the nearshore gap
    const { g, codes } = grid((_, c) => (c < 5 ? 0.2 + c * 0.1 : null));
    const n = neighbourhood(g, codes, 36.6, -121.88, 15);
    expect(n.here).toBeNull();
    expect(n.n).toBeGreaterThan(0);
    expect(n.cells.every((x) => x.cell.col < 5)).toBe(true);
    expect(n.min).toBeGreaterThanOrEqual(0.2 - 1e-9);
    expect(n.max).toBeLessThanOrEqual(0.6 + 1e-9);
    expect(n.median).not.toBeNull();
    expect(n.nearest!.km).toBeGreaterThan(0);
  });

  it("outside the model: no cells within the radius, the nearest found farther and named with its distance", () => {
    const { g, codes } = grid((r, c) => (r === 0 && c === 0 ? 0.8 : null));
    const n = neighbourhood(g, codes, 36.72, -121.8, 5);
    expect(n.n).toBe(0);
    expect(n.median).toBeNull();
    expect(n.nearest?.value).toBeCloseTo(0.8);
    expect(n.nearest!.km).toBeGreaterThan(5);
  });
});

describe("words and shapes", () => {
  it("states zeros as not quantified or none counted, never as clean", () => {
    expect(fmtMeasured("pDA", 0)).toBe("0 (not quantified)");
    expect(fmtMeasured("pn_seriata", 0)).toBe("0 cells/L (none counted)");
    expect(fmtMeasured("pn_seriata", 4900)).toBe("4,900 cells/L");
  });
  it("says ages in words a reader can weigh", () => {
    expect(ageWords(0)).toBe("today");
    expect(ageWords(13)).toBe("13 days ago");
    expect(ageWords(420)).toBe("14 months ago");
    expect(ageWords(1500)).toBe("4 years ago");
  });
  it("draws the neighbourhood ring as a closed polygon of the right size", () => {
    const ring = ringPolygon(36.6, -121.9, 15).coordinates[0];
    expect(ring[0]).toEqual(ring.at(-1));
    const north = Math.max(...ring.map((p) => p[1]));
    expect((north - 36.6) * 111.2).toBeCloseTo(15, 0);
  });
});
