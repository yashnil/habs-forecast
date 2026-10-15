import { describe, expect, it } from "vitest";
import type { StationEvidence } from "@/lib/hab";
import { harborEvidence, representative, stationSetting, type EvidenceInputs } from "@/lib/harborEvidence";

/**
 * The harbor evidence graph: one place, every reading typed by how it is known, with its distance,
 * age and representativeness, and claims that cite the nodes they rest on. The acceptance
 * statement for Monterey is the product brief's own example.
 */
const MONTEREY = { name: "Monterey", lat: 36.60024, lon: -121.89469 };
const NOW = new Date("2026-10-15T20:00:00Z");

const santaCruz: StationEvidence = {
  kind: "cells",
  station: { station_id: "SantaCruzWharf", name: "Santa Cruz Wharf", lat: 36.958, lon: -122.017, status: "", source_url: "https://erddap.sccoos.org/erddap/tabledap/HABs-SantaCruzWharf.html", latest: [] },
  km: 41.2,
  bearing: "NNW",
  date: "2026-09-30",
  values: [{ variable: "pn_seriata", value: 4900 }],
  ageDays: 15,
  series: {},
  near: true,
};
const montereyWharf = { station: { station_id: "MontereyWharf", name: "Monterey Wharf", lat: 36.605, lon: -121.89, status: "", source_url: "", latest: [] }, km: 0.6, date: "2025-08-12", ageDays: 429 };
const offshore = { n: 27, median: 0.77, min: 0.73, max: 0.79, nearest: { cell: { row: 1, col: 1, lat: 36.55, lon: -121.95 }, value: 0.74, km: 6.6 }, here: null, cells: [] };

function inputs(over: Partial<EvidenceInputs> = {}): EvidenceInputs {
  return {
    place: MONTEREY,
    now: NOW,
    tripDate: "2026-10-15",
    measured: { cells: santaCruz, toxin: null, nearest: montereyWharf },
    model: { days: [{ lead: 2, valid: "2026-10-15", pda: offshore, pn: null }], issued: "2026-10-14", fresh: { state: "current", basisDate: "2026-10-14", ageDays: 1 } },
    sat: null,
    conditions: null,
    official: null,
    ...over,
  };
}

describe("harbor evidence graph", () => {
  it("states the measurement with its place, date and distance, names the closer silent station, and says the model has no estimate here", () => {
    const g = harborEvidence(inputs());
    const measured = g.claims.find((c) => c.id === "measured")!;
    expect(measured.text).toBe("Santa Cruz Wharf reported 4,900 cells/L Pseudo-nitzschia on September 30, 41 km NNW of Monterey. Monterey Wharf is closer (0.6 km) but has no comparable sample since August 12, 2025.");
    expect(measured.supports).toEqual(["pn_cells:SantaCruzWharf"]);
    const model = g.claims.find((c) => c.id === "model")!;
    expect(model.kind).toBe("gap");
    expect(model.text).toMatch(/^NOAA's C-HARM model has no valid estimate in these nearshore waters\. Its nearest cell is 6\.6 km out; the 27 offshore cells within 15 km read 73%–79% for October 15, context for the open water, not the harbor\.$/);
    // no claim speaks of safety, catch or a harbour-wide risk score
    for (const c of g.claims) expect(c.text).not.toMatch(/\b(safe|unsafe|risk score|catch)\b/i);
    // every supporting id is a node in the graph
    const ids = new Set(g.nodes.map((n) => n.id));
    for (const c of g.claims) for (const s of c.supports) expect(ids.has(s)).toBe(true);
  });

  it("types each node by how it is known and never presents offshore cells as the waters' own", () => {
    const g = harborEvidence(inputs());
    const sc = g.nodes.find((n) => n.id === "pn_cells:SantaCruzWharf")!;
    expect(sc.class).toBe("measurement");
    expect(sc.where).toMatchObject({ km: 41.2, bearing: "NNW", representative: "context", setting: "exposed-pier" });
    expect(sc.caveats[0]).toMatch(/Seawater, not seafood/);
    const m = g.nodes.find((n) => n.id.startsWith("charm:"))!;
    expect(m.id).toBe("charm:offshore:2026-10-15");
    expect(m.class).toBe("published_forecast");
    expect(m.where.representative).toBe("context");
    expect(m.support).toMatchObject({ kind: "cells", n: 27, radiusKm: 15, nativeM: 3000 });
    expect(m.text).toBe("73%–79%");
    expect(g.gaps.find((x) => x.id === "model-here")!.reason).toMatch(/no valid estimate in these nearshore waters; its nearest cell with a value is 6\.6 km out/);
  });

  it("with the waters' own cell, the model is 'at' and the claim says cell, probability, not measurement", () => {
    const g = harborEvidence(inputs({ model: { days: [{ lead: 2, valid: "2026-10-15", pda: { ...offshore, here: 0.31 }, pn: null }], issued: "2026-10-14", fresh: { state: "current", basisDate: "2026-10-14", ageDays: 1 } } }));
    const m = g.nodes.find((n) => n.id === "charm:own:2026-10-15")!;
    expect(m.where.representative).toBe("at");
    expect(g.claims.find((c) => c.id === "model")!.text).toBe("NOAA's C-HARM gives the 3 km cell containing Monterey a 31% chance that particulate domoic acid exceeds its threshold on October 15. A probability, not a measurement.");
  });

  it("an old run is named by its date and age in the claim and the node, not called a failure", () => {
    const g = harborEvidence(inputs({ model: { days: [{ lead: 2, valid: "2026-10-15", pda: { ...offshore, here: 0.31 }, pn: null }], issued: "2026-10-12", fresh: { state: "stale", basisDate: "2026-10-12", ageDays: 3 } } }));
    expect(g.claims.find((c) => c.id === "model")!.text).toMatch(/\(run of Oct 12, 3 days old\)/);
    expect(g.nodes.find((n) => n.id.startsWith("charm:"))!.freshness).toEqual({ state: "stale", note: "run of Oct 12, 3 days old" });
  });

  it("says a gap as a gap: no recent sample, nothing near, a model day beyond the run, no clear satellite view", () => {
    const g = harborEvidence(
      inputs({
        measured: { cells: { ...santaCruz, ageDays: 200, date: "2026-03-29" }, toxin: null, nearest: null },
        model: { days: [{ lead: 3, valid: "2026-10-16", pda: offshore, pn: null }], issued: "2026-10-14", fresh: null },
        tripDate: "2026-10-18",
        sat: { value: null, sensor: "Sentinel-3 OLCI, 300 m", windowDays: 7, flagged: false, radiusKm: 5 },
      }),
    );
    expect(g.claims.find((c) => c.id === "measured")).toBeUndefined();
    expect(g.gaps.map((x) => x.id)).toEqual(["pn_cells-recent", "model-day", "satellite"]);
    expect(g.gaps[0].reason).toMatch(/No Pseudo-nitzschia cells sample within 60 km in the last 30 days; the latest nearby is 7 months ago at Santa Cruz Wharf \(41 km\)/);
    expect(g.gaps[2].reason).toMatch(/not low chlorophyll/);
    // the old sample is still a node, flagged historical, so nothing is hidden
    expect(g.nodes.find((n) => n.id === "pn_cells:SantaCruzWharf")!.freshness.state).toBe("historical");
  });

  it("representativeness is per variable, not one radius", () => {
    expect(representative("pn_cells", 1.5)).toBe("at");
    expect(representative("pn_cells", 12)).toBe("near");
    expect(representative("pn_cells", 41)).toBe("context");
    expect(representative("pn_cells", 90)).toBe("far");
    expect(representative("chl_satellite", 3)).toBe("near");
    expect(representative("chl_satellite", 12)).toBe("context");
    expect(representative("wind", 25)).toBe("near");
    expect(representative("water_temp", 25)).toBe("context");
    expect(stationSetting("Santa Cruz Wharf")).toBe("exposed-pier");
    expect(stationSetting("Moss Landing, South Harbor, CA")).toBe("sheltered-harbor");
    expect(stationSetting("Monterey Canyon Outer, CA (156)")).toBe("offshore");
    expect(stationSetting("MBM1")).toBe("unclassified");
  });
});
