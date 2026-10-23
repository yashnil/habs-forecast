import { readFileSync } from "node:fs";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { LayerArtifact, Manifest } from "@/generated/schema";
import type { OfficialDataset } from "@/generated/official";
import { speciesById } from "@/content/species";
import { associationsAt, localNotices } from "@/lib/applicability";
import { defaultHour, observedCells, usableHours } from "@/lib/currents";
import { checkGraph } from "@/lib/evidenceCheck";
import { ageOn, dayRange, harborEvidence, type EvidenceInputs } from "@/lib/harborEvidence";
import { clockPT, dtf } from "@/lib/time";
import { resolve } from "@/v4/explore/layers";
import { timeSteps } from "@/v4/explore/Timebar";
import { clockLabel, dayLabel, pacificDay } from "@/v4/ui/time";

/**
 * Track S of the v4 final pass: the evidence wording says what is known, with absolute times and
 * real spans, and the currents default to an hour that actually saw the sea.
 */
const MONTEREY = { name: "Monterey", lat: 36.60024, lon: -121.89469 };
const failed = (rel: string) => JSON.parse(readFileSync(path.resolve(__dirname, "../fixture-data-failed/v1", rel), "utf8"));
const fmf = failed("manifest.json") as { official_url: string };
const disputedDs: OfficialDataset = failed(fmf.official_url);
const ID = "cdph-2026-sn26-018-monterey-bivalves";

function inputs(over: Partial<EvidenceInputs> = {}): EvidenceInputs {
  return {
    place: MONTEREY,
    now: new Date("2026-10-19T22:00:00Z"),
    today: "2026-10-19",
    tripDate: "2026-10-19",
    measured: { cells: null, toxin: null, nearest: null },
    model: null,
    sat: null,
    conditions: null,
    official: null,
    ...over,
  };
}

describe("S1: a disputed lift never leads as lifted", () => {
  const g = harborEvidence(inputs({ official: { assocs: associationsAt(disputedDs, "port-550", MONTEREY.lat, MONTEREY.lon), verification: null, audience: "all", species: speciesById("any") } }));
  it("the disputed line leads with 'Status unclear' and quotes the notice's own title", () => {
    const d = g.claims.find((c) => c.id === `disputed-${ID}`)!;
    expect(d.headline).toBe("Status unclear: CDPH notice “Do not eat sport-harvested bivalve shellfish from Monterey County”");
    expect(d.headline).not.toMatch(/lift/i);
    expect(d.detail).toMatch(/^CDPH release SN26-021 appears to lift it; no person has reviewed that release\./);
    expect(d.tone).toBe("alert");
  });
  it("the notice count says the unclear one exists without counting it", () => {
    const n = g.claims.find((c) => c.id === "notices")!;
    expect(n.headline).toMatch(/^\d+ notices? covers? these waters, plus 1 whose status is unclear$/);
    expect(n.supports).not.toContain(`official:${ID}`);
  });
});

describe("S2: the currents open on the newest hour with usable coverage", () => {
  const cov = (f: number) => ({ domain_observed_fraction: f, domain_reference_cells: 39563, reference: "", regions: [] });
  const hour = (h: string, f: number | null): LayerArtifact => ({ layer_id: `hfr2km_currents_20261019T${h}Z`, group_id: "currents", vectors: {}, time: { valid_time: `2026-10-19T${h}:00:00Z` }, coverage: f == null ? null : cov(f) }) as unknown as LayerArtifact;
  const m = { layers: [hour("09", 0.1066), hour("10", 0.1079), hour("11", 0.1095), hour("12", 0)] } as unknown as Manifest;
  it("a near-empty newest hour is not the default; it stays selectable, labelled incomplete with its cells", () => {
    expect(usableHours(m).get("hfr2km_currents_20261019T12Z")).toBe(false);
    expect(defaultHour(m)?.layer_id).toBe("hfr2km_currents_20261019T11Z");
    expect(resolve(m, { kind: "currents", hour: null, mean: false }, "").curLayer?.layer_id).toBe("hfr2km_currents_20261019T11Z");
    // an explicitly chosen incomplete hour is shown as chosen
    expect(resolve(m, { kind: "currents", hour: "20261019T12Z", mean: false }, "").curLayer?.layer_id).toBe("hfr2km_currents_20261019T12Z");
    const t = timeSteps(m, { kind: "currents", hour: null, mean: false }, null)!;
    expect(t.current).toBe("20261019T11Z");
    const last = t.steps.find((s) => s.key === "20261019T12Z")!;
    expect(last.sub).toBe("incomplete (0 cells)");
    expect(last.disabled).toBeFalsy();
    expect(observedCells(m.layers[2])).toBe(Math.round(0.1095 * 39563));
  });
  it("hours without a published coverage figure count as usable", () => {
    const m2 = { layers: [hour("10", null), hour("11", null)] } as unknown as Manifest;
    expect(defaultHour(m2)?.layer_id).toBe("hfr2km_currents_20261019T11Z");
  });
});

describe("S4–S6: named days, real spans, absolute times", () => {
  it("a satellite median names its span of days and its nearest clear pixel, and passes the evidence rules", () => {
    const g = harborEvidence(inputs({ sat: { value: 1.23, date: "2026-10-16", km: 0.849, sensor: "Sentinel-3 OLCI, 300 m", layerId: "olci300_latest", basis: { n: 40, min: 0.5, max: 3, radiusKm: 5, oldest: "2026-10-13", most: "2026-10-16" }, aside: null } }));
    const s = g.claims.find((c) => c.id === "satellite")!;
    expect(s.headline).toBe("Satellite chlorophyll 1.2 mg/m³, Oct 13–16, most Oct 16");
    expect(s.detail).toMatch(/^Median of 40 clear pixels within 5 km, nearest clear pixel 849 m away · Sentinel-3 OLCI, 300 m · 3 days old on Oct 19\. Algae of every kind, not toxin\.$/);
    expect(checkGraph(g).filter((v) => v.claim === "satellite")).toEqual([]);
    // one day only: just that day
    const one = harborEvidence(inputs({ sat: { value: 1.23, date: "2026-10-16", km: 0, sensor: "VIIRS, 750 m", layerId: "v", basis: null, aside: null } })).claims.find((c) => c.id === "satellite")!;
    expect(one.headline).toBe("Satellite chlorophyll 1.2 mg/m³, Oct 16");
    expect(dayRange("2026-09-29", "2026-10-02")).toBe("Sep 29–Oct 2");
  });
  it("the model's missing day is named: 'NOAA's newest run ends Oct 18; nothing for today (Oct 19)'", () => {
    const g = harborEvidence(inputs({ modelBeyond: true, model: { days: [{ lead: 0, valid: "2026-10-18", pda: null, pn: null }], issued: "2026-10-18", fresh: null } }));
    expect(g.claims.find((c) => c.id === "model")!.headline).toBe("NOAA's newest run ends Oct 18; nothing for today (Oct 19)");
  });
  it("ages are stated against the briefing's day, never 'today' or 'N days ago'", () => {
    expect(ageOn(0, "2026-10-19")).toBe("from the briefing's day (Oct 19)");
    expect(ageOn(4, "2026-10-19")).toBe("4 days old on Oct 19");
    expect(ageOn(21, "2026-10-19")).toBe("3 weeks old on Oct 19");
    const g = harborEvidence(inputs({ sat: { value: 2, date: "2026-10-19", km: 0, sensor: "VIIRS, 750 m", layerId: "v", basis: null, aside: null } }));
    for (const c of g.claims) expect(`${c.headline} ${c.detail ?? ""} ${c.text}`).not.toMatch(/\b(now|today|yesterday|days ago|weeks ago)\b/);
  });
});

describe("S11: date formatters are built once and reused", () => {
  afterEach(() => vi.restoreAllMocks());
  it("dtf returns the same formatter for the same options, and the helpers build none on repeat calls", () => {
    expect(dtf("en-US", { month: "short", timeZone: "UTC" })).toBe(dtf("en-US", { month: "short", timeZone: "UTC" }));
    // warm every formatter once, then count constructions
    const run = () => [dayLabel("2026-10-16", true), dayLabel("2025-08-27"), clockLabel("2026-10-16T16:24:00Z"), pacificDay("2026-10-17T02:00:00Z"), clockPT("2026-10-16T22:00:00Z")];
    const first = run();
    const spy = vi.spyOn(Intl, "DateTimeFormat");
    expect(run()).toEqual(first);
    expect(spy).not.toHaveBeenCalled();
    expect(first).toEqual(["Fri, Oct 16", "Aug 27, 2025", "Oct 16, 9:24 AM PDT", "2026-10-16", "3:00 PM PDT"]);
  });
});

describe("S9: any place on the map can lead the notices control", () => {
  it("localNotices gives a point the same tiers as a harbour", () => {
    const l = localNotices(disputedDs, "point-36.6,-121.9", "this point", MONTEREY.lat, MONTEREY.lon, "water")!;
    expect(l.name).toBe("this point");
    expect(l.covering.length).toBeGreaterThan(0);
    expect(localNotices(null, "x", "x", 0, 0)).toBeNull();
  });
});

describe("S7/S8: plain counts, named groups and years, honest coverage percents", () => {
  it("coverage is floored, never rounded up to a whole sea", async () => {
    const { coveragePct } = await import("@/lib/layers");
    expect(coveragePct(0.996)).toBe(">99%");
    expect(coveragePct(1)).toBe("100%");
    expect(coveragePct(0.987)).toBe("98%");
    expect(coveragePct(0.004)).toBe("<1%");
    expect(coveragePct(0)).toBe("0%");
  });
  it("a cell count names its size group, and a sample from another year carries its year", () => {
    const cells = { kind: "cells" as const, station: { station_id: "SantaCruzWharf", name: "Santa Cruz Wharf", lat: 36.958, lon: -122.017, status: "", source_url: "https://x", latest: [] }, km: 41.2, bearing: "NNW", date: "2026-10-12", values: [{ variable: "pn_seriata", value: 4900 }], ageDays: 7, series: {}, near: true };
    const closer = { station: { station_id: "MontereyWharf", name: "Monterey Wharf", lat: 36.605, lon: -121.89, status: "", source_url: "", latest: [] }, km: 0.6, date: "2025-08-12", ageDays: 433 };
    const g = harborEvidence(inputs({ measured: { cells, toxin: null, nearest: closer } as unknown as EvidenceInputs["measured"] }));
    const hab = g.claims.find((c) => c.id === "hab")!;
    expect(hab.headline).toBe("Santa Cruz Wharf, 41 km NNW: Pseudo-nitzschia (seriata group) 4,900 cells/L, Oct 12");
    expect(hab.detail).toContain("its last toxin or Pseudo-nitzschia sample is from Aug 12, 2025");
    expect(checkGraph(g).filter((v) => v.claim === "hab")).toEqual([]);
  });
});
