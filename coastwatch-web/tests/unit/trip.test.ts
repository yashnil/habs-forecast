import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import type { OfficialDataset } from "@/generated/official";
import type { LayerArtifact } from "@/generated/schema";
import { speciesById } from "@/content/species";
import { associationsAt } from "@/lib/applicability";
import { nextSteps, speciesStatements } from "@/lib/nextSteps";
import type { StationEvidence, Neighbourhood } from "@/lib/hab";
import { checklist, checklistKey, parseTripDate, periodFor, topicWords, tripDays, whatMatters, type MatterInput } from "@/lib/trip";
import type { Conditions } from "@/lib/conditions";
import { coastHref } from "@/lib/coast";

const FIX = path.resolve(__dirname, "../fixture-data/v1");
const m = JSON.parse(readFileSync(path.join(FIX, "manifest.json"), "utf8")) as { official_url: string };
const official: OfficialDataset = JSON.parse(readFileSync(path.join(FIX, m.official_url), "utf8"));
const MONTEREY = { lat: 36.60024, lon: -121.89469 };
const CRESCENT = { lat: 41.74901, lon: -124.20398 };

const lead = (valid: string) => ({ time: { valid_date: valid } }) as unknown as LayerArtifact;
const nb = (over: Partial<Neighbourhood>): Neighbourhood => ({ n: 0, median: null, min: null, max: null, nearest: null, here: null, cells: [], ...over });
const ev = (over: Partial<StationEvidence>): StationEvidence =>
  ({
    kind: "cells",
    station: { station_id: "s", name: "Santa Cruz Wharf", lat: 36.96, lon: -122.02, status: "", source_url: "", latest: [] },
    km: 41,
    bearing: "NNW",
    date: "2026-09-30",
    values: [{ variable: "pn_seriata", value: 4900 }],
    ageDays: 13,
    series: {},
    near: true,
    ...over,
  }) as StationEvidence;

function input(over: Partial<MatterInput> & { at?: { lat: number; lon: number } }): MatterInput {
  const at = over.at ?? MONTEREY;
  const species = over.species ?? speciesById("any");
  const audience = over.audience ?? "all";
  return {
    assocs: associationsAt(official, "x", at.lat, at.lon),
    audience,
    species,
    statements: speciesStatements(official, species, audience),
    verification: null,
    hab: { cells: null, toxin: null },
    model: null,
    conditions: null,
    tripDate: "2026-10-13",
    fmt: (s) => s,
    ...over,
  };
}

describe("trip days", () => {
  it("offers today and the next three days, each with the model day that covers it", () => {
    const days = tripDays("2026-10-13", [lead("2026-10-11"), lead("2026-10-12"), lead("2026-10-13"), lead("2026-10-14")]);
    expect(days.map((d) => d.date)).toEqual(["2026-10-13", "2026-10-14", "2026-10-15", "2026-10-16"]);
    expect(days.map((d) => d.label)).toEqual(["Today", "Tomorrow", "Thu", "Fri"]);
    expect(days.map((d) => d.lead)).toEqual([2, 3, null, null]);
  });
  it("keeps a linked day only when it can be planned; links carry it", () => {
    const days = tripDays("2026-10-13", []);
    expect(parseTripDate("2026-10-15", days)).toBe("2026-10-15");
    expect(parseTripDate("2026-11-30", days)).toBe("2026-10-13");
    expect(parseTripDate(null, days)).toBe("2026-10-13");
    expect(coastHref({ place: "port-201", date: "2026-10-15" })).toBe("/coast?place=port-201&date=2026-10-15");
    expect(coastHref({ place: "port-201", date: "bad" })).toBe("/coast?place=port-201");
  });
});

describe("what matters right now", () => {
  it("a Dungeness crab trip leads with CDFW's season status, quoted, with its source", () => {
    const items = whatMatters(input({ at: CRESCENT, species: speciesById("dungeness"), audience: "commercial" }));
    const first = items[0];
    expect(first.id).toBe("season");
    expect(first.kind).toBe("official");
    expect(first.quote).toBeTruthy();
    expect(first.source?.url).toMatch(/^https:\/\/wildlife\.ca\.gov/);
    // the quote is the agency's words, unchanged
    const st = speciesStatements(official, speciesById("dungeness"), "commercial").find((s) => /season status/i.test(s.topic))!;
    expect(first.quote).toBe(st.statement);
  });

  it("a missing sample is a gap, never reassurance", () => {
    for (const at of [MONTEREY, CRESCENT]) {
      const items = whatMatters(input({ at }));
      const hab = items.find((x) => x.id === "hab")!;
      expect(hab.kind).toBe("gap");
      expect(`${hab.headline} ${hab.detail}`).toMatch(/not a clean result/);
      for (const x of items) expect(`${x.headline} ${x.detail ?? ""}`).not.toMatch(/\b(safe|all clear|clean)\b(?! result)/i);
    }
  });

  it("an offshore model value is never presented as the waters' own", () => {
    const items = whatMatters(input({ hab: { cells: null, toxin: null }, model: { valid: "2026-10-13", pda: nb({ n: 27, median: 0.74, min: 0.6, max: 0.79 }) } }));
    const hab = items.find((x) => x.id === "hab")!;
    expect(hab.kind).toBe("gap");
    expect(hab.detail).toMatch(/no value at these nearshore waters/i);
    expect(hab.detail).toMatch(/offshore cells within 15 km: median 74%/);
    // with the waters' own cell, it says so
    const own = whatMatters(input({ model: { valid: "2026-10-13", pda: nb({ here: 0.31, n: 5, median: 0.3 }) } })).find((x) => x.id === "hab")!;
    expect(own.kind).toBe("model");
    expect(own.headline).toMatch(/31%/);
    expect(own.detail).toMatch(/not a measurement/);
  });

  it("a recent nearby sample leads the HAB line with station, distance and age, and says seawater is not seafood", () => {
    const items = whatMatters(input({ hab: { cells: ev({}), toxin: null } }));
    const hab = items.find((x) => x.id === "hab")!;
    expect(hab.kind).toBe("measured");
    expect(hab.headline).toBe("Pseudo-nitzschia 4,900 cells/L at Santa Cruz Wharf, 13 days ago");
    expect(hab.detail).toMatch(/^41 km NNW of these waters/);
    expect(hab.detail).toMatch(/Seawater, not seafood/);
    // a sample far away or old never leads as "near these waters"
    const far = whatMatters(input({ hab: { cells: ev({ km: 394, near: false }), toxin: null } })).find((x) => x.id === "hab")!;
    expect(far.kind).toBe("gap");
    const old = whatMatters(input({ hab: { cells: ev({ ageDays: 200 }), toxin: null } })).find((x) => x.id === "hab")!;
    expect(old.kind).toBe("gap");
  });

  it("a day beyond the model says the model does not reach it", () => {
    const hab = whatMatters(input({ model: null, modelBeyond: true })).find((x) => x.id === "hab")!;
    expect(hab.detail).toMatch(/does not reach this day/);
  });

  it("notices that apply by their own geography are named, restrictive ones as alerts", () => {
    const items = whatMatters(input({ species: speciesById("anchovy-sardine"), audience: "recreational" }));
    const n = items.find((x) => x.id === "notices")!;
    expect(n.kind).toBe("official");
    expect(n.tone).toBe("alert");
    expect(n.headline).toMatch(/notices cover these waters for anchovy or sardine/);
  });

  it("conditions: NWS's period for the trip day, quoted, and an advisory as an alert", () => {
    const periods = [
      { name: "Tonight", text: "NW wind 10 to 15 kt. Seas 3 to 5 ft." },
      { name: "Wed", text: "NE wind around 5 kt. Seas 3 to 4 ft." },
      { name: "Wed Night", text: "SW wind around 5 kt. Seas 3 to 4 ft." },
      { name: "Thu", text: "NE wind 5 to 10 kt. Seas around 3 ft." },
    ];
    expect(periodFor(periods, "2026-10-13", "2026-10-13")?.name).toBe("Tonight");
    expect(periodFor(periods, "2026-10-15", "2026-10-13")?.name).toBe("Thu");
    expect(periodFor(periods, "2026-10-16", "2026-10-13")).toBeNull();
    const rest = whatMatters(input({ conditions: { lat: 0, lon: 0, marine: { status: "ok", data: { zone: "PZZ450", zoneName: "x", updated: null, periods: [{ name: "Rest Of Tonight", text: "N wind 5 to 10 kt. Seas 6 ft." }] }, source: "", url: "u", fetchedAt: "" }, alerts: { status: "ok", data: [], source: "", url: "", fetchedAt: "" }, buoy: { status: "none", reason: "", source: "" }, tides: { status: "none", reason: "", source: "" } }, tripDate: "2026-10-13", today: "2026-10-13" })).find((x) => x.id === "conditions")!;
    expect(rest.headline).toBe("NWS rest of tonight: N wind 5 to 10 kt · Seas 6 ft");
    const src = "NOAA National Weather Service";
    const cond = (alerts: { event: string; headline: string }[]): Conditions => ({
      lat: 0,
      lon: 0,
      marine: { status: "ok", data: { zone: "PZZ535", zoneName: "Monterey Bay", updated: null, periods }, source: src, url: "https://forecast.weather.gov/shmrn.php?mz=pzz535", fetchedAt: "" },
      alerts: { status: "ok", data: alerts.map((a) => ({ ...a, onset: null, ends: null, url: "" })), source: src, url: "", fetchedAt: "" },
      buoy: { status: "none", reason: "", source: src },
      tides: { status: "none", reason: "", source: src },
    });
    const calm = whatMatters(input({ conditions: cond([]), tripDate: "2026-10-15", today: "2026-10-13" })).find((x) => x.id === "conditions")!;
    expect(calm.headline).toBe("NWS for Thu: NE wind 5 to 10 kt · Seas around 3 ft");
    expect(calm.tone).toBe("note");
    expect(calm.detail).toMatch(/not a HAB or fishing forecast/);
    expect(calm.source?.label).toBe("NWS forecast, Monterey Bay");
    const sca = whatMatters(input({ conditions: cond([{ event: "Small craft advisory", headline: "Small craft advisory in effect until 9 pm pdt this evening" }]), tripDate: "2026-10-13", today: "2026-10-13" })).find((x) => x.id === "conditions")!;
    expect(sca.tone).toBe("alert");
    expect(sca.headline).toMatch(/^Small craft advisory · NWS tonight: NW wind 10 to 15 kt/);
  });

  it("topic words read inside a sentence", () => {
    expect(topicWords("Dungeness crab, commercial — marine toxin closures")).toBe("marine toxin closures");
  });
});

describe("before you leave", () => {
  const steps = (over: Partial<Parameters<typeof nextSteps>[0]>) =>
    nextSteps({
      official,
      verification: { state: "unverified", reasons: [], reviewedAt: null, reviewedBy: null, reviewAgeDays: null, lastCheckedAt: null },
      assocs: associationsAt(official, "x", CRESCENT.lat, CRESCENT.lon),
      audience: "commercial",
      species: speciesById("dungeness"),
      lat: CRESCENT.lat,
      conditions: null,
      lastToxin: null,
      fmt: (s) => s,
      ...over,
    });

  it("commercial Dungeness: season status first, the closure line, at most five, each with a source", () => {
    const { shown, more } = checklist(steps({}));
    expect(shown.length).toBeLessThanOrEqual(5);
    expect(shown[0].title).toMatch(/^Confirm the season status on CDFW's page/);
    expect(shown.some((s) => s.id === "closure-line")).toBe(true);
    for (const s of [...shown, ...more]) expect("url" in s.source ? s.source.url : s.source.tel).toBeTruthy();
    // one step per page: nothing opens the same page twice
    const keys = [...shown, ...more].map((s) => ("url" in s.source ? s.source.url : s.source.tel));
    expect(new Set(keys).size).toBe(keys.length);
    // buoy readings and the list reminder are not actions
    expect([...shown, ...more].some((s) => s.id === "buoy" || s.id === "verify")).toBe(false);
  });

  it("shellfish harvest puts CDPH's advisories and testing ahead of fishing rules", () => {
    const { shown } = checklist(steps({ audience: "shellfish", species: speciesById("razor-clam") }));
    const cdph = shown.findIndex((s) => s.id === "cdph-map");
    expect(cdph).toBeGreaterThanOrEqual(0);
    const rules = shown.findIndex((s) => /^(species-rules|region-)/.test(s.id));
    if (rules >= 0) expect(cdph).toBeLessThan(rules);
  });

  it("ticks are kept per trip", () => {
    expect(checklistKey("port-201", "commercial", "dungeness", "2026-10-14")).not.toBe(checklistKey("port-201", "commercial", "dungeness", "2026-10-15"));
  });
});
