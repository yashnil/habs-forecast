import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import type { OfficialDataset } from "@/generated/official";
import type { LayerArtifact } from "@/generated/schema";
import { speciesById } from "@/content/species";
import { associationsAt } from "@/lib/applicability";
import { nextSteps } from "@/lib/nextSteps";
import { checklist, checklistKey, parseTripDate, tripDays } from "@/lib/trip";
import { coastHref } from "@/lib/coast";

const FIX = path.resolve(__dirname, "../fixture-data/v1");
const m = JSON.parse(readFileSync(path.join(FIX, "manifest.json"), "utf8")) as { official_url: string };
const official: OfficialDataset = JSON.parse(readFileSync(path.join(FIX, m.official_url), "utf8"));
const CRESCENT = { lat: 41.74901, lon: -124.20398 };

const lead = (valid: string) => ({ time: { valid_date: valid } }) as unknown as LayerArtifact;

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
