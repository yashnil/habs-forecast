import { describe, expect, it } from "vitest";
import { forecastFromSea } from "@/nx/trips/forecast";
import { advisoryEvidence, forecastChanges, observationEvidence } from "@/nx/trips/forecastDiff";
import type { EvidenceItem, ForecastSnap, Snapshot, SourceKey, SourceRead } from "@/nx/trips/types";
import { NOW, recordedSea, recordedZone, seaOf } from "./seaFixture";

const read = (state: SourceRead["state"] = "ok"): SourceRead => ({ state, at: NOW, detail: null });
const SOURCES = (["season", "whale", "cdphTest", "nwsForecast", "nwsAlerts", "buoy", "tides"] as SourceKey[]);
function snap(forecast: ForecastSnap | null, o: { takenAt?: string; depart?: string | null; evidence?: EvidenceItem[]; sources?: Partial<Record<SourceKey, SourceRead>> } = {}): Snapshot {
  return {
    schema: 1,
    id: Math.random().toString(36),
    tripId: "t1",
    kind: "planning",
    takenAt: o.takenAt ?? NOW,
    serverNow: NOW,
    mode: "live",
    savedFrom: null,
    sources: { ...Object.fromEntries(SOURCES.map((k) => [k, read()])), ...o.sources } as Snapshot["sources"],
    evidence: o.evidence ?? [],
    forecast,
    tides: null,
    assumptions: { limits: { windKt: 20, seasFt: 6 }, depart: o.depart === null ? null : { date: o.depart ?? "2026-10-14", time: null, timeZone: "America/Los_Angeles" }, economics: null },
    limitations: [],
  };
}

const rec = recordedZone("PZZ535");
const reissued = (edit: (name: string, text: string) => string, updated = "2026-10-13T03:30:00-07:00") => forecastFromSea(seaOf("PZZ535", updated, rec.periods.map((p) => ({ name: p.name, text: edit(p.name, p.text) }))));

describe("forecastChanges", () => {
  const base = snap(forecastFromSea(recordedSea("PZZ535")));

  it("a reissue with identical numbers is no change", () => {
    const next = snap(reissued((_, t) => t), { takenAt: NOW });
    expect(forecastChanges(base, next, NOW)).toEqual([]);
  });

  it("reworded text with the same numbers is no change", () => {
    const next = snap(reissued((_, t) => t.replace("Wave Detail:", "Wave detail:")));
    expect(forecastChanges(base, next, NOW)).toEqual([]);
  });

  it("stronger wind in the departure period: priority 1, says so, before/after, url and period key", () => {
    const next = snap(reissued((n, t) => (n === "Wed" ? t.replace("NE wind 5 to 10 kt", "NE wind 15 to 25 kt") : t)));
    const cs = forecastChanges(base, next, NOW);
    expect(cs).toHaveLength(1);
    expect(cs[0]).toMatchObject({ priority: 1, category: "forecast", kind: "revised", periodKey: "2026-10-14|day", before: "5–10 kt", after: "15–25 kt", url: "https://api.weather.gov/zones/forecast/PZZ535/forecast" });
    expect(cs[0].title).toBe("Marine forecast updated since you planned: stronger winds during your selected departure period.");
  });

  it("higher seas and a direction change outside the departure date are lower priority, each its own change", () => {
    const next = snap(reissued((n, t) => (n === "Tue Night" ? t.replace("Seas 3 to 4 ft", "Seas 5 to 7 ft").replace("becoming NE", "becoming SE") : t)));
    const cs = forecastChanges(base, next, NOW);
    expect(cs.map((c) => [c.periodKey, c.priority, c.before, c.after])).toEqual([
      ["2026-10-13|night", 2, "3–4 ft", "5–7 ft"],
      ["2026-10-13|night", 3, "N → NE", "N → SE"],
    ]);
    expect(cs[0].title).toContain("higher seas for Tue 13, night");
  });

  it("different valid times are never compared (a bulletin shifted by a day)", () => {
    const shifted = forecastFromSea(seaOf("PZZ535", "2026-10-13T20:10:00-07:00", rec.periods.slice(2).map((p, i) => ({ name: rec.periods[i].name, text: p.text }))));
    // shifted: the 13th's night now holds what was the 14th's night text etc. Only the same keys are compared.
    const cs = forecastChanges(base, snap(shifted), NOW).filter((c) => c.kind === "revised");
    for (const c of cs) {
      const a = base.forecast!.periods.find((p) => p.key === c.periodKey)!;
      const b = shifted.periods.find((p) => p.key === c.periodKey)!;
      expect(a.key).toBe(b.key);
    }
    expect(cs.every((c) => c.periodKey !== null)).toBe(true);
  });

  it("a different zone is said once and periods are not compared", () => {
    const cs = forecastChanges(base, snap(forecastFromSea(recordedSea("PZZ540"))), NOW);
    expect(cs).toHaveLength(1);
    expect(cs[0].title).toContain("PZZ535 → PZZ540");
  });

  it("periods ended since planning, and a stale forecast", () => {
    const later = "2026-10-15T02:00:00Z";
    const cs = forecastChanges(base, snap(forecastFromSea(recordedSea("PZZ535"))), later);
    const expired = cs.filter((c) => c.kind === "expired");
    expect(expired.map((c) => c.title)).toEqual([
      "The forecast period you planned to depart in has ended (Wed 14, day).",
      "3 forecast periods you planned with have ended (Mon 12, night to Tue 13, night).",
    ]);
    expect(expired[0].periodKey).toBe("2026-10-14|day");
    expect(cs.find((c) => c.kind === "stale")!.title).toMatch(/issued \d+ hours ago/);
  });

  it("new, revised and ended advisories, only when both alert reads completed", () => {
    const sca = advisoryEvidence({ event: "Small Craft Advisory", headline: "Small Craft Advisory until 3 AM PDT", url: "u1", onset: "2026-10-12T21:06:00-07:00", ends: "2026-10-13T03:00:00-07:00" }, "PZZ535");
    const ext = advisoryEvidence({ event: "Small Craft Advisory", headline: "Small Craft Advisory until Wednesday 3 PM", url: "u2", onset: "2026-10-12T21:06:00-07:00", ends: "2026-10-14T15:00:00-07:00" }, "PZZ535");
    const f = forecastFromSea(recordedSea("PZZ535"));
    expect(sca.key).toBe(ext.key);
    const n = forecastChanges(snap(f), snap(f, { evidence: [ext] }), NOW);
    expect(n[0]).toMatchObject({ category: "advisory", kind: "new", priority: 1, url: "u2" });
    expect(n[0].title).toBe("NWS issued a Small Craft Advisory during your selected departure period since you planned.");
    expect(forecastChanges(snap(f, { evidence: [sca] }), snap(f, { evidence: [ext] }), NOW)[0].kind).toBe("revised");
    expect(forecastChanges(snap(f, { evidence: [sca] }), snap(f, { evidence: [sca] }), NOW)).toEqual([]);
    expect(forecastChanges(snap(f, { evidence: [sca] }), snap(f), NOW)[0].kind).toBe("removed");
    // alerts did not answer: no claim that it ended
    expect(forecastChanges(snap(f, { evidence: [sca] }), snap(f, { sources: { nwsAlerts: read("failed") } }), NOW)).toEqual([]);
  });

  it("a newer buoy reading near port: one observation change per station, priority 3", () => {
    const f = forecastFromSea(recordedSea("PZZ535"));
    const b = { id: "46042", name: "Monterey", km: 30, url: "https://www.ndbc.noaa.gov/station_page.php?station=46042", windKt: 12, windDir: "NW", gustKt: null, periodS: 10, waveFt: 4.6 };
    const o1 = observationEvidence({ ...b, at: "2026-10-13T05:00:00Z" });
    const o2 = observationEvidence({ ...b, at: "2026-10-13T06:00:00Z", waveFt: 5.2 });
    const cs = forecastChanges(snap(f, { evidence: [o1] }), snap(f, { evidence: [o2, o2] }), NOW);
    expect(cs).toHaveLength(1);
    expect(cs[0]).toMatchObject({ category: "observation", priority: 3, evidenceKey: "observation:ndbc:46042" });
    expect(cs[0].title).toBe("Newer buoy reading near port: waves 5.2 ft at 10 s, wind NW 12 kt.");
    expect(forecastChanges(snap(f, { evidence: [o1] }), snap(f, { evidence: [o1] }), NOW)).toEqual([]);
  });

  it("change ids are stable", () => {
    const next = snap(reissued((n, t) => (n === "Wed" ? t.replace("NE wind 5 to 10 kt", "NE wind 15 to 25 kt") : t)));
    expect(forecastChanges(base, next, NOW)[0].id).toBe(forecastChanges(base, next, NOW)[0].id);
  });
});
