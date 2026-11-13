import { describe, expect, it } from "vitest";
import { datesLine, pacificMs, pennantsFor, posterParts, seaBars, spanWords, zoneCaption, compassDeg, glyphFor } from "@/nx/today/ticket";
import type { LanePeriod } from "@/pilot/today/marine";

const lane = (date: string, night: boolean, lo: number | null, hi: number | null, name = "Tuesday"): LanePeriod =>
  ({ name, short: name.slice(0, 3), date, night, text: `${name}: seas ${lo} to ${hi} ft.`, dir: null, windLo: null, windHi: null, winds: [], gust: null, seasLo: lo, seasHi: hi, swell: null }) as LanePeriod;

describe("v14 Today ticket arrangements", () => {
  it("lifts only a trailing calendar date out of the record's statement", () => {
    expect(posterParts("Season delayed to at least Jan 1, 2027")).toEqual({ lede: "Season delayed to at least", hero: "Jan 1, 2027", date: "2027-01-01" });
    expect(posterParts("Closed until December 31", 2026)).toEqual({ lede: "Closed until", hero: "December 31", date: "2026-12-31" });
    expect(posterParts("No closure or restriction on record")).toBeNull();
    expect(posterParts("Sources differ")).toBeNull();
  });

  it("places Pacific wall-clock times across daylight saving", () => {
    expect(new Date(pacificMs("2026-10-13", 6)).toISOString()).toBe("2026-10-13T13:00:00.000Z");
    expect(new Date(pacificMs("2026-12-15", 6)).toISOString()).toBe("2026-12-15T14:00:00.000Z");
    expect(new Date(pacificMs("2026-10-13", 30)).toISOString()).toBe("2026-10-14T13:00:00.000Z");
  });

  it("draws the dates on record only from the record's dates", () => {
    const d = datesLine({ today: "2026-11-12", effective: "2026-10-30", earliest: "2027-01-01", window: { start: "2026-12-15", end: "2026-12-18" }, windowWords: "next whale assessment" })!;
    expect(d.span?.fromLabel).toBe("Oct 30");
    expect(d.span?.toLabel).toBe("earliest start");
    expect(d.window!.from).toBeLessThan(d.window!.to);
    expect(d.domain[0]).toBeLessThan(d.span!.from);
    expect(d.domain[1]).toBeGreaterThan(d.span!.to);
    expect(d.label).toContain("next whale assessment Dec 15 to Dec 18");
    expect(datesLine({ today: "2026-11-12", effective: null, earliest: null, window: null })).toBeNull();
    // an effective date in the future is not drawn as past
    expect(datesLine({ today: "2026-11-12", effective: "2026-11-20", earliest: "2027-01-01", window: null })!.span!.fromLabel).toBe("");
  });

  it("puts NWS periods, the buoy's one point and the advisories on the sea bars; a period without seas is left out", () => {
    const m = seaBars({
      lanes: [lane("2026-11-12", false, 9, 11, "Today"), lane("2026-11-12", true, 11, 11, "Tonight"), lane("2026-11-13", false, null, null, "Friday"), lane("2026-11-13", true, 18, 18, "Friday Night")],
      now: "2026-11-12T18:00:00Z",
      alerts: [{ event: "Small Craft Advisory", onset: null, ends: "2026-11-13T08:00:00Z" }, { event: "Gale Warning", onset: "2026-11-13T08:00:00Z", ends: null }],
      buoy: { id: "46014", waveFt: 13.1, at: "2026-11-12T18:00:00Z" },
    })!;
    expect(m.periods.map((p) => p.id)).toEqual(["2026-11-12d", "2026-11-12n", "2026-11-13n"]);
    expect(m.days.map((d) => d.label)).toEqual(["Thu 12", "Fri 13"]);
    expect(m.measured).toMatchObject({ v: 13.1, label: "Buoy 46014 · 10:00 AM" });
    expect(m.advisories.map((a) => [a.label, a.level])).toEqual([["Small craft", 1], ["Gale", 2]]);
    expect(m.advisories[1].to).toBe(m.domain[1]);
    expect(m.max).toBeGreaterThanOrEqual(18);
    expect(seaBars({ lanes: [], now: "2026-11-12T18:00:00Z", alerts: [], buoy: null })).toBeNull();
  });

  it("words and marks", () => {
    expect(pennantsFor("Small Craft Advisory")).toBe(1);
    expect(pennantsFor("Gale Warning")).toBe(2);
    expect(spanWords("Small Craft Advisory")).toBe("Small craft");
    expect(zoneCaption("delayed", "Fleet Advisory")).toBe("Season delayed · Fleet Advisory");
    expect(zoneCaption(null, null)).toBeNull();
    expect(compassDeg("SSE")).toBe(157.5);
    expect(compassDeg("calm")).toBeNull();
    expect(glyphFor({ kind: "official" })).toBe("unverified");
    expect(glyphFor({ kind: "measured", stale: true })).toBe("old");
    expect(glyphFor(null)).toBe("none");
  });
});
