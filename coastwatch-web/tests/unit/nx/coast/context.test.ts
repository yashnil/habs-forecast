import { describe, expect, it } from "vitest";
import type { SeasonGate } from "@/nx/coast/brief";
import { carriedQuery, carriedWords, HYPOTHETICAL, readCarried, tripFlag } from "@/nx/coast/context";
import { tripFishery } from "@/nx/coast/fisheries";

const crab = tripFishery("dungeness_commercial")!;
const season = (o: Partial<SeasonGate> = {}): SeasonGate => ({
  text: "Commercial season delayed to at least Jan 1, 2027 (humpback whale entanglements).",
  tone: "regulation",
  level: "delayed",
  conflict: false,
  seasonal: false,
  source: "CDFW",
  url: "https://wildlife.ca.gov/",
  checked: "2026-11-08T00:00:00Z",
  verified: false,
  stale: false,
  differ: null,
  ...o,
});

describe("a delayed or closed season makes the hypothetical unmistakable", () => {
  it("quotes an unverified record as the agency's, not verified, and says the trip cannot be made now", () => {
    const f = tripFlag(season(), crab)!;
    expect(f).toMatchObject({ level: "delayed", verified: false, source: "CDFW", short: "Season delayed · CDFW · not verified" });
    expect(f.words).toBe("CDFW's record, which nobody at Neritic has verified, says the dungeness crab season here is delayed. These figures are hypothetical: they do not describe a trip that can be made now. Check CDFW before you plan around them.");
    expect(tripFlag(season({ verified: null }), crab)!.verified).toBe(false);
  });
  it("a record a person reviewed is still the agency's record", () => {
    const f = tripFlag(season({ level: "closed", verified: true }), crab)!;
    expect(f.short).toBe("Season closed · CDFW");
    expect(f.words.startsWith("CDFW's record says the dungeness crab season here is closed.")).toBe(true);
    expect(tripFlag(season({ level: "closed", seasonal: true }), tripFishery("dungeness_recreational"))!.words).toContain("the dungeness crab season here is closed for the season");
  });
  it("says nothing for an open or restricted season, a missing record, or sources that differ", () => {
    expect(tripFlag(season({ level: "no_restriction_on_record" }), crab)).toBeNull();
    expect(tripFlag(season({ level: "watch" }), crab)).toBeNull();
    expect(tripFlag(season({ level: "unknown" }), crab)).toBeNull();
    expect(tripFlag(null, crab)).toBeNull();
    expect(tripFlag(season({ conflict: true }), crab)).toBeNull();
    expect(HYPOTHETICAL).toBe("Hypothetical, from your numbers.");
  });
});

describe("the point and date carried from Explore", () => {
  it("reads /coast?place=…&at=<lat>,<lon>&date=<iso>", () => {
    const c = readCarried(new URLSearchParams("place=port-592&at=36.8,-121.9&date=2026-11-14"));
    expect(c).toEqual({ at: { lat: 36.8, lon: -121.9 }, date: "2026-11-14" });
    expect(carriedWords(c)).toEqual({ point: "Point from Explore: 36°48.0′ N, 121°54.0′ W", date: "Date from Explore: Nov 14" });
    expect(carriedQuery(c)).toEqual({ at: "36.8,-121.9", date: "2026-11-14" });
    expect(readCarried({ at: "36.8,-121.9", date: "2026-11-14T18:00:00Z" }).date).toBe("2026-11-14T18:00:00Z");
    expect(carriedWords({ at: null, date: "2026-11-14T18:00:00Z" }).date).toBe("Date from Explore: Nov 14, 10:00 AM PST");
  });
  it("drops anything malformed instead of guessing", () => {
    expect(readCarried(new URLSearchParams("at=91,10&date=tomorrow"))).toEqual({ at: null, date: null });
    expect(readCarried(new URLSearchParams("at=36.8&date=2026-13-45"))).toEqual({ at: null, date: null });
    expect(readCarried(new URLSearchParams("at=36.8,-121.9,5"))).toEqual({ at: null, date: null });
    expect(readCarried({ at: ["36.8,-121.9"] })).toEqual({ at: null, date: null });
    expect(readCarried(new URLSearchParams(""))).toEqual({ at: null, date: null });
    expect(carriedWords({ at: null, date: null })).toEqual({ point: null, date: null });
    expect(carriedQuery({ at: null, date: null })).toEqual({});
  });
});
