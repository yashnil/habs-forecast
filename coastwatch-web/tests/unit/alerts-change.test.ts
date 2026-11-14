import { describe, expect, it } from "vitest";
import { asOfWord, recordWords } from "@/nx/alerts/words";
import { alertChange, daysWords, firstDate, shortTitle } from "@/nx/alerts/change";

/** The Alerts ticket's before → after (src/nx/alerts/change.ts): dates read from the record's own words, never made up. */
const iso = (d: Date | null) => (d ? d.toISOString().slice(0, 10) : null);
const ref = new Date("2026-12-19T14:42:00Z");

describe("firstDate", () => {
  it("reads a named date with its year, and an ISO day", () => {
    expect(iso(firstDate("Commercial season delayed to at least Jan 15, 2027", ref))).toBe("2027-01-15");
    expect(iso(firstDate("Opens January 1, 2027 with a 30-fathom depth limit.", ref))).toBe("2027-01-01");
    expect(iso(firstDate("Source: CDFW, 2026-12-18.", ref))).toBe("2026-12-18");
  });
  it("takes the first date the words name", () => {
    expect(iso(firstDate("Delayed to Jan 15, 2027; next assessment Dec 18, 2026", ref))).toBe("2027-01-15");
  });
  it("gives a date without a year the year nearest the record", () => {
    expect(iso(firstDate("next assessment Dec 15", new Date("2027-01-03T12:00:00Z")))).toBe("2026-12-15");
    expect(iso(firstDate("opens Jan 1", ref))).toBe("2027-01-01");
  });
  it("names no date when the words name none, and never invents an impossible one", () => {
    expect(firstDate("Fleet advisory: humpback whales present", ref)).toBeNull();
    expect(firstDate("30-fathom depth limit", ref)).toBeNull();
    expect(firstDate("Feb 30, 2027", ref)).toBeNull();
    expect(firstDate(null, ref)).toBeNull();
  });
});

describe("alertChange", () => {
  const item = { created_at: "2026-12-19T14:42:00Z", as_of: "2026-12-18T18:00:00Z" };
  it("a moved date: was, now and the days between", () => {
    const c = alertChange({ ...item, change: "changed", title: "Fort Bragg: Dungeness crab (commercial): Commercial season delayed to at least Jan 15, 2027", was: "Dungeness crab (commercial): Commercial season delayed to at least Jan 1, 2027" });
    expect([iso(c.was), iso(c.now), c.days]).toEqual(["2027-01-01", "2027-01-15", 14]);
  });
  it("the same date in new words (delayed → opens) is a change of 0 days", () => {
    const c = alertChange({ ...item, change: "changed", title: "Bodega Bay: Dungeness crab (commercial): Commercial season opens Jan 1, 2027 with a 30-fathom depth limit.", was: "Dungeness crab (commercial): Commercial season delayed to at least Jan 1, 2027" });
    expect(c.days).toBe(0);
  });
  it("a new record has no before; a record no longer listed has neither", () => {
    expect(alertChange({ ...item, change: "new", title: "X: Season delayed to at least Jan 1, 2027", was: null })).toMatchObject({ was: null, days: null });
    expect(alertChange({ ...item, change: "gone", title: "X: no longer listed — Season delayed to at least Jan 1, 2027", was: null })).toEqual({ now: null, was: null, days: null });
  });
});

describe("daysWords", () => {
  it("says the gap in the reader's words", () => {
    expect(daysWords(14)).toEqual({ figure: "+14 days", note: "later" });
    expect(daysWords(1)).toEqual({ figure: "+1 day", note: "later" });
    expect(daysWords(-3)).toEqual({ figure: "−3 days", note: "earlier" });
    expect(daysWords(0).figure).toBe("Same date");
  });
});

describe("shortTitle", () => {
  const base = { place_label: "Fort Bragg", created_at: "2026-12-19T14:42:00Z", as_of: null, trigger: "status_change" as const };
  it("the record's subject and what changed", () => {
    expect(shortTitle({ ...base, change: "changed", title: "Fort Bragg: Dungeness crab (commercial): Commercial season delayed to at least Jan 15, 2027", was: "Dungeness crab (commercial): Commercial season delayed to at least Jan 1, 2027" })).toBe("Commercial Dungeness delay moved");
    expect(shortTitle({ ...base, change: "changed", title: "Fort Bragg: Dungeness crab (commercial): Commercial season opens Jan 1, 2027", was: "Dungeness crab (commercial): Commercial season delayed to at least Jan 1, 2027" })).toBe("Commercial Dungeness status changed");
    expect(shortTitle({ ...base, change: "gone", title: "Fort Bragg: no longer listed — Rock crab: Advisory", was: null })).toBe("Rock crab record no longer listed");
    expect(shortTitle({ ...base, trigger: "new_toxin_result", change: "new", title: "Fort Bragg: Dungeness crab test 9.8 ppm", was: null })).toBe("New toxin test");
  });
});

describe("the record's words on the ticket", () => {
  it("a quotation that ends a sentence ends it once, and days read as the reader's", () => {
    expect(recordWords("Changed since 2026-11-08. Was: “Commercial season delayed to at least Jan 1, 2027.”. Source: CDFW, 2026-11-08.", "2026-11-13T12:00:00Z")).toBe(
      "Changed since Nov 8. Was: “Commercial season delayed to at least Jan 1, 2027.” Source: CDFW, Nov 8.",
    );
    expect(recordWords("Was: “Season delayed”.")).toBe("Was: “Season delayed”.");
  });
  it("a record's date is when Neritic checked it; a toxin result's is its sample date", () => {
    expect(asOfWord({ trigger: "status_change" })).toBe("checked");
    expect(asOfWord({ trigger: "ramp_action" })).toBe("checked");
    expect(asOfWord({ trigger: "new_toxin_result" })).toBe("sampled");
  });
});
