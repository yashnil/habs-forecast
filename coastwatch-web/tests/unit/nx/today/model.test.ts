import { readFileSync } from "node:fs";
import path from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import type { Conditions } from "@/lib/conditions";
import { loadToday } from "@/pilot/today/load";
import { snapFacts } from "@/pilot/facts";
import { buildToday, comingThatMatter, othersLine, parseWindow, scopeWords, splitHeadline, suggestFisheries, type TodayInputs } from "@/nx/today/model";
import { changeLine, sinceWords } from "@/nx/today/change";
import { distanceKm, phoneBounds, todayBounds } from "@/nx/today/camera";
import { kindOfFact, provenanceOf, shortSource } from "@/nx/today/provenance";
import { periodNumbers, type LanePeriod } from "@/pilot/today/marine";
import { fixtureServerEnv } from "../../fixtures/portToday";

/**
 * Today v12's view model (src/nx/today/model.ts) on the committed fixture, loaded the way the page loads it
 * (src/pilot/today/load.ts), with the recorded NOAA answers. Truth rules from docs/coastwatch/v12/research/
 * E-science-and-perf.md section 2 are checked where the model makes a choice.
 */
const env = fixtureServerEnv();
const saved: Record<string, string | undefined> = {};
let inputs: TodayInputs;
const sea = JSON.parse(readFileSync(path.join(__dirname, "..", "..", "fixtures", "today-conditions.json"), "utf8")) as Conditions;
const place = { key: "port-592", name: "Moss Landing" };

beforeAll(async () => {
  for (const [k, v] of Object.entries(env)) {
    saved[k] = process.env[k];
    process.env[k] = v;
  }
  const l = await loadToday("port-592");
  if (!l.ok || !l.inputs) throw new Error("fixture did not load");
  inputs = l.inputs;
}, 30_000);
afterAll(() => {
  for (const k of Object.keys(env)) {
    if (saved[k] === undefined) delete process.env[k];
    else process.env[k] = saved[k];
  }
});

const ALL_WORDS = (v: ReturnType<typeof buildToday>) =>
  JSON.stringify({
    s: v.statement.state === "ok" ? [v.statement.title, v.statement.why] : null,
    a: v.agenda.map((x) => [x.title, x.detail]),
    r: v.rows.map((r) => (r.state === "unavailable" ? r.note : [r.value, r.detail.map((d) => [d.text, d.provenance?.caveats])])),
    m: [v.models.value, v.models.charm?.headline, v.models.charm?.words, v.models.outlook?.headline, v.models.outlook?.words, v.models.outlook?.provenance?.caveats],
    so: v.soundings.map((x) => [x.label, x.unit, x.caption, x.short]),
  });

describe("the status statement", () => {
  it("leads with the reader's fishery, in the record's words, official and not verified", () => {
    const v = buildToday(inputs, { fishery: "dungeness_commercial", sea: null, place });
    expect(v.statement.state).toBe("ok");
    if (v.statement.state !== "ok") return;
    expect(v.statement.title).toBe("Season delayed to at least Jan 1, 2027");
    expect(v.statement.why).toBe("Humpback whale entanglements");
    expect(v.statement.tone).toBe("regulation");
    expect(v.statement.provenance?.kind).toBe("official");
    expect(v.statement.source).toEqual({ agency: "CDFW", url: "https://wildlife.ca.gov/Conservation/Marine/Whale-Safe-Fisheries" });
  });

  it("a fishery with nothing on record says so, and that it does not mean open (T37)", () => {
    const v = buildToday(inputs, { fishery: "market_squid", sea: null, place });
    if (v.statement.state !== "ok") throw new Error("expected a statement");
    expect(v.statement.title).toBe("No closure or restriction on record");
    expect(v.statement.why).toMatch(/does not mean the season is open/);
    expect(v.statement.tone).toBe("ink");
  });

  it("an unreadable season list is a designed unavailable state, never a calm one", () => {
    const v = buildToday({ ...inputs, fisheries: null }, { fishery: "groundfish", sea: null, place });
    expect(v.statement.state).toBe("unavailable");
  });

  it("splits only a trailing reason and keeps the words", () => {
    expect(splitHeadline("Commercial season delayed to at least Jan 1, 2027 (humpback whale entanglements).")).toEqual({ title: "Commercial season delayed to at least Jan 1, 2027", why: "Humpback whale entanglements" });
    expect(splitHeadline("No take (bait only) since Sep 11.")).toEqual({ title: "No take (bait only) since Sep 11", why: null });
    expect(scopeWords("Dungeness crab (commercial)")).toBe("Dungeness crab, commercial");
  });
});

describe("the next decision date", () => {
  it("reads CDFW's window into dates, the first as the BigDate and the next as an agenda item", () => {
    expect(parseWindow("December 15-18 (then January 13-15), subject to change", "2026-12-15")).toEqual([
      { start: "2026-12-15", end: "2026-12-18" },
      { start: "2027-01-13", end: "2027-01-15" },
    ]);
    expect(parseWindow("unreadable", "2026-12-15")).toEqual([]);
    const v = buildToday(inputs, { fishery: "dungeness_commercial", sea: null, place });
    expect(v.decision).toMatchObject({ start: "2026-12-15", end: "2026-12-18", tentative: true, label: "Next whale assessment, subject to change", sameAs: true });
    expect(v.agenda.find((a) => a.id === "assessment-1")).toMatchObject({ start: "2027-01-13", title: "Following whale assessment", tentative: true });
  });

  it("is absent when the record names no future date", () => {
    expect(buildToday(inputs, { fishery: "market_squid", sea: null, place }).decision).toBeNull();
  });
});

describe("the agenda and the soundings", () => {
  it("the latest test: at its place on the chart with max of the action level and age (T8), its agenda line without the figure", () => {
    const v = buildToday(inputs, { fishery: "dungeness_commercial", sea: null, place });
    const t = v.soundings.find((s) => s.id === "test")!;
    expect(t).toMatchObject({ value: "9.8", unit: "of 30 ppm", label: "Dungeness viscera, highest of 6", lon: -121.9, lat: 36.8 });
    expect(t.caption.join(" ")).toMatch(/CDPH, Sep 28, 14 days ago/);
    expect(t.provenance?.kind).toBe("measured");
    const a = v.agenda.find((x) => x.id === "test")!;
    expect(a.detail).toBe("All 6 under the action level. 14 days ago.");
    expect(a.detail).not.toMatch(/9\.8/);
  });

  it("the buoy is a sounding only once the sea has loaded, measured, with its own time; NWS today is a prediction", () => {
    expect(buildToday(inputs, { fishery: "dungeness_commercial", sea: null, place }).soundings.map((s) => s.id)).toEqual(["test"]);
    const v = buildToday(inputs, { fishery: "dungeness_commercial", sea, place });
    const b = v.soundings.find((s) => s.id === "buoy")!;
    expect(b.label).toMatch(/^(Waves|Wind), buoy 46236$/);
    expect(b.caption[0]).toMatch(/^NDBC, Oct \d+, \d+:\d\d (AM|PM) PT$/);
    expect(v.soundings.length).toBeLessThanOrEqual(2);
    const today = v.agenda.find((x) => x.id === "today");
    expect(today?.provenance?.kind).toBe("prediction");
  });

  it("a sea that could not be read is a worded row, and nothing about it is invented", () => {
    const v = buildToday(inputs, { fishery: "dungeness_commercial", sea: "error", place });
    expect(v.rows.find((r) => r.id === "sea")).toMatchObject({ state: "unavailable" });
    expect(v.agenda.some((a) => a.id === "today")).toBe(false);
  });

  it("no CDPH sidecar: the tests row says so", () => {
    const v = buildToday({ ...inputs, biotoxin: null }, { fishery: "dungeness_commercial", sea: null, place });
    expect(v.rows.find((r) => r.id === "tests")).toMatchObject({ state: "unavailable" });
    expect(v.soundings).toEqual([]);
  });

  it("coming NWS periods: only those well above today, at most two", () => {
    const lane = (date: string, text: string, night = false): LanePeriod => ({ ...periodNumbers(text), name: date, short: "", date, night, text });
    const first = lane("2026-10-12", "NW wind 5 to 10 kt. Seas 3 to 5 ft.");
    const out = comingThatMatter([first, lane("2026-10-13", "Seas 4 to 6 ft."), lane("2026-10-14", "NW wind 20 to 25 kt. Seas 6 to 8 ft."), lane("2026-10-15", "Seas 11 to 15 ft."), lane("2026-10-16", "Seas 12 ft.")], first, "2026-10-12");
    expect(out.map((p) => p.date)).toEqual(["2026-10-14", "2026-10-15"]);
  });
});

describe("models behind a tap (T3, T5, T9–T12)", () => {
  it("C-HARM is named with its event, says past run when past, and carries the mandatory words; the outlook says algae, not toxin", () => {
    const v = buildToday(inputs, { fishery: "dungeness_commercial", sea: null, place });
    expect(v.models.charm?.past).toBe(true);
    expect(v.models.charm?.headline).toMatch(/^Past run of C-HARM/);
    expect(v.models.charm?.words).toMatch(/P\(pDA > 500 ng\/L\)/);
    expect(v.models.charm?.words).toMatch(/probability of conditions in the water, not toxin in seafood/);
    expect(v.models.outlook?.signal).toBe("higher");
    expect(v.models.value).toBe("Algae higher than usual, not toxin");
    expect(v.models.outlook?.provenance?.kind).toBe("outlook");
  });

  it("an outlook about usual is said as no signal", () => {
    const o = { ...inputs.outlook!, value: 1, typical: 1 };
    const v = buildToday({ ...inputs, outlook: o }, { fishery: "dungeness_commercial", sea: null, place });
    expect(v.models.value).toBe("No signal");
    expect(v.models.outlook?.headline).toMatch(/^No signal/);
  });

  it("no forbidden verdict words anywhere in the view (T36)", () => {
    for (const f of ["dungeness_commercial", "market_squid", "anchovy_sardine", "rock_crab"] as const) {
      const words = ALL_WORDS(buildToday(inputs, { fishery: f, sea, place }));
      expect(words).not.toMatch(/\b(safe|all clear|no-go|good day|hotspot|best spot|recommended|low risk|toxicity|toxin forecast)\b/i);
    }
  });
});

describe("other fisheries and suggestions", () => {
  it("one compact line, the flagged one first", () => {
    const v = buildToday(inputs, { fishery: "dungeness_commercial", sea: null, place });
    expect(othersLine(v.others)).toMatch(/^.+: (delayed|restricted|closed|watch|season closed|sport harvest watch).*, \d+ more$/i);
    expect(v.others.find((o) => o.fishery === "dungeness_commercial")).toBeUndefined();
  });

  it("suggests only fisheries Neritic has a test site or a record for at this port, in the list's order", () => {
    expect(suggestFisheries(inputs)).toEqual(["dungeness_commercial", "rock_crab", "anchovy_sardine"]);
    expect(suggestFisheries({ ...inputs, biotoxin: null, fisheries: null })).toEqual([]);
  });
});

describe("since your last visit (Journey 2)", () => {
  const now = "2026-10-13T06:00:00Z";
  it("no earlier visit: no line", () => {
    const v = buildToday(inputs, { fishery: "dungeness_commercial", sea, place });
    expect(changeLine(null, v.allFacts, now)).toBeNull();
  });

  it("nothing changed: one line that says so", () => {
    const v = buildToday(inputs, { fishery: "dungeness_commercial", sea, place });
    const l = changeLine({ at: "2026-10-08T17:00:00Z", facts: snapFacts(v.allFacts) }, v.allFacts, now)!;
    expect(l.text).toBe("Since Oct 8: no change in season status, tests, notices or models.");
    expect(l.changes).toEqual([]);
  });

  it("a status that changed is named, computed from the stored facts; a change of fishery is not news", () => {
    const v = buildToday(inputs, { fishery: "dungeness_commercial", sea, place });
    const other = buildToday(inputs, { fishery: "rock_crab", sea, place });
    expect(changeLine({ at: "2026-10-08T17:00:00Z", facts: snapFacts(other.allFacts) }, v.allFacts, now)!.changes).toEqual([]);
    const before = snapFacts(v.allFacts).map((f) => (f.id === "status:dungeness_commercial" ? { ...f, value: "No restriction on record", headline: "x" } : f));
    const l = changeLine({ at: "2026-10-08T17:00:00Z", facts: before }, v.allFacts, now)!;
    expect(l.text).toBe("Since Oct 8: Dungeness crab (commercial) is now delayed.");
    expect(sinceWords("2026-10-13T04:10:00Z", now)).toBe("9:10 PM PT");
  });
});

describe("camera", () => {
  const port = { lon: -121.787, lat: 36.804 };
  it("fits the port and the water that matters, never closer than about 20 km either side", () => {
    const [[w, s], [e, n]] = todayBounds(port, [{ lon: -121.9, lat: 36.8 }, { lon: -121.95, lat: 36.759 }]);
    expect(w).toBeLessThan(-121.95);
    expect(e).toBeGreaterThan(port.lon);
    expect(distanceKm({ lon: w, lat: s }, { lon: e, lat: s })).toBeGreaterThanOrEqual(39);
    expect(n).toBeGreaterThan(port.lat);
  });

  it("leaves out a point far from the port", () => {
    const near = todayBounds(port, [{ lon: -121.9, lat: 36.8 }]);
    expect(todayBounds(port, [{ lon: -121.9, lat: 36.8 }, { lon: -123.5, lat: 37.5 }])).toEqual(near);
  });

  it("phone: the port has room to its east, and open water west and south of the points takes their type", () => {
    const [[w, s], [e, n]] = phoneBounds(port, [{ lon: -121.9, lat: 36.8 }]);
    expect(distanceKm({ lon: port.lon, lat: port.lat }, { lon: e, lat: port.lat })).toBeGreaterThan(20);
    expect(port.lat - s).toBeGreaterThan(n - port.lat);
    expect(distanceKm({ lon: -121.9, lat: 36.8 }, { lon: w, lat: 36.8 })).toBeGreaterThan(15);
  });
});

describe("provenance", () => {
  it("kind words are T1's, sources short", () => {
    expect(kindOfFact({ basis: "forecast", kind: "bloom_forecast" })).toBe("model");
    expect(kindOfFact({ basis: "forecast", kind: "tide" })).toBe("prediction");
    expect(kindOfFact({ basis: "official", kind: "status" })).toBe("official");
    expect(shortSource("NOAA National Weather Service")).toBe("NWS");
    expect(shortSource("CDPH Summary of Domoic Acid Levels in Crab")).toBe("CDPH");
    expect(provenanceOf({ id: "x", place_id: "p", kind: "status", severity: "info", headline: "h", basis: "official", verified: false, source: { name: "CDFW", url: "/x" }, as_of: null }, "t")).toBeNull();
  });
});
