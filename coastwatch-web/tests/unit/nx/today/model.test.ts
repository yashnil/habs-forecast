import { readFileSync } from "node:fs";
import path from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import type { Conditions } from "@/lib/conditions";
import { loadToday } from "@/pilot/today/load";
import { snapFacts } from "@/pilot/facts";
import { alertWords, buildToday, comingThatMatter, othersLine, parseWindow, periodNameWords, scopeWords, splitHeadline, suggestFisheries, windWords, type TodayInputs } from "@/nx/today/model";
import { advisoriesOf, buildGates, gateHrefs, orderSeaFacts, RELATION, type GateFact } from "@/nx/today/gates";
import { changeLine, sinceWords } from "@/nx/today/change";
import { distanceKm, phoneBounds, todayBounds } from "@/nx/today/camera";
import { kindOfFact, provenanceOf, shortSource } from "@/nx/today/provenance";
import { periodNumbers, type LanePeriod } from "@/pilot/today/marine";
import { fixtureServerEnv } from "../../fixtures/portToday";
import { gatesFrom } from "@/nx/coast/brief";
import { coastFishery } from "@/nx/coast/fisheries";

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

  it("a fishery with nothing on record is attributed to CDFW, which acts on fisheries, never CDPH (T42)", () => {
    const v = buildToday(inputs, { fishery: "dungeness_commercial", sea: null, place });
    const g = v.others.find((o) => o.fishery === "groundfish")!;
    expect(g.word).toBe("No restriction on record");
    expect(g.provenance?.source).toBe("CDFW");
    expect(g.provenance?.sourceLong).not.toMatch(/CDPH/);
    const sq = buildToday(inputs, { fishery: "market_squid", sea: null, place });
    if (sq.statement.state !== "ok") throw new Error("expected a statement");
    expect(sq.statement.provenance?.source).toBe("CDFW");
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
    // phones keep "highest of n" and the age; nothing speaks an average (T8)
    expect(t.short).toBe("Dungeness, highest of 6 · CDPH, Sep 28 (14 days)");
    expect(t.spoken).not.toMatch(/average|avg/);
  });

  it("the other test sites lead with the highest result, the censored count and the age, never the average (T7, T8)", () => {
    const v = buildToday(inputs, { fishery: "dungeness_commercial", sea: null, place });
    const r = v.rows.find((x) => x.id === "tests");
    if (!r || r.state === "unavailable") throw new Error("expected a tests row");
    const rock = r.detail.map((d) => d.text).find((x) => /rock crab/.test(x))!;
    expect(rock).toBe("Monterey Bay rock crab tested under the action level on Sep 28: highest 20 ppm of 30, all 6 samples under, 4 below detection. 14 days ago.");
    expect(JSON.stringify(r)).not.toMatch(/3\.8|avg|average/);
    expect(r.detail.find((d) => /rock crab/.test(d.text))!.provenance?.method).toMatch(/<2\.5, ND, ND, 20, <2\.5, 2\.6/);
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

describe("whale measures (T42)", () => {
  it("a commercial crabber gets the commercial clauses only, and Today and My Coast say the same thing", () => {
    const v = buildToday(inputs, { fishery: "dungeness_commercial", sea: null, place });
    const w = v.rows.find((r) => r.id === "whale");
    if (!w || w.state === "unavailable") throw new Error("expected a whale row");
    expect(w.value).toBe("Zone 4: Fleet Advisory, no depth or gear limit on record");
    expect(JSON.stringify(w)).not.toMatch(/recreational crab traps|traps prohibited/i);
    expect(w.detail[0].text).toMatch(/^CDFW, Zone 4, for commercial crabbers: “Commercial season delayed .*; Fleet Advisory\.” No depth limit or gear reduction on record\.$/);
    const g = gatesFrom(v.model, coastFishery("dungeness_commercial")!, { now: inputs.now, fixture: true });
    expect(g.whale?.measures).toBe("Fleet Advisory. No depth limit or gear reduction on record.");
  });

  it("a recreational crabber gets the recreational clause", () => {
    const v = buildToday(inputs, { fishery: "dungeness_recreational", sea: null, place });
    const w = v.rows.find((r) => r.id === "whale");
    if (!w || w.state === "unavailable") throw new Error("expected a whale row");
    expect(w.value).toBe("Zone 4: Recreational crab traps prohibited");
    expect(JSON.stringify(w)).not.toMatch(/Fleet Advisory|fathom/);
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

  it("an outlook whose dates have passed says 'past run', in Explore's words (T3, one helper)", () => {
    const v = buildToday({ ...inputs, now: "2026-10-22T18:00:00Z" }, { fishery: "dungeness_commercial", sea: null, place });
    expect(v.models.value).toMatch(/^Algae outlook, past run \(Oct \d+\)$/);
  });

  it("an outlook about usual is said as no signal", () => {
    const o = { ...inputs.outlook!, value: 1, typical: 1 };
    const v = buildToday({ ...inputs, outlook: o }, { fishery: "dungeness_commercial", sea: null, place });
    expect(v.models.value).toBe("Algae about usual for the season, not toxin");
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

describe("the three gates (v13 lead decision 6)", () => {
  /** the same fixture a month and a half later: the Sep 28 test is past TEST_STALE_DAYS */
  const later = () => ({ ...inputs, now: "2026-11-12T18:00:00Z" });
  const withAlert = (): Conditions => {
    const c = JSON.parse(JSON.stringify(sea)) as Conditions;
    const a = (c as unknown as { alerts: { status: string; data: unknown[] } }).alerts;
    if (a?.status === "ok" && Array.isArray(a.data) && !a.data.length) a.data.push({ event: "Small Craft Advisory", headline: "Small Craft Advisory in effect from Thursday morning through Friday morning", url: "https://www.weather.gov/mtr/", ends: "2026-10-14T16:00:00Z", severity: "Minor" });
    return c;
  };

  it("whale, then domoic acid, then sea; each with its source, date and age, and every in-app step keeps the port", () => {
    const v = buildToday(inputs, { fishery: "dungeness_commercial", sea, place });
    expect(v.gates.map((g) => g.id)).toEqual(["whale", "domoic", "sea"]);
    const [whale, domoic, seaGate] = v.gates;
    expect(whale).toMatchObject({ authority: "CDFW RAMP", dateRole: "assessed", state: "ok" });
    expect(whale.headline).toBe("Zone 4: Fleet Advisory, no depth or gear limit on record");
    expect(whale.facts[0].text).toMatch(/^Next whale assessment, subject to change: Dec 15–18$/);
    // CDFW's own clauses for the reader's sector, quoted, and never the recreational rule (T42)
    expect(whale.facts.find((f) => f.id === "cdfw")!.text).toMatch(/^CDFW, for commercial crabbers: “Commercial season delayed/);
    expect(whale.facts.map((f) => f.text).join(" ")).not.toMatch(/recreational/i);
    // the port's other CDPH sites sit in the domoic gate, highest result first, with their age
    expect(domoic.facts.find((f) => /rock crab/.test(f.text))!.text).toMatch(/highest 20 ppm of 30.*14 days ago\.$/);
    expect(whale.relation).toEqual([RELATION.whaleDecides]);
    // the three-line summary under the status: a few words each, never a verdict
    expect(v.gates.map((g) => g.summary)).toEqual(["Delay · next Dec 15–18", "Test 14 days ago", expect.stringMatching(/^Seas \d+–\d+ ft forecast$/)]);
    for (const g of v.gates) expect(g.summary.length).toBeLessThanOrEqual(25);
    // the whale stamp's sheet: the reader's own clauses first, the whole official sentence labelled (T42)
    const wp = whale.provenance!;
    expect(wp.value).not.toMatch(/recreational/i);
    expect(wp.caveats![0]).toMatch(/^For commercial crabbers, in CDFW's words: “Commercial season delayed/);
    expect(wp.caveats![0]).not.toMatch(/recreational/i);
    expect(wp.caveats![1]).toMatch(/^Full CDFW text \(all sectors\): “.*recreational/i);
    expect(domoic).toMatchObject({ authority: "CDPH tests", dateRole: "sampled", date: "2026-09-28", age: "14 days ago", state: "ok" });
    expect(domoic.headline).toBe("Dungeness test, Monterey Bay, Sep 28: highest 9.8 of 30 ppm, 6 samples");
    expect(seaGate).toMatchObject({ authority: "NWS and NDBC", dateRole: "issued", state: "ok" });
    for (const g of v.gates) {
      expect(g.actions.length).toBeGreaterThan(0);
      for (const a of g.actions) expect(a.external ? /^https:\/\//.test(a.href) : a.href.includes("place=port-592")).toBe(true);
    }
    expect(domoic.actions.find((a) => a.id === "research")!.href).toBe("/research?place=port-592&var=pDA");
    expect(seaGate.actions.find((a) => a.id === "explore")!.href).toBe("/explore?place=port-592&layer=sst");
    expect(seaGate.actions.find((a) => a.id === "trip")!.href).toBe("/coast?place=port-592&view=trip");
  });

  it("a test never softens the whale delay (T44): the fixed sentence is on the domoic gate whatever the result", () => {
    const d = buildToday(inputs, { fishery: "dungeness_commercial", sea: null, place }).gates.find((g) => g.id === "domoic")!;
    expect(d.relation).toContain(RELATION.testDoesNotLift);
    expect(d.relation).toContain(RELATION.testWhoDecides);
    expect(d.relation.join(" ")).toMatch(/usually/);
    // a fishery RAMP does not govern has no whale gate and no whale sentence
    const r = buildToday(inputs, { fishery: "rock_crab", sea: null, place });
    expect(r.gates.map((g) => g.id)).toEqual(["domoic", "sea"]);
    expect(r.gates[0].relation).not.toContain(RELATION.testDoesNotLift);
  });

  it("a test older than 30 days is history: worded as old in its gate and its row, never a figure on the chart", () => {
    const v = buildToday(later(), { fishery: "dungeness_commercial", sea: null, place });
    expect(v.soundings.find((s) => s.id === "test")).toBeUndefined();
    const d = v.gates.find((g) => g.id === "domoic")!;
    expect(d.state).toBe("old");
    expect(d.summary).toBe("Test 45 days ago");
    expect(d.age).toBe("45 days ago");
    expect(d.relation[0]).toBe("This test is older than 30 days: it is history, not a reading of the water now.");
    expect(d.headline).toMatch(/highest 9\.8 of 30 ppm/);
    // its words are not lost with the figure: the tests row now names it, with (old)
    const r = v.rows.find((x) => x.id === "tests");
    if (!r || r.state === "unavailable") throw new Error("expected a tests row");
    expect(r.state).toBe("stale");
    expect(r.value).toBe("2 sites nearby");
    expect(r.detail.some((x) => /Dungeness/.test(x.text) && /\(old\)/.test(x.text))).toBe(true);
    // the fresh test stays where it was
    expect(buildToday(inputs, { fishery: "dungeness_commercial", sea: null, place }).soundings.map((s) => s.id)).toEqual(["test"]);
  });

  it("an NWS advisory leads the sea gate and the page, stamped with NWS and its issue time, never 'not verified'", () => {
    const v = buildToday(inputs, { fishery: "dungeness_commercial", sea: withAlert(), place });
    const g = v.gates.find((x) => x.id === "sea")!;
    expect(g.facts[0].role).toBe("advisory");
    expect(g.headline).toBe("NWS Small Craft Advisory in effect from Thursday morning through Friday morning");
    expect(alertWords("Small craft advisory", "Small craft advisory in effect until 3 PM.")).toBe("NWS Small craft advisory in effect until 3 PM");
    expect(alertWords("Gale Warning", "Winds 35 kt expected")).toBe("NWS Gale Warning: Winds 35 kt expected");
    expect(v.advisories.map((a) => a.id)).toEqual(["alert-Small Craft Advisory"]);
    expect(g.summary).toBe("Small Craft Advisory");
    expect(v.advisories[0].provenance).toMatchObject({ kind: "advisory", source: "NWS", dateRole: "issued" });
    expect(v.agenda.find((a) => a.id === "alert-Small Craft Advisory")!.provenance?.kind).toBe("advisory");
    // only the "official" kind carries "not verified" on its stamp (ds Stamp): an advisory never does
    expect(v.advisories[0].provenance!.kind).not.toBe("official");
    expect(buildToday(inputs, { fishery: "dungeness_commercial", sea, place }).advisories).toEqual([]);
  });

  it("advisories sort first whatever order the facts arrive in", () => {
    const f = (id: string, role: GateFact["role"]): GateFact => ({ id, role, text: id });
    expect(orderSeaFacts([f("tide", "predicted"), f("waves", "measured"), f("sca", "advisory"), f("nws", "forecast"), f("gale", "advisory")]).map((x) => x.id)).toEqual(["sca", "gale", "nws", "waves", "tide"]);
    const gates = buildGates({ place: "p", whale: null, test: { state: "none" }, sea: { state: "ok", facts: [f("waves", "measured"), f("sca", "advisory")], marineUrl: null, issued: null, age: null } });
    expect(advisoriesOf(gates).map((x) => x.id)).toEqual(["sca"]);
  });

  it("the measured waves are in the sea gate and in a row for phones, where the chart drops the buoy figure", () => {
    const v = buildToday(inputs, { fishery: "dungeness_commercial", sea, place });
    const waves = v.gates.find((g) => g.id === "sea")!.facts.find((f) => f.id === "waves")!;
    expect(waves.text).toMatch(/^Waves \d+\.\d ft( at \d+ s)? at buoy 46236$/);
    expect(waves.provenance?.kind).toBe("measured");
    const row = v.rows.find((r) => r.id === "waves");
    if (!row || row.state === "unavailable") throw new Error("expected a waves row");
    expect(row.compactOnly).toBe(true);
    expect(row.value).toMatch(/^\d+\.\d ft( at \d+ s)?, buoy 46236$/);
  });

  it("states that are not values: a sea still loading, a sea that failed, no CDPH sidecar", () => {
    expect(buildToday(inputs, { fishery: "dungeness_commercial", sea: null, place }).gates.find((g) => g.id === "sea")!.state).toBe("loading");
    const e = buildToday(inputs, { fishery: "dungeness_commercial", sea: "error", place }).gates.find((g) => g.id === "sea")!;
    expect(e.state).toBe("unavailable");
    expect(e.actions[0]).toMatchObject({ id: "nws", external: true });
    const t = buildToday({ ...inputs, biotoxin: null }, { fishery: "dungeness_commercial", sea: null, place }).gates.find((g) => g.id === "domoic")!;
    expect(t.state).toBe("unavailable");
    expect(t.date).toBeNull();
  });

  it("gate addresses keep the port, encoded", () => {
    expect(gateHrefs.trip("port 5/92")).toBe("/coast?place=port%205%2F92&view=trip");
    expect(gateHrefs.alerts("port-592")).toBe("/alerts?place=port-592");
  });

  it("no forbidden verdict words in any gate (T36)", () => {
    for (const s of [sea, withAlert(), null, "error"] as const) {
      const words = JSON.stringify(buildToday(later(), { fishery: "dungeness_commercial", sea: s, place }).gates.map((g) => [g.title, g.headline, g.relation, g.facts.map((f) => f.text), g.actions.map((a) => a.label)]));
      expect(words).not.toMatch(/\b(safe|all clear|no-go|good day|hotspot|best spot|recommended|low risk|toxicity|toxin forecast|clean)\b/i);
    }
  });
});

describe("small words (v13)", () => {
  it("the other fisheries keep their sector, so a recreational rule never reads as the commercial one (T42)", () => {
    const v = buildToday(inputs, { fishery: "dungeness_commercial", sea: null, place });
    const line = othersLine(v.others);
    if (/^Dungeness/.test(line)) expect(line).toMatch(/^Dungeness crab, recreational: /);
    expect(othersLine([{ fishery: "dungeness_recreational", label: "Dungeness crab (recreational)", word: "Restricted", headline: "", flagged: true, provenance: null }, { fishery: "groundfish", label: "Groundfish", word: "No restriction on record", headline: "", flagged: false, provenance: null }])).toBe("Dungeness crab, recreational: restricted, 1 more");
  });

  it("NWS period names keep a weekday's capital; calm wind has no direction", () => {
    expect(periodNameWords("Tonight")).toBe("tonight");
    expect(periodNameWords("This Afternoon")).toBe("this afternoon");
    expect(periodNameWords("Thu")).toBe("Thu");
    expect(periodNameWords("Thursday Night")).toBe("Thursday night");
    expect(windWords(0.3, "E")).toBe("Calm");
    expect(windWords(6.2, "NW")).toBe("NW 6 kt");
    expect(windWords(12, null)).toBe("12 kt");
  });
});
