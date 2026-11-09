import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import path from "node:path";
import type { BiotoxinIndex, FisheriesStatusIndex, PortFact } from "@/pilot/contracts";
import type { Conditions } from "@/lib/conditions";
import { buildPortToday, buildTodayModel, diffFacts, parseFisheries, rampZoneFor, skillWords, snapFacts, testSentence, type CharmInput, type NoticeInput, type OutlookInput, type PortTodayInputs } from "@/pilot/facts";
import { parseBiotoxin, parseFisheriesStatus } from "@/pilot/data";

// the data track's fixtures (pilot/data: `cwp fixture`), as the page reads them
const fx = (rel: string) => JSON.parse(readFileSync(path.join(__dirname, "..", "fixture-data", "v1", rel), "utf8"));
const BIO = parseBiotoxin(fx("biotoxin/index.json"))! as BiotoxinIndex;
const FISH = parseFisheriesStatus(fx("fisheries-status/index.json"))! as FisheriesStatusIndex;
const NOW = "2026-10-13T13:00:00Z"; // 6 AM PDT

const charm = (p: number, state: CharmInput["state"] = "current", here = false): CharmInput => ({
  issued: "2026-10-12",
  state,
  ageDays: state === "current" ? 1 : 9,
  url: "https://coastalscience.noaa.gov",
  days: [0, 1, 2, 3].map((lead) => ({ lead, date: `2026-10-${11 + lead}`, pda: { here: here ? p : null, median: p, max: p + 0.05, n: 20, nearestKm: 3 }, pn: { here: null, median: p / 2, max: p, n: 20, nearestKm: 3 } })),
});
const OUTLOOK: OutlookInput = {
  id: "chl_outlook_8d",
  label: "Chlorophyll outlook, next 8 days",
  issued: "2026-10-12",
  valid_from: "2026-10-13",
  valid_to: "2026-10-20",
  value: 3.2,
  range: null,
  basis: "nearby",
  skill: { metric: "RMSE ln(mg m-3)", model: 0.754, climatology: 0.85, persistence: 0.892, n: 185000, test_period: "2019 to 2021-06" },
  caveats: [],
  method: "climatology + AR(1)",
};
const base = (over: Partial<PortTodayInputs> = {}): PortTodayInputs => ({
  place: { place_id: "port-592", label: "Moss Landing", lat: 36.8044, lon: -121.78691 },
  now: NOW,
  dataGeneratedAt: "2026-10-12T18:00:00Z",
  fisheries: FISH,
  biotoxin: BIO,
  notices: [],
  charm: charm(0.12),
  outlook: OUTLOOK,
  conditions: null,
  ...over,
});
const set = (date: string, values: (number | null)[], lvl = 30): BiotoxinIndex["sites"][number]["sets"][number] => {
  const nums = values.filter((v): v is number => v != null);
  return { date, kind: "routine", tissue: "viscera", values_ppm: values, nd_count: values.length - nums.length, avg_ppm: Number((nums.reduce((a, b) => a + b, 0) / nums.length).toFixed(1)), max_ppm: Math.max(...nums), pct_over_action: null, action_level_ppm: lvl };
};
const PROGRESS: BiotoxinIndex["sites"][number] = { id: "test-progress-dungeness", bma: "G", bma_label: "Monterey Bay", port_label: null, site_label: "Test site", species: "Dungeness crab", lat: null, lon: null, informs_places: ["port-592"], sets: [set("2026-09-28", [3.1, 4.8, 9.2, 2.5, 6.0, 7.4]), set("2026-09-14", [null, 4.0, 12.1, 6.2, 8.3, 3.9]), set("2026-08-31", [14.2, 22.0, 9.8, 31.5, 18.0, 12.4])] };
const ANCHOVY: BiotoxinIndex["sites"][number] = { id: "test-anchovy", bma: null, bma_label: "Monterey Bay", port_label: null, site_label: "Monterey Bay (Block 509)", species: "Northern anchovy", lat: null, lon: null, informs_places: ["port-592"], sets: [set("2026-10-05", [85, 210, 44, 120, 66, 31], 20)] };
const BIO_PLUS: BiotoxinIndex = { ...BIO, sites: [...BIO.sites, PROGRESS, ANCHOVY] };
const SAFETY = /\b(safe|safety|all clear|clear to|good to go|go fishing|green light|no risk)\b/i;
const text = (f: PortFact) => [f.headline, f.value, f.detail].filter(Boolean).join(" ");

describe("buildPortToday: hierarchy", () => {
  it("puts an official closure or delay first, ahead of every test, forecast and outlook", () => {
    const t = buildPortToday(base({ charm: charm(0.9) }));
    expect(t.facts[0].basis).toBe("official");
    expect(t.facts[0].severity).toBe("red");
    expect(t.facts[0].id).toBe("status:dungeness_commercial");
    // every official fact comes before every non-official one
    const firstNon = t.facts.findIndex((f) => f.basis !== "official");
    expect(t.facts.slice(firstNon).every((f) => f.basis !== "official")).toBe(true);
    // a high C-HARM probability is amber and still comes after the official facts
    const ch = t.facts.find((f) => f.id === "charm:pda")!;
    expect(ch.severity).toBe("amber");
    expect(t.facts.indexOf(ch)).toBeGreaterThan(t.facts.findIndex((f) => f.kind === "test"));
  });

  it("an official notice that closes a fishery outranks a calmer curated record", () => {
    const n: NoticeInput = { id: "cdph-mussels", agency: "CDPH", title: "Annual quarantine of sport-harvested mussels", summary: "Quarantine.", action: "quarantine", species: ["mussels (all species)"], fishery: "sport_harvest", effective: "2026-05-01", url: "https://cdph.ca.gov", verified: false };
    const m = buildTodayModel(base({ notices: [n] }));
    const row = m.status.find((r) => r.fishery === "bivalve_shellfish")!;
    expect(row.severity).toBe("red");
    expect(row.word).toBe("Closed");
    expect(row.verified).toBe(false);
    expect(m.today.facts.some((f) => f.id === "notice:cdph-mussels" && f.basis === "official" && f.verified === false)).toBe(true);
  });

  it("when the registry and the curated status disagree, the row keeps the more restrictive and says so, with both dates", () => {
    // the registry still lists the annual mussel quarantine; the curated record (checked later) says CDPH announced its end
    const n: NoticeInput = { id: "cdph-mussels", agency: "CDPH", title: "Annual quarantine of sport-harvested mussels", summary: "Sport-harvested mussels may not be harvested for human consumption.", action: "quarantine", species: ["mussels (all species)"], fishery: "sport_harvest", effective: "2026-05-01", url: "https://cdph.ca.gov", verified: false, registry_checked: "2026-10-08T19:32:08Z" };
    const m = buildTodayModel(base({ notices: [n] }));
    const row = m.status.find((r) => r.fishery === "bivalve_shellfish")!;
    const curated = FISH.records.find((r) => r.fishery === "bivalve_shellfish")!;
    expect(curated.level).toBe("watch");
    // never silently one or the other: the stricter level, and both statements with who said what and when
    expect(row.level).toBe("closed");
    expect(row.conflict).toMatchObject({ shown: "notice", notice: { word: "Closed", agency: "CDPH", effective: "2026-05-01", checked: "2026-10-08T19:32:08Z", verified: false }, record: { word: "Watch", source: "CDPH", checked: curated.checked_at, verified: false } });
    expect(row.checked_at).toBe("2026-10-08T19:32:08Z");
    const d = row.fact.detail!;
    expect(d).toContain("Sources differ: the official-notices registry lists “Annual quarantine of sport-harvested mussels” as in effect (CDPH, in effect since May 1, registry last read Oct 8, not verified)");
    expect(d).toContain(`while Neritic's curated status says “${curated.headline.replace(/\.$/, "")}” (CDPH, checked Nov 8, not verified)`);
    expect(d).toContain("; Neritic shows the more restrictive of the two until a person has checked both, so check CDPH before you act.");
    // the disagreement is never worded as a lift, and nothing calls the fishery open
    expect(d).not.toMatch(/\b(has been|was|is) lifted\b|\bis open\b/i);
    expect(row.fact.as_of).toBe("2026-10-08T19:32:08Z");
    // a closure beside a restriction is two instruments, not a disagreement (anchovy: both restrict)
    const anchovy: NoticeInput = { ...n, id: "cdph-anchovy", title: "Do not eat northern anchovies", action: "consumption_advisory", species: ["northern anchovy"], fishery: "all" };
    expect(buildTodayModel(base({ notices: [anchovy] })).status.find((r) => r.fishery === "anchovy_sardine")!.conflict).toBeNull();
    // the curated record says "none on record" while a notice restricts: also said
    const rock: NoticeInput = { ...n, id: "cdph-rock", title: "Do not eat rock crab viscera", action: "consumption_advisory", species: ["rock crab"], fishery: "all" };
    const rc = buildTodayModel(base({ notices: [rock] })).status.find((r) => r.fishery === "rock_crab")!;
    expect(rc.level).toBe("restricted");
    expect(rc.conflict?.shown).toBe("notice");
  });

  it("a clean test never softens an official delay of the same fishery", () => {
    const m = buildTodayModel(base());
    const t = m.tests.find((x) => x.site.id === "bma-g-monterey-bay-dungeness-crab")!;
    expect(t.fact.detail).toMatch(/still delayed/);
    expect(m.lede.label).toBe("Dungeness crab (commercial)");
    expect(m.lede.text).toMatch(/^Commercial season delayed to at least Jan 1, 2027/);
    expect(m.lede.more).toBe(3); // salmon (season over), anchovy/sardine, mussels (watch) below
  });

  it("a fishery closed by its season dates is said calmly, and never leads over an emergency or a restriction", () => {
    const m = buildTodayModel(base());
    const salmon = m.status.find((r) => r.fishery === "salmon_commercial")!;
    // still closed, still not open: only the manner differs from an emergency closure
    expect(salmon).toMatchObject({ level: "closed", seasonal: true, word: "Season closed" });
    expect(salmon.fact).toMatchObject({ severity: "red", value: "Season closed" });
    expect(m.status.find((r) => r.fishery === "dungeness_commercial")).toMatchObject({ level: "delayed", seasonal: false, word: "Delayed" });
    // alone, it leads; beside a restriction, the restriction leads
    const only = buildTodayModel(base({ selected: ["salmon_commercial", "groundfish"] }));
    expect(only.lede).toMatchObject({ label: "Salmon (commercial)", seasonal: true, more: 0 });
    const both = buildTodayModel(base({ selected: ["salmon_commercial", "anchovy_sardine"] }));
    expect(both.lede).toMatchObject({ label: "Anchovy & sardine", seasonal: false, more: 1 });
  });

  it("when Neritic's season list is missing, an official notice still leads and the gap is said after it", () => {
    const n: NoticeInput = { id: "cdph-mussels", agency: "CDPH", title: "Annual quarantine of sport-harvested mussels", summary: "Quarantine.", action: "quarantine", species: ["mussels (all species)"], fishery: "sport_harvest", effective: "2026-05-01", url: "https://cdph.ca.gov", verified: false, registry_checked: "2026-09-01T00:00:00Z" };
    const m = buildTodayModel(base({ fisheries: null, notices: [n] }));
    expect(m.lede).toMatchObject({ label: "Shellfish (mussels, clams, oysters)", severity: "red", caution: "The rest of Neritic's season list did not load. Check CDFW and CDPH before you go." });
    // a registry not read for over two weeks is flagged like any other old check
    const row = m.status.find((r) => r.fishery === "bivalve_shellfish")!;
    expect(row.stale).toBe(true);
    expect(m.today.facts.some((f) => f.id === "age:fisheries-status" && f.severity === "amber")).toBe(true);
    expect(m.freshness.find((f) => f.label === "Status")).toMatchObject({ when: "not available", missing: true });
    // with nothing at all to lead, the answer is that the status could not be loaded
    const none = buildTodayModel(base({ fisheries: null }));
    expect(none.lede).toMatchObject({ label: null, severity: "amber", caution: null });
    expect(none.lede.text).toMatch(/^Fishery status could not be loaded/);
  });

  it("keeps the strip order of the reader's fisheries and stable ids", () => {
    const a = buildTodayModel(base({ selected: ["groundfish", "dungeness_commercial"] }));
    expect(a.status.map((r) => r.fact.id)).toEqual(["status:groundfish", "status:dungeness_commercial"]);
    const b = buildTodayModel(base({ selected: ["groundfish", "dungeness_commercial"], now: "2026-10-14T13:00:00Z" }));
    expect(b.status.map((r) => r.fact.id)).toEqual(a.status.map((r) => r.fact.id));
  });
});

describe("buildPortToday: words", () => {
  it("never calls anything safe, clear or go, whatever the data", () => {
    for (const over of [{}, { fisheries: { ...FISH, records: [] } }, { charm: charm(0.02, "current", true) }, { notices: null }, { fisheries: null, biotoxin: null, charm: null, outlook: null }]) {
      for (const f of buildPortToday(base(over as Partial<PortTodayInputs>)).facts) expect(text(f)).not.toMatch(SAFETY);
    }
  });

  it("the calmest status is 'No restriction on record' in neutral, never green", () => {
    const m = buildTodayModel(base({ fisheries: { ...FISH, records: [] } }));
    const g = m.status.find((r) => r.fishery === "groundfish")!;
    expect(g.word).toBe("No restriction on record");
    expect(g.severity).toBe("neutral");
    expect(g.fact.detail).toMatch(/not a guarantee/);
    expect(m.lede.severity).toBe("neutral");
    expect(m.lede.status).toMatch(/No closure or restriction on record/);
  });

  it("unknown is not open: missing status data says 'not available', never 'no restriction'", () => {
    const m = buildTodayModel(base({ fisheries: null }));
    expect(m.statusError).toBe(true);
    expect(m.status.every((r) => r.level === "unknown" && r.word === "Status not available")).toBe(true);
    expect(m.lede.status).toMatch(/could not be loaded/);
    expect(m.today.facts.some((f) => f.id === "age:fisheries-status" && f.severity === "amber")).toBe(true);
  });

  it("labels the chlorophyll outlook a bloom indicator, never a toxin forecast, with its skill from the data", () => {
    const f = buildPortToday(base()).facts.find((x) => x.kind === "bloom_outlook")!;
    expect(f.headline).toMatch(/Chlorophyll outlook/);
    expect(f.headline).toMatch(/not a toxin forecast/);
    expect(text(f).replace(/not a toxin forecast/g, "")).not.toMatch(/toxin forecast/i);
    expect(f.detail).toContain("0.754");
    expect(f.detail).toContain("0.85");
    expect(f.basis).toBe("outlook");
    // a different skill in the file is quoted as it is, not a remembered number
    const g = buildPortToday(base({ outlook: { ...OUTLOOK, skill: { metric: "RMSE ln(mg m-3)", model: 0.7, climatology: 0.81 } } })).facts.find((x) => x.kind === "bloom_outlook")!;
    expect(g.detail).toContain("0.7 for this outlook");
    expect(g.detail).not.toContain("0.754");
    expect(skillWords(OUTLOOK.skill)).toBe("Held-out RMSE ln(mg m-3): 0.754 for this outlook, against 0.85 climatology and 0.892 persistence (n = 185,000; 2019 to 2021-06). Lower is better.");
  });

  it("C-HARM is a probability of conditions in the water, not toxin in seafood, and says when it is not at the harbour", () => {
    const f = buildPortToday(base()).facts.find((x) => x.id === "charm:pda")!;
    expect(f.headline).toMatch(/12% probability of elevated particulate domoic acid in the water near Moss Landing/);
    expect(f.detail).toMatch(/not toxin in seafood/);
    expect(f.detail).toMatch(/not a value at the harbour/);
    expect(f.valid_for).toEqual({ from: "2026-10-11", to: "2026-10-14" });
  });

  it("states a test plainly first: place, species, verdict against the limit, date, average", () => {
    const site = BIO.sites.find((s) => s.id === "bma-g-monterey-bay-dungeness-crab")!;
    expect(testSentence(site, site.sets[0], 30, NOW)).toBe("Monterey Bay Dungeness tested well under the limit on Sep 28: avg 5.5 ppm, limit 30.");
    expect(testSentence(ANCHOVY, ANCHOVY.sets[0], 20, NOW)).toMatch(/tested over the limit on Oct 5: 6 of 6 samples at or over 20 ppm, max 210 ppm/);
  });

  it("observations are measured with their own time; forecasts are forecasts", () => {
    const cond = JSON.parse(readFileSync(path.join(__dirname, "fixtures", "today-conditions.json"), "utf8")) as Conditions;
    const m = buildTodayModel(base({ conditions: cond }));
    const buoy = m.today.facts.find((f) => f.id === "sea:buoy")!;
    expect(buoy.basis).toBe("measured");
    expect(buoy.headline).toMatch(/^Measured at buoy \d+: waves/);
    expect(buoy.as_of).toMatch(/^2026-10-13T/);
    // the nearest buoy off Moss Landing (46236) is a wave buoy: no wind, so nothing is claimed for it here
    expect(m.sea.buoy).toMatchObject({ id: "46236", windKt: null });
    expect(m.sea.windBuoy).toBeNull();
    expect(m.today.facts.some((f) => f.id === "sea:wind")).toBe(false);
    // with the nearest station that does measure wind: its own id, distance and time, never merged into 46236's
    const withWind: Conditions = { ...cond, buoyWind: { status: "ok", source: "NOAA National Data Buoy Center", url: "https://www.ndbc.noaa.gov/station_page.php?station=46092", fetchedAt: "2026-10-13T06:00:00.000Z", station: { id: "46092", name: "MBM1", km: 21.6, lat: 36.75, lon: -122.03 }, data: { windKt: { value: 5.83, at: "2026-10-13T03:18:00.000Z" }, windDirDeg: { value: 320, at: "2026-10-13T03:18:00.000Z" } } } };
    const w = buildTodayModel(base({ conditions: withWind }));
    expect(w.sea.windBuoy).toMatchObject({ id: "46092", km: 21.6, windDir: "NW", at: "2026-10-13T03:18:00.000Z" });
    const wind = w.today.facts.find((f) => f.id === "sea:wind")!;
    expect(wind).toMatchObject({ kind: "observation", basis: "measured", headline: "Measured at buoy 46092, 22 km from Moss Landing: wind NW 6 kt.", as_of: "2026-10-13T03:18:00.000Z" });
    expect(wind.detail).toContain("The nearest buoy (46236) has no wind sensor");
    expect(w.today.facts.find((f) => f.id === "sea:buoy")!.headline).toMatch(/^Measured at buoy 46236: waves/);
    expect(m.today.facts.find((f) => f.id === "weather:nws")!.basis).toBe("forecast");
    expect(m.today.facts.find((f) => f.id === "tide:next")!.headline).toMatch(/^Next tide: (high|low) /);
    expect(m.sea.lanes.length).toBeGreaterThanOrEqual(5);
    // issued last evening: at 6 AM "Tonight" (Oct 12) is over and not shown; at 11 PM it is
    expect(m.sea.lanes[0].name).toBe("Tue");
    expect(buildTodayModel(base({ conditions: cond, now: "2026-10-13T06:00:00Z" })).sea.lanes[0].name).toBe("Tonight");
    expect(m.lede.sea).toMatch(/kt/);
  });
});

describe("buildPortToday: stale and missing data", () => {
  it("flags a stale C-HARM run and an out-of-date one differently", () => {
    const s = buildPortToday(base({ charm: charm(0.2, "stale") }));
    expect(s.facts.find((f) => f.id === "charm:pda")!.detail).toMatch(/NOAA has not published a newer one/);
    expect(s.facts.some((f) => f.id === "age:charm")).toBe(true);
    const h = buildPortToday(base({ charm: charm(0.9, "historical") }));
    const ch = h.facts.find((f) => f.id === "charm:pda")!;
    expect(ch.headline).toMatch(/out of date/);
    expect(ch.severity).toBe("info"); // an old run never raises a watch
    expect(h.facts.find((f) => f.id === "age:charm")!.severity).toBe("amber");
  });

  it("flags old tests, old status checks and an old refresh", () => {
    const later = "2026-11-25T16:00:00Z";
    const m = buildTodayModel(base({ now: later }));
    const t = m.tests.find((x) => x.site.id === "bma-g-monterey-bay-dungeness-crab")!;
    expect(t.stale).toBe(true);
    expect(t.fact.detail).toMatch(/over 30 days old/);
    expect(m.status.find((r) => r.fishery === "dungeness_commercial")!.stale).toBe(true);
    expect(m.today.facts.some((f) => f.id === "age:fisheries-status")).toBe(true);
    expect(m.today.facts.some((f) => f.id === "age:data")).toBe(true);
    expect(m.freshness.find((x) => x.label === "Tests")!.stale).toBe(true);
  });

  it("missing sidecars are unavailable sections, not empty ones", () => {
    const m = buildTodayModel(base({ biotoxin: null, outlook: null, charm: null }));
    expect(m.testsError).toBe(true);
    expect(m.tests).toEqual([]);
    expect(m.bloom.outlookFact).toBeNull();
    expect(m.bloom.charmFact.headline).toMatch(/No C-HARM run/);
    expect(m.today.facts.some((f) => f.id === "age:biotoxin")).toBe(true);
  });

  it("shows progress toward two clean sets after a set over the action level", () => {
    const m = buildTodayModel(base({ biotoxin: BIO_PLUS }));
    const t = m.tests.find((x) => x.site.id === "test-progress-dungeness")!;
    expect(t.cleanStreak).toBe(2);
    expect(t.showProgress).toBe(true);
    expect(t.progressWords).toMatch(/^Two consecutive sets under the action level \(Sep 14 and Sep 28\)/);
    const a = m.tests.find((x) => x.site.id === "test-anchovy")!;
    expect(a.over).toBe(true);
    expect(a.fact.severity).toBe("amber");
    expect(a.cleanStreak).toBe(0);
  });
});

describe("whale rules and helpers", () => {
  it("places ports in their RAMP zone and reports the zone's action", () => {
    expect(rampZoneFor(36.8044).zone).toBe("4"); // Moss Landing
    expect(rampZoneFor(37.5037).zone).toBe("3"); // Half Moon Bay
    expect(rampZoneFor(41.749).zone).toBe("1"); // Crescent City
    const m = buildTodayModel(base());
    expect(m.whale!.zone).toBe("4");
    expect(m.whale!.fact.severity).toBe("red");
    expect(m.whale!.fact.detail).toMatch(/Next CDFW risk assessment December 15-18/);
    expect(m.whale!.verified).toBe(false);
  });

  it("parses the fisheries list from a link", () => {
    expect(parseFisheries("groundfish,dungeness_commercial,nope,groundfish")).toEqual(["groundfish", "dungeness_commercial"]);
    expect(parseFisheries("")).toBeNull();
  });
});

describe("what changed since the last visit", () => {
  it("lists new, changed and gone facts by id, ignoring the hourly sea", () => {
    const prev = buildPortToday(base({ fisheries: { ...FISH, records: FISH.records.filter((r) => r.fishery !== "anchovy_sardine") } }));
    const next = buildPortToday(base({ charm: { ...charm(0.12), issued: "2026-10-13" } }));
    const d = diffFacts(snapFacts(prev.facts), next.facts);
    expect(d.find((c) => c.id === "status:anchovy_sardine")?.change).toBe("changed");
    expect(d.find((c) => c.id === "charm:pda")?.change).toBe("changed");
    expect(d.some((c) => c.id.startsWith("sea:") || c.id.startsWith("tide:"))).toBe(false);
    expect(diffFacts(snapFacts(next.facts), next.facts)).toEqual([]);
    const gone = diffFacts(snapFacts(next.facts), buildPortToday(base({ biotoxin: null })).facts);
    expect(gone.filter((c) => c.change === "gone").map((c) => c.id)).toContain("test:bma-g-monterey-bay-dungeness-crab");
  });
});

describe("the data track's fixtures, as Today reads them", () => {
  it("Monterey Bay's ports get their crab sites and RAMP zone 4; Half Moon Bay gets its own", () => {
    expect(BIO.sites.filter((x) => x.informs_places.includes("port-592")).map((x) => x.id)).toEqual(expect.arrayContaining(["bma-g-monterey-bay-dungeness-crab", "bma-g-monterey-bay-rock-crab"]));
    const hmb = buildTodayModel(base({ place: { place_id: "port-452", label: "Princeton / Half Moon Bay", lat: 37.5037, lon: -122.48185 } }));
    expect(hmb.whale!.zone).toBe("3");
    expect(hmb.tests.map((t) => t.site.id)).toEqual(expect.arrayContaining(["bma-f-pillar-point-dungeness-crab", "bma-f-pigeon-point-dungeness-crab"]));
    expect(hmb.status.find((r) => r.fishery === "anchovy_sardine")!.level).toBe("no_restriction_on_record");
    // every curated record is pending, so every one says Not verified
    expect(buildTodayModel(base()).status.filter((r) => r.records.length).every((r) => r.verified === false)).toBe(true);
  });
});
