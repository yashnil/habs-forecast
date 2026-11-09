import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import path from "node:path";
import type { BiotoxinIndex, FisheriesStatusIndex, PortFact } from "@/pilot/contracts";
import type { Conditions } from "@/lib/conditions";
import { buildPortToday, buildTodayModel, diffFacts, parseFisheries, rampZoneFor, skillWords, snapFacts, testSentence, type CharmInput, type NoticeInput, type OutlookInput, type PortTodayInputs } from "@/pilot/facts";
import { parseBiotoxin, parseFisheriesStatus } from "@/pilot/today/sidecars";

const fx = (rel: string) => JSON.parse(readFileSync(path.join(__dirname, "..", "fixture-derived", "v1", rel), "utf8"));
const BIO = parseBiotoxin(fx("biotoxin/index.json"))!;
const FISH = parseFisheriesStatus(fx("fisheries-status/index.json"))!;
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

  it("a clean test never softens an official delay of the same fishery", () => {
    const m = buildTodayModel(base());
    const t = m.tests.find((x) => x.site.id === "bma-g-monterey-bay-dungeness")!;
    expect(t.fact.detail).toMatch(/still delayed/);
    expect(m.lede.label).toBe("Dungeness crab (commercial)");
    expect(m.lede.text).toMatch(/^Commercial season delayed to at least Jan 1, 2027/);
    expect(m.lede.more).toBe(2); // anchovy/sardine and salmon restrictions below
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
    const site = BIO.sites.find((s) => s.id === "bma-g-monterey-bay-dungeness")!;
    expect(testSentence(site, site.sets[0], 30, NOW)).toBe("Monterey Bay Dungeness tested well under the limit on Sep 28: avg 5.5 ppm, limit 30.");
    const anchovy = BIO.sites.find((s) => s.id === "monterey-bay-northern-anchovy")!;
    expect(testSentence(anchovy, anchovy.sets[0], 20, NOW)).toMatch(/tested over the limit on Oct 5: 6 of 6 samples at or over 20 ppm, max 210 ppm/);
  });

  it("observations are measured with their own time; forecasts are forecasts", () => {
    const cond = JSON.parse(readFileSync(path.join(__dirname, "fixtures", "today-conditions.json"), "utf8")) as Conditions;
    const m = buildTodayModel(base({ conditions: cond }));
    const buoy = m.today.facts.find((f) => f.id === "sea:buoy")!;
    expect(buoy.basis).toBe("measured");
    expect(buoy.headline).toMatch(/^Measured at buoy \d+: waves/);
    expect(buoy.as_of).toMatch(/^2026-10-13T/);
    expect(m.today.facts.find((f) => f.id === "weather:nws")!.basis).toBe("forecast");
    expect(m.today.facts.find((f) => f.id === "tide:next")!.headline).toMatch(/^Next tide: (high|low) /);
    expect(m.sea.lanes.length).toBeGreaterThanOrEqual(5);
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
    const later = "2026-11-20T16:00:00Z";
    const m = buildTodayModel(base({ now: later }));
    const t = m.tests.find((x) => x.site.id === "bma-g-monterey-bay-dungeness")!;
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
    const m = buildTodayModel(base());
    const t = m.tests.find((x) => x.site.id === "bma-g-monterey-bay-dungeness")!;
    expect(t.cleanStreak).toBe(2);
    expect(t.showProgress).toBe(true);
    expect(t.progressWords).toMatch(/^Two consecutive sets under the action level \(Sep 14 and Sep 28\)/);
    const a = m.tests.find((x) => x.site.id === "monterey-bay-northern-anchovy")!;
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
    expect(m.whale!.fact.detail).toMatch(/Next CDFW risk assessment Dec 15/);
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
    expect(gone.filter((c) => c.change === "gone").map((c) => c.id)).toContain("test:bma-g-monterey-bay-dungeness");
  });
});

describe("temporary sidecar checks", () => {
  it("rejects malformed sidecars", () => {
    expect(parseBiotoxin({ version: 2 })).toBeNull();
    expect(parseFisheriesStatus({ version: 1, records: [{ id: "x", level: "safe" }] })!.records).toEqual([]);
    expect((parseBiotoxin(fx("biotoxin/index.json")) as BiotoxinIndex).sites.length).toBe(4);
    expect((parseFisheriesStatus(fx("fisheries-status/index.json")) as FisheriesStatusIndex).ramp!.zones.length).toBe(5);
  });
});
