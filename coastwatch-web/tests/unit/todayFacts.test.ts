import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import path from "node:path";
import type { BiotoxinIndex, FisheriesStatusIndex, PortFact } from "@/pilot/contracts";
import type { Conditions } from "@/lib/conditions";
import { FISHERY_OPTIONS, OUTLOOK_SIGNAL_RATIO, alertEndWords, rowHeadline, rowWord, unverifiedCount, buildPortToday, buildTodayModel, diffFacts, factorWords, ledeParts, outlookErrorFactor, outlookRead, parseFisheries, statusTally, rampZoneFor, skillWords, snapFacts, testSentence, type CharmInput, type NoticeInput, type OutlookInput, type PortTodayInputs } from "@/pilot/facts";
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

describe("Today's honesty rules", () => {
  it("counts the group's not-verified marker, with no-record rows as not verified and exceptions only in a mixed group", () => {
    expect(unverifiedCount([{ verified: false, known: true }, { verified: null, known: true }, { verified: true, known: false }])).toMatchObject({ n: 2, of: 2, words: "2 of 2", mixed: false });
    expect(unverifiedCount([{ verified: false, known: true }, { verified: true, known: true }])).toMatchObject({ n: 1, of: 2, words: "1 of 2", mixed: true });
    expect(unverifiedCount([{ verified: true, known: true }])).toMatchObject({ n: 0, words: null, mixed: false });
  });

  it("the lede carries its own source, date and review state, and 'none on record' carries its caveat in the same sentence", () => {
    const m = buildTodayModel(base({ selected: ["dungeness_commercial"] }));
    expect(m.lede.source?.name).toBe("CDFW");
    expect(m.lede.checked_at).toBeTruthy();
    expect(m.lede.verified).toBe(false);
    const quiet = buildTodayModel(base({ selected: ["groundfish"] }));
    expect(quiet.lede.text).toMatch(/No closure or restriction on record.*Not a promise they are open/);
    expect(quiet.lede.verified).not.toBe(true);
  });

  it("a C-HARM run whose days have all passed is a past run, never an outlook, and raises no watch", () => {
    const oldRun: CharmInput = { ...charm(0.9, "stale"), days: charm(0.9, "stale").days.map((d, i) => ({ ...d, date: `2026-10-0${i + 1}` })) };
    const v = buildTodayModel(base({ charm: oldRun }));
    expect(v.bloom.charmPast).toBe(true);
    expect(v.bloom.charmFact.headline).toMatch(/past run/);
    expect(v.bloom.charmFact.headline).toMatch(/Those days have passed/);
    expect(v.bloom.charmFact.severity).toBe("info");
  });

  it("says an advisory's end only as NWS stated it", () => {
    expect(alertEndWords(null, "2026-10-13T06:00:00Z")).toBe("no end time given");
    expect(alertEndWords("2026-10-13T15:00:00-07:00", "2026-10-13T06:00:00Z")).toBe("until Oct 13, 3:00 PM");
  });
});

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
    expect(row.level).toBe("closed");
    expect(row.verified).toBe(false);
    expect(m.today.facts.some((f) => f.id === "notice:cdph-mussels" && f.basis === "official" && f.verified === false)).toBe(true);
  });

  it("a rule on sport harvest is said as one: in the pill, the headline and the count, never as a commercial closure", () => {
    const n: NoticeInput = { id: "cdph-mussels", agency: "CDPH", title: "Annual quarantine of sport-harvested mussels", summary: "Sport-harvested mussels may not be harvested for human consumption (bait only).", action: "quarantine", species: ["mussels (all species)"], fishery: "sport_harvest", effective: "2026-05-01", url: "https://cdph.ca.gov", verified: false, registry_checked: "2026-10-08T19:32:08Z" };
    const m = buildTodayModel(base({ notices: [n] }));
    const row = m.status.find((r) => r.fishery === "bivalve_shellfish")!;
    // the stricter source still leads and keeps its level and colour: only the scope is now said
    expect(row).toMatchObject({ level: "closed", severity: "red", scope: "Sport harvest", word: "Sport harvest closed", verified: false });
    expect(row.headline).toBe("Annual quarantine of sport-harvested mussels (sport harvest only).");
    expect(row.conflict?.shown).toBe("notice");
    expect(row.fact).toMatchObject({ id: "status:bivalve_shellfish", value: "Sport harvest closed", severity: "red", verified: false });
    expect(row.fact.headline).toBe("Shellfish (mussels, clams, oysters): Annual quarantine of sport-harvested mussels (sport harvest only).");
    expect(row.fact.detail).toContain("This is a rule for sport harvest, not a closure of a commercial fishery.");
    expect(m.today.facts.find((f) => f.id === "notice:cdph-mussels")).toMatchObject({ value: "Sport harvest closed", severity: "red" });
    // the count is in the pills' own words: the delay is the one commercial action, the quarantine is counted as what it is
    // (here the registry and the curated record disagree, so the row counts as unclear, never as settled)
    expect(statusTally(m.status)).toBe("1 delayed · 1 status unclear · 1 restricted · 1 season closed · 3 none on record");
    expect(statusTally(m.status)).not.toMatch(/closed or delayed/);
    // without the notice the curated record leads: a watch on sport harvest, counted as a watch (not a restriction)
    const calm = buildTodayModel(base());
    expect(calm.status.find((r) => r.fishery === "bivalve_shellfish")).toMatchObject({ level: "watch", severity: "amber", scope: "Sport harvest", word: "Sport harvest watch" });
    expect(statusTally(calm.status)).toBe("1 delayed · 1 restricted · 1 sport harvest watch · 1 season closed · 3 none on record");
    // a notice on the whole fishery is not scoped, and the recreational Dungeness row needs no second label
    const all: NoticeInput = { ...n, id: "cdph-rock", title: "Do not eat rock crab viscera", action: "consumption_advisory", species: ["rock crab"], fishery: "all" };
    expect(buildTodayModel(base({ notices: [all] })).status.find((r) => r.fishery === "rock_crab")).toMatchObject({ scope: null, word: "Restricted", headline: "Do not eat rock crab viscera." });
    const rec = buildTodayModel(base({ selected: ["dungeness_recreational"] })).status[0];
    expect(rec.scope).toBeNull();
    // nothing could be loaded: said as that
    expect(statusTally(buildTodayModel(base({ fisheries: null })).status)).toBe("7 not available");
  });

  it("the hero sentence says the sector once and reads as one sentence for every fishery", () => {
    expect(ledeParts("Dungeness crab (commercial)", "Commercial season delayed to at least Jan 1, 2027 (humpback whale entanglements).")).toEqual({ label: "Dungeness crab", text: "commercial season delayed to at least Jan 1, 2027 (humpback whale entanglements)." });
    expect(ledeParts("Dungeness crab (recreational)", "Recreational season open since Nov 7 with crab traps prohibited; hoop nets and snares allowed.")).toEqual({ label: "Dungeness crab", text: "recreational season open since Nov 7 with crab traps prohibited; hoop nets and snares allowed." });
    expect(ledeParts("Salmon (commercial)", "The 2026 commercial ocean salmon season has concluded.")).toEqual({ label: "Salmon", text: "the 2026 commercial ocean salmon season has concluded." });
    // the sector stays in the label when the headline does not say it; a name keeps its capital; a species list is not a sector
    expect(ledeParts("Salmon (commercial)", "Closed south of Pigeon Point.")).toEqual({ label: "Salmon (commercial)", text: "closed south of Pigeon Point." });
    // a headline that names its own subject in the fishery's words leads with it: one colon, not two
    expect(ledeParts("Anchovy & sardine", "Monterey Bay anchovy: no take for human consumption since Sep 11 (domoic acid); bait only.")).toEqual({ label: "Monterey Bay anchovy", text: "no take for human consumption since Sep 11 (domoic acid); bait only." });
    expect(ledeParts("Groundfish", "Zones 3-4: closed inside 20 fathoms.")).toEqual({ label: "Groundfish", text: "Zones 3-4: closed inside 20 fathoms." });
    expect(ledeParts("Rock crab", "CDPH advises against eating the viscera.")).toEqual({ label: "Rock crab", text: "CDPH advises against eating the viscera." });
    expect(ledeParts("Shellfish (mussels, clams, oysters)", "Annual quarantine of sport-harvested mussels (sport harvest only).")).toEqual({ label: "Shellfish (mussels, clams, oysters)", text: "annual quarantine of sport-harvested mussels (sport harvest only)." });
    // every fishery's lead, from the fixture records: no sector said twice, no "X: X …"
    for (const f of FISHERY_OPTIONS) {
      const l = buildTodayModel(base({ selected: [f.id] })).lede;
      if (!l.label) continue;
      expect(l.status).toBe(`${l.label}: ${l.text}`);
      expect(l.status.split(": ").length, l.status).toBeLessThanOrEqual(2);
      expect(l.status.match(/\bcommercial\b/gi)?.length ?? 0).toBeLessThanOrEqual(1);
      expect(l.status.match(/\brecreational\b/gi)?.length ?? 0).toBeLessThanOrEqual(1);
    }
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
    // the row never states it as settled: unclear, awaiting a person's review (the sheet shows both records)
    expect(rowWord(row)).toBe("Status unclear");
    expect(rowHeadline(row)).toBe("Sport harvest: sources differ, awaiting a person's review.");
    expect(statusTally([row])).toBe("1 status unclear");
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
    expect(m.lede.label).toBe("Dungeness crab");
    expect(m.lede.text).toMatch(/^commercial season delayed to at least Jan 1, 2027/);
    expect(m.lede.status).toBe("Dungeness crab: commercial season delayed to at least Jan 1, 2027 (humpback whale entanglements).");
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
    expect(only.lede).toMatchObject({ label: "Salmon", text: "the 2026 commercial ocean salmon season has concluded.", seasonal: true, more: 0 });
    const both = buildTodayModel(base({ selected: ["salmon_commercial", "anchovy_sardine"] }));
    expect(both.lede).toMatchObject({ label: "Monterey Bay anchovy", text: "no take for human consumption since Sep 11 (domoic acid); bait only.", seasonal: false, more: 1 });
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

  it("says the outlook in three words against the usual value, and only calls a clear departure a signal", () => {
    // the thresholds: 1.5x usual or more, two-thirds of usual or less; anything between is "about usual"
    expect(OUTLOOK_SIGNAL_RATIO).toBe(1.5);
    expect(outlookRead(3.0, 2.0)).toMatchObject({ level: "higher", phrase: "higher than usual", ratioWords: "about 1.5× the typical level" });
    expect(outlookRead(2.98, 2.0)).toMatchObject({ level: "usual", phrase: "about usual", ratioWords: null });
    expect(outlookRead(2.4, 2.0)!.level).toBe("usual"); // 1.2x is inside the error: not a signal
    expect(outlookRead(1.34, 2.0)!.level).toBe("usual"); // 0.67x: just inside
    expect(outlookRead(1.3, 2.0)).toMatchObject({ level: "lower", phrase: "lower than usual", ratioWords: "about 65% of the typical level" });
    expect(outlookRead(24, 2.0)!.ratioWords).toBe("about 12× the typical level");
    // nothing to compare with, nothing said
    for (const [v, t] of [[3.2, null], [3.2, undefined], [null, 2], [3.2, 0], [0, 2], [3.2, Number.NaN]] as const) expect(outlookRead(v, t)).toBeNull();
  });

  it("leads the outlook with a plain sentence, then what it is and how sure, and keeps its id for alerts and Ask", () => {
    // the clock here is Oct 13, the first day of the outlook's period
    const m = buildTodayModel(base({ outlook: { ...OUTLOOK, typical: 2.0 } }));
    const f = m.bloom.outlookFact!;
    expect(f.id).toBe("outlook:chl8d");
    expect(f.headline).toBe("This 8-day period (Oct 13–20): chlorophyll near Moss Landing is expected to run higher than usual for this time of year, about 1.6× the typical level.");
    expect(m.bloom.outlookRead).toMatchObject({ level: "higher", phrase: "higher than usual" });
    expect(m.bloom.outlookRead!.ratio).toBeCloseTo(1.6, 5);
    // what it is, how sure (the factor comes from the published RMSE: e^0.754 = 2.1), and a neutral use
    expect(m.bloom.outlookHonest).toBe("A bloom indicator, not a toxin forecast: chlorophyll is algae of every kind. It is one 8-day average, not a day-by-day forecast, and its typical error is about a factor of 2 either way.");
    expect(m.bloom.outlookUse).toBe("A higher-than-usual outlook is a reason to watch the next CDPH test results, not a sign of toxin.");
    expect(m.bloom.errorFactor).toBeCloseTo(Math.exp(0.754), 6);
    expect(f.detail).toContain(m.bloom.outlookHonest!);
    expect(f.detail).toContain(m.bloom.outlookUse!);
    expect(f.detail).toContain("About 3.2 mg/m³ against a usual 2.0 mg/m³ (the median of the water within 15 km).");
    expect(f.detail).toContain("0.754 for this outlook"); // the skill is still quoted from the data
    expect(f.value).toBe("3.2 mg/m³");
    expect(f.basis).toBe("outlook");
    expect(f.severity).toBe("info"); // never a status colour: chlorophyll is not toxin
    expect(text(f)).not.toMatch(SAFETY);
    expect(text(f).replace(/not a toxin forecast/g, "")).not.toMatch(/toxin forecast/i);

    const usual = buildTodayModel(base({ outlook: { ...OUTLOOK, typical: 2.9 } })).bloom;
    expect(usual.outlookFact!.headline).toBe("This 8-day period (Oct 13–20): chlorophyll near Moss Landing is expected to run about usual for this time of year.");
    expect(usual.outlookFact!.headline).not.toMatch(/×|%/);
    expect(usual.outlookUse).toMatch(/says nothing about toxin either way: only CDPH's lab tests of seafood measure it/);

    const lower = buildTodayModel(base({ outlook: { ...OUTLOOK, value: 1.0, typical: 2.0 } })).bloom;
    expect(lower.outlookFact!.headline).toBe("This 8-day period (Oct 13–20): chlorophyll near Moss Landing is expected to run lower than usual for this time of year, about 50% of the typical level.");
    expect(lower.outlookUse).toMatch(/does not rule out toxin/);
    expect(text(lower.outlookFact!)).not.toMatch(SAFETY);
  });

  it("words the outlook's window as it is: not yet begun, already begun, or past", () => {
    const o = { ...OUTLOOK, typical: 2.0 };
    const at = (now: string) => buildTodayModel(base({ now, outlook: o })).bloom.outlookFact!.headline;
    expect(at("2026-10-13T06:00:00Z")).toMatch(/^Next 8 days \(Oct 13–20\): chlorophyll near Moss Landing is expected/); // Oct 12, 11 PM Pacific
    expect(at("2026-10-15T18:00:00Z")).toMatch(/^This 8-day period \(Oct 13–20\): chlorophyll near Moss Landing is expected/);
    expect(at("2026-10-22T18:00:00Z")).toMatch(/^The 8-day period Oct 13–20 \(now past\): chlorophyll near Moss Landing was expected/);
    // the year's last period is shorter than 8 days and is not called 8
    const last = buildTodayModel(base({ now: "2026-12-20T18:00:00Z", outlook: { ...o, valid_from: "2026-12-27", valid_to: "2026-12-31" } })).bloom;
    expect(last.outlookFact!.headline).toMatch(/^Next 5 days \(Dec 27–31\)/);
    expect(last.outlookHonest).toMatch(/one 5-day average/);
  });

  it("falls back to the plain value when the data carries no usual value (older data), with no comparison", () => {
    for (const typical of [undefined, null] as const) {
      const m = buildTodayModel(base({ outlook: { ...OUTLOOK, typical } }));
      expect(m.bloom.outlookRead).toBeNull();
      expect(m.bloom.outlookUse).toBeNull();
      expect(m.bloom.outlookFact!.id).toBe("outlook:chl8d");
      expect(m.bloom.outlookFact!.headline).toBe("Chlorophyll outlook, Oct 13–20: about 3.2 mg/m³ near Moss Landing. A bloom indicator, not a toxin forecast.");
      expect(text(m.bloom.outlookFact!)).not.toMatch(/usual|typical level/);
    }
    // no value at the port: nothing is compared, even with a usual value
    expect(buildTodayModel(base({ outlook: { ...OUTLOOK, value: null, typical: 2 } })).bloom.outlookFact!.headline).toMatch(/no value near Moss Landing/);
  });

  it("takes the size of the outlook's error from the published skill, and says nothing when it cannot", () => {
    expect(factorWords(outlookErrorFactor({ metric: "RMSE ln(mg m-3)", model: 0.754 })!)).toBe("2");
    expect(factorWords(outlookErrorFactor({ metric: "RMSE ln(mg m-3)", model: 0.95 })!)).toBe("2.5");
    expect(outlookErrorFactor({ metric: "RMSE log10(mg m-3)", model: 0.33 })).toBeCloseTo(10 ** 0.33, 6);
    expect(outlookErrorFactor({ metric: "correlation", model: 0.6 })).toBeNull();
    expect(outlookErrorFactor({ metric: "RMSE mg m-3", model: 1.2 })).toBeNull();
    const m = buildTodayModel(base({ outlook: { ...OUTLOOK, typical: 2.0, skill: { metric: "skill score", model: 0.2, climatology: 0 } } }));
    expect(m.bloom.errorFactor).toBeNull();
    expect(m.bloom.outlookHonest).toBe("A bloom indicator, not a toxin forecast: chlorophyll is algae of every kind. It is one 8-day average, not a day-by-day forecast.");
  });

  it("C-HARM is a probability of conditions in the water, not toxin in seafood, and says when it is not at the harbour", () => {
    const f = buildPortToday(base()).facts.find((x) => x.id === "charm:pda")!;
    expect(f.headline).toMatch(/12% probability of elevated particulate domoic acid in the water near Moss Landing/);
    expect(f.detail).toMatch(/not toxin in seafood/);
    expect(f.detail).toMatch(/not a value at the harbour/);
    expect(f.valid_for).toEqual({ from: "2026-10-11", to: "2026-10-14" });
  });

  it("states a test plainly first: place, species, verdict against the action level, date, the highest result and the count (T8)", () => {
    const site = BIO.sites.find((s) => s.id === "bma-g-monterey-bay-dungeness-crab")!;
    expect(testSentence(site, site.sets[0], 30, NOW)).toBe("Monterey Bay Dungeness tested under the action level on Sep 28: highest 9.8 ppm of 30, all 6 samples under.");
    expect(testSentence(ANCHOVY, ANCHOVY.sets[0], 20, NOW)).toMatch(/tested over the action level on Oct 5: highest 210 ppm of 20, 6 of 6 samples at or over/);
  });

  it("never leads with an average that counts non-detects as zero: the highest result and the censored count instead (T7, T8)", () => {
    const rock = BIO.sites.find((s) => s.id === "bma-g-monterey-bay-rock-crab");
    const set = { date: "2026-09-28", kind: "routine" as const, tissue: "viscera" as const, values_ppm: [null, null, null, 20, null, 2.6], nd_count: 4, avg_ppm: 3.8, max_ppm: 20, pct_over_action: 0, action_level_ppm: 30 };
    const site = rock ?? { ...BIO.sites[0], id: "rock", species: "rock crab", site_label: "Monterey Bay" };
    const s = testSentence(site, set, 30, NOW);
    expect(s).toMatch(/highest 20 ppm of 30, all 6 samples under, 4 below detection\.$/);
    expect(s).not.toMatch(/3\.8|avg|average/);
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
    // NWS's evening bulletin can still open with "This Afternoon": by 6 PM that period is over and does not lead
    const evening: Conditions = {
      ...cond,
      marine: { status: "ok", source: "NWS", url: "https://x", fetchedAt: "2026-11-10T05:45:00Z", data: { zone: "PZZ535", zoneName: "Monterey Bay", updated: "2026-11-10T04:55:00.000Z", periods: [{ name: "This Afternoon", text: "NW wind 5 to 10 kt. Seas 5 to 7 ft." }, { name: "Tonight", text: "N wind 10 to 15 kt. Seas 5 to 7 ft." }] } },
      alerts: { status: "ok", source: "NWS", url: "https://x", fetchedAt: "2026-11-10T05:45:00Z", data: [{ event: "Small craft advisory", headline: "Small craft advisory in effect until 9 pm pst this evening", onset: null, ends: "2026-11-10T05:00:00.000Z", url: "https://x" }] },
    } as Conditions;
    const night = buildTodayModel(base({ conditions: evening, now: "2026-11-10T05:45:00Z" }));
    expect(night.sea.lanes[0].name).toBe("Tonight");
    expect(night.sea.alerts).toEqual([]); // its stated end has passed
    const before = buildTodayModel(base({ conditions: evening, now: "2026-11-10T04:58:00Z" }));
    expect(before.sea.alerts).toHaveLength(1);
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
