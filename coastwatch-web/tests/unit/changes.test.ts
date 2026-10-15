import { readFileSync } from "node:fs";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { Manifest, PortsCollection } from "@/generated/schema";
import type { OfficialDataset } from "@/generated/official";
import type { ObservationDataset } from "@/generated/observations";
import { buildSnapshot, type SnapNotice } from "@/lib/briefingSnapshot";
import { diffSnapshots, type Change } from "@/lib/changes";
import { slimStations } from "@/lib/data";
import { KEPT_WARNING, keptHtml, keptSections, makeKept } from "@/lib/keptBriefing";
import { forgetAll, readWaters, recordVisit, rememberedFor, storageAvailable } from "@/lib/savedWaters";

const FIX = path.resolve(__dirname, "../fixture-data/v1");
const read = <T>(rel: string): T => JSON.parse(readFileSync(path.join(FIX, rel), "utf8")) as T;
const manifest = read<Manifest>("manifest.json");
const input = {
  manifest,
  ports: read<PortsCollection>(manifest.ports_url!),
  official: read<OfficialDataset>(manifest.official_url!),
  stations: slimStations(read<ObservationDataset>(manifest.observations_url!)),
};
const NOW = new Date("2026-10-09T18:00:00Z");
const LAST = "2026-10-08T20:00:00Z"; // the previous evening, after CoastWatch's check of the fixture (Oct 8, 18:00 UTC)
const fmt = { fmtDate: (d: string) => d, fmtTime: (t: string) => t };
const clone = <T>(x: T): T => JSON.parse(JSON.stringify(x));
const snap = (place = "port-201", audience: "all" | "commercial" | "recreational" | "shellfish" = "commercial", species = "dungeness") => buildSnapshot(input, place, audience, species, NOW)!;
const kinds = (c: Change[]) => c.map((x) => x.kind);
const notice = (over: Partial<SnapNotice>): SnapNotice => ({ id: "x", agency: "CDPH", title: "Do not eat sport-harvested razor clams from Del Norte County", status: "active", published: null, effective: null, lifted: null, url: "https://www.cdph.ca.gov/x", tier: "applies", forYou: true, ...over });

describe("briefing snapshot", () => {
  it("is null for an unknown place, and carries the agency statements and dates for Crescent City Dungeness", () => {
    expect(buildSnapshot(input, "port-99999", "all", "any", NOW)).toBeNull();
    const s = snap();
    expect(s.place.name).toBe("Crescent City");
    expect(s.official.ok).toBe(true);
    expect(s.official.statements.map((x) => x.id)).toContain("cdfw-dungeness-2026-27-season-status");
    expect(s.official.checkedAt).toMatch(/^2026-/);
    expect(s.official.hotlines.length).toBeGreaterThan(0);
    for (const n of s.official.notices) expect(n.url ?? "https://").toMatch(/^https:\/\//);
  });

  it("keeps lifted notices for the place, with their status, so a later visit can say 'lifted'", () => {
    const ds = clone(input.official);
    const r = ds.registry.records.find((x) => x.id === "cdph-2026-sn26-020-razor-clam-del-norte")!;
    r.status = "lifted";
    r.lifted_date = "2026-10-12";
    const s = buildSnapshot({ ...input, official: ds }, "port-201", "shellfish", "razor-clam", NOW)!;
    expect(s.official.notices.find((n) => n.id === r.id)).toMatchObject({ status: "lifted", lifted: "2026-10-12" });
  });

  it("lists shore stations within 60 km nearest first, with sample dates (not publication dates)", () => {
    const s = snap("port-550", "all", "any");
    expect(s.stations.length).toBeGreaterThan(0);
    expect(s.stations.map((x) => x.km)).toEqual([...s.stations.map((x) => x.km)].sort((a, b) => a - b));
    for (const st of s.stations) expect(st.km).toBeLessThanOrEqual(60);
  });
});

describe("since you last checked", () => {
  it("lists nothing when nothing differs, and nothing across different places", () => {
    const s = snap();
    expect(diffSnapshots(s, clone(s), LAST, fmt)).toEqual([]);
    expect(diffSnapshots(snap("port-550"), s, LAST, fmt)).toEqual([]);
  });

  it("calls a notice issued since the last check only when the agency's own date is on or after it", () => {
    const prev = snap();
    const next = clone(prev);
    next.official.notices.push(notice({ id: "new-1", published: "2026-10-09" }));
    const c = diffSnapshots(prev, next, LAST, fmt);
    expect(kinds(c)).toEqual(["agency-issued"]);
    expect(c[0].dates).toContainEqual({ label: "CDPH dated", date: "2026-10-09" });
  });

  it("never describes a previously unseen old notice as new: it is 'newly listed by Neritic Signal' with the agency's date", () => {
    const prev = snap();
    const next = clone(prev);
    next.official.notices.push(notice({ id: "old-1", published: "2026-09-14", title: "Do not eat sport-harvested bivalve shellfish from Monterey County" }));
    next.official.notices.push(notice({ id: "undated", published: null, effective: null, title: "Commercial rock crab: domoic acid closure" }));
    const c = diffSnapshots(prev, next, LAST, fmt);
    expect(kinds(c)).toEqual(["listed-late", "listed-late"]);
    expect(c[0].title).toMatch(/^Newly listed by Neritic Signal/);
    expect(c[0].detail).toMatch(/Not new from the agency: CDPH dated it 2026-09-14, before your last check/);
    expect(c[1].detail).toMatch(/gives no date/);
    for (const x of c) expect(`${x.title} ${x.detail}`).not.toMatch(/\bissued since|\bnew notice\b|after your last check/i);
  });

  it("distinguishes an agency lift after the last check from a lift Neritic Signal learned of late", () => {
    const base = snap();
    base.official.notices.push(notice({ id: "n1" }), notice({ id: "n2" }));
    const next = clone(base);
    Object.assign(next.official.notices.find((n) => n.id === "n1")!, { status: "lifted", lifted: "2026-10-09" });
    Object.assign(next.official.notices.find((n) => n.id === "n2")!, { status: "lifted", lifted: "2026-09-30" });
    const c = diffSnapshots(base, next, LAST, fmt);
    expect(c.find((x) => x.kind === "agency-lifted")?.detail).toMatch(/2026-10-09, after your last check/);
    expect(c.find((x) => x.kind === "listed-late")?.detail).toMatch(/now lists it as lifted\. CDPH dated that 2026-09-30, before your last check/);
  });

  it("says a notice is no longer listed without guessing why", () => {
    const prev = snap();
    prev.official.notices.push(notice({ id: "gone" }));
    const c = diffSnapshots(prev, snap(), LAST, fmt);
    expect(kinds(c)).toEqual(["removed"]);
    expect(c[0].detail).toMatch(/does not know why/);
  });

  it("reports a release that appears to lift a notice here as awaiting review, not as lifted", () => {
    const prev = snap("port-550", "all", "any");
    const next = clone(prev);
    const id = next.official.notices[0].id;
    next.official.disputes.push({ recordId: id, agency: "CDPH", release: "SN26-021", title: "CDPH Lifts Warning", url: "https://www.cdph.ca.gov/SN26-021" });
    const c = diffSnapshots(prev, next, LAST, fmt);
    expect(kinds(c)).toEqual(["release-pending"]);
    expect(c[0].detail).toMatch(/No person has reviewed it/);
    expect(c[0].source?.label).toBe("CDPH SN26-021");
  });

  it("quotes a changed agency status statement first, with both wordings and no invented date", () => {
    const prev = snap();
    const next = clone(prev);
    next.official.statements.find((s) => s.id === "cdfw-dungeness-2026-27-season-status")!.statement = "Commercial Fishery: Season opens in Zones 1 and 2 on December 1.";
    next.stations.push({ id: "s", name: "S", km: 1, url: "https://x", toxin: null, cells: null });
    next.official.notices.push(notice({ id: "z", published: "2026-10-09" }));
    const c = diffSnapshots(prev, next, LAST, fmt);
    expect(c[0].kind).toBe("agency-statement");
    expect(c[0].title).toMatch(/now says: “Commercial Fishery: Season opens/);
    expect(c[0].detail).toMatch(/At your last check it said: “Commercial Fishery: Season is closed/);
    expect(c[0].dates.map((d) => d.label)).toEqual(["Neritic Signal checked"]);
  });

  it("names a new sample by its sample date, saying when it was sampled before the last check", () => {
    const prev = snap("port-550", "all", "any");
    const next = clone(prev);
    prev.stations[0].toxin = "2026-09-01";
    next.stations[0].toxin = "2026-09-30";
    const c = diffSnapshots(prev, next, LAST, fmt);
    expect(c[0].kind).toBe("observation-new");
    expect(c[0].dates).toEqual([{ label: "sampled", date: "2026-09-30" }]);
    expect(c[0].detail).toMatch(/before your last check; the result was published since/);
    expect(c[0].detail).toMatch(/not a test of seafood/);
  });

  it("says when the newest toxin result has aged past 30 days since the last check", () => {
    const prev = snap("port-550", "all", "any");
    for (const s of prev.stations) s.toxin = null;
    prev.stations[0].toxin = "2026-09-08"; // 30 days old at the last check, 31 now
    const next = clone(prev);
    const c = diffSnapshots(prev, next, LAST, fmt);
    expect(kinds(c)).toEqual(["became-stale"]);
    expect(c[0].detail).toMatch(/No recent sample is not the same as no toxin/);
  });

  it("separates a new model run and source outages from agency information", () => {
    const prev = snap();
    const next = clone(prev);
    prev.forecast = { issued: "2026-10-05", outcome: "updated" };
    next.forecast = { issued: "2026-10-12", outcome: "failed" };
    prev.sources = [{ id: "charm", title: "C-HARM", outcome: "updated", lastSuccess: "2026-10-06T14:00:00Z", lastAttempt: "2026-10-06T14:00:00Z", latestValid: null, latestIssued: null }];
    next.sources = [{ id: "charm", title: "C-HARM", outcome: "failed", lastSuccess: "2026-10-13T14:14:44Z", lastAttempt: "2026-10-13T22:47:03Z", latestValid: null, latestIssued: null }];
    const c = diffSnapshots(prev, next, LAST, fmt);
    expect(kinds(c)).toEqual(["source-outage", "forecast-new"]);
    expect(c[1].detail).toMatch(/not a measurement of toxin/);
    const back = diffSnapshots(next, prev, LAST, fmt).map((x) => x.kind);
    expect(back).toContain("source-recovered");
  });

  it("reports official notices becoming unavailable rather than 'no notices'", () => {
    const prev = snap();
    const next = clone(prev);
    next.official = { ...next.official, ok: false, notices: [], statements: [], disputes: [], pages: [] };
    expect(kinds(diffSnapshots(prev, next, LAST, fmt))).toEqual(["unavailable"]);
  });

  it("puts the chosen species' items before other species, and never says open, safe or good fishing", () => {
    const prev = snap();
    const next = clone(prev);
    next.official.notices.push(notice({ id: "other", forYou: false, published: "2026-10-09" }), notice({ id: "mine", forYou: true, published: "2026-10-09" }));
    const c = diffSnapshots(prev, next, LAST, fmt);
    expect(c.map((x) => x.forYou)).toEqual([true, false]);
    for (const x of c) expect(`${x.title} ${x.detail}`).not.toMatch(/\b(is|are) (open|safe)\b|safe to eat|good (fishing|day)/i);
  });
});

describe("kept briefing", () => {
  it("keeps the time it was made, the waters, activity and species, every source link, and the warning", () => {
    const s = snap();
    const k = makeKept(s, NOW);
    const html = keptHtml(k);
    expect(html).toContain("Crescent City · Commercial fishing · Dungeness crab");
    expect(html).toContain(KEPT_WARNING);
    expect(html).toMatch(/Saved Oct 9/);
    for (const st of s.official.statements) expect(html).toContain(st.url);
    expect(html).toContain("Season is closed");
    expect(html).toMatch(/not a legal authorization to fish or harvest, and not proof that seafood is safe/);
    expect(html).not.toMatch(/<script|<link |src=/i);
  });

  it("escapes agency text, and states plainly when no notice was listed (never 'open')", () => {
    const s = snap("port-550", "commercial", "squid");
    s.official.statements = [];
    s.official.notices = s.official.notices.map((n) => ({ ...n, forYou: false }));
    s.place.name = "<img onerror=x>";
    const html = keptHtml(makeKept(s, NOW));
    expect(html).not.toContain("<img onerror");
    const lines = keptSections(makeKept(s, NOW)).flatMap((x) => x.lines.map((l) => l.text)).join(" ");
    expect(lines).toMatch(/No notice in Neritic Signal's list covered these waters for Market squid when this was saved\. That is not the same as open or safe\./);
  });
});

describe("remembered waters", () => {
  const mem = new Map<string, string>();
  const storage = { getItem: (k: string) => mem.get(k) ?? null, setItem: (k: string, v: string) => void mem.set(k, v), removeItem: (k: string) => void mem.delete(k) };
  beforeEach(() => {
    mem.clear();
    (globalThis as unknown as { window: unknown }).window = { localStorage: storage, sessionStorage: storage };
  });
  afterEach(() => delete (globalThis as unknown as { window?: unknown }).window);

  it("returns the previous visit for the same waters, activity and species, and remembers the activity and species for the place", () => {
    expect(recordVisit(snap(), new Date(LAST)).previous).toBeNull();
    const second = recordVisit(snap(), NOW);
    expect(second.previous?.lastSeenAt).toBe(new Date(LAST).toISOString());
    expect(readWaters()).toHaveLength(1);
    expect(rememberedFor("port-201")).toMatchObject({ audience: "commercial", species: "dungeness" });
    forgetAll();
    expect(readWaters()).toEqual([]);
  });

  it("behaves as a first visit, without errors, when storage is blocked", () => {
    const throwing = { getItem: () => { throw new Error("blocked"); }, setItem: () => { throw new Error("blocked"); }, removeItem: () => { throw new Error("blocked"); } };
    (globalThis as unknown as { window: unknown }).window = { localStorage: throwing, sessionStorage: throwing };
    expect(storageAvailable()).toBe(false);
    expect(readWaters()).toEqual([]);
    expect(recordVisit(snap(), NOW)).toEqual({ previous: null, saved: false });
    expect(() => forgetAll()).not.toThrow();
  });
});

