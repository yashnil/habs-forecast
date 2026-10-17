import { describe, expect, it } from "vitest";
import type { StationEvidence } from "@/lib/hab";
import { readFileSync } from "node:fs";
import path from "node:path";
import type { OfficialDataset } from "@/generated/official";
import { speciesById } from "@/content/species";
import { associationsAt } from "@/lib/applicability";
import { speciesStatements } from "@/lib/nextSteps";
import type { Conditions } from "@/lib/conditions";
import type { Neighbourhood } from "@/lib/hab";
import { harborEvidence, pageClaims, periodFor, representative, stationSetting, topicWords, type EvidenceInputs } from "@/lib/harborEvidence";

/**
 * The harbor evidence graph: one place, every reading typed by how it is known, with its distance,
 * age and representativeness, and claims that cite the nodes they rest on. The acceptance
 * statement for Monterey is the product brief's own example.
 */
const MONTEREY = { name: "Monterey", lat: 36.60024, lon: -121.89469 };
const NOW = new Date("2026-10-15T20:00:00Z");

const santaCruz: StationEvidence = {
  kind: "cells",
  station: { station_id: "SantaCruzWharf", name: "Santa Cruz Wharf", lat: 36.958, lon: -122.017, status: "", source_url: "https://erddap.sccoos.org/erddap/tabledap/HABs-SantaCruzWharf.html", latest: [] },
  km: 41.2,
  bearing: "NNW",
  date: "2026-09-30",
  values: [{ variable: "pn_seriata", value: 4900 }],
  ageDays: 15,
  series: {},
  near: true,
};
const montereyWharf = { station: { station_id: "MontereyWharf", name: "Monterey Wharf", lat: 36.605, lon: -121.89, status: "", source_url: "", latest: [] }, km: 0.6, date: "2025-08-12", ageDays: 429 };
const offshore = { n: 27, median: 0.77, min: 0.73, max: 0.79, nearest: { cell: { row: 1, col: 1, lat: 36.55, lon: -121.95 }, value: 0.74, km: 6.6 }, here: null, cells: [] };

function inputs(over: Partial<EvidenceInputs> = {}): EvidenceInputs {
  return {
    place: MONTEREY,
    now: NOW,
    tripDate: "2026-10-15",
    measured: { cells: santaCruz, toxin: null, nearest: montereyWharf },
    model: { days: [{ lead: 2, valid: "2026-10-15", pda: offshore, pn: null }], issued: "2026-10-14", fresh: { state: "current", basisDate: "2026-10-14", ageDays: 1 } },
    sat: null,
    conditions: null,
    official: null,
    ...over,
  };
}

describe("harbor evidence graph", () => {
  it("states the measurement with its place, date and distance, names the closer silent station, and says the model has no estimate here", () => {
    const g = harborEvidence(inputs());
    const measured = g.claims.find((c) => c.id === "hab")!;
    expect(measured.text).toBe("Santa Cruz Wharf reported 4,900 cells/L Pseudo-nitzschia on September 30, 41 km NNW of Monterey. Monterey Wharf is closer (0.6 km) but has no comparable sample since August 12, 2025.");
    expect(measured.supports).toEqual(["pn_cells:SantaCruzWharf"]);
    const model = g.claims.find((c) => c.id === "model")!;
    expect(model.kind).toBe("gap");
    expect(model.text).toMatch(/^NOAA's C-HARM model has no valid estimate in these nearshore waters\. Its nearest cell is 6\.6 km out; the 27 offshore cells within 15 km read 73%–79% for October 15, context for the open water, not the harbor\.$/);
    // no claim speaks of safety, catch or a harbour-wide risk score
    for (const c of g.claims) expect(c.text).not.toMatch(/\b(safe|unsafe|risk score|catch)\b/i);
    // every supporting id is a node in the graph
    const ids = new Set(g.nodes.map((n) => n.id));
    for (const c of g.claims) for (const s of c.supports) expect(ids.has(s)).toBe(true);
  });

  it("types each node by how it is known and never presents offshore cells as the waters' own", () => {
    const g = harborEvidence(inputs());
    const sc = g.nodes.find((n) => n.id === "pn_cells:SantaCruzWharf")!;
    expect(sc.class).toBe("measurement");
    expect(sc.where).toMatchObject({ km: 41.2, bearing: "NNW", representative: "context", setting: "exposed-pier" });
    expect(sc.caveats[0]).toMatch(/Seawater, not seafood/);
    const m = g.nodes.find((n) => n.id.startsWith("charm:"))!;
    expect(m.id).toBe("charm:offshore:2026-10-15");
    expect(m.class).toBe("published_forecast");
    expect(m.where.representative).toBe("context");
    expect(m.support).toMatchObject({ kind: "cells", n: 27, radiusKm: 15, nativeM: 3000 });
    expect(m.text).toBe("73%–79%");
    expect(g.gaps.find((x) => x.id === "model-here")!.reason).toMatch(/no valid estimate in these nearshore waters; its nearest cell with a value is 6\.6 km out/);
  });

  it("with the waters' own cell, the model is 'at' and the claim says cell, probability, not measurement", () => {
    const g = harborEvidence(inputs({ model: { days: [{ lead: 2, valid: "2026-10-15", pda: { ...offshore, here: 0.31 }, pn: null }], issued: "2026-10-14", fresh: { state: "current", basisDate: "2026-10-14", ageDays: 1 } } }));
    const m = g.nodes.find((n) => n.id === "charm:own:2026-10-15")!;
    expect(m.where.representative).toBe("at");
    expect(g.claims.find((c) => c.id === "model")!.text).toBe("NOAA's C-HARM gives the 3 km cell containing Monterey a 31% chance that particulate domoic acid exceeds its threshold on October 15. A probability, not a measurement.");
  });

  it("an old run is named by its date and age in the claim and the node, not called a failure", () => {
    const g = harborEvidence(inputs({ model: { days: [{ lead: 2, valid: "2026-10-15", pda: { ...offshore, here: 0.31 }, pn: null }], issued: "2026-10-12", fresh: { state: "stale", basisDate: "2026-10-12", ageDays: 3 } } }));
    expect(g.claims.find((c) => c.id === "model")!.text).toMatch(/\(run of Oct 12, 3 days old\)/);
    expect(g.nodes.find((n) => n.id.startsWith("charm:"))!.freshness).toEqual({ state: "stale", note: "run of Oct 12, 3 days old" });
  });

  it("says a gap as a gap: no recent sample, nothing near, a model day beyond the run, no clear satellite view", () => {
    const g = harborEvidence(
      inputs({
        measured: { cells: { ...santaCruz, ageDays: 200, date: "2026-03-29" }, toxin: null, nearest: null },
        model: { days: [{ lead: 3, valid: "2026-10-16", pda: offshore, pn: null }], issued: "2026-10-14", fresh: null },
        tripDate: "2026-10-18",
        sat: { value: null, sensor: "Sentinel-3 OLCI, 300 m", windowDays: 7, flagged: false, radiusKm: 5 },
      }),
    );
    expect(g.claims.find((c) => c.id === "hab")!.kind).toBe("gap");
    expect(g.gaps.map((x) => x.id)).toEqual(["pn_cells-recent", "model-day", "satellite"]);
    expect(g.gaps[0].reason).toMatch(/No Pseudo-nitzschia cells sample within 60 km in the last 30 days; the latest nearby is 7 months ago at Santa Cruz Wharf \(41 km\)/);
    expect(g.gaps[2].reason).toMatch(/not low chlorophyll/);
    // the old sample is still a node, flagged historical, so nothing is hidden
    expect(g.nodes.find((n) => n.id === "pn_cells:SantaCruzWharf")!.freshness.state).toBe("historical");
  });

  it("water temperature from a tide station is the harbour's own within 2 km, the same bay to 6 km, another water body beyond", () => {
    expect(representative("water_temp", 1.0)).toBe("at");
    expect(representative("water_temp", 5.7)).toBe("near");
    expect(representative("water_temp", 10.9)).toBe("context");
  });

  it("representativeness is per variable, not one radius", () => {
    expect(representative("pn_cells", 1.5)).toBe("at");
    expect(representative("pn_cells", 12)).toBe("near");
    expect(representative("pn_cells", 41)).toBe("context");
    expect(representative("pn_cells", 90)).toBe("far");
    expect(representative("chl_satellite", 3)).toBe("near");
    expect(representative("chl_satellite", 12)).toBe("context");
    expect(representative("wind", 25)).toBe("near");
    expect(representative("water_temp", 25)).toBe("far");
    expect(stationSetting("Santa Cruz Wharf")).toBe("exposed-pier");
    expect(stationSetting("Moss Landing, South Harbor, CA")).toBe("sheltered-harbor");
    expect(stationSetting("Monterey Canyon Outer, CA (156)")).toBe("offshore");
    expect(stationSetting("MBM1")).toBe("unclassified");
  });
});


// ---------------------------------------------------------------- the briefing's claims (moved from lib/trip)

const FIX = path.resolve(__dirname, "../fixture-data/v1");
const mf = JSON.parse(readFileSync(path.join(FIX, "manifest.json"), "utf8")) as { official_url: string };
const official: OfficialDataset = JSON.parse(readFileSync(path.join(FIX, mf.official_url), "utf8"));
const CRESCENT = { name: "Crescent City", lat: 41.74901, lon: -124.20398 };
const nb = (over: Partial<Neighbourhood>): Neighbourhood => ({ n: 0, median: null, min: null, max: null, nearest: null, here: null, cells: [], ...over });
const CLOCK = new Date("2026-10-13T20:00:00Z");

function brief(over: Partial<EvidenceInputs> & { at?: { name: string; lat: number; lon: number }; audience?: "all" | "commercial" | "recreational" | "shellfish"; speciesId?: string } = {}): EvidenceInputs {
  const at = over.at ?? MONTEREY;
  const species = speciesById(over.speciesId ?? "any");
  const audience = over.audience ?? "all";
  return {
    place: at,
    now: CLOCK,
    today: "2026-10-13",
    tripDate: "2026-10-13",
    measured: { cells: null, toxin: null, nearest: null },
    model: null,
    sat: null,
    conditions: null,
    official: { assocs: associationsAt(official, "x", at.lat, at.lon), verification: null, audience, species, statements: speciesStatements(official, species, audience) },
    ...over,
  };
}
const claim = (x: EvidenceInputs, id: string) => harborEvidence(x).claims.find((c) => c.id === id)!;

describe("the briefing's claims", () => {
  it("a Dungeness crab trip leads with CDFW's season status, quoted, with its source", () => {
    const g = harborEvidence(brief({ at: CRESCENT, audience: "commercial", speciesId: "dungeness" }));
    const first = g.claims[0];
    expect(first.id).toBe("season");
    expect(first.kind).toBe("official");
    expect(first.quote).toMatch(/season|closed|delay/i);
    expect(first.source?.url).toMatch(/^https:\/\/wildlife\.ca\.gov/);
    expect(first.supports[0]).toMatch(/^statement:/);
    expect(g.nodes.some((n) => n.id === first.supports[0])).toBe(true);
  });

  it("a missing sample is a gap, never reassurance", () => {
    const g = harborEvidence(brief({ at: CRESCENT }));
    const hab = claim(brief({ at: CRESCENT }), "hab");
    expect(hab.kind).toBe("gap");
    expect(`${hab.headline} ${hab.detail}`).toMatch(/not a clean result/);
    for (const c of g.claims) expect(`${c.headline} ${c.detail ?? ""} ${c.text}`).not.toMatch(/\b(safe|all clear|clean)\b(?! result)/i);
  });

  it("an offshore model value is never presented as the waters' own, and a stale run is named by its date, not called stale", () => {
    const model = (pda: Neighbourhood, fresh: EvidenceInputs["model"] extends infer M ? (M extends { fresh: infer F } ? F : never) : never) => ({ days: [{ lead: 2, valid: "2026-10-13", pda, pn: null }], issued: "2026-10-12", fresh });
    const offshore = claim(brief({ model: model(nb({ n: 27, median: 0.74, min: 0.6, max: 0.79, nearest: { cell: { row: 0, col: 0, lat: 36.55, lon: -121.95 }, value: 0.7, km: 6.6 } }), { state: "stale", basisDate: "2026-10-12", ageDays: 1 }) }), "hab");
    expect(offshore.kind).toBe("gap");
    // the HAB line says what the shore stations say; the model is its own line, said once
    expect(offshore.detail).not.toMatch(/C-HARM|offshore cells/);
    const offshoreModel = claim(brief({ model: model(nb({ n: 27, median: 0.74, min: 0.6, max: 0.79, nearest: { cell: { row: 0, col: 0, lat: 36.55, lon: -121.95 }, value: 0.7, km: 6.6 } }), { state: "stale", basisDate: "2026-10-12", ageDays: 1 }) }), "model");
    expect(offshoreModel.kind).toBe("gap");
    expect(offshoreModel.headline).toMatch(/no valid NOAA model estimate for these nearshore waters/i);
    expect(offshoreModel.detail).toMatch(/27 offshore cells within 15 km read 60%–79% for Oct 13, context for the open water, not the harbour/);
    expect(`${offshoreModel.headline} ${offshoreModel.detail}`).not.toMatch(/median|stale/);
    const page = pageClaims(harborEvidence(brief({ model: model(nb({ n: 27, median: 0.74, min: 0.6, max: 0.79, nearest: { cell: { row: 0, col: 0, lat: 36.55, lon: -121.95 }, value: 0.7, km: 6.6 } }), { state: "stale", basisDate: "2026-10-12", ageDays: 1 }) })));
    expect(page.map((c) => c.id)).toEqual(expect.arrayContaining(["hab", "model"]));
    // no fact is said twice across the page's lines
    const sentences = page.flatMap((c) => `${c.headline}. ${c.detail ?? ""}`.split(/(?<=[.;])\s+/).map((x) => x.trim().toLowerCase()).filter((x) => x.split(" ").length >= 5));
    expect(new Set(sentences).size).toBe(sentences.length);
    const own = claim(brief({ model: model(nb({ here: 0.31, n: 5, median: 0.3 }), { state: "current", basisDate: "2026-10-12", ageDays: 1 }) }), "hab");
    expect(own.kind).toBe("model");
    expect(own.headline).toMatch(/31%/);
    expect(own.detail).toMatch(/not a measurement/);
    expect(own.supports).toEqual(["charm:own:2026-10-13"]);
    // the model's own claim is folded into the HAB line on the page, but stays in the graph for the API
    const g = harborEvidence(brief({ model: model(nb({ here: 0.31, n: 5, median: 0.3 }), { state: "current", basisDate: "2026-10-12", ageDays: 1 }) }));
    expect(g.claims.map((c) => c.id)).toContain("model");
    expect(pageClaims(g).map((c) => c.id)).not.toContain("model");
  });

  it("a day beyond the model says the model does not reach it", () => {
    const hab = claim(brief({ tripDate: "2026-10-16", modelBeyond: true, model: { days: [{ lead: 3, valid: "2026-10-14", pda: nb({ n: 3, median: 0.5 }), pn: null }], issued: "2026-10-11", fresh: null } }), "hab");
    // the HAB gap line leaves the model to the model line, which the page keeps beside it
    expect(hab.kind).toBe("gap");
    expect(hab.detail).not.toMatch(/model/);
    const model = claim(brief({ tripDate: "2026-10-16", modelBeyond: true, model: { days: [{ lead: 3, valid: "2026-10-14", pda: nb({ n: 3, median: 0.5 }), pn: null }], issued: "2026-10-11", fresh: null } }), "model");
    expect(model.headline).toBe("NOAA's model does not reach Oct 16");
    expect(model.detail).toBe("Its newest run ends Oct 14.");
  });

  it("more than three covering notices: the three named are followed by how many more, none dropped silently", () => {
    const n = harborEvidence(brief({})).claims.find((c) => c.id === "notices")!;
    expect(n.supports.length).toBeGreaterThan(3);
    expect(n.detail).toContain(`and ${n.supports.length - 3} more, all in Notices`);
    expect(n.text).toContain(`and ${n.supports.length - 3} more`);
  });

  it("notices that apply by their own geography are named, restrictive ones as alerts, and a disputed one leads", () => {
    const g = harborEvidence(brief({ speciesId: "anchovy-sardine" }));
    const n = g.claims.find((c) => c.id === "notices")!;
    expect(n.headline).toMatch(/notices cover these waters for anchovy or sardine|Do not eat northern anchovies/);
    expect(n.tone).toBe("alert");
    for (const id of n.supports) expect(g.nodes.find((x) => x.id === id)?.variable).toBe("official notice");
    const d = g.claims.find((c) => c.id.startsWith("disputed-"));
    if (d) {
      expect(g.claims.indexOf(d)).toBeLessThan(g.claims.indexOf(n));
      expect(d.headline).toMatch(/appears to have lifted/);
    }
  });

  it("conditions: NWS's period for the trip day, the harbour's water before the buoy, and an advisory as an alert", () => {
    const cond = (alerts: { event: string; headline: string }[], harbor = true): Conditions => ({
      lat: 36.6,
      lon: -121.9,
      marine: { status: "ok", data: { zone: "PZZ535", zoneName: "Monterey Bay", updated: null, periods: [{ name: "Tonight", text: "NW wind 10 to 15 kt. Seas 4 to 6 ft." }, { name: "Thursday", text: "NE wind 5 to 10 kt. Seas around 3 ft." }] }, source: "NWS", url: "https://marine.weather.gov/x", fetchedAt: "2026-10-13T20:00:00Z" },
      alerts: { status: "ok", data: alerts.map((a) => ({ ...a, onset: null, ends: null, url: "https://api.weather.gov/a" })), source: "NWS", url: "https://x", fetchedAt: "2026-10-13T20:00:00Z" },
      buoy: { status: "ok", data: { windKt: { value: 9.7, at: "2026-10-13T19:50:00Z" }, windDirDeg: { value: 330, at: "2026-10-13T19:50:00Z" }, waveFt: { value: 3.6, at: "2026-10-13T19:50:00Z" } }, source: "NDBC", url: "https://ndbc/46236", fetchedAt: "2026-10-13T20:00:00Z", station: { id: "46236", name: "Monterey Canyon Outer, CA (156)", km: 18, lat: 36.759, lon: -121.95 } },
      tides: { status: "ok", data: [{ at: "2026-10-13 04:51", ft: 2.2, type: "L" }, { at: "2026-10-13 11:02", ft: 5.8, type: "H" }], source: "CO-OPS", url: "https://tides/x", fetchedAt: "2026-10-13T20:00:00Z", station: { id: "9413450", name: "Monterey", km: 1 } },
      harbor: harbor ? { status: "ok", data: { waterC: { value: 17.6, at: "2026-10-13T19:54:00Z" }, flags: "0,0,0" }, source: "NOAA CO-OPS water temperature", url: "https://tidesandcurrents.noaa.gov/stationhome.html?id=9413450", fetchedAt: "2026-10-13T20:00:00Z", station: { id: "9413450", name: "Monterey", km: 1.0, lat: 36.605, lon: -121.889 } } : undefined,
    });
    const g = harborEvidence(brief({ conditions: cond([]) }));
    const c = g.claims.find((x) => x.id === "conditions")!;
    expect(c.headline).toBe("NWS tonight: NW wind 10 to 15 kt · Seas 4 to 6 ft");
    expect(c.detail).toMatch(/^Water 17\.6 °C in the harbour · Wind 10 kt NNW, waves 3\.6 ft at Monterey Canyon Outer buoy now · tides low 2\.2 ft, high 5\.8 ft\./);
    expect(c.supports).toEqual(expect.arrayContaining(["nws:PZZ535:Tonight", "coops:9413450:water_temperature", "buoy:46236:windKt", "buoy:46236:waveFt"]));
    expect(g.nodes.find((n) => n.id === "coops:9413450:water_temperature")!.where.representative).toBe("at");
    // the trip day's own period, by name
    const thu = harborEvidence(brief({ conditions: cond([]), tripDate: "2026-10-15" })).claims.find((x) => x.id === "conditions")!;
    expect(thu.headline).toBe("NWS for Thursday: NE wind 5 to 10 kt · Seas around 3 ft");
    expect(periodFor([{ name: "Tonight", text: "" }, { name: "Thu", text: "" }], "2026-10-15", "2026-10-13")?.name).toBe("Thu");
    const sca = harborEvidence(brief({ conditions: cond([{ event: "Small craft advisory", headline: "Small craft advisory in effect until 9 pm pdt this evening" }]) })).claims.find((x) => x.id === "conditions")!;
    expect(sca.tone).toBe("alert");
    expect(sca.headline).toMatch(/^Small craft advisory · NWS tonight/);
    // shellfish harvest leads with the lowest tide
    const shell = harborEvidence(brief({ conditions: cond([], false), audience: "shellfish" })).claims.find((x) => x.id === "conditions")!;
    expect(shell.detail).toMatch(/^Lowest tide 2\.2 ft at 4:51 AM/);
    // tides alone (no harbour thermometer, no buoy: Shelter Cove) still read as a sentence
    const tidesOnly = harborEvidence(brief({ conditions: { ...cond([], false), buoy: { status: "none", source: "NDBC", url: null, fetchedAt: "2026-10-13T20:00:00Z" } as unknown as Conditions["buoy"] } })).claims.find((x) => x.id === "conditions")!;
    expect(tidesOnly.detail).toMatch(/^Tides low 2\.2 ft, high 5\.8 ft\. Wind and seas/);
  });

  it("topic words read inside a sentence", () => {
    expect(topicWords("Dungeness crab, commercial — marine toxin closures")).toBe("marine toxin closures");
    expect(topicWords("Season status")).toBe("season status");
  });

  it("the claims never speak of safety or catch, and every supporting id is a node", () => {
    for (const x of [brief(), brief({ at: CRESCENT, audience: "commercial", speciesId: "dungeness" }), brief({ speciesId: "anchovy-sardine", audience: "recreational" })]) {
      const g = harborEvidence(x);
      const ids = new Set(g.nodes.map((n) => n.id));
      for (const c of g.claims) {
        expect(c.text).not.toMatch(/\b(safe|unsafe|risk score|catch)\b/i);
        for (const s of c.supports) expect(ids.has(s)).toBe(true);
      }
    }
  });
});
