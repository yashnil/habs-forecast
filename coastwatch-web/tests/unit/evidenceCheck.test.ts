import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import path from "node:path";
import type { ObservationDataset } from "@/generated/observations";
import type { OfficialDataset } from "@/generated/official";
import { associationsAt } from "@/lib/applicability";
import type { Audience } from "@/lib/coast";
import type { Conditions } from "@/lib/conditions";
import { slimStations } from "@/lib/data";
import { checkGraph, datesIn, numbersIn } from "@/lib/evidenceCheck";
import { measuredEvidence, stationsAround, EVIDENCE_RADIUS_KM, type Neighbourhood } from "@/lib/hab";
import { harborEvidence, type EvidenceGraph, type EvidenceInputs } from "@/lib/harborEvidence";
import { speciesStatements } from "@/lib/nextSteps";
import { SPECIES } from "@/content/species";

/**
 * The consistency rules (lib/evidenceCheck) over every harbour in the fixture data, every activity,
 * every species and each state the model, the satellite and the live conditions can be in: no
 * graph may break a rule. Then each rule is shown to catch the mistake it exists for.
 */
const FIX = path.resolve(__dirname, "../fixture-data/v1");
const mf = JSON.parse(readFileSync(path.join(FIX, "manifest.json"), "utf8")) as { official_url: string; observations_url?: string };
const official: OfficialDataset = JSON.parse(readFileSync(path.join(FIX, mf.official_url), "utf8"));
const obs = JSON.parse(readFileSync(path.join(FIX, "observations-6d325da89a.json"), "utf8")) as ObservationDataset;
const ports = (JSON.parse(readFileSync(path.join(FIX, "ports-53ac2a4e2e.geojson"), "utf8")) as { features: { geometry: { coordinates: [number, number] }; properties: { port_code: number; display_name: string } }[] }).features.map((f) => ({
  key: `port-${f.properties.port_code}`,
  name: f.properties.display_name,
  lat: f.geometry.coordinates[1],
  lon: f.geometry.coordinates[0],
}));
const stations = slimStations(obs);

const nb = (over: Partial<Neighbourhood>): Neighbourhood => ({ n: 0, median: null, min: null, max: null, nearest: null, here: null, cells: [], ...over });
const MODELS = (valid: string): EvidenceInputs["model"][] => [
  { days: [{ lead: 0, valid, pda: nb({ here: 0.31, n: 6, median: 0.3, min: 0.22, max: 0.41 }), pn: null }], issued: valid, fresh: { state: "current", basisDate: valid, ageDays: 0 } },
  { days: [{ lead: 0, valid, pda: nb({ n: 27, median: 0.74, min: 0.6, max: 0.79, nearest: { cell: { row: 0, col: 0, lat: 0, lon: 0 }, value: 0.7, km: 6.6 } }), pn: null }], issued: valid, fresh: { state: "stale", basisDate: valid, ageDays: 3 } },
  { days: [{ lead: 0, valid, pda: nb({ nearest: { cell: { row: 0, col: 0, lat: 0, lon: 0 }, value: 0.12, km: 31.4 } }), pn: null }], issued: valid, fresh: null },
  { days: [{ lead: 0, valid, pda: nb({}), pn: null }], issued: valid, fresh: null },
  null,
  "error",
];
const SATS: EvidenceInputs["sat"][] = [
  { value: 2.4, date: "2026-10-11", km: 0, sensor: "Sentinel-3 OLCI, 300 m", layerId: "olci300_chl", basis: { n: 14, min: 1.1, max: 3.9, radiusKm: 5 }, aside: { value: 22, reason: "shore" } },
  { value: 0.83, date: "2026-10-04", km: 1.2, sensor: "VIIRS, 750 m", layerId: "viirs_chl", basis: null, aside: null },
  { value: null, sensor: "VIIRS, 750 m", windowDays: 7, flagged: false, radiusKm: 5 },
  { value: null, sensor: "VIIRS, 750 m", windowDays: 7, flagged: true, radiusKm: 5 },
  null,
  "error",
];
const at = "2026-10-13T05:42:00Z";
const CONDITIONS: Conditions = {
  lat: 0,
  lon: 0,
  marine: { status: "ok", source: "NWS", url: "https://www.weather.gov/marine/", fetchedAt: at, data: { zone: "PZZ535", zoneName: "Monterey Bay", updated: at, periods: [{ name: "Today", text: "NW wind 10 to 15 kt. Seas 5 ft at 9 seconds." }] } },
  alerts: { status: "ok", source: "NWS", url: "https://www.weather.gov/", fetchedAt: at, data: [] },
  buoy: { status: "ok", source: "NDBC", url: "https://www.ndbc.noaa.gov/station_page.php?station=46042", fetchedAt: at, station: { id: "46042", name: "MONTEREY - 27NM WNW of Monterey, CA", km: 41.9, lat: 36.785, lon: -122.398 }, data: { windKt: { value: 13.6, at }, windDirDeg: { value: 310, at }, waveFt: { value: 6.2, at }, periodS: { value: 11, at } } },
  tides: { status: "ok", source: "CO-OPS", url: "https://tidesandcurrents.noaa.gov/", fetchedAt: at, data: [{ at: "2026-10-13 04:12", ft: 5.1, type: "H" }, { at: "2026-10-13 10:40", ft: -0.3, type: "L" }] },
  harbor: { status: "ok", source: "CO-OPS", url: "https://tidesandcurrents.noaa.gov/stationhome.html?id=9413450", fetchedAt: at, station: { id: "9413450", name: "Monterey", km: 0.9, lat: 36.6089, lon: -121.8914 }, basis: "in-harbour", data: { waterC: { value: 15.8, at }, flags: "0,0,0" } },
};

function graphs(now: Date): EvidenceGraph[] {
  const out: EvidenceGraph[] = [];
  const day = now.toISOString().slice(0, 10);
  const audiences: Audience[] = ["all", "commercial", "recreational", "shellfish"];
  let i = 0;
  for (const p of ports) {
    const measured = {
      cells: measuredEvidence(stations, p.lat, p.lon, "cells", now),
      toxin: measuredEvidence(stations, p.lat, p.lon, "toxin", now),
      nearest: stationsAround(stations, p.lat, p.lon, now, EVIDENCE_RADIUS_KM)[0] ?? null,
    };
    for (const audience of audiences) {
      for (const species of SPECIES.filter((s) => s.id === "any" || audience === "all" || s.activities.includes(audience as "commercial"))) {
        // cycle the source states through the combinations instead of a full product (ports × 4 × species × 36 would be slow)
        const model = MODELS(day)[i % 6];
        const sat = SATS[Math.floor(i / 6) % 6];
        const conditions = i % 3 === 0 ? null : CONDITIONS;
        i++;
        out.push(
          harborEvidence({
            place: { name: p.name, lat: p.lat, lon: p.lon },
            now,
            today: day,
            tripDate: day,
            measured,
            model,
            sat,
            conditions,
            official: { assocs: associationsAt(official, p.key, p.lat, p.lon), verification: null, audience, species, statements: speciesStatements(official, species, audience) },
          }),
        );
      }
    }
  }
  return out;
}

describe("evidence consistency rules", () => {
  it("finds numbers and dates the way a reader would", () => {
    expect(numbersIn("Santa Cruz Wharf, 41 km NNW: Pseudo-nitzschia 4,900 cells/L, Sep 30")).toEqual(["41", "4900"]);
    expect(numbersIn("higher than 98% of the Aug–Oct samples there in 2014–2025. Lowest tide -0.3 ft at 10:40 AM")).toEqual(["98", "0.3"]);
    expect(datesIn("Sep 30 and October 15, 2025")).toEqual(["09-30", "10-15"]);
    // a release id is a name: found on live data (CDPH SN26-021 disputing SN26-018), absent from the fixtures
    expect(numbersIn("Release SN26-021. Confirm with CDPH which applies")).toEqual([]);
    expect(numbersIn("CDPH SN26-018: 4 sites, 0.21 ng/mL")).toEqual(["4", "0.21"]);
  });

  for (const now of [new Date("2026-10-13T06:00:00Z"), new Date("2026-09-03T06:00:00Z"), new Date("2026-04-20T06:00:00Z")]) {
    it(`holds for every harbour, activity, species and source state (${now.toISOString().slice(0, 10)})`, () => {
      const gs = graphs(now);
      expect(gs.length).toBeGreaterThan(500);
      const bad = gs.flatMap((g) => checkGraph(g).map((v) => `${g.place.name}: ${v.rule} ${v.claim}: ${v.message}`));
      expect([...new Set(bad)].slice(0, 20)).toEqual([]);
    });
  }

  it("catches each mistake it exists for", () => {
    const g = graphs(new Date("2026-10-13T06:00:00Z")).find((x) => x.claims.some((c) => c.id === "hab" && c.kind === "measured") && x.claims.some((c) => c.id === "satellite"))!;
    const plant = (f: (g: EvidenceGraph) => void) => {
      const h: EvidenceGraph = structuredClone(g);
      f(h);
      return checkGraph(h).map((v) => v.rule);
    };
    const hab = (h: EvidenceGraph) => h.claims.find((c) => c.id === "hab")!;
    expect(checkGraph(g)).toEqual([]);
    expect(plant((h) => (hab(h).headline += " (9,999 cells/L)"))).toContain("numbers");
    expect(plant((h) => (hab(h).headline += ", Feb 2"))).toContain("dates");
    expect(plant((h) => hab(h).supports.push("pda:nowhere"))).toContain("cites");
    expect(plant((h) => (hab(h).supports = []))).toContain("cites");
    expect(plant((h) => (h.claims.find((c) => c.id === "satellite")!.detail = "Chlorophyll was high."))).toContain("kind-words");
    expect(plant((h) => (hab(h).history = "A forecast of more to come."))).toContain("kind-words");
    expect(plant((h) => (hab(h).detail = "The water is safe to harvest. Seawater, not seafood."))).toContain("safety");
    expect(plant((h) => (hab(h).detail = "A low probability does not mean the water is safe. Seawater, not seafood."))).not.toContain("safety");
    expect(plant((h) => (h.claims.find((c) => c.id === "satellite")!.detail = hab(h).detail))).toContain("once");
  });

  // the #66 review's species-line mistakes, each caught: 3.4 readings said to be about toxin, 3.2 an agency
  // page line without its "Neritic's copy, not reviewed" provenance, 3.1 an end date in a standing line
  it("catches the species-line mistakes the #66 review found", () => {
    const g = graphs(new Date("2026-10-13T06:00:00Z")).find((x) => x.claims.some((c) => c.id === "species" && c.kind === "agency_page"))!;
    const plant = (f: (sp: EvidenceGraph["claims"][number]) => void) => {
      const h: EvidenceGraph = structuredClone(g);
      f(h.claims.find((c) => c.id === "species")!);
      return checkGraph(h).map((v) => `${v.rule}: ${v.message}`);
    };
    expect(checkGraph(g)).toEqual([]);
    expect(plant((c) => (c.detail = "Neritic's shore readings and NOAA's model are about domoic acid in the water, not in the crab."))).toContain("kind-words: readings, cells or chlorophyll said to be about toxin");
    expect(plant((c) => (c.detail = "The chlorophyll fluorescence at the pier is about toxin."))).toContain("kind-words: readings, cells or chlorophyll said to be about toxin");
    expect(plant((c) => (c.provenance = undefined))).toContain("kind-words: an agency-page line without “Neritic's copy … not reviewed by a person”");
    expect(plant((c) => (c.provenance = "The agency's page."))).toContain("kind-words: an agency-page line without “Neritic's copy … not reviewed by a person”");
    expect(plant((c) => (c.detail = "CDPH quarantines sport-harvested mussels from May 1 to October 31 every year."))).toContain("kind-words: an agency-page line with a calendar end date (reads as over once it passes)");
  });
});
