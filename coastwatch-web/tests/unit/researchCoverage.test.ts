import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import type { ObservationDataset, ObsStation } from "@/generated/observations";
import { cadenceWords, directionWords, modelCoverage, usefulStation, placeCoverage, programmeVariables, reachOf, reachWords, stationsAround, valueTimes, varCoverage, whereWords, type Located } from "@/nx/research/coverage";
import { located } from "@/nx/research/model";

/**
 * Research coverage (v15.1): per variable, per station, around a place. Fixture stations (2026-10-08 copy) plus
 * synthetic stations for the states the fixture does not hold (failed source, a burst of samples).
 */
const dir = path.join(__dirname, "../fixture-data/v1");
const observations = JSON.parse(readFileSync(path.join(dir, readdirSync(dir).find((f) => f.startsWith("observations-"))!), "utf8")) as ObservationDataset;
const stations = located(observations) as unknown as Located[];
const NOW = Date.parse("2026-10-13T06:00:00Z");
const DAY = 86_400_000;
const st = (id: string) => stations.find((s) => s.station_id === id)!;
const CRESCENT = { lat: 41.74901, lon: -124.20398 };
const MOSS = { lat: 36.8044, lon: -121.78691 };
const FORT_BRAGG = { lat: 39.42518, lon: -123.80376 };

function synth(times: string[], values: (number | null)[], extra: Partial<ObsStation> = {}): Located {
  return {
    station_id: "HABs-Synth",
    name: "Synthetic",
    lat: 37,
    lon: -122,
    status: "updated",
    sample_times: times,
    series: [{ variable: "pDA", values, qualifiers: {} }],
    ...extra,
  } as unknown as Located;
}

describe("one variable at one station", () => {
  it("a weekly, current record is regular; its cadence is weekly", () => {
    const c = varCoverage(st("HABs-CalPolyPier"), "pn_seriata", NOW);
    expect(c.state).toBe("recent");
    expect(cadenceWords(c.cadenceDays)).toBe("weekly");
    expect(c.nYear).toBeGreaterThan(40);
  });

  it("Monterey Wharf: temperature is current, pDA is an old but valid record ending Aug 31, 2022", () => {
    expect(varCoverage(st("HABs-MontereyWharf"), "temp", NOW).state).toBe("recent");
    const p = varCoverage(st("HABs-MontereyWharf"), "pDA", NOW);
    expect(p.state).toBe("old");
    expect(p.last?.slice(0, 10)).toBe("2022-08-31");
    expect(p.nYear).toBe(0);
  });

  it("a variable a station never published is 'not collected', never zero", () => {
    const c = varCoverage(st("HABs-TrinidadPier"), "pn_seriata", NOW);
    expect(c).toMatchObject({ state: "notCollected", n: 0, last: null, cadenceDays: null });
  });

  it("Trinidad Pier pDA (last Apr 17, 2026) is old at the fixture's clock, sampled every few weeks", () => {
    const c = varCoverage(st("HABs-TrinidadPier"), "pDA", NOW);
    expect(c.state).toBe("old");
    expect(c.last?.slice(0, 10)).toBe("2026-04-17");
    expect(c.cadenceDays).toBeGreaterThan(14);
  });

  it("recent but far apart is 'sparse', not 'regular'", () => {
    const times = [0, 30, 60, 90, 120].map((d) => new Date(NOW - (120 - d) * DAY - 3 * DAY).toISOString().replace(/\.\d+Z$/, "Z"));
    const c = varCoverage(synth(times, [0.1, 0.2, 0, 0.3, 0.1]), "pDA", NOW);
    expect(c.state).toBe("sparse");
    expect(cadenceWords(c.cadenceDays)).toBe("every 30 days");
  });

  it("a rejected sample and an unmeasured visit are not values", () => {
    const times = ["2026-10-01T18:00:00Z", "2026-10-08T18:00:00Z", "2026-10-10T18:00:00Z"];
    const s = synth(times, [0.1, -0.2, null]);
    (s.series[0] as { qualifiers: Record<string, string> }).qualifiers = { "1": "rejected_negative" };
    expect(valueTimes(s, "pDA")).toEqual([Date.parse(times[0])]);
    expect(varCoverage(s, "pDA", NOW).last).toBe(times[0]);
  });

  it("a failed source with nothing in the copy is 'source unavailable', not 'not collected'", () => {
    const c = varCoverage(synth([], [], { status: "failed" }), "pDA", NOW);
    expect(c.state).toBe("unavailable");
  });
});

describe("stations around a place", () => {
  it("Crescent City: nothing within 60 km; the three nearest are listed with distance and direction", () => {
    const rows = stationsAround(stations, CRESCENT, NOW);
    expect(rows).toHaveLength(3);
    expect(rows.every((r) => !r.near)).toBe(true);
    expect(rows[0].station.station_id).toBe("HABs-TrinidadPier");
    expect(Math.round(rows[0].km)).toBe(77);
    expect(rows[0].dir).toBe("south");
    expect(whereWords(rows[0])).toBe("77 km south");
    expect(rows[0].measures).toEqual(["pDA", "tDA"]);
  });

  it("Moss Landing: the stations within 60 km, nearest first", () => {
    const rows = stationsAround(stations, MOSS, NOW, 1);
    expect(rows.map((r) => r.station.station_id)).toEqual(["HABs-MontereyWharf", "HABs-SantaCruzWharf"]);
    expect(rows.every((r) => r.near)).toBe(true);
  });

  it("the programme's variables are those some station publishes (no station publishes dDA)", () => {
    const v = programmeVariables(stations);
    expect(v).not.toContain("dDA");
    expect(v).toEqual(expect.arrayContaining(["pDA", "tDA", "pn_seriata", "pn_delicatissima", "chl_extracted", "temp"]));
  });
});

describe("per variable: the nearest station that measures it, in words that never stand in for the place", () => {
  it("Crescent City pDA: Trinidad Pier, 77 km south; nothing nearer samples it", () => {
    const r = reachOf(stations, CRESCENT, "pDA", NOW);
    expect(r.nNear).toBe(0);
    expect(reachWords(r, "pDA", null)).toBe("The nearest station measuring pDA is Trinidad Pier, 77 km south; nothing nearer samples it.");
  });

  it("Crescent City Pseudo-nitzschia: the nearest station counting it is Bodega Marine Lab, hundreds of km south", () => {
    const r = reachOf(stations, CRESCENT, "pn_seriata", NOW);
    expect(r.nearest?.station.station_id).toBe("HABs-BodegaMarineLab");
    expect(r.nearest!.km).toBeGreaterThan(350);
    expect(r.nearest!.near).toBe(false);
  });

  it("Fort Bragg: no station within 60 km for any variable", () => {
    const rows = placeCoverage(stations, FORT_BRAGG, NOW);
    expect(rows.every((r) => r.nearest && !r.nearest.near)).toBe(true);
  });

  it("near: the station and its distance, as a plain statement", () => {
    const r = reachOf(stations, MOSS, "temp", NOW);
    expect(reachWords(r, "water temperature", null)).toBe("Monterey Wharf, 24 km south, measures water temperature.");
  });

  it("when the nearest record of a variable is old, a farther station with a current one is offered beside it", () => {
    const r = reachOf(stations, MOSS, "pDA", NOW);
    expect(r.nearest?.station.station_id).toBe("HABs-MontereyWharf");
    expect(r.recent?.station.station_id).toBe("HABs-SantaCruzWharf");
  });

  it("no station at all: says so", () => {
    expect(reachWords({ nearest: null }, "dDA", null)).toBe("No CalHABMAP station measures dDA.");
  });
});

describe("words and the model", () => {
  it("directions in words", () => {
    expect(directionWords(41.75, -124.2, 41.05, -124.15)).toBe("south");
    expect(directionWords(36.8, -121.79, 36.96, -122.03)).toBe("northwest");
    expect(directionWords(36.8, -121.79, 36.8001, -121.79)).toBe("here");
  });

  it("C-HARM history is counted apart from samples, as model days", () => {
    const m = modelCoverage(st("HABs-SantaCruzWharf") as unknown as ObsStation);
    expect(m.days).toBeGreaterThan(0);
    expect(m.radiusKm).toBe(15);
    expect(modelCoverage({ charm: null } as unknown as ObsStation).days).toBe(0);
  });
});

describe("the station a place opens on, by usefulness", () => {
  it("Moss Landing opens on Santa Cruz Wharf (recent pDA), naming Monterey Wharf, nearer, whose record is older", () => {
    const p = usefulStation(stations, MOSS, null, NOW)!;
    expect(p.station.station_id).toBe("HABs-SantaCruzWharf");
    expect(p.passed?.station.station_id).toBe("HABs-MontereyWharf");
    expect(p.passed!.km).toBeLessThan(p.km);
  });

  it("Fort Bragg opens on Bodega Marine Lab (Apr 2026), not the buoy beside it (Jan 2026)", () => {
    const p = usefulStation(stations, FORT_BRAGG, null, NOW)!;
    expect(p.station.station_id).toBe("HABs-BodegaMarineLab");
    expect(p.last.slice(0, 10)).toBe("2026-04-13");
  });

  it("Crescent City opens on Trinidad Pier, the nearest, with nothing passed over", () => {
    const p = usefulStation(stations, CRESCENT, null, NOW)!;
    expect(p.station.station_id).toBe("HABs-TrinidadPier");
    expect(p.passed).toBeNull();
  });

  it("an asked variable leads: temperature at Moss Landing is Monterey Wharf's, current and nearest", () => {
    const p = usefulStation(stations, MOSS, "temp", NOW)!;
    expect(p.station.station_id).toBe("HABs-MontereyWharf");
    expect(p.passed).toBeNull();
  });

  it("two current stations a few km apart: the nearer stays the choice (Morro Bay front bay, not back bay)", () => {
    expect(usefulStation(stations, { lat: 35.36581, lon: -120.84991 }, null, NOW)!.station.station_id).toBe("HABs-MorroBayFrontBay");
  });

  it("a station at the port with a current record stays the choice (Avila: Cal Poly Pier)", () => {
    expect(usefulStation(stations, { lat: 35.17498, lon: -120.75519 }, null, NOW)!.station.station_id).toBe("HABs-CalPolyPier");
  });
});
