import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import path from "node:path";
import type { ObservationDataset } from "@/generated/observations";
import type { StationEvidence } from "@/lib/hab";
import { slimStations } from "@/lib/data";
import { harborEvidence } from "@/lib/harborEvidence";
import { rankWords, recordContext, recordSamples, seasonWords, MIN_SEASON_SAMPLES } from "@/lib/history";

/**
 * A station's newest HAB sample against its own record: the counting rules (zeros count, rejected
 * values do not, the two Pseudo-nitzschia classes are summed), the minimum record, and the words,
 * which never claim more than a rank in the station's own past.
 */
const obs = JSON.parse(readFileSync(path.join(__dirname, "../fixture-data/v1/observations-6d325da89a.json"), "utf8")) as ObservationDataset;

const st = (rows: [string, number | null, number | null, string?][]) => ({
  sample_times: rows.map((r) => `${r[0]}T17:00:00Z`),
  series: [
    { variable: "pn_seriata", values: rows.map((r) => r[1]), qualifiers: Object.fromEntries(rows.flatMap((r, i) => (r[3] ? [[String(i), r[3]]] : []))) },
    { variable: "pn_delicatissima", values: rows.map((r) => r[2]) },
  ],
});

/** n same-season samples a year for `years` years, values 0..n-1 ×100, then a newest October sample. */
function synthetic(years: number, perYear: number, newest: number) {
  const rows: [string, number | null, number | null][] = [];
  for (let y = 0; y < years; y++) for (let i = 0; i < perYear; i++) rows.push([`${2015 + y}-10-${String(1 + i).padStart(2, "0")}`, i * 100, null]);
  rows.push(["2026-06-01", 99999, null]); // a different season: never compared
  rows.push(["2026-10-05", newest, null]);
  return st(rows);
}

describe("station record", () => {
  it("sums the two size classes on a visit, keeps reported zeros, drops rejected values and unmeasured visits", () => {
    const s = st([
      ["2024-01-01", 100, 50],
      ["2024-01-08", null, 70],
      ["2024-01-15", 0, null],
      ["2024-01-22", -5, null, "rejected_negative"],
      ["2024-01-29", null, null],
    ]);
    expect(recordSamples(s, "pn")).toEqual([
      { date: "2024-01-01", value: 150 },
      { date: "2024-01-08", value: 70 },
      { date: "2024-01-15", value: 0 },
    ]);
  });

  it("ranks the newest sample among the same months of earlier years only", () => {
    const r = recordContext(synthetic(4, 5, 250), "pn")!;
    expect(r.season.months).toEqual([9, 10, 11]);
    expect(seasonWords(r.season.months)).toBe("Sep–Nov");
    expect(r.season.n).toBe(20);
    expect(r.season.years).toBe(4);
    // 0,100,200 of each year's 0..400 are below 250: 12 of 20
    expect(r.season.below).toBeCloseTo(0.6);
    expect(rankWords(r)).toBe("higher than 60% of the Sep–Nov samples there in 2015–2018");
    // the June 2026 sample (99,999) is earlier and higher: the last sample at least as high
    expect(r.lastAsHigh).toEqual({ date: "2026-06-01", value: 99999 });
    expect(r.previous).toEqual({ date: "2026-06-01", value: 99999 });
  });

  it("refuses to rank a record that is too short, and says so", () => {
    const r = recordContext(synthetic(2, 10, 250), "pn")!; // 20 samples but only 2 years
    expect(r.season.below).toBeNull();
    expect(rankWords(r)).toBeNull();
    const r2 = recordContext(synthetic(4, 2, 250), "pn")!; // 4 years but 8 samples
    expect(r2.season.n).toBeLessThan(MIN_SEASON_SAMPLES);
    expect(rankWords(r2)).toBeNull();
  });

  it("never rounds to 0% or 100% unless every sample is lower (or none is)", () => {
    const top = recordContext(synthetic(4, 5, 100000), "pn")!;
    expect(rankWords(top)).toBe("higher than all 20 Sep–Nov samples there in 2015–2018");
    const low = recordContext(synthetic(4, 5, 0), "pn")!;
    expect(rankWords(low)).toBe("none counted, as in 20% of the Sep–Nov samples there in 2015–2018");
    const rows = synthetic(4, 5, 1);
    expect(rankWords(recordContext(rows, "pn")!)).toBe("higher than 20% of the Sep–Nov samples there in 2015–2018");
  });

  it("is computed for the published stations (Santa Cruz Wharf's record starts in 2014)", () => {
    const slim = slimStations(obs);
    const sc = slim.find((s) => s.station_id === "HABs-SantaCruzWharf")!;
    const r = sc.record?.pn;
    expect(r?.since.slice(0, 4)).toBe("2014");
    expect(r!.season.years).toBeGreaterThanOrEqual(MIN_SEASON_YEARS_IN_FIXTURE);
    // the newest record sample is the station's newest Pseudo-nitzschia sample
    const latest = sc.latest.filter((l) => l.variable.startsWith("pn_") && l.value != null).map((l) => l.date).sort().at(-1);
    expect(r!.date).toBe(latest);
  });

  it("becomes one sentence on the HAB line and its own node, never part of the line's plain text", () => {
    const slim = slimStations(obs);
    const sc = slim.find((s) => s.station_id === "HABs-SantaCruzWharf")!;
    const r = sc.record!.pn!;
    const now = new Date(`${r.date}T20:00:00Z`);
    const e: StationEvidence = { kind: "cells", station: sc, km: 41.2, bearing: "NNW", date: r.date, values: [{ variable: "pn_seriata", value: r.value }], ageDays: 0, series: {}, near: true };
    const g = harborEvidence({ place: { name: "Monterey", lat: 36.6, lon: -121.89 }, now, tripDate: r.date, measured: { cells: e, toxin: null, nearest: null }, model: null, sat: null, conditions: null, official: null });
    const hab = g.claims.find((c) => c.id === "hab")!;
    expect(hab.history).toMatch(/^Compared with earlier years at Santa Cruz Wharf: higher than \d+% of the Aug–Oct samples there in 2014–2025\. /);
    expect(hab.text).not.toContain(hab.history!);
    const node = g.nodes.find((n) => n.id === `record:pn:HABs-SantaCruzWharf`)!;
    expect(node.class).toBe("record_statistic");
    expect(hab.supports).toContain(node.id);
    expect(node.caveats.join(" ")).toMatch(/not a forecast/);
    for (const w of [hab.history!, node.text]) expect(w).not.toMatch(/\b(safe|unsafe|unusual|dangerous|risk|alarming)\b/i);
  });
});

const MIN_SEASON_YEARS_IN_FIXTURE = 3;
