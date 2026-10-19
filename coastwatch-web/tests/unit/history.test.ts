import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import path from "node:path";
import type { ObservationDataset } from "@/generated/observations";
import type { StationEvidence } from "@/lib/hab";
import { slimStations } from "@/lib/data";
import { harborEvidence, historySentence } from "@/lib/harborEvidence";
import { rankWords, recordContext, recordSamples, seasonWords, smallWords, MIN_SEASON_SAMPLES } from "@/lib/history";

/**
 * A station's newest HAB sample against its own record as Neritic holds it: the counting rules
 * (reported zeros are censored and never ranked as lower, rejected values are left out, the two
 * Pseudo-nitzschia classes are summed), the minimum record, and the words, which never claim more
 * than a rank in the station's own past and never call Neritic's 2014 window the station's record.
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
    // each year's 0..400: the 0 is censored (not ranked); of the 16 quantified, 100 and 200 are below 250: 8 of 16
    expect(r.season.zeros).toBe(4);
    expect(r.season.quantified).toBe(16);
    expect(r.season.below).toBeCloseTo(0.5);
    expect(rankWords(r)).toBe("higher than 50% of the 16 quantified Sep–Nov samples there in 2015–2018 (a small sample: each of the 16 is 6 points of the rank); another 4 had no cells counted (not necessarily none present) and are not ranked");
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
    expect(rankWords(top)).toBe("higher than all 16 quantified Sep–Nov samples there in 2015–2018 (a small sample: each of the 16 is 6 points of the rank); another 4 had no cells counted (not necessarily none present) and are not ranked");
    const low = recordContext(synthetic(4, 5, 0), "pn")!;
    expect(rankWords(low)).toBe("no cells counted, as in 20% of the 20 Sep–Nov samples there in 2015–2018 (a small sample: each of the 20 is 5 points of the rank)");
    // 1 cell/L: higher than the four zeros only, which are censored, so it is higher than none of the quantified
    const rows = synthetic(4, 5, 1);
    expect(rankWords(recordContext(rows, "pn")!)).toBe("none of the 16 quantified Sep–Nov samples there in 2015–2018 (a small sample: each of the 16 is 6 points of the rank) was lower; another 4 had no cells counted (not necessarily none present) and are not ranked");
  });

  // #66 review 1.1: a reported 0 is "at or below the laboratory's limit, not that toxin was absent" (pipeline)
  it("never counts a reported zero as lower than a measured value: Trinidad Pier's 0.0057 ng/mL is said as counts", () => {
    const slim = slimStations(obs);
    const tr = slim.find((x) => x.station_id === "HABs-TrinidadPier")!.record!.pDA!;
    expect(tr.value).toBeCloseTo(0.00567, 5);
    // the review's recomputation: 28 Mar–May samples in 2019–2024, 24 "lower" before, of which 23 were reported zeros
    expect([tr.season.n, tr.season.zeros, tr.season.quantified, tr.season.quantifiedBelow]).toEqual([28, 23, 5, 1]);
    expect(tr.season.below).toBeNull(); // 5 quantified samples are too few to rank
    expect(rankWords(tr)).toBe("of the 28 Mar–May samples there in 2019–2024, 23 were reported 0 (not quantified, not absent), and 1 of the 5 quantified was lower");
    expect(rankWords(tr)).not.toMatch(/%/);
    // Santa Cruz Wharf: of 139 samples "lower" before, 126 were zero counts; ranked among the 16 quantified only
    const sc = slim.find((x) => x.station_id === "HABs-SantaCruzWharf")!.record!.pn!;
    expect([sc.season.n, sc.season.zeros, sc.season.quantified, sc.season.quantifiedBelow]).toEqual([142, 126, 16, 13]);
    expect(rankWords(sc)).toBe("higher than 81% of the 16 quantified Aug–Oct samples there in 2014–2025 (a small sample: each of the 16 is 6 points of the rank); another 126 had no cells counted (not necessarily none present) and are not ranked");
  });

  it("says the counts when the season has zeros but too few quantified samples, and stays silent when the season itself is short", () => {
    // 4 years × 4 samples: three zeros and one 500 each year; newest 300
    const rows: [string, number | null, number | null][] = [];
    for (let y = 0; y < 4; y++) for (let i = 0; i < 4; i++) rows.push([`${2015 + y}-10-${String(1 + i).padStart(2, "0")}`, i === 3 ? 500 : 0, null]);
    rows.push(["2026-10-05", 300, null]);
    const r = recordContext(st(rows), "pn")!;
    expect(rankWords(r)).toBe("of the 16 Sep–Nov samples there in 2015–2018, 12 had no cells counted (not necessarily none present), and 0 of the 4 quantified were lower");
    expect(rankWords(recordContext(synthetic(2, 10, 250), "pn")!)).toBeNull();
  });

  // v4 integration, owner's check 7: a rank says its sample size, says a small one is small, and says how
  // values below the detection limit were handled (counted apart, never ranked)
  it("discloses a small sample and the below-detection-limit handling in the same sentence", () => {
    const slim = slimStations(obs);
    const sc = slim.find((x) => x.station_id === "HABs-SantaCruzWharf")!.record!;
    const pn = rankWords(sc.pn!)!;
    expect(pn).toContain("16 quantified");
    expect(pn).toContain("(a small sample: each of the 16 is 6 points of the rank)");
    expect(pn).toContain("another 126 had no cells counted (not necessarily none present) and are not ranked");
    const pda = rankWords(sc.pDA!)!; // 57 quantified: no small-sample clause, the zeros still said
    expect(pda).toContain("57 quantified");
    expect(pda).not.toContain("small sample");
    expect(pda).toContain("were reported 0 (not quantified, not absent) and are not ranked");
    expect(smallWords(29)).toBe(" (a small sample: each of the 29 is 3 points of the rank)");
    expect(smallWords(30)).toBe("");
  });

  // #66 review 1.2: 2014 is where Neritic's copy starts, not where the station's record began
  it("says the record is Neritic's copy, which starts in 2014, never that the station's record began then", () => {
    const now = new Date("2026-10-06T20:00:00Z");
    const top = recordContext(synthetic(4, 5, 100000), "pn", "2014-01-01")!;
    top.lastAsHigh = null; // the June sample aside: the highest in the copy
    const words = historySentence(top, "Santa Cruz Wharf", now);
    expect(words).toContain("The highest in Neritic's copy of its record, which starts in 2015.");
    const short = recordContext(synthetic(2, 10, 250), "pn", "2014-01-01")!;
    expect(historySentence(short, "Bodega Marine Lab buoy", now)).toContain("Neritic's copy of its record, which starts in 2015)");
    for (const w of [words, historySentence(short, "x", now)]) expect(w).not.toMatch(/record began|its record starts|since the record/);
    // no "highest" claim when every earlier sample was a reported zero (a zero may hide any value up to the limit)
    const allZero = recordContext(st([["2020-10-01", 0, null], ["2021-10-01", 0, null], ["2026-10-01", 40, null]]), "pn")!;
    expect(historySentence(allZero, "x", now)).not.toMatch(/highest/);
  });

  it("is computed for the published stations (Neritic's copy of Santa Cruz Wharf starts in 2014; the station's own series in 2011)", () => {
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
    expect(hab.history).toMatch(/^Compared with earlier years at Santa Cruz Wharf: higher than \d+% of the \d+ quantified Aug–Oct samples there in 2014–2025( \(a small sample: each of the \d+ is \d+ points of the rank\))?; another \d+ had no cells counted \(not necessarily none present\) and are not ranked\. /);
    expect(hab.text).not.toContain(hab.history!);
    const node = g.nodes.find((n) => n.id === `record:pn:HABs-SantaCruzWharf`)!;
    expect(node.class).toBe("record_statistic");
    expect(hab.supports).toContain(node.id);
    expect(node.caveats.join(" ")).toMatch(/not a forecast/);
    for (const w of [hab.history!, node.text]) expect(w).not.toMatch(/\b(safe|unsafe|unusual|dangerous|risk|alarming)\b/i);
  });
});

const MIN_SEASON_YEARS_IN_FIXTURE = 3;
