import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { LayerArtifact, Manifest, ValueGrid } from "@/generated/schema";
import type { ObservationDataset, ObsStation } from "@/generated/observations";
import { areaCells, areaStats } from "@/lib/area";
import { cellAt } from "@/lib/grid";
import type { Sample } from "@/lib/observations";
import { SeriesChart, measuredSegments, monthTicks, sampleOnDay } from "@/v4/charts/SeriesChart";
import { AXES } from "@/lib/bloom";
import { LinkedSeries, qcWords, readingAt, realSamples } from "@/v4/charts/LinkedSeries";
import { parseDay, researchWindow } from "@/v4/research/Research";
import { transect } from "@/v4/research/transect";
import { areaCsv, cellLeadsCsv, modelHistoryCsv, stationCsv, transectCsv } from "@/v4/research/exports";
import { layerLineage } from "@/v4/research/layerLineage";
import manifestJson from "../fixture-data/v1/manifest.json";

/** Research tools: the exact transect walk, area cells, CSV provenance headers, layer lineage and
 *  per-sample QC marks. Synthetic grids unless the fixture is named. */
const manifest = manifestJson as unknown as Manifest;
const dir = path.join(__dirname, "../fixture-data/v1");
const observations = JSON.parse(readFileSync(path.join(dir, readdirSync(dir).find((f) => f.startsWith("observations-"))!), "utf8")) as ObservationDataset;
const charm = (v: string, lead: number) => manifest.layers.find((l) => l.group_id === "charm" && l.variable === v && l.time.lead_days === lead)! as LayerArtifact;

/** A 4 × 4 grid of 1° cells centred on integer lat/lon 0..3; `holes` have no value. */
function grid(holes: [number, number][] = []): { g: ValueGrid; codes: Uint16Array } {
  const g = { url: "", encoding: "uint16le+gzip", width: 4, height: 4, lat_first: 0, lat_step: 1, lon_first: 0, lon_step: 1, scale_factor: 0.001, add_offset: 0, nodata: 65535, max_quantization_error: 0.0005, transform: "none", chunks: null } as unknown as ValueGrid;
  const codes = new Uint16Array(16).map((_, i) => 100 + i);
  for (const [r, c] of holes) codes[r * 4 + c] = 65535;
  return { g, codes };
}

/** Every cell a dense walk along the line visits, in order (the reference for the exact walk). */
function dense(g: ValueGrid, a: { lat: number; lon: number }, b: { lat: number; lon: number }, n = 200_000) {
  const out: string[] = [];
  for (let i = 0; i <= n; i++) {
    const c = cellAt(g, a.lat + ((b.lat - a.lat) * i) / n, a.lon + ((b.lon - a.lon) * i) / n);
    const k = c ? `${c.row},${c.col}` : "out";
    if (out[out.length - 1] !== k) out.push(k);
  }
  return out;
}

describe("transect: every crossed cell, exactly once, in order", () => {
  it("lists a cell the line clips only at a corner", () => {
    const { g, codes } = grid();
    const a = { lat: 0, lon: 0 };
    const b = { lat: 3, lon: 2.9 };
    const cells = transect(g, codes, a, b).map((c) => `${c.row},${c.col}`);
    // row 1 is entered just before column 1 (lat crosses 0.5 at 1/6 of the line, lon at 0.172)
    expect(cells.slice(0, 3)).toEqual(["0,0", "1,0", "1,1"]);
    expect(cells).toEqual(dense(g, a, b));
    // the old 1/6-cell sampler stepped over that corner
    expect(cells).toContain("1,0");
  });

  it("a line through exact corner points crosses no zero-length cell", () => {
    const { g, codes } = grid();
    expect(transect(g, codes, { lat: 0, lon: 0 }, { lat: 3, lon: 3 }).map((c) => `${c.row},${c.col}`)).toEqual(["0,0", "1,1", "2,2", "3,3"]);
  });

  it("keeps gaps as null and marks the stretch outside the grid once", () => {
    const { g, codes } = grid([[1, 1]]);
    const t = transect(g, codes, { lat: 1, lon: -3 }, { lat: 1, lon: 3 });
    expect(t[0]).toMatchObject({ inDomain: false, value: null, km: 0 });
    expect(t.filter((c) => !c.inDomain)).toHaveLength(1);
    expect(t.slice(1).map((c) => c.col)).toEqual([0, 1, 2, 3]);
    expect(t.find((c) => c.col === 1)!.value).toBeNull();
    expect(t.find((c) => c.col === 2)!.value).toBeCloseTo(0.106, 6);
    // distances increase along the line
    expect(t.every((c, i) => i === 0 || c.km > t[i - 1].km)).toBe(true);
  });
});

describe("area cells", () => {
  it("lists every cell centre in the box, counts gaps, never fills", () => {
    const { g, codes } = grid([[1, 1]]);
    const cells = areaCells({ grid: g, codes }, [0.5, 0.5, 2.5, 2.5])!;
    expect(cells.map((c) => `${c.row},${c.col}`)).toEqual(["1,1", "1,2", "2,1", "2,2"]);
    expect(cells[0].value).toBeNull();
    expect(areaStats({ grid: g, codes }, [0.5, 0.5, 2.5, 2.5])).toMatchObject({ domain: 4, valued: 3 });
  });
});

const header = (csv: string) => csv.split("\n").filter((l) => l.startsWith("#"));
const columns = (csv: string) => csv.split("\n").find((l) => !l.startsWith("#"));

describe("exports carry their provenance", () => {
  const l = charm("pseudo_nitzschia", 0);
  const modelLines = (csv: string) => {
    const h = header(csv).join("\n");
    expect(h).toContain("Product class: model output");
    expect(h).toContain("units: probability (0-1)");
    expect(h).toContain("wvcharmV3_0day");
    expect(h).toContain(`Licence: ${l.provenance.license}`);
    expect(h).toContain(`Pipeline run: ${manifest.pipeline_run_id}`);
    expect(h).toContain("(inferred: nowcast valid day + 1 day");
    expect(h).toContain("Empty value = no published value");
    expect(h).toContain("Model probability, not a toxin measurement.");
    expect(h).toContain("Not a seafood test");
  };

  it("transect: columns, and a gap is an empty value", () => {
    const { g, codes } = grid([[1, 1]]);
    const t = transect(g, codes, { lat: 1, lon: -1 }, { lat: 1, lon: 3 });
    const csv = transectCsv({ lat: 1, lon: -1 }, { lat: 1, lon: 3 }, t, l, manifest);
    modelLines(csv);
    expect(columns(csv)).toBe("km,row,col,lat,lon,value,in_domain");
    const body = csv.trim().split("\n").filter((x) => !x.startsWith("#")).slice(1);
    expect(body[0]).toMatch(/^0,,,1,-1,,no$/);
    expect(body.find((r) => r.split(",")[2] === "1")!.split(",")[5]).toBe("");
  });

  it("area and cell leads", () => {
    const { g, codes } = grid();
    const a = areaCsv([0.5, 0.5, 1.5, 1.5], areaCells({ grid: g, codes }, [0.5, 0.5, 1.5, 1.5])!, l, manifest);
    modelLines(a);
    expect(columns(a)).toBe("row,col,lat,lon,value");
    const layers = [0, 1, 2, 3].map((k) => charm("pseudo_nitzschia", k));
    const c = cellLeadsCsv({ row: 1, col: 2, lat: 36.04, lon: -123.03 }, [
      { lead: 0, date: "2026-10-07", value: 0.5, issued: true },
      { lead: 1, date: "2026-10-08", value: null, issued: true },
    ], layers, manifest, "pseudo_nitzschia");
    modelLines(c);
    expect(header(c).join("\n")).toContain("wvcharmV3_0day, wvcharmV3_1day, wvcharmV3_2day, wvcharmV3_3day");
    expect(header(c).join("\n")).toContain("Quantisation error");
    expect(c).toContain("lead_days,kind,valid_date,value\n0,nowcast,2026-10-07,0.5\n1,forecast,2026-10-08,\n");
  });

  it("model history of a station with one", () => {
    const st = observations.stations.find((s) => s.charm?.history?.pseudo_nitzschia?.length)!;
    const csv = modelHistoryCsv(st, "pseudo_nitzschia", l, manifest, observations.method.charm);
    modelLines(csv);
    expect(columns(csv)).toBe("date,value,n_cells,partial");
    expect(csv).toContain("never compared with them as numbers");
    expect(csv.trim().split("\n").filter((x) => !x.startsWith("#")).length - 1).toBe(st.charm!.history!.pseudo_nitzschia.length);
  });

  it("station window: observation class, units in words, licence, run id", () => {
    const st = observations.stations.find((s) => s.station_id === "HABs-SantaCruzWharf")!;
    const v = observations.variables.find((x) => x.id === "temp")!;
    const csv = stationCsv(observations, st, v, -Infinity, Infinity);
    const h = header(csv).join("\n");
    expect(h).toContain("temp in °C");
    expect(h).not.toContain("degree_C");
    expect(h).toContain("Product class: observation");
    expect(h).toContain(`Pipeline run: ${observations.provenance.pipeline_run_id}`);
    expect(h).toContain("Licence: ");
    expect(h).toContain("empty value = not measured (never zero)");
    expect(columns(csv)).toBe("sample_time_utc,value,qualifier");
  });
});

describe("layer lineage: only what the manifest holds", () => {
  it("C-HARM: dataset, inferred issue date, producer's gap-filling note, every QC check, run id", () => {
    const l = charm("particulate_domoic", 2);
    const L = layerLineage(l, manifest);
    expect(L.source.find((r) => r.label === "Dataset")!.value).toBe("wvcharmV3_2day");
    expect(L.time.find((r) => r.label === "Run issued")!.value).toContain("inferred");
    expect(L.time.find((r) => r.label === "Valid")!.value).toContain("forecast, 2 days ahead");
    expect(L.upstreamNote).toContain("DINEOF gap filling");
    expect(L.checks.map((c) => c.name)).toEqual(l.qc!.checks.map((c) => c.name));
    expect(L.quantErr).toBe(l.grid!.max_quantization_error);
    expect(L.pipeline[0]).toMatchObject({ label: "Pipeline run", value: manifest.pipeline_run_id });
    expect(L.caveats).toEqual(l.caveats);
    expect(L.licence).toBe(l.provenance.license);
  });

  it("failed checks are counted, never hidden", () => {
    const l = charm("pseudo_nitzschia", 0);
    const broken = { ...l, qc: { ...l.qc!, checks: [...l.qc!.checks, { name: "plausible_values", passed: false, detail: "x" }] } } as LayerArtifact;
    const L = layerLineage(broken, manifest);
    expect(L.failed).toBe(1);
    expect(L.checks.find((c) => c.name === "plausible_values")!.passed).toBe(false);
  });

  it("no producer note → none is invented", () => {
    const l = charm("pseudo_nitzschia", 0);
    const bare = { ...l, provenance: { ...l.provenance, upstream_metadata: {} } } as LayerArtifact;
    expect(layerLineage(bare, manifest).upstreamNote).toBeNull();
  });
});

describe("station chart: per-sample QC and units", () => {
  const T = (d: string) => Date.parse(`${d}T19:00:00Z`);
  const S = (d: string, value: number | null, q: Sample["q"], index: number): Sample => ({ t: T(d), time: `${d}T19:00:00Z`, value, q, index });
  const ss = [S("2026-06-01", 0.2, null, 0), S("2026-06-08", 40, "flag_high", 1), S("2026-06-15", null, "rejected_negative", 2), S("2026-06-22", 0, "reported_zero", 3)];

  it("rings a flagged-high value, marks a rejected sample on the floor, never as a value", () => {
    const html = renderToStaticMarkup(createElement(SeriesChart, { samples: ss, variable: "pDA", units: "ng/mL", label: "pDA", from: T("2026-05-25"), to: T("2026-06-30"), axisTitle: "pDA (ng/mL, log scale)" }));
    expect(html).toContain('data-testid="series-flagged"');
    expect(html).toContain('data-testid="series-rejected"');
    expect(html).toContain("rejected (QC)");
    expect(html).toContain("flagged high by QC");
    expect(html).toContain("pDA (ng/mL, log scale)");
    // the table: a rejected sample has no value
    expect(html).toMatch(/2026-06-15<\/td><td[^>]*>—<\/td><td[^>]*>rejected \(QC\)/);
  });

  it("human units (°C, mg/m³), never the raw codes", () => {
    const t = renderToStaticMarkup(createElement(SeriesChart, { samples: [S("2026-06-01", 14.2, null, 0)], variable: "temp", units: "degree_C", label: "Water temperature", from: T("2026-05-25"), to: T("2026-06-30") }));
    expect(t).toContain("°C");
    expect(t).not.toContain("degree_C");
    const c = renderToStaticMarkup(createElement(SeriesChart, { samples: [S("2026-06-01", 2, null, 0)], variable: "chl_extracted", units: "mg/m3", label: "Chlorophyll", from: T("2026-05-25"), to: T("2026-06-30") }));
    expect(c).toContain("mg/m³");
    expect(c).not.toContain("mg/m3");
  });

  it("the shared cursor finds the sample of the same UTC day, or none", () => {
    expect(sampleOnDay(ss, Date.parse("2026-06-08T12:00:00Z"))).toBe(1);
    expect(sampleOnDay(ss, Date.parse("2026-06-09T12:00:00Z"))).toBeNull();
  });

  it("month ticks: monthly for a short window", () => {
    expect(monthTicks(Date.parse("2026-04-10T00:00:00Z"), Date.parse("2026-10-10T00:00:00Z"), 700).map((t) => new Date(t).toISOString().slice(0, 7))).toEqual(["2026-05", "2026-06", "2026-07", "2026-08", "2026-09", "2026-10"]);
  });
});

describe("no model-vs-observation arithmetic in Research", () => {
  it("the Research source carries no skill, correlation or accuracy wording", () => {
    const src = [
      ...["Research.tsx", "parts.tsx", "CoastLadder.tsx", "Stations.tsx", "Lineage.tsx", "exports.ts", "layerLineage.ts"].map((f) => path.join(__dirname, "../../src/v4/research", f)),
      ...["LinkedSeries.tsx", "BrushStrip.tsx"].map((f) => path.join(__dirname, "../../src/v4/charts", f)),
    ]
      .map((f) => readFileSync(f, "utf8"))
      .join("\n");
    expect(src).not.toMatch(/correlat|skill score|hit rate|accuracy|R²|r-squared/i);
    expect(src).not.toMatch(/\b(ROC|AUC|RMSE)\b/);
  });
});

describe("the linked instrument (v10): gaps, QC, model beside the samples", () => {
  const T = (d: string) => Date.parse(`${d}T19:00:00Z`);
  const S = (d: string, value: number | null, q: Sample["q"], index: number): Sample => ({ t: T(d), time: `${d}T19:00:00Z`, value, q, index });
  // weekly samples, then nine weeks with none, then weekly again; a bare visit; a reported zero; a rejected sample
  const ss = [S("2026-01-05", 0.1, null, 0), S("2026-01-12", 0.2, null, 1), S("2026-01-19", 0.3, null, 2), S("2026-03-23", 0.4, null, 3), S("2026-03-30", 0.5, null, 4), S("2026-04-06", null, null, 5), S("2026-04-13", 0, "reported_zero", 6), S("2026-04-20", null, "rejected_negative", 7)];
  const html = (extra: Record<string, unknown> = {}) =>
    renderToStaticMarkup(
      createElement(LinkedSeries, { samples: ss, variable: "pDA", units: "ng/mL", label: "pDA", axisTitle: "pDA, ng/mL · log scale", from: T("2026-01-01"), to: T("2026-04-30"), model: null, cursorT: null, pinnedT: null, onHover: () => undefined, onPin: () => undefined, source: "CalHABMAP", ...extra }),
    );

  it("never draws a line across missing samples: two runs, and the long gap is drawn as not measured", () => {
    const h = html();
    const lines = [...h.matchAll(/<polyline points="([^"]+)"[^>]*data-testid="series-line"/g)].map((m) => m[1].split(" ").length);
    expect(lines).toEqual([3, 2]);
    expect(h).toContain('data-testid="series-gap"');
    expect(h).toContain("not measured");
  });

  it("never joins across a reported 0: Apr 8 0.003, Apr 15 reported 0, Apr 22 0.26 are two marks and a floor mark, not one line", () => {
    const zz = [S("2026-04-01", 0.01, null, 0), S("2026-04-08", 0.003, null, 1), S("2026-04-15", 0, "reported_zero", 2), S("2026-04-22", 0.26, null, 3), S("2026-04-29", 0.3, null, 4)];
    const runs = measuredSegments(zz, AXES.pDA, T("2026-03-25"), T("2026-05-05")).map((r) => r.map((p) => p.time.slice(0, 10)));
    expect(runs).toEqual([
      ["2026-04-01", "2026-04-08"],
      ["2026-04-22", "2026-04-29"],
    ]);
    // the same for a rejected sample; a bare visit (nothing measured) does not break a short run
    const rr = [S("2026-04-08", 0.003, null, 0), S("2026-04-15", null, "rejected_negative", 1), S("2026-04-22", 0.26, null, 2)];
    expect(measuredSegments(rr, AXES.pDA, T("2026-03-25"), T("2026-05-05"))).toHaveLength(2);
    const bv = [S("2026-04-08", 0.003, null, 0), S("2026-04-15", null, null, 1), S("2026-04-22", 0.26, null, 2)];
    expect(measuredSegments(bv, AXES.pDA, T("2026-03-25"), T("2026-05-05"))).toHaveLength(1);
    // drawn: no polyline passes over Apr 15, and the zero is its own mark on the floor
    const h = renderToStaticMarkup(createElement(LinkedSeries, { samples: zz, variable: "pDA", units: "ng/mL", label: "pDA", axisTitle: "pDA", from: T("2026-03-25"), to: T("2026-05-05"), model: null, cursorT: null, pinnedT: null, onHover: () => undefined, onPin: () => undefined, source: "CalHABMAP" }));
    expect([...h.matchAll(/<polyline points="([^"]+)"[^>]*data-testid="series-line"/g)].map((m) => m[1].split(" ").length)).toEqual([2, 2]);
    expect(h.match(/data-testid="series-zero"/g)).toHaveLength(1);
  });

  it("a reported zero sits on the floor, a rejected sample is a cross, a bare visit is only a rug tick", () => {
    const h = html();
    expect(h.match(/data-testid="series-zero"/g)).toHaveLength(1);
    expect(h.match(/data-testid="series-rejected"/g)).toHaveLength(1);
    // every visit is in the rug (8), only real samples are stops for the crosshair (7)
    expect(realSamples(ss)).toHaveLength(7);
    expect(qcWords(ss[6])).toBe("reported 0: not quantified, not absent");
    expect(qcWords(ss[7])).toBe("rejected by QC, not a measurement");
    expect(qcWords(ss[0])).toBe("no QC flag");
  });

  it("the model has its own panel and axis; a day's reading never mixes the two", () => {
    const model = { points: [{ t: Date.parse("2026-03-30T12:00:00Z"), date: "2026-03-30", v: 0.7, n: 20, partial: false }], historyStart: Date.parse("2026-03-01T12:00:00Z"), what: "Domoic acid (pDA)", radiusKm: 15 };
    const h = html({ model });
    expect(h).toContain('data-testid="model-panel-svg"');
    expect(h).toContain("no model history kept before Mar 1, 2026");
    const r = readingAt(T("2026-03-30"), ss, model);
    expect(r.sample?.value).toBe(0.5);
    expect(r.model?.v).toBe(0.7);
    expect(readingAt(T("2026-01-12"), ss, model).modelWhy).toBe("before the model history Neritic keeps");
    expect(readingAt(T("2026-04-06"), ss, model).sample).toBeNull();
  });

  it("the window: a brushed link is clamped to the record; presets and the default stay", () => {
    const first = Date.parse("2014-01-08T12:00:00Z");
    const last = Date.parse("2026-09-30T12:00:00Z");
    const base = { first, last, range: null, recent: true, askedDay: null };
    const w = researchWindow({ ...base, start: parseDay("2025-03-01"), end: parseDay("2025-09-01") });
    expect(w.range).toBeNull();
    expect(new Date(w.from).toISOString().slice(0, 10)).toBe("2025-03-01");
    expect(researchWindow({ ...base, start: parseDay("2010-01-01"), end: parseDay("2015-01-01") }).from).toBe(first - 4 * 86_400_000);
    expect(researchWindow({ ...base, start: parseDay("2025-03-01"), end: parseDay("2025-03-03") }).range).toBe("1y");
    expect(researchWindow({ ...base, start: null, end: null, range: "3y" }).range).toBe("3y");
    expect(researchWindow({ ...base, start: null, end: null, recent: false }).range).toBe("all");
    expect(parseDay("2025-13-45")).toBeNull();
    expect(parseDay("nonsense")).toBeNull();
  });
});
