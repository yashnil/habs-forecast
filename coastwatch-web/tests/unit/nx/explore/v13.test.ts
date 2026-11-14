import { describe, expect, it } from "vitest";
import type { Manifest } from "@/generated/schema";
import type { StationLite } from "@/lib/data";
import { parseDerived } from "@/lib/derived";
import { FlowSim, HIST, HIST_MS, SEG, TRAIL_MIN_PX, trailCap } from "@/lib/flowSim";
import { FLOW_TIME_SCALE, TRAIL_SAMPLE_S } from "@/nx/chart/layers/FlowLayer";
import type { CurrentField } from "@/lib/currents";
import type { ValueGrid } from "@/generated/schema";
import manifest from "../../../fixture-data/v1/manifest.json";
import derivedRaw from "../../../fixture-derived/v1/derived/index.json";
import type { FieldGrid } from "@/nx/chart/layers/grid";
import { FIELD_FS, fieldQuadBounds } from "@/nx/chart/layers/FieldLayer";
import { CHART_FRAME } from "@/nx/chart/style";
import { buildExploreModel, shortSensor, SST_FRESHNESS, type ExploreFrame, type ExploreStation } from "@/nx/explore/model";
import { cellGeometry, cellSizeWords, coastHref, gapPosition, NEARBY_KM, stationFor } from "@/nx/explore/read";
import { qualityHours } from "@/nx/explore/timeline";
import { coversPort, exploreBounds, withinFor } from "@/nx/explore/camera";
import { coverageNote, coverageWords, currentsCaption, currentsStill, currentsToggleWords, legendCaption, noValueWords, OUTSIDE_KEY, readingStamp, unitOf, valueWords } from "@/nx/explore/words";

/**
 * Explore's v13 logic fixes (audit-C E-1 … E-13 and lead decision 7), each a rule the v12 baseline broke.
 */
const build = (now: string, over: { manifest?: Manifest; derived?: unknown; stations?: StationLite[] } = {}) =>
  buildExploreModel({ manifest: over.manifest ?? (manifest as unknown as Manifest), derived: parseDerived(over.derived ?? derivedRaw), stations: over.stations ?? [], baseUrl: "/data/fixture/v1", now: new Date(now) });

describe("E-1: a reading's figure and unit", () => {
  it("a C-HARM probability is a whole percent with one percent sign, never '%%'", () => {
    expect(valueWords("charm", 0.724)).toBe("72");
    expect(`${valueWords("charm", 0.724)} ${unitOf("charm")}`).toMatch(/^\d{1,3} %$/);
    expect(valueWords("charm", 1)).toBe("100");
    for (const v of [0, 0.004, 0.5, 0.999]) expect(`${valueWords("charm", v)}${unitOf("charm")}`).not.toMatch(/%.*%/);
  });
  it("chlorophyll to two figures, the outlook to one (T15), the temperature in °F to a tenth", () => {
    expect(valueWords("chl", 0.2136)).toBe("0.21");
    expect(valueWords("outlook", 1.6)).toBe("2");
    expect(valueWords("sst", 11.7)).toBe("53.1");
    expect([unitOf("chl"), unitOf("outlook"), unitOf("sst")]).toEqual(["mg/m³", "mg/m³", "°F"]);
  });
});

describe("E-2: outside a model's footprint is marked, never blank water", () => {
  const grid = { width: 4, height: 4, lat_first: 37, lat_step: -0.03, lon_first: -122.5, lon_step: 0.03 };
  it("a model, an outlook and an observed layer are drawn to the chart's edge; isotherms stop at their grid", () => {
    for (const k of ["classes", "outlook", "ramp"] as const) expect(fieldQuadBounds(k, grid)).toEqual(CHART_FRAME.bounds);
    const [w, , e] = fieldQuadBounds("isolines", grid);
    expect(w).toBeCloseTo(-122.515, 6);
    expect(e).toBeCloseTo(-122.395, 6);
  });
  it("the shader draws the 'outside the model' rules where a model or outlook cell has no value, and discards nothing there", () => {
    const branch = (n: number) => FIELD_FS.slice(FIELD_FS.indexOf(`if (u_variant == ${n}) {`), FIELD_FS.indexOf(n === 1 ? "if (u_variant == 2) {" : "// isolines"));
    for (const n of [1, 2]) {
      expect(branch(n)).toMatch(/if \(nodata\) \{ o = outsideModel\(\); return; \}/);
      expect(branch(n)).not.toMatch(/discard/);
    }
    expect(FIELD_FS).toMatch(/if \(!within && u_variant == 3\) discard;/);
  });
  it("the legend keys it in words that say it is not a value", () => {
    expect(OUTSIDE_KEY).toMatch(/outside the model/);
    expect(OUTSIDE_KEY).toMatch(/no value/);
  });
});

describe("E-3: why a cell has no value comes from where the gap is", () => {
  // 1 x 7 row, west to east; N = no value
  const N = 65535;
  const g: FieldGrid = { width: 7, height: 1, lat_first: 36.6, lat_step: -0.03, lon_first: -122.6, lon_step: 0.03, scale_factor: 0.001, add_offset: 0, nodata: N, transform: "none" };
  const row = Uint16Array.from([N, 500, 500, N, 500, N, N]);
  it("values only to the west: between the layer and the shore; only to the east: beyond its seaward edge; both: a hole", () => {
    expect(gapPosition(g, row, 0, 6)).toBe("shoreward");
    expect(gapPosition(g, row, 0, 5)).toBe("shoreward");
    expect(gapPosition(g, row, 0, 0)).toBe("seaward");
    expect(gapPosition(g, row, 0, 3)).toBe("inside");
    expect(gapPosition(g, Uint16Array.from([N, N, N, N, N, N, N]), 0, 3)).toBe("row-empty");
  });
  it("reads east and west from the grid's own direction", () => {
    const flipped: FieldGrid = { ...g, lon_first: -122.42, lon_step: -0.03 };
    // column 0 is now the easternmost cell: values lie only to its west
    expect(gapPosition(flipped, Uint16Array.from([N, 500, 500, 500, 500, 500, 500]), 0, 0)).toBe("shoreward");
  });
  it("an offshore gap in the outlook is never called the nearshore band; only a shoreward C-HARM gap is", () => {
    const outlook = { id: "outlook", kind: "outlook" } as const, charm = { id: "charm", kind: "model" } as const, sst = { id: "sst", kind: "analysis" } as const;
    expect(noValueWords(outlook, "seaward")).toBe("Beyond the seaward edge of the outlook at this latitude. No value is not a low value.");
    expect(noValueWords(outlook, "inside")).toBe("A gap inside the area the outlook covers. No value is not a low value.");
    expect(noValueWords(charm, "shoreward")).toMatch(/^Between C-HARM's cells and the shore: the model leaves out the nearshore band and the bays\./);
    expect(noValueWords(sst, "shoreward")).toMatch(/^Between the analysis's cells and the shore: the analysis has no cell this close to land\./);
    expect(noValueWords(sst, "outside-grid")).toMatch(/^Outside the area the analysis covers\./);
    for (const pos of ["seaward", "inside", "row-empty", "outside-grid"] as const) for (const p of [outlook, sst]) expect(noValueWords(p, pos)).not.toMatch(/nearshore|model's area/);
  });
});

describe("E-4: the temperature is aged by a freshness limit, never 'current' whatever its date", () => {
  it("the fixture's Oct 12 analysis is current on Oct 12, old five days on, historical past a week", () => {
    const sst = (now: string) => build(now).products.find((p) => p.id === "sst")!;
    expect(sst("2026-10-13T06:00:00Z").frames[0].fresh).toBe("current");
    expect(sst("2026-10-17T18:00:00Z").frames[0].fresh).toBe("stale");
    expect(sst("2026-10-25T18:00:00Z").frames[0].fresh).toBe("historical");
    const old = sst("2026-10-17T18:00:00Z");
    expect(legendCaption(old, old.frames[0])).toMatch(/· Oct 12 \(old\)\./);
    const fresh = sst("2026-10-13T06:00:00Z");
    expect(legendCaption(fresh, fresh.frames[0])).not.toMatch(/\(old\)/);
  });
  it("a policy the layer publishes wins over the default", () => {
    const raw = JSON.parse(JSON.stringify(derivedRaw)) as { layers: { kind: string; freshness?: unknown }[] };
    raw.layers.find((l) => l.kind === "sst")!.freshness = { basis: "observed_date", current_max_age_days: 0, stale_max_age_days: 1, note: "" };
    expect(build("2026-10-13T18:00:00Z", { derived: raw }).products.find((p) => p.id === "sst")!.frames[0].fresh).toBe("stale");
    expect(SST_FRESHNESS.current_max_age_days).toBeLessThan(SST_FRESHNESS.stale_max_age_days);
  });
  it("the legend names the product, not its expansion and route", () => {
    expect(shortSensor("JPL MUR SST v4.1 (Multi-scale Ultra-high Resolution analysis, GHRSST L4), via NOAA CoastWatch West Coast ERDDAP jplMURSST41")).toBe("JPL MUR SST v4.1");
    expect(shortSensor("JPL MUR SST v4.1 analysis (fixture)")).toBe("JPL MUR SST v4.1 analysis");
  });
});

describe("E-5: what a readout's stamp says the date is", () => {
  const f = (o: Partial<ExploreFrame>) => ({ kind: "measured", start: "2026-11-05", end: "2026-11-12", issued: null, sensor: "S", ...o }) as ExploreFrame;
  it("an outlook is dated by its issue and names its window, never its last day", () => {
    const s = readingStamp(f({ kind: "outlook", issued: "2026-11-09", sensor: "Neritic outlook" }), null);
    expect(s).toEqual({ date: "2026-11-09", dateRole: "issued", source: "Neritic outlook, mean of Nov 5–12" });
  });
  it("a model lead is its valid day with its run; a latest view's pixel its own day", () => {
    expect(readingStamp(f({ kind: "model", start: "2026-11-05", end: "2026-11-05", issued: "2026-11-03", sensor: "C-HARM v3.1" }), null)).toEqual({ date: "2026-11-05", dateRole: "valid", source: "C-HARM v3.1, run of Nov 3" });
    expect(readingStamp(f({ start: "2026-11-02", end: "2026-11-07" }), "2026-11-06")).toMatchObject({ date: "2026-11-06", dateRole: "measured" });
    expect(readingStamp(f({ kind: "analysis", start: "2026-11-07", end: "2026-11-07" }), null)).toMatchObject({ date: "2026-11-07", dateRole: undefined });
  });
  it("the fixture's outlook and C-HARM frames carry their issue dates", () => {
    const m = build("2026-10-13T06:00:00Z");
    expect(m.products.find((p) => p.id === "outlook")!.frames[0].issued).toBe("2026-10-12");
    // round 2 (C): C-HARM's issue date is inferred in the fixture (issued_date_derived): no frame claims one
    expect(m.products.find((p) => p.id === "charm")!.frames.every((x) => x.issued === null)).toBe(true);
  });
});

describe("E-6: a 'nearby' record is within 25 km and measures the variable (Research's own rule)", () => {
  const st = (id: string, lat: number, lon: number, vars: string[]): ExploreStation => ({ id, name: id, lat, lon, last: null, latest: vars.map((variable) => ({ variable, value: 1, date: "2026-10-01", n_last_365d: 3 })) });
  const stations = [st("SantaCruz", 36.958, -122.017, ["pDA", "pn_seriata"]), st("Monterey", 36.605, -121.889, ["pDA"]), st("MossShore", 36.802, -121.79, [])];
  it("links the nearest station that measures it, not a nearer one that does not", () => {
    // mid-bay: MossShore is nearest but has never published pDA; Santa Cruz has, about 18 km away
    const c = stationFor(stations, 36.8, -121.95, "pDA");
    expect(c.link?.id).toBe("SantaCruz");
    expect(c.link!.km).toBeLessThan(NEARBY_KM);
    expect(c.far).toBeNull();
    expect(c.unknown).toBeNull();
  });
  it("beyond 25 km nothing is called nearby: the nearest that measures it is named as far", () => {
    const c = stationFor(stations, 36.6, -122.4, "pDA");
    expect(c.link).toBeNull();
    expect(c.far?.id).toBe("Monterey");
    expect(c.far!.km).toBeGreaterThan(NEARBY_KM);
  });
  it("a variable the list knows but no station has: nothing is offered", () => {
    expect(stationFor(stations, 36.8, -121.95, "pn_delicatissima")).toEqual({ link: null, far: null, unknown: null });
    expect(stationFor([], 36.8, -121.95, "pDA")).toEqual({ link: null, far: null, unknown: null });
  });
  it("a variable the station list does not carry (chlorophyll, temperature) is never claimed: the nearest station is only a place to look", () => {
    for (const v of ["chl_extracted", "temp"] as const) {
      const c = stationFor(stations, 36.8, -121.95, v);
      expect(c.link).toBeNull();
      expect(c.far).toBeNull();
      expect(c.unknown?.id).toBe("MossShore");
    }
  });
});

describe("E-7: the hand-off to My Coast", () => {
  it("carries the port, the read cell's centre and the reading's date", () => {
    expect(coastHref({ place: "port-592", lat: 36.80012, lon: -121.95191, date: "2026-11-06" })).toBe("/coast?place=port-592&at=36.8001,-121.9519&date=2026-11-06");
    expect(coastHref({ place: null, lat: 36.8, lon: -121.95, date: "2026-11-06" })).toBe("/coast?at=36.8000,-121.9500&date=2026-11-06");
  });
});

describe("E-8: the cell that was read", () => {
  const g: FieldGrid = { width: 10, height: 10, lat_first: 37, lat_step: -0.03, lon_first: -122.6, lon_step: 0.03, scale_factor: 1, add_offset: 0, nodata: 65535, transform: "none" };
  it("has its centre, its size on the ground and a closed outline half a step around the centre", () => {
    const c = cellGeometry(g, 2, 3);
    expect(c.lat).toBeCloseTo(36.94, 6);
    expect(c.lon).toBeCloseTo(-122.51, 6);
    expect(c.heightKm).toBeCloseTo(3.34, 2);
    expect(c.widthKm).toBeCloseTo(3.34 * Math.cos((36.94 * Math.PI) / 180), 2);
    expect(c.outline).toHaveLength(5);
    expect(c.outline[0]).toEqual(c.outline[4]);
    const lons = c.outline.map((p) => p[0]), lats = c.outline.map((p) => p[1]);
    expect(Math.min(...lons)).toBeCloseTo(-122.525, 6);
    expect(Math.max(...lons)).toBeCloseTo(-122.495, 6);
    expect(Math.min(...lats)).toBeCloseTo(36.925, 6);
    expect(Math.max(...lats)).toBeCloseTo(36.955, 6);
  });
  it("says its size in the fewest honest figures", () => {
    expect(cellSizeWords({ widthKm: 0.27, heightKm: 0.3 })).toBe("300 m");
    expect(cellSizeWords({ widthKm: 0.9, heightKm: 1.11 })).toBe("1 km");
    expect(cellSizeWords({ widthKm: 2.7, heightKm: 3.34 })).toBe("3 km");
    expect(cellSizeWords({ widthKm: 1.2, heightKm: 1.5 })).toBe("1.5 km");
  });
});

describe("E-12: currents say what is drawn", () => {
  const hour = { id: "h", label: "5:00 AM PT", stamp: "Oct 8, 5:00 AM PT", date: "2026-10-08", time: "2026-10-08T12:00:00Z", mean: false, volume: 0.12, fresh: "current" as const, u: null as never, v: null as never };
  it("a failed or empty hour is never 'on, measured'", () => {
    expect(currentsToggleWords("failed", hour)).toBe("unavailable: this hour could not be read");
    expect(currentsToggleWords("empty", hour)).toBe("no measurement in this hour");
    expect(currentsToggleWords("loading", hour)).not.toMatch(/measured/);
    expect(currentsToggleWords("on", hour)).toBe("on, measured");
    expect(currentsToggleWords("on", { ...hour, fresh: "historical" })).toBe("on, measured (old)");
    expect(currentsToggleWords("off", null)).toBe("off");
  });
  it("the caption states how much of the coastal ocean the radar measured", () => {
    expect(coverageWords(hour)).toBe("The radar measured 12 % of the coastal ocean this hour.");
    expect(coverageWords({ volume: 0.004, mean: false })).toBe("The radar measured under 1 % of the coastal ocean this hour.");
    expect(coverageWords({ volume: null, mean: false })).toBeNull();
    expect(currentsCaption(hour)).toMatch(/The radar measured 12 % of the coastal ocean this hour\. Relative speed/);
  });
  it("the 24 h mean is drawn still and says why (lead decision 7)", () => {
    const mean = { ...hour, mean: true, label: "24 h mean" };
    expect(currentsStill(mean)).toBe(true);
    expect(currentsStill(hour)).toBe(false);
    expect(currentsStill({ ...hour, fresh: "stale" })).toBe(true);
    expect(currentsCaption(mean)).toMatch(/A mean of 24 hours is not one moment of the water, so it is drawn still\./);
    expect(currentsToggleWords("on", mean)).toBe("on, measured, 24 h mean");
  });
  it("the steadiness mask reads at most eight hours, the ones around the hour on show", () => {
    const hours = Array.from({ length: 24 }, (_, i) => ({ id: `h${i}`, time: `2026-10-08T${String(i).padStart(2, "0")}:00:00Z`, mean: false }));
    const all = [...hours, { id: "mean", time: "2026-10-08T23:00:00Z", mean: true }];
    expect(qualityHours(all, "h12").map((h) => h.id)).toEqual(["h8", "h9", "h10", "h11", "h12", "h13", "h14", "h15"]);
    expect(qualityHours(all, "h23").map((h) => h.id)).toEqual(["h16", "h17", "h18", "h19", "h20", "h21", "h22", "h23"]);
    expect(qualityHours(all, "h0")).toHaveLength(8);
    expect(qualityHours(all.slice(0, 3), "h1").map((h) => h.id)).toEqual(["h0", "h1", "h2"]);
    expect(qualityHours(all, null).some((h) => h.mean)).toBe(false);
  });
});

describe("E-13: a layer missing from a run stays in the switch, with the reason", () => {
  it("no C-HARM in the manifest, no outlook in the derived index: both are listed as missing, in the layer order", () => {
    const m = JSON.parse(JSON.stringify(manifest)) as { layers: { group_id: string; layer_id: string }[] };
    m.layers = m.layers.filter((l) => !/charm/i.test(`${l.group_id} ${l.layer_id}`));
    const d = JSON.parse(JSON.stringify(derivedRaw)) as { layers: { kind: string }[] };
    d.layers = d.layers.filter((l) => l.kind !== "chl_outlook");
    const model = build("2026-10-13T06:00:00Z", { manifest: m as unknown as Manifest, derived: d });
    expect(model.products.map((p) => p.id)).toEqual(["chl", "sst"]);
    expect(model.missing).toEqual([
      { id: "charm", label: "C-HARM", why: "Not in this data run" },
      { id: "outlook", label: "Outlook", why: "Not in this data run" },
    ]);
  });
  it("a full run has none missing", () => {
    expect(build("2026-10-13T06:00:00Z").missing).toEqual([]);
  });
});

describe("lead decision 7: a streak is about one cell long", () => {
  const g2 = { url: "", lat_first: 36, lat_step: 0.018, lon_first: -123, lon_step: 0.0208, width: 40, height: 40, scale_factor: 1, add_offset: 0 } as unknown as ValueGrid;
  const field = (): CurrentField => {
    const u = new Float32Array(1600), v = new Float32Array(1600);
    for (let r = 0; r < 40; r++) for (let c = 0; c < 40; c++) [u[r * 40 + c], v[r * 40 + c]] = [0, c < 20 ? 0.05 : 0.5];
    return { g: g2, u, v };
  };
  const rng = () => {
    let seed = 7;
    return () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  };
  const drawnLengths = (sim: FlowSim) => {
    const out = new Float32Array(300 * HIST * SEG);
    const n = sim.build(out, { colour: false, alpha: 1 });
    // segments of one particle are contiguous: a new trail starts where a segment does not begin at the last one's end
    const lens: number[] = [];
    let len = 0;
    for (let i = 0; i < n; i++) {
      const q = i * SEG;
      const joined = i > 0 && out[q] === out[q - SEG + 2] && out[q + 1] === out[q - SEG + 3];
      if (!joined && i > 0) {
        lens.push(len);
        len = 0;
      }
      len += Math.hypot(out[q + 2] - out[q], out[q + 3] - out[q + 1]);
    }
    if (n) lens.push(len);
    return lens;
  };

  /** the still frame as the chart's streak layer settles it: trails as long as its moving ones (FlowLayer `settle`) */
  const settleAsChart = (sim: FlowSim, worldPx: number) => sim.settle((worldPx * HIST_MS) / 1000 / (TRAIL_SAMPLE_S * FLOW_TIME_SCALE));

  it("the cap is one cell, and never under 10 px on screen", () => {
    const cell = 0.018 / 360;
    expect(trailCap(cell, 512 * 2 ** 11)).toBe(cell);
    expect(trailCap(cell, 512 * 2 ** 6)).toBeCloseTo(TRAIL_MIN_PX / (512 * 2 ** 6), 12);
    expect(trailCap(cell, 512 * 2 ** 6)).toBeGreaterThan(cell);
  });

  it("uncapped (the legacy map) a fast trail runs across several cells; capped, no drawn trail is longer than a cell", () => {
    const worldPx = 512 * 2 ** 10;
    const sim = new FlowSim(300, rng());
    sim.setField(field());
    sim.setCount(300);
    settleAsChart(sim, worldPx);
    const cell = sim.cellSize;
    expect(cell).toBeCloseTo(0.018 / 360, 12);
    expect(Math.max(...drawnLengths(sim))).toBeGreaterThan(cell * 1.5);
    sim.setTrailCap(trailCap(cell, worldPx));
    const capped = drawnLengths(sim);
    expect(capped.length).toBeGreaterThan(100);
    // float32 positions: allow a hair over
    expect(Math.max(...capped)).toBeLessThanOrEqual(cell * 1.001);
    for (const s of sim.streaks(worldPx)) expect(s.px).toBeLessThanOrEqual(cell * worldPx * 1.001);
    // a slow streak is still shorter than the cap: the cap cuts, it does not stretch
    expect(Math.min(...capped)).toBeLessThan(cell * 0.5);
  });

  it("a capped trail still ends at the particle's head and brightens toward it", () => {
    const worldPx = 512 * 2 ** 10;
    const sim = new FlowSim(40, rng());
    sim.setField(field());
    sim.setCount(40);
    settleAsChart(sim, worldPx);
    sim.setTrailCap(trailCap(sim.cellSize, worldPx));
    const out = new Float32Array(40 * HIST * SEG);
    const n = sim.build(out, { colour: false, alpha: 1 });
    expect(n).toBeGreaterThan(0);
    const heads = new Set(sim.heads().map(([lon, lat]) => `${lon.toFixed(5)},${lat.toFixed(5)}`));
    expect(heads.size).toBeGreaterThan(0);
    for (let i = 1; i < n; i++) {
      const q = i * SEG;
      const joined = out[q] === out[q - SEG + 2] && out[q + 1] === out[q - SEG + 3];
      if (joined) expect(out[q + 4]).toBeGreaterThanOrEqual(out[q - SEG + 4]);
    }
  });
});

describe("a port the layers do not reach (Bodega Bay over a Monterey Bay grid)", () => {
  const monterey: [number, number, number, number] = [-122.6, 36.4, -121.75, 37.2];
  const bodega = { lon: -123.05, lat: 38.33 }, moss = { lon: -121.79, lat: 36.8 };
  it("is not covered, so the view is not held inside the data; a port in the data, or no port, is", () => {
    expect(coversPort(monterey, bodega)).toBe(false);
    expect(coversPort(monterey, moss)).toBe(true);
    expect(withinFor(monterey, bodega)).toBeNull();
    expect(withinFor(monterey, moss)).not.toBeNull();
    expect(withinFor(monterey, null)).not.toBeNull();
    expect(withinFor(null, moss)).toBeNull();
  });
  it("the opening view frames the port itself", () => {
    const [[w, s], [e, n]] = exploreBounds(bodega, monterey);
    expect(bodega.lon).toBeGreaterThan(w);
    expect(bodega.lon).toBeLessThan(e);
    expect(bodega.lat).toBeGreaterThan(s);
    expect(bodega.lat).toBeLessThan(n);
  });
  it("the page says what is missing and where the layer is", () => {
    expect(coverageNote("Chlorophyll", "Bodega Bay", monterey)).toBe('Chlorophyll is not published for the water at Bodega Bay in this data run: its cells run from 36°24′ N to 37°12′ N. The chart is on Bodega Bay; "Fit the data" on the chart goes to where the layer is.');
  });
});
