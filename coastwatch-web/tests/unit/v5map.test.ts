import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import type { ValueGrid } from "@/generated/schema";
import { contextIndex, contextWords } from "@/lib/gridTilesContext";
import { OUTSIDE_TINT, renderGridTile } from "@/lib/gridTiles";
import { checkShowsSkill, checkWords, nearshoreCheck, parseDerived, skillRows } from "@/lib/derived";
import { choiceParams, estimateFits, parseChoice } from "@/v4/explore/layers";

const g = (w: number, h: number): ValueGrid => ({ width: w, height: h, lat_first: 36.0, lat_step: 0.03, lon_first: -123.0, lon_step: 0.03, scale_factor: 0.01, add_offset: 0, nodata: 65535, max_quantization_error: 0.005, url: "x" });

describe("C-HARM nearshore context (lib/gridTilesContext)", () => {
  it("points each cell without a value to the nearest valued cell within 6 km, and only those", () => {
    const grid = g(8, 1);
    const N = 65535;
    // one valued cell at column 0; cells 1 and 2 (2.7 and 5.4 km away) get it, column 3 (8.1 km) does not
    const codes = new Uint16Array([50, N, N, N, N, N, N, 70]);
    const idx = contextIndex(grid, codes);
    expect(Array.from(idx)).toEqual([-1, 0, 0, -1, -1, 7, 7, -1]);
  });

  it("tints water without a value in one neutral colour, never with a probability colour", () => {
    const grid = { ...g(2, 1), lat_first: 2, lon_first: 2, lat_step: 1, lon_step: 1 };
    const codes = new Uint16Array([10, 65535]);
    const lut = new Uint32Array(65536);
    lut[10] = 0xff0000ff; // opaque red
    const px = new Uint32Array(renderGridTile(grid, codes, lut, 6, 32, 31, 512, { tint: OUTSIDE_TINT }).buffer);
    const seen = new Set<number>();
    for (const v of px) if (v) seen.add(v);
    expect([...seen].sort()).toEqual([lut[10], OUTSIDE_TINT].sort());
  });

  it("gives the nearest NOAA value as text only, with the true distance", () => {
    expect(contextWords({ value: 0.234, km: 3.42 })).toBe("NOAA's model does not reach this water. The nearest NOAA cell, 3.4 km away, reads 23%: text only, not a forecast for this water.");
  });
});

describe("derived index (lib/derived)", () => {
  const fixture = JSON.parse(readFileSync(path.resolve(__dirname, "../fixture-data/v1/derived/index.json"), "utf8"));

  it("reads the fixture's outlook, estimate and nearshore check", () => {
    const d = parseDerived(fixture)!;
    expect(d.layers.map((l) => l.kind)).toEqual(["chl_outlook", "chl_estimate", "sst"]);
    const check = nearshoreCheck(d)!;
    expect(checkShowsSkill(check)).toBe(false);
    expect(checkWords(check)).toMatch(/^Context only/);
    expect(checkWords({ ...check, readout: undefined })).toMatch(/^Checked against 1,058 CalHABMAP pier samples/);
  });

  it("offers nothing for a missing, foreign or malformed index, and never an outlook without a baseline", () => {
    expect(parseDerived(null)).toBeNull();
    expect(parseDerived({ version: 2, layers: [] })).toBeNull();
    const noBaseline = structuredClone(fixture);
    delete noBaseline.layers[0].skill.climatology;
    delete noBaseline.layers[0].skill.persistence;
    expect(parseDerived(noBaseline)!.layers.map((l) => l.kind)).toEqual(["chl_estimate", "sst"]);
    const badGrid = structuredClone(fixture);
    badGrid.layers[1].grid.encoding = "png";
    expect(parseDerived(badGrid)!.layers.map((l) => l.kind)).toEqual(["chl_outlook", "sst"]);
  });

  it("reads sea temperature as an observation, or as an analysis when the source is MUR", async () => {
    const { sstIsAnalysis } = await import("@/lib/derived");
    const t = parseDerived(fixture)!.layers[2];
    if (t.kind !== "sst") throw new Error("sst third");
    expect(sstIsAnalysis(t)).toBe(false);
    expect(sstIsAnalysis({ ...t, source: "NASA JPL MUR SST v4.1" })).toBe(true);
  });

  it("lists the outlook's skill with every baseline the index reports", () => {
    const o = parseDerived(fixture)!.layers[0];
    if (o.kind !== "chl_outlook") throw new Error("outlook first");
    expect(skillRows(o.skill).map((r) => r.label)).toEqual(["This outlook", "Climatology + AR(1)", "Climatology", "Persistence (last view)"]);
  });
});

describe("layer model (v4/explore/layers)", () => {
  it("round-trips the outlook and a satellite view with the estimate turned off", () => {
    expect(parseChoice(new URLSearchParams("layer=outlook"))).toEqual({ kind: "outlook" });
    const sat = parseChoice(new URLSearchParams("layer=olci300&est=0"))!;
    expect(sat).toMatchObject({ kind: "satellite", estimate: false });
    expect(choiceParams(sat)).toEqual({ layer: "olci300", est: "0" });
    expect(choiceParams({ kind: "satellite", product: "olci300", day: null })).toEqual({ layer: "olci300" });
  });

  it("puts the estimate only under the latest chlorophyll view of its own product", () => {
    const est = { observed_layer_id: "olci300_chl_latest" } as Parameters<typeof estimateFits>[0];
    const olci = { layer_id: "olci300_chl_latest" } as Parameters<typeof estimateFits>[2];
    const viirs = { layer_id: "viirs750_chl_latest" } as Parameters<typeof estimateFits>[2];
    expect(estimateFits(est, { kind: "satellite", product: "olci300", day: null }, olci)).toBe(true);
    expect(estimateFits(est, { kind: "satellite", product: "viirs750", day: null }, viirs)).toBe(false);
    expect(estimateFits(est, { kind: "satellite", product: "olci300", day: "2026-10-06" }, olci)).toBe(false);
    expect(estimateFits(est, { kind: "satellite", product: "olci300", day: null, view: "age" }, olci)).toBe(false);
  });
});
