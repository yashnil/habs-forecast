import { readFileSync } from "node:fs";
import path from "node:path";
import { gunzipSync } from "node:zlib";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { GET as gridGET } from "@/app/(neritic)/explore/model/chlorophyll/grid/route";
import { GET as indexGET } from "@/app/(neritic)/explore/model/chlorophyll/route";
import { chlEstimateIndex, decodeDineof, DINEOF, dineofFieldUrl, packChl, setChlSource, type ChlEstimateSource } from "@/lib/ocean/chlorophyll";
import { chlEstimateLabel, MASK_LAND, MASK_NEAR_SHORE, type ChlEstimateIndex } from "@/lib/ocean/contract";
import { PACK_CHL_LOG10, unpackCode } from "@/lib/ocean/pack";

const FX = path.resolve(__dirname, "../../ocean-fixtures");
const body = () => new Uint8Array(gunzipSync(readFileSync(path.join(FX, "dineof-20261102.dods.gz"))));
const codesOf = (b: Uint8Array) => { const raw = gunzipSync(b); return new Uint16Array(raw.buffer, raw.byteOffset, raw.length / 2); };

const env = { ...process.env };
beforeEach(() => {
  setChlSource(DINEOF);
  process.env.CW_OCEAN_FIXTURE = FX;
  delete process.env.CW_OCEAN_FIXED_NOW;
  vi.stubGlobal("fetch", vi.fn(async () => { throw new Error("network used in fixture mode"); }));
});
afterEach(() => {
  process.env = { ...env };
  vi.unstubAllGlobals();
});

describe("NOAA DINEOF gap-filled chlorophyll (recorded 2026-11-02 California subset)", () => {
  const f = decodeDineof(body());

  it("reads the grid geometry from the ERDDAP axes (2.08 km, latitude descending)", () => {
    expect(f.geometry.width).toBe(409);
    expect(f.geometry.height).toBe(466);
    expect(f.geometry.lat_first).toBeCloseTo(42.0937, 3);
    expect(f.geometry.lat_step).toBeCloseTo(-1 / 48, 6);
    expect(f.geometry.lon_step).toBeCloseTo(1 / 48, 6);
  });

  it("keeps the recorded value at 36.70°N 122.10°W (0.330 mg/m³) through packing", () => {
    const g = f.geometry;
    const row = Math.round((36.7 - g.lat_first) / g.lat_step), col = Math.round((-122.1 - g.lon_first) / g.lon_step);
    expect(f.values[row * g.width + col]).toBeCloseTo(0.32995, 4);
    const p = packChl(f, 2);
    expect(unpackCode(codesOf(p.chl)[row * g.width + col], PACK_CHL_LOG10)).toBeCloseTo(0.32995, 3);
  });

  it("leaves land as nodata and flags cells touching land as near shore", () => {
    const p = packChl(f, 2);
    const chl = codesOf(p.chl), mask = codesOf(p.mask);
    let near = 0;
    for (let k = 0; k < mask.length; k++) {
      if (mask[k] === MASK_LAND) expect(chl[k]).toBe(65535);
      if (mask[k] === MASK_NEAR_SHORE) near++;
    }
    expect(near).toBeGreaterThan(400); // the California coastline at about 2 km
  });
});

describe("chlorophyll.estimate index and routes", () => {
  it("carries the real date, its age and a stale flag (fixture recorded 2026-11-14; newest day 11-02)", async () => {
    const r = await indexGET();
    const idx = (await r.json()) as ChlEstimateIndex;
    expect(idx.status).toBe("stale");
    expect(idx.date).toBe("2026-11-02");
    expect(idx.age_days).toBe(12);
    expect(idx.reason).toMatch(/12 days old/);
    expect(idx.per_pixel_uncertainty).toBe(false);
    expect(idx.filled_flag).toBe(false);
    expect(idx.label).toBe("Gap-filled estimate (NOAA DINEOF) · Nov 2 · reconstructed where satellites saw nothing, not observed · no per-pixel uncertainty");
    expect(idx.grids?.chl.transform).toBe("log10");
    expect(idx.grids?.chl.url).toBe("/explore/model/chlorophyll/grid?date=2026-11-02&var=chl");
    expect(idx.caveats.join(" ")).toMatch(/3× low/);
  });

  it("serves the day's grid and refuses dates outside the window", async () => {
    const ok = await gridGET(new Request("http://x/explore/model/chlorophyll/grid?date=2026-11-02&var=mask"));
    expect(ok.status).toBe(200);
    expect(codesOf(new Uint8Array(await ok.arrayBuffer())).length).toBe(409 * 466);
    expect((await gridGET(new Request("http://x/explore/model/chlorophyll/grid?date=2026-11-20&var=chl"))).status).toBe(400);
    expect((await gridGET(new Request("http://x/explore/model/chlorophyll/grid?date=2026-11-01&var=chl"))).status).toBe(503); // not recorded
  });

  it("puts the source behind an interface (a fresher source with uncertainty is current, and says so in its label)", async () => {
    const fresh: ChlEstimateSource = {
      ...DINEOF,
      id: "test-fresh",
      short: "Test L4",
      perPixelUncertainty: true,
      latestDate: async () => "2026-11-13",
      field: async () => ({ geometry: { width: 2, height: 1, lat_first: 36, lat_step: 0.04, lon_first: -123, lon_step: 0.04 }, values: [0.5, -999], fill: -999 }),
    };
    setChlSource(fresh);
    const idx = await chlEstimateIndex();
    expect(idx.status).toBe("current");
    expect(idx.label).toBe(chlEstimateLabel("2026-11-13", "Test L4", true));
    expect(idx.label).not.toMatch(/no per-pixel uncertainty/);
  });

  it("answers unavailable when the source cannot be read and nothing is cached", async () => {
    setChlSource({ ...DINEOF, id: "down", latestDate: async () => { throw new Error("HTTP 502"); } });
    const idx = await chlEstimateIndex();
    expect(idx.status).toBe("unavailable");
    expect(idx.grids).toBeNull();
    expect(idx.reason).toMatch(/HTTP 502/);
  });

  it("asks ERDDAP for one day's California box only", () => {
    expect(dineofFieldUrl("2026-11-02")).toBe(
      "https://coastwatch.noaa.gov/erddap/griddap/noaacwNPPN20S3ASCIDINEOF2kmDaily.dods?chlor_a%5B(2026-11-02T12:00:00Z)%5D%5B(0.0)%5D%5B(42.1):1:(32.4)%5D%5B(-125.5):1:(-117.0)%5D",
    );
  });
});
