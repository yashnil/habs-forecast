import { readFileSync } from "node:fs";
import path from "node:path";
import { gunzipSync } from "node:zlib";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { MASK_LAND, MASK_NEAR_SHORE, MASK_WATER, wcofsLabel } from "@/lib/ocean/contract";
import { PACK_TEMPERATURE, PACK_VELOCITY, unpackCode } from "@/lib/ocean/pack";
import { buildIndex, catalogUrl, discoverCycle, GEOMETRY, packStep, parseCatalog, resetWcofsCaches, stepUrl, STEP_IDS, validTime, wcofsIndex } from "@/lib/ocean/wcofs";
import { GET as gridGET } from "@/app/(neritic)/explore/model/wcofs/grid/route";
import { GET as indexGET } from "@/app/(neritic)/explore/model/wcofs/route";

const FX = path.resolve(__dirname, "../../ocean-fixtures");
const gz = (f: string) => new Uint8Array(gunzipSync(readFileSync(path.join(FX, f))));
const CATALOG = new TextDecoder().decode(gz("wcofs-catalog-20261114.xml.gz"));
const codesOf = (b: Uint8Array) => { const raw = gunzipSync(b); return new Uint16Array(raw.buffer, raw.byteOffset, raw.length / 2); };

/** The audit's sample: 36.70°N 122.10°W is row (36.70 − 32.38) / 0.04 = 108, column (−122.10 + 125.02) / 0.04 = 73. */
const SAMPLE = 108 * GEOMETRY.width + 73;

const env = { ...process.env };
beforeEach(() => resetWcofsCaches());
afterEach(() => {
  process.env = { ...env };
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe("WCOFS contract", () => {
  it("names steps and their valid times from the cycle", () => {
    expect(STEP_IDS).toHaveLength(32);
    expect(validTime("2026111403", "n003")).toBe("2026-11-13T06:00:00.000Z");
    expect(validTime("2026111403", "n024")).toBe("2026-11-14T03:00:00.000Z");
    expect(validTime("2026111403", "f024")).toBe("2026-11-15T03:00:00.000Z");
    expect(validTime("2026111403", "f072")).toBe("2026-11-17T03:00:00.000Z");
    expect(GEOMETRY).toEqual({ width: 201, height: 244, lat_first: 32.38, lat_step: 0.04, lon_first: -125.02, lon_step: 0.04 });
  });

  it("reads the newest cycle and its regulargrid steps from a recorded THREDDS catalog", () => {
    const c = parseCatalog(CATALOG)!;
    expect(c.cycleId).toBe("2026111403");
    expect(c.steps).toEqual(STEP_IDS);
    expect(parseCatalog("<catalog/>")).toBeNull();
  });

  it("labels a step as developmental model guidance, never a measurement or a bloom prediction", () => {
    const l = wcofsLabel("2026111403", { kind: "forecast", valid_time: "2026-11-15T03:00:00.000Z" });
    expect(l).toBe("NOAA WCOFS model guidance (developmental) · cycle Nov 14 03z · valid Nov 14, 7:00 PM PST · a model forecast, not a measurement; not a bloom or toxin prediction");
    expect(wcofsLabel("2026111403", { kind: "nowcast", valid_time: "2026-11-14T03:00:00.000Z" })).toContain("a model nowcast, not a measurement");
  });
});

describe("WCOFS step packing (recorded f024, the full California box)", () => {
  const packed = packStep(gz("wcofs-20261114-f024.dods.gz"), "2026111403", "f024");

  it("matches the audit's sample at 36.70°N 122.10°W: u +0.049, v +0.094 m/s, SST 19.71 °C", () => {
    expect(unpackCode(codesOf(packed.u)[SAMPLE], PACK_VELOCITY)).toBeCloseTo(0.049, 3);
    expect(unpackCode(codesOf(packed.v)[SAMPLE], PACK_VELOCITY)).toBeCloseTo(0.094, 3);
    expect(unpackCode(codesOf(packed.sst)[SAMPLE], PACK_TEMPERATURE)).toBeCloseTo(19.71, 2);
    expect(packed.valid_time).toBe("2026-11-15T03:00:00.000Z");
  });

  it("keeps land as nodata and flags water within about 6 km of shore", () => {
    const sst = codesOf(packed.sst), mask = codesOf(packed.mask);
    const land = [...mask].filter((m) => m === MASK_LAND).length / mask.length;
    expect(land).toBeGreaterThan(0.5); // the box holds most of California
    for (let k = 0; k < mask.length; k++) if (mask[k] === MASK_LAND) expect(sst[k]).toBe(65535);
    expect(mask[SAMPLE]).toBe(MASK_WATER); // the Monterey Bay mouth is about 15 km from shore
    // Monterey harbor (36.61°N 121.89°W) sits on land at 4 km; the nearest wet cell offshore of it is near shore
    const row = Math.round((36.61 - 32.38) / 0.04);
    const cols = Array.from({ length: 6 }, (_, i) => Math.round((-121.89 + 125.02) / 0.04) - i);
    const firstWet = cols.map((c) => mask[row * GEOMETRY.width + c]).find((m) => m !== MASK_LAND);
    expect(firstWet).toBe(MASK_NEAR_SHORE);
    expect(packed.u.length).toBeLessThan(150_000); // compact: well under the 196 kB float32 subset per variable
  });

  it("refuses a file whose own time does not match the step", () => {
    expect(() => packStep(gz("wcofs-20261114-f024.dods.gz"), "2026111403", "f021")).toThrow(/expected 2026-11-15T00:00/);
  });
});

describe("WCOFS cycle discovery with fallback", () => {
  const stubFetch = (answers: Record<string, () => Response>) => {
    const calls: string[] = [];
    vi.stubGlobal("fetch", vi.fn(async (url: string) => {
      calls.push(url);
      const a = answers[url];
      if (!a) throw new TypeError("fetch failed");
      return a();
    }));
    return calls;
  };

  it("probes today, then yesterday, when today's cycle has not landed", async () => {
    process.env.CW_OCEAN_FIXTURE = "off";
    const calls = stubFetch({
      [catalogUrl(new Date("2026-11-15T02:00:00Z"))]: () => new Response("nope", { status: 404 }),
      [catalogUrl(new Date("2026-11-14T02:00:00Z"))]: () => new Response(CATALOG),
    });
    const c = await discoverCycle(new Date("2026-11-15T02:00:00Z"));
    expect(c?.cycleId).toBe("2026111403");
    expect(calls).toHaveLength(2);
  });

  it("throws when no catalog answers at all, and finds nothing when three days are empty", async () => {
    process.env.CW_OCEAN_FIXTURE = "off";
    stubFetch({});
    await expect(discoverCycle(new Date("2026-11-15T02:00:00Z"))).rejects.toThrow(/unreachable/);
    stubFetch(Object.fromEntries([15, 14, 13].map((d) => [catalogUrl(new Date(`2026-11-${d}T02:00:00Z`)), () => new Response("x", { status: 404 })])));
    expect(await discoverCycle(new Date("2026-11-15T02:00:00Z"))).toBeNull();
  });

  it("serves the last good cycle, marked as a fallback, when THREDDS stops answering; unavailable with nothing cached", async () => {
    process.env.CW_OCEAN_FIXTURE = "off";
    process.env.CW_OCEAN_FIXED_NOW = "2026-11-14T12:00:00Z";
    stubFetch({});
    const none = await wcofsIndex();
    expect(none.status).toBe("unavailable");
    expect(none.steps).toEqual([]);
    expect(none.reason).toMatch(/could not be read/);

    resetWcofsCaches();
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.setSystemTime(new Date("2026-11-14T12:00:00Z"));
    stubFetch({ [catalogUrl(new Date("2026-11-14T12:00:00Z"))]: () => new Response(CATALOG) });
    expect((await wcofsIndex()).fallback).toBe(false);
    vi.setSystemTime(new Date("2026-11-14T14:00:00Z")); // past the hour's cache
    stubFetch({});
    const fb = await wcofsIndex();
    expect(fb.status).toBe("current");
    expect(fb.fallback).toBe(true);
    expect(fb.cycle?.id).toBe("2026111403");
    expect(fb.reason).toMatch(/last good cycle/);
  });

  it("says expired, with the valid interval, when the newest cycle ends before now", () => {
    const idx = buildIndex({ cycleId: "2026111403", steps: STEP_IDS }, new Date("2026-11-18T00:00:00Z"), null);
    expect(idx.status).toBe("expired");
    expect(idx.valid).toEqual({ start: "2026-11-13T06:00:00.000Z", end: "2026-11-17T03:00:00.000Z" });
    expect(idx.reason).toMatch(/valid only until 2026-11-17T03:00/);
  });

  it("uses an encoded, box-only OPeNDAP constraint (one step, three variables at 0 m)", () => {
    expect(stepUrl("2026111403", "f024")).toBe(
      "https://opendap.co-ops.nos.noaa.gov/thredds/dodsC/NOAA/WCOFS/MODELS/2026/11/14/wcofs.t03z.20261114.regulargrid.f024.nc.dods?u_eastward%5B0%5D%5B0%5D%5B349:1:592%5D%5B509:1:709%5D,v_northward%5B0%5D%5B0%5D%5B349:1:592%5D%5B509:1:709%5D,temp%5B0%5D%5B0%5D%5B349:1:592%5D%5B509:1:709%5D,time",
    );
  });
});

describe("WCOFS routes in fixture mode (no network)", () => {
  beforeEach(() => {
    process.env.CW_OCEAN_FIXTURE = FX;
    delete process.env.CW_OCEAN_FIXED_NOW;
    vi.stubGlobal("fetch", vi.fn(async () => { throw new Error("network used in fixture mode"); }));
  });

  it("answers the index from the recorded cycle, defaulting to the newest step valid at the recording time", async () => {
    const r = await indexGET();
    const idx = await r.json();
    expect(idx.status).toBe("current");
    expect(idx.cycle).toEqual({ id: "2026111403", issued: "2026-11-14T03:00:00.000Z", steps_available: 32, steps_expected: 32 });
    expect(idx.default_step).toBe("f006"); // recorded 10:20Z; f006 is valid at 09:00Z
    expect(idx.steps[0].track).toBe("estimates");
    expect(idx.steps.at(-1).track).toBe("forecasts");
    expect(idx.families["currents.predicted"].vars).toEqual(["u", "v"]);
    expect(idx.steps.find((s: { id: string }) => s.id === "f024").grids.sst.url).toBe("/explore/model/wcofs/grid?cycle=2026111403&step=f024&var=sst");
    expect(r.headers.get("Cache-Control")).toMatch(/s-maxage=900/);
  });

  it("serves a recorded step's grid and answers an unrecorded one as unavailable", async () => {
    const ok = await gridGET(new Request("http://x/explore/model/wcofs/grid?cycle=2026111403&step=f024&var=sst"));
    expect(ok.status).toBe(200);
    const sst = codesOf(new Uint8Array(await ok.arrayBuffer()));
    expect(unpackCode(sst[SAMPLE], PACK_TEMPERATURE)).toBeCloseTo(19.71, 2);
    const miss = await gridGET(new Request("http://x/explore/model/wcofs/grid?cycle=2026111403&step=f027&var=u"));
    expect(miss.status).toBe(503);
    expect((await miss.json()).status).toBe("unavailable");
    const bad = await gridGET(new Request("http://x/explore/model/wcofs/grid?cycle=2020010103&step=f024&var=u"));
    expect(bad.status).toBe(400);
  });

  it("is on by default whenever the conditions fixture is (the e2e server)", async () => {
    delete process.env.CW_OCEAN_FIXTURE;
    process.env.CW_CONDITIONS_FIXTURE = "tests/conditions-fixtures.json";
    const { fixtureDir } = await import("@/lib/ocean/upstream");
    expect(fixtureDir()).toBe("tests/ocean-fixtures");
  });
});
