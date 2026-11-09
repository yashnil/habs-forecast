import { cpSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { buildTodayModel } from "@/pilot/facts";
import { loadToday } from "@/pilot/today/load";
import { fixtureServerEnv } from "./fixtures/portToday";

/**
 * The 8-day outlook as Today loads it from the data (src/pilot/today/load.ts `outlookAt`): the value and
 * the usual value for its period are read in the same cells of the derived fixture, and an index
 * without a usual-value grid (data published before the field existed) still gives an outlook.
 */
const env = fixtureServerEnv();
const saved: Record<string, string | undefined> = {};
beforeAll(() => {
  for (const [k, v] of Object.entries(env)) {
    saved[k] = process.env[k];
    process.env[k] = v;
  }
});
afterAll(() => {
  for (const k of Object.keys(env)) {
    if (saved[k] === undefined) delete process.env[k];
    else process.env[k] = saved[k];
  }
});

async function bloomAt(placeId: string) {
  const load = await loadToday(placeId);
  if (!load.ok || !load.inputs) throw new Error("fixture did not load");
  return { outlook: load.inputs.outlook, bloom: buildTodayModel({ ...load.inputs, conditions: null }).bloom };
}

describe("the outlook against usual, from the derived fixture", () => {
  it("Moss Landing: 1.6 times usual in the fixture, said as higher than usual", async () => {
    const { outlook, bloom } = await bloomAt("port-592");
    expect(outlook!.value! / outlook!.typical!).toBeCloseTo(1.6, 2);
    expect(outlook!.typical_basis).toMatch(/usual chlorophyll for this 8-day period/);
    expect(bloom.outlookRead!.level).toBe("higher");
    expect(bloom.outlookFact!.headline).toBe("Next 8 days (Oct 13–20): chlorophyll near Moss Landing is expected to run higher than usual for this time of year, about 1.6× the typical level.");
  });

  it("Half Moon Bay: 1.1 times usual in the fixture is not a signal", async () => {
    const { outlook, bloom } = await bloomAt("port-452");
    expect(outlook!.value! / outlook!.typical!).toBeCloseTo(1.1, 2);
    expect(bloom.outlookRead!.level).toBe("usual");
    expect(bloom.outlookFact!.headline).toBe("Next 8 days (Oct 13–20): chlorophyll near Princeton / Half Moon Bay is expected to run about usual for this time of year.");
  });

  it("an index without the usual-value grid (older data) still gives the outlook, without a comparison", async () => {
    const dir = mkdtempSync(path.join(tmpdir(), "neritic-older-"));
    cpSync(env.CW_DATA_DIR, dir, { recursive: true });
    const f = path.join(dir, "derived", "index.json");
    const idx = JSON.parse(readFileSync(f, "utf8"));
    for (const l of idx.layers) {
      delete l.typical_grid;
      delete l.typical_basis;
    }
    writeFileSync(f, JSON.stringify(idx));
    const keep = process.env.CW_DATA_DIR;
    process.env.CW_DATA_DIR = dir;
    try {
      const { outlook, bloom } = await bloomAt("port-592");
      expect(outlook!.value).toBeGreaterThan(0);
      expect(outlook!.typical).toBeNull();
      expect(bloom.outlookRead).toBeNull();
      expect(bloom.outlookFact!.id).toBe("outlook:chl8d");
      expect(bloom.outlookFact!.headline).toMatch(/^Chlorophyll outlook, Oct 13–20: about [\d.]+ mg\/m³ near Moss Landing\. A bloom indicator, not a toxin forecast\.$/);
    } finally {
      process.env.CW_DATA_DIR = keep;
    }
  });
});
