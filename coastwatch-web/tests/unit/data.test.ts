import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { checkManifest, checkPorts, loadData } from "@/lib/data";
import { charmLayer, charmRun, isFixture } from "@/lib/layers";
import type { Manifest } from "@/generated/schema";

const FIX = path.resolve(__dirname, "../fixture-data/v1");
const read = (rel: string) => JSON.parse(readFileSync(path.join(FIX, rel), "utf8"));

afterEach(() => {
  delete process.env.CW_DATA_DIR;
  delete process.env.CW_DATA_BASE_URL;
});

describe("schema validation (shared JSON Schema from the pipeline)", () => {
  it("accepts the pipeline's fixture manifest and ports", () => {
    expect(checkManifest(read("manifest.json")).ok).toBe(true);
    expect(checkPorts(read(read("manifest.json").ports_url)).ok).toBe(true);
  });

  it.each([
    ["missing provenance", (m: Manifest) => delete (m.layers[0] as Partial<typeof m.layers[0]>).provenance],
    ["unknown product class", (m: Manifest) => ((m.layers[0] as { product_class: string }).product_class = "safe_zone")],
    ["bad palette colour", (m: Manifest) => (m.layers[0].palette!.stops[0].color = "magenta")],
    ["wrong schema version", (m: Manifest) => ((m as { schema_version: number }).schema_version = 2)],
    ["extra field", (m: Manifest) => ((m as unknown as Record<string, unknown>).recommendation = "go fishing")],
    ["non-numeric lead", (m: Manifest) => ((m.layers[0].time as { lead_days: unknown }).lead_days = "tomorrow")],
  ])("rejects a manifest with %s", (_name, mutate) => {
    const m = read("manifest.json");
    mutate(m);
    expect(checkManifest(m).ok).toBe(false);
  });
});

describe("loadData", () => {
  it("loads and validates artifacts from a directory", async () => {
    process.env.CW_DATA_DIR = FIX;
    const d = await loadData();
    expect(d.ok).toBe(true);
    if (!d.ok) return;
    expect(isFixture(d.manifest)).toBe(true);
    expect(d.ports?.features).toHaveLength(22);
    expect(charmRun(d.manifest)?.issued_date).toBe("2026-10-08");
    expect(charmLayer(d.manifest, "particulate_domoic", 1)?.time.valid_date).toBe("2026-10-08");
  });

  it("reports a missing manifest instead of throwing", async () => {
    process.env.CW_DATA_DIR = mkdtempSync(path.join(tmpdir(), "cw-empty-"));
    const d = await loadData();
    expect(d.ok).toBe(false);
    if (!d.ok) expect(d.error).toMatch(/manifest not available/);
  });

  it("reports a corrupt manifest instead of rendering it", async () => {
    const dir = mkdtempSync(path.join(tmpdir(), "cw-bad-"));
    writeFileSync(path.join(dir, "manifest.json"), JSON.stringify({ schema_version: 1, layers: "oops" }));
    process.env.CW_DATA_DIR = dir;
    const d = await loadData();
    expect(d.ok).toBe(false);
    if (!d.ok) expect(d.error).toMatch(/schema validation/);
  });
});
