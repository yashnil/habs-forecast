/** v10 fix 2: every route reports the freshness of the published sources it shows, by one rule. */
import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import type { Manifest } from "@/generated/schema";
import { charmRun } from "@/lib/layers";
import { currentsEntry, fieldState, modelEntry, satelliteEntry } from "@/v4/readiness/shown";
import { overall } from "@/v4/readiness/store";
import type { SourceEntry } from "@/v4/readiness/store";

const manifest = JSON.parse(readFileSync(path.resolve(__dirname, "../fixture-data/v1/manifest.json"), "utf8")) as Manifest;
const issued = charmRun(manifest)!.issued_date;
const at = (d: string, days: number) => new Date(Date.parse(`${d}T19:00:00Z`) + days * 86_400_000);

describe("the shown sources' freshness", () => {
  it("a model run issued today is ready; one five days old is stale, with its age", () => {
    expect(modelEntry(manifest, at(issued, 0), "here")!.state).toBe("ready");
    const old = modelEntry(manifest, at(issued, 5), "In the lanes")!;
    expect(old.state).toBe("stale");
    expect(old.detail).toMatch(/days\)/);
    expect(old.detail).toContain("In the lanes");
    expect(old.asOf).toBe(issued);
  });
  it("satellite and radar entries carry the shown layer's own date", () => {
    const sat = satelliteEntry(manifest, new Date("2026-10-13T06:00:00Z"), "On the map");
    expect(sat?.id).toBe("satellite");
    expect(sat?.asOf).toMatch(/^\d{4}-\d{2}-\d{2}$/);
    const cur = currentsEntry(manifest, new Date("2027-01-01T00:00:00Z"), "x");
    if (cur) expect(cur.state).toBe("stale");
  });
  it("without a clock yet nothing is called old (no invented age)", () => {
    expect(fieldState(null, null)).toEqual({ state: "ready", note: null });
  });
  it("the chip's overall state is not Ready when a shown source is stale", () => {
    const data: SourceEntry = { id: "data", label: "d", state: "ready" };
    const model = modelEntry(manifest, at(issued, 5), "x")!;
    expect(overall({ sources: { data }, online: true, version: 1 })).toBe("ready");
    expect(overall({ sources: { data, model }, online: true, version: 2 })).toBe("stale");
  });
});
