/** M3: observation helpers, fisheries aggregation helpers, copy and artifact integrity. */
import { readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import type { ObservationDataset } from "@/generated/observations";
import type { FisheriesDataset } from "@/generated/fisheries";
import * as copy from "@/content/copy";
import { checkFisheries, checkManifest, checkObservations } from "@/lib/data";
import { REGION_BOUNDS, cadenceLabel, coverage, formatObs, logDomain, samples, stationFreshness, summaryOf, toxinStatus } from "@/lib/observations";
import { fisheriesValue, selectedTotal } from "@/lib/fisheries";

const DIR = path.resolve(__dirname, "../fixture-data/v1");
const read = (prefix: string) => JSON.parse(readFileSync(path.join(DIR, readdirSync(DIR).find((f) => f.startsWith(prefix))!), "utf8"));
const obs = read("observations-") as ObservationDataset;
const fish = read("fisheries-") as FisheriesDataset;
const manifest = JSON.parse(readFileSync(path.join(DIR, "manifest.json"), "utf8"));
const st = (id: string) => obs.stations.find((s) => s.station_id === id)!;
const v = (id: string) => obs.variables.find((x) => x.id === id)!;

describe("artifacts validate in the browser-side schema", () => {
  it("observations and fisheries", () => {
    expect(checkObservations(obs).ok).toBe(true);
    expect(checkFisheries(fish).ok).toBe(true);
    expect(checkManifest(manifest).ok).toBe(true);
  });
  it("the published M2 manifest (no M3 artifacts) still validates", () => {
    const m2 = JSON.parse(readFileSync(path.resolve(__dirname, "../../../pipeline/tests/fixtures/compat/m2/manifest.json"), "utf8"));
    const r = checkManifest(m2);
    expect(r.ok).toBe(true);
    if (r.ok) expect(r.manifest.observations_url ?? null).toBeNull();
  });
  it("a corrupted observation artifact is rejected", () => {
    const bad = structuredClone(obs) as unknown as { stations: { series: { values: unknown[] }[] }[] };
    bad.stations[0].series[0].values[0] = "0.2";
    expect(checkObservations(bad).ok).toBe(false);
  });
});

describe("observation semantics", () => {
  it("not measured is null and formatted as 'not measured', never 0", () => {
    const sc = samples(st("HABs-SantaCruzWharf"), "pDA");
    const missing = sc.filter((s) => s.value == null && s.q == null);
    expect(missing.length).toBeGreaterThan(0);
    expect(formatObs(v("pDA"), null)).toBe("not measured");
  });
  it("a reported zero is never shown as a bare 0", () => {
    expect(formatObs(v("pDA"), 0, "reported_zero")).toBe("reported 0 (not quantified)");
    expect(formatObs(v("pDA"), null, "rejected_negative")).toBe("rejected (QC)");
    const zeros = samples(st("HABs-SantaCruzWharf"), "pDA").filter((s) => s.q === "reported_zero");
    expect(zeros.every((s) => s.value === 0)).toBe(true);
  });
  it("coverage counts visits, measurements and zeros in a window", () => {
    const ss = samples(st("HABs-SantaCruzWharf"), "pDA");
    const c = coverage(ss, 0, Date.parse("2100-01-01"));
    expect(c.visits).toBe(ss.length);
    expect(c.measured).toBe(summaryOf(st("HABs-SantaCruzWharf"), "pDA")!.n_measured);
    expect(c.zeros).toBe(summaryOf(st("HABs-SantaCruzWharf"), "pDA")!.n_reported_zero);
  });
  it("long absence is described as absence", () => {
    const sm = summaryOf(st("HABs-MontereyWharf"), "pDA");
    expect(toxinStatus(v("pDA"), sm)).toMatch(/^pDA last measured \w{3} 2022$/);
    expect(toxinStatus(v("pDA"), { ...sm!, n_measured: 0 })).toContain("not measured");
  });
  it("station freshness comes from the newest sample and the published policy", () => {
    const policy = manifest.sources.find((s: { source_id: string }) => s.source_id === "calhabmap").freshness;
    expect(stationFreshness(st("HABs-SantaCruzWharf"), policy, new Date("2026-10-08T18:00:00Z"))?.state).toBe("current");
    expect(stationFreshness(st("HABs-TrinidadPier"), policy, new Date("2026-10-08T18:00:00Z"))?.state).toBe("historical");
  });
  it("helpers", () => {
    expect(logDomain([0.003, 12])).toEqual([0.001, 100]);
    expect(logDomain([0, 0])).toBeNull();
    expect(cadenceLabel(7)).toBe("about weekly");
    expect(cadenceLabel(null)).toContain("no regular sampling");
  });
});

describe("fisheries semantics", () => {
  it("selected totals skip missing values and never invent zeros", () => {
    const t1 = fish.groups.filter((g) => g.tier === 1);
    const y = fish.years.at(-1)!;
    const expected = t1.reduce((s, g) => s + (g.annual.find((a) => a.year === y)!.dollars_real ?? 0), 0);
    expect(selectedTotal(t1, y, "real")).toBeCloseTo(expected, 2);
    const empty = t1.map((g) => ({ ...g, annual: g.annual.map((a) => ({ ...a, dollars_real: null, dollars_nominal: null })) }));
    expect(selectedTotal(empty, y, "real")).toBeNull();
  });
  it("base-year real equals nominal", () => {
    const a = fish.groups[0].annual.find((x) => x.year === fish.deflator.base_year)!;
    expect(fisheriesValue(a, "real")).toBe(fisheriesValue(a, "nominal"));
  });
  it("port-level is unavailable and withheld is separate", () => {
    expect(fish.port_level.status).toBe("unavailable");
    expect(fish.withheld.length).toBe(fish.years.length);
  });
});

describe("M3 copy", () => {
  const texts = [copy.BLOOM_COPY, copy.FISHERIES_COPY].flatMap((o) => Object.values(o) as string[]);
  it("never presents exposure as predicted loss", () => {
    for (const t of [...texts, fish.terminology, ...fish.caveats])
      for (const s of t.split(/(?<=[.;])\s+/)) if (/\blosse?s?\b/i.test(s)) expect(s).toMatch(/\bnot\b/);
  });
  it("states that absence of measurement is not absence of toxin", () => {
    expect(copy.BLOOM_COPY.absence).toContain("not the same as no toxin");
    expect(copy.BLOOM_COPY.reviewPending).toContain("not yet been reviewed");
  });
});

describe("station map regions", () => {
  it("region bounds match the curated regions", () => {
    const curated = JSON.parse(readFileSync(path.resolve(__dirname, "../../../data/curated/ports.json"), "utf8")).regions;
    expect(Object.fromEntries(curated.map((r: { id: string; bounds: unknown }) => [r.id, r.bounds]))).toEqual(REGION_BOUNDS);
    for (const st of obs.stations) if (st.region) expect(REGION_BOUNDS[st.region]).toBeDefined();
  });
});
