import { readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import type { ObservationDataset } from "@/generated/observations";
import { citation, modelCsv, SAMPLE_COLUMNS, windowCsv } from "@/nx/research/exports";
import { buildStrips, findStation, located } from "@/nx/research/model";

const dir = path.resolve(__dirname, "../../../fixture-data/v1");
const ds = JSON.parse(readFileSync(path.join(dir, readdirSync(dir).find((f) => f.startsWith("observations-"))!), "utf8")) as ObservationDataset;
const st = findStation(located(ds), "HABs-SantaCruzWharf")!;
const strips = buildStrips(st, "pDA");
const D = (s: string) => Date.parse(`${s}T12:00:00Z`);
const win: [number, number] = [D("2025-06-01"), D("2026-06-01")];
const body = (csv: string) => csv.split("\n").filter((l) => l && !l.startsWith("#"));

describe("the window's CSV", () => {
  const csv = windowCsv(ds, st, strips, win);
  const [head, ...rows] = body(csv);
  it("opens with # lines naming kind, source, request, licence and run", () => {
    expect(csv).toMatch(/^# Neritic research export: Santa Cruz Wharf/);
    expect(csv).toContain("# Kind: measured");
    expect(csv).toContain("# Request: https://erddap.sccoos.org/");
    expect(csv).toContain("# Licence:");
    expect(csv).toContain(`# Pipeline run: ${ds.provenance.pipeline_run_id}`);
  });
  it("has source columns on every row", () => {
    expect(head).toBe(SAMPLE_COLUMNS.join(","));
    expect(rows.every((r) => r.includes(",measured,HABs-SantaCruzWharf,Santa Cruz Wharf,") && r.includes("CalHABMAP via SCCOOS ERDDAP"))).toBe(true);
  });
  it("holds exactly the samples inside the window, every quantity", () => {
    const want = strips.filter((s) => s.kind === "measured").reduce((a, s) => a + s.points.filter((p) => p.t >= win[0] && p.t <= win[1]).length, 0);
    expect(rows).toHaveLength(want);
    for (const r of rows) {
      const t = Date.parse(r.split(",")[0]);
      expect(t).toBeGreaterThanOrEqual(win[0]);
      expect(t).toBeLessThanOrEqual(win[1]);
    }
    expect(new Set(rows.map((r) => r.split(",")[1]))).toEqual(new Set(["pDA", "pn_seriata", "chl_extracted", "temp"]));
  });
  it("keeps a reported 0 with its qualifier, never an empty value", () => {
    const zero = rows.find((r) => r.includes(",reported_zero,"));
    expect(zero).toBeDefined();
    expect(zero!.split(",")[2]).toBe("0");
  });
});

describe("the model CSV and the citation", () => {
  it("writes the model days in the window as probabilities, kind model", () => {
    const m = strips.find((s) => s.kind === "model")!;
    const csv = modelCsv(st, m, [D("2026-05-01"), D("2026-06-01")], "particulate_domoic");
    expect(csv).toContain("not toxin in seafood");
    const rows = body(csv).slice(1);
    expect(rows.length).toBeGreaterThan(0);
    expect(rows.every((r) => Number(r.split(",")[2]) >= 0 && Number(r.split(",")[2]) <= 1 && r.includes(",model,"))).toBe(true);
  });
  it("cites the program, the station, the window and the link back", () => {
    const c = citation(ds, st, strips, win, "https://example.org/research?station=HABs-SantaCruzWharf");
    expect(c).toContain("CalHABMAP");
    expect(c).toContain("Santa Cruz Wharf (HABs-SantaCruzWharf)");
    expect(c).toContain("Jun 1, 2025–Jun 1, 2026");
    expect(c).toContain("https://example.org/research?station=HABs-SantaCruzWharf");
  });
});
