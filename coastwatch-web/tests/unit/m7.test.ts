import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import type { FisheriesDataset, SpeciesGroup } from "@/generated/fisheries";
import * as copy from "@/content/copy";
import geo from "@/content/fisheries-geo.json";
import {
  fisheriesValue,
  formatTick,
  formatValue,
  groupColor,
  groupsInTier,
  niceTicks,
  parseView,
  pct,
  rowsWithoutValue,
  selectedTotal,
  shareOfState,
  span,
  stateTotal,
  tierInk,
  viewQuery,
  withheldValue,
  yearFraction,
} from "@/lib/fisheries";

/** Fisheries & Economic Exposure (M7): units, totals, shares, link state, the documented events and
 *  the reference map. Values are checked against the fixture dataset, never re-derived by hand. */
const FIX = path.resolve(__dirname, "../fixture-data/v1");
const ds = JSON.parse(readFileSync(path.join(FIX, readdirSync(FIX).find((f) => f.startsWith("fisheries-"))!), "utf8")) as FisheriesDataset;
const latest = ds.years.at(-1)!;
const g = (id: string) => ds.groups.find((x) => x.id === id)!;

describe("units", () => {
  it("reads revenue in base-year or nominal dollars, or pounds, and never invents a value", () => {
    const a = g("dungeness_crab").annual[0];
    expect(fisheriesValue(a, "real")).toBe(a.dollars_real);
    expect(fisheriesValue(a, "nominal")).toBe(a.dollars_nominal);
    expect(fisheriesValue(a, "pounds")).toBe(a.pounds);
    expect(fisheriesValue({ dollars_real: null, dollars_nominal: null, pounds: null }, "pounds")).toBeNull();
  });
  it("formats dollars and pounds distinctly; a missing value reads 'no value', never $0", () => {
    expect(formatValue(49_745_602, "real")).toBe("$49.7M");
    expect(formatValue(627_641, "nominal")).toBe("$628k");
    expect(formatValue(14_173_705, "pounds")).toBe("14.2M lb");
    expect(formatValue(null, "real")).toBe("no value");
    expect(formatTick(40e6, "real")).toBe("$40M");
    expect(formatTick(2.5e6, "pounds")).toBe("2.5M lb");
    expect(formatTick(0, "real")).toBe("$0");
  });
  it("in the base year, real and nominal are equal", () => {
    for (const grp of ds.groups) expect(selectedTotal([grp], ds.deflator.base_year, "real")).toBe(selectedTotal([grp], ds.deflator.base_year, "nominal"));
  });
});

describe("totals and shares", () => {
  it("tier sets: Tier 1 is the closure fisheries; Tiers 1 + 2 is all six", () => {
    expect(groupsInTier(ds.groups, "1").map((x) => x.id)).toEqual(["dungeness_crab", "rock_crab", "northern_anchovy"]);
    expect(groupsInTier(ds.groups, "12")).toHaveLength(6);
  });
  it("selected totals skip missing values; all-missing is null, not zero", () => {
    const t1 = groupsInTier(ds.groups, "1");
    const expected = t1.reduce((s, x) => s + (x.annual.find((a) => a.year === latest)!.pounds ?? 0), 0);
    expect(selectedTotal(t1, latest, "pounds")).toBeCloseTo(expected, 3);
    const blank = t1.map((x) => ({ ...x, annual: x.annual.map((a) => ({ ...a, pounds: null })) }));
    expect(selectedTotal(blank, latest, "pounds")).toBeNull();
  });
  it("share of the statewide total uses NOAA's total (which includes the withheld row) and is a lower bound", () => {
    const t1 = groupsInTier(ds.groups, "1");
    const part = selectedTotal(t1, latest, "real");
    const total = stateTotal(ds, latest, "real");
    expect(shareOfState(part, total)).toBeCloseTo(part! / total!, 10);
    expect(shareOfState(null, total)).toBeNull();
    expect(shareOfState(part, 0)).toBeNull();
    // the withheld amount is part of the state total, never of any group
    const w = withheldValue(ds, latest, "nominal")!;
    expect(w).toBeGreaterThan(0);
    const all = selectedTotal(ds.groups, latest, "nominal")!;
    expect(all + w).toBeLessThanOrEqual(stateTotal(ds, latest, "nominal")! + 1);
  });
  it("percentages: under 1% reads <1%, missing reads —", () => {
    expect(pct(0.262)).toBe("26%");
    expect(pct(0.004)).toBe("<1%");
    expect(pct(null)).toBe("—");
  });
  it("span: mean, lowest and highest years; missing years are left out", () => {
    const s = span([2015, 2016, 2017], (y) => ({ 2015: 3, 2016: null, 2017: 1 })[y] ?? null);
    expect(s).toEqual({ mean: 2, low: { year: 2017, v: 1 }, high: { year: 2015, v: 3 }, n: 2 });
    expect(span([2015], () => null)).toEqual({ mean: null, low: null, high: null, n: 0 });
  });
  it("counts source rows without a published value (the bivalve years)", () => {
    const n = ds.groups.reduce((s, x) => s + (x.annual.find((a) => a.year === latest)?.n_rows_without_value ?? 0), 0);
    expect(rowsWithoutValue(ds.groups, latest)).toBe(n);
  });
  it("nice ticks cover the maximum with round steps", () => {
    expect(niceTicks(112.8e6)).toEqual([0, 50e6, 100e6, 150e6]);
    expect(niceTicks(61.3e6)).toEqual([0, 20e6, 40e6, 60e6, 80e6]);
    expect(niceTicks(0)).toEqual([0, 1]);
  });
});

describe("link state", () => {
  it("defaults to 2024 dollars, Tier 1, no group, the latest year", () => {
    expect(parseView(ds, {})).toEqual({ unit: "real", tiers: "1", group: null, year: latest });
  });
  it("restores every parameter and ignores unknown values", () => {
    expect(parseView(ds, { unit: "pounds", tiers: "all", group: "rock_crab", year: "2016" })).toEqual({ unit: "pounds", tiers: "12", group: "rock_crab", year: 2016 });
    expect(parseView(ds, { unit: "euros", tiers: "9", group: "../etc", year: "1900" })).toEqual({ unit: "real", tiers: "1", group: null, year: latest });
  });
  it("a Tier 2 group in the link brings Tier 2 into the chart", () => {
    expect(parseView(ds, { group: "bivalves" }).tiers).toBe("12");
  });
  it("round-trips, and the default view is a bare link", () => {
    const v = parseView(ds, { unit: "nominal", tiers: "all", group: "spiny_lobster", year: "2019" });
    expect(parseView(ds, Object.fromEntries(new URLSearchParams(viewQuery(v, latest))))).toEqual(v);
    expect(viewQuery(parseView(ds, {}), latest)).toBe("");
  });
});

describe("documented events", () => {
  it("are dated, sourced to an official state record, tied to species groups that exist", () => {
    for (const e of copy.FISHERIES_EVENTS) {
      expect(yearFraction(e.start)).toBeLessThan(yearFraction(e.end));
      expect(e.source.url).toMatch(/^https:\/\/fisheries\.legislature\.ca\.gov\//);
      for (const id of e.groups) expect(ds.groups.map((x) => x.id)).toContain(id);
    }
  });
  it("place 2015-11-06 late in 2015 and 2016-05-26 in the first half of 2016", () => {
    expect(yearFraction("2015-11-06")).toBeCloseTo(2015 + 309 / 365, 6);
    expect(yearFraction("2016-05-26") - 2016).toBeGreaterThan(0.39);
    expect(yearFraction("2016-05-26") - 2016).toBeLessThan(0.41);
  });
  it("never attribute a change in landings to the event", () => {
    for (const e of copy.FISHERIES_EVENTS) expect(`${e.label} ${e.detail}`).not.toMatch(/\b(caus|loss|lost|drop|declin|impact|because the|led to)/i);
  });
});

describe("copy", () => {
  const strings = Object.values(copy.FISHERIES_COPY).flat() as string[];
  it("never presents landings as losses or forecasts", () => {
    for (const t of strings) for (const s of t.split(/(?<=[.;:])\s+/)) if (/\blosse?s?\b/i.test(s)) expect(s).toMatch(/\bnot\b/);
    expect(copy.FISHERIES_COPY.cannot.join(" ")).toMatch(/does not show a cause/);
    expect(copy.FISHERIES_COPY.cannot.join(" ")).toMatch(/not profit|Profit or income/);
    expect(copy.FISHERIES_COPY.cannot.join(" ")).toMatch(/Fishing effort/);
  });
});

describe("reference map (decision F2)", () => {
  it("uses CDFW's published port points, attributed and licensed, with no values attached", () => {
    expect(geo.ports_source.license).toBe("CC BY 4.0");
    expect(geo.ports_source.url).toContain("gis.data.ca.gov");
    expect(geo.ports.length).toBeGreaterThan(100);
    for (const p of geo.ports) expect(Object.keys(p).sort()).toEqual(["area", "area_code", "code", "lat", "lon", "name"]);
  });
  it("has CDFW's nine coastal port areas plus the two inland ones, inside California", () => {
    const areas = [...new Set(geo.ports.map((p) => p.area))];
    expect(areas).toEqual(["Eureka", "Fort Bragg", "Bodega Bay", "San Francisco", "Monterey", "Morro Bay", "Santa Barbara", "Los Angeles", "San Diego", "Sacramento Delta", "Inland Waters"]);
    for (const p of geo.ports) {
      expect(p.lat).toBeGreaterThan(32.5);
      expect(p.lat).toBeLessThan(42);
      expect(p.lon).toBeGreaterThan(-124.5);
      expect(p.lon).toBeLessThan(-114);
    }
  });
});

describe("colours", () => {
  it("keep tiers in two families and avoid the reserved measured, model and official colours", () => {
    const reserved = ["#0d756c", "#6a3fb0", "#9a4a00"];
    for (const grp of ds.groups as SpeciesGroup[]) expect(reserved).not.toContain(groupColor(grp.id));
    expect(tierInk(1)).not.toBe(tierInk(2));
  });
});
