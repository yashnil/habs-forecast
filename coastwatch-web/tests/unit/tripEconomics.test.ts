import { describe, expect, it } from "vitest";
import {
  computeTrip,
  fuelGallons,
  kgToLb,
  lbToKg,
  parseField,
  sanitizeInputs,
  usd,
  usdPerLbToPerKg,
  usdPerLbToPerTon,
  usdPerTonToPerLb,
  type TripInputs,
} from "@/lib/tripEconomics";
import { DIESEL_REF, PRICE_REFS, dieselAge, priceRef } from "@/content/referenceValues";
import { FISHERY_PRESETS, SOURCES, presetForSpecies } from "@/content/fisheryPresets";

/**
 * Hand-worked trip (the same boat and costs for every crew model):
 *   steaming 20 nm at 8 kt = 2.5 h × 7.8 gph = 19.5 gal; fishing 8 h × 4.8 gph = 38.4 gal; 57.9 gal × $5 = $289.50
 *   bait $200 (shared), insurance $100 (boat-paid)
 *   catch 1,000 lb × $3.50 = $3,500 gross
 */
const trip = (over: Partial<TripInputs> = {}): TripInputs => ({
  roundTripNm: 20,
  cruiseKt: 8,
  steamGph: 7.8,
  fishingHours: 8,
  fishingGph: 4.8,
  fuelUsdPerGal: 5,
  costs: [
    { label: "Bait", usd: 200, shared: true },
    { label: "Insurance (day)", usd: 100, shared: false },
  ],
  catch: [{ species: "Dungeness crab", ref: "dungeness", lb: 1000, usdPerLb: 3.5 }],
  crew: { kind: "afterExpenses", pct: 25 },
  ...over,
});

describe("fuel", () => {
  it("adds steaming and fishing burn", () => {
    expect(fuelGallons(trip())).toBeCloseTo(57.9, 9);
  });
  it("uses gallons entered directly instead", () => {
    expect(fuelGallons(trip({ fuelGal: 40 }))).toBe(40);
  });
  it("is unknown when a started estimate is incomplete", () => {
    expect(fuelGallons(trip({ cruiseKt: null }))).toBeNull();
    expect(fuelGallons(trip({ cruiseKt: 0 }))).toBeNull();
    expect(fuelGallons({ fishingHours: 6, fishingGph: 3 })).toBe(18);
  });
});

describe("crew model A: share after expenses", () => {
  it("matches the hand-worked example", () => {
    const r = computeTrip(trip());
    // D = 289.5 + 200 = 489.5; N = 3500 − 489.5 = 3010.5; crew = 0.25 N = 752.625; net = N − crew − 100
    expect(r.fuelUsd).toBeCloseTo(289.5, 9);
    expect(r.tripCostUsd).toBeCloseTo(589.5, 9);
    expect(r.grossUsd).toBe(3500);
    expect(r.crewUsd).toBeCloseTo(752.625, 9);
    expect(r.netUsd).toBeCloseTo(2157.875, 9);
    // (1 − s)(3.5 q − 489.5) = 100  →  q = (489.5 + 133.33…) / 3.5
    expect(r.breakEvenLb.value).toBeCloseTo((489.5 + 100 / 0.75) / 3.5, 9);
    expect(r.breakEvenUsdPerLb.value).toBeCloseTo((489.5 + 100 / 0.75) / 1000, 9);
    // 0.75 (3300 − 57.9 f) = 100  →  f = (3300 − 133.33…) / 57.9
    expect(r.breakEvenFuelUsdPerGal.value).toBeCloseTo((3300 - 100 / 0.75) / 57.9, 9);
    expect(r.missing).toEqual([]);
  });
  it("pays no crew share when the trip loses money before shares", () => {
    const r = computeTrip(trip({ catch: [{ species: "Dungeness crab", lb: 100, usdPerLb: 3.5 }] }));
    // N = 350 − 489.5 = −139.5 → crew 0, net = −139.5 − 100
    expect(r.crewUsd).toBe(0);
    expect(r.netUsd).toBeCloseTo(-239.5, 9);
  });
});

describe("crew model B: share of gross", () => {
  it("matches the hand-worked example, with fees as % of gross and $/lb", () => {
    const r = computeTrip(trip({ crew: { kind: "ofGross", pct: 25 }, feesPctOfGross: 2, feesUsdPerLb: 0.05 }));
    // net = 3500 (1 − 0.25 − 0.02) − 0.05 × 1000 − 489.5 − 100 = 1915.5
    expect(r.crewUsd).toBeCloseTo(875, 9);
    expect(r.feesUsd).toBeCloseTo(120, 9);
    expect(r.netUsd).toBeCloseTo(1915.5, 9);
    // q (3.5 × 0.73 − 0.05) = 489.5 + 100 (the $/lb fee moves with q)
    expect(r.breakEvenLb.value).toBeCloseTo(589.5 / 2.505, 9);
    // 1000 p × 0.73 = 489.5 + 100 + 0.05 × 1000
    expect(r.breakEvenUsdPerLb.value).toBeCloseTo(639.5 / 730, 9);
  });
});

describe("crew model C: day rate", () => {
  it("matches the hand-worked example (the crew's pay is boat-paid)", () => {
    const r = computeTrip(trip({ crew: { kind: "dayRate", crew: 2, usdPerDay: 250, days: 1 } }));
    expect(r.crewUsd).toBe(500);
    expect(r.netUsd).toBeCloseTo(2410.5, 9);
    expect(r.breakEvenLb.value).toBeCloseTo(1089.5 / 3.5, 9);
    expect(r.breakEvenUsdPerLb.value).toBeCloseTo(1089.5 / 1000, 9);
  });
  it("asks for the day rate when it is missing", () => {
    expect(computeTrip(trip({ crew: { kind: "dayRate", crew: 2 } })).missing).toContain("dayRate");
    expect(computeTrip(trip({ crew: { kind: "dayRate", crew: 0 } })).missing).not.toContain("dayRate");
  });
});

describe("boat-paid fuel", () => {
  it("moves fuel out of the crew's deductions", () => {
    const r = computeTrip(trip({ fuelShared: false }));
    // D = 200, B = 100 + 289.5; N = 3300; crew = 825; net = 3300 − 825 − 389.5
    expect(r.crewUsd).toBeCloseTo(825, 9);
    expect(r.netUsd).toBeCloseTo(2085.5, 9);
  });
});

describe("multi-species", () => {
  it("adds lines to gross; break-even catch is the primary line's, others held fixed", () => {
    const r = computeTrip(trip({ catch: [{ species: "Dungeness crab", lb: 1000, usdPerLb: 3.5 }, { species: "Rock crab", lb: 200, usdPerLb: 2 }] }));
    expect(r.grossUsd).toBe(3900);
    expect(r.totalLb).toBe(1200);
    expect(r.breakEvenLb.value).toBeCloseTo((489.5 + 100 / 0.75 - 400) / 3.5, 9);
  });
  it("says the other lines already cover the trip", () => {
    const r = computeTrip(trip({ catch: [{ species: "Dungeness crab", lb: 1000, usdPerLb: 3.5 }, { species: "Rock crab", lb: 2000, usdPerLb: 2 }] }));
    expect(r.breakEvenLb).toEqual({ value: 0, note: "covered" });
    expect(r.breakEvenUsdPerLb).toEqual({ value: 0, note: "covered" });
  });
});

describe("no break-even", () => {
  it("when fees and shares take all of the gross", () => {
    const r = computeTrip(trip({ crew: { kind: "ofGross", pct: 90 }, feesPctOfGross: 10 }));
    expect(r.breakEvenLb.note).toBe("none");
    expect(r.breakEvenUsdPerLb.note).toBe("none");
    expect(r.warnings.join(" ")).toMatch(/take all of the gross/);
  });
  it("when the per-lb fee is above the price", () => {
    const r = computeTrip(trip({ feesUsdPerLb: 4 }));
    expect(r.breakEvenLb).toEqual({ value: null, note: "none" });
  });
  it("for fuel, when the trip loses money even with free fuel", () => {
    const r = computeTrip(trip({ catch: [{ species: "Dungeness crab", lb: 50, usdPerLb: 3.5 }] }));
    expect(r.breakEvenFuelUsdPerGal.note).toBe("none");
  });
});

describe("sensitivity", () => {
  it("has a 3 × 3 price × catch grid whose centre is the net", () => {
    const r = computeTrip(trip());
    expect(r.sensitivity).toHaveLength(9);
    const centre = r.sensitivity!.find((c) => c.priceDelta === 0 && c.catchDelta === 0)!;
    expect(centre.netUsd).toBe(r.netUsd);
    // price +20 %, catch +30 %: R = 3500 × 1.2 × 1.3 = 5460; N = 4970.5; net = 0.75 N − 100
    const top = r.sensitivity!.find((c) => c.priceDelta === 0.2 && c.catchDelta === 0.3)!;
    expect(top.netUsd).toBeCloseTo(0.75 * (5460 - 489.5) - 100, 9);
  });
  it("moves fuel ±20 %", () => {
    const r = computeTrip(trip());
    expect(r.fuelSensitivity!.up).toBeCloseTo(0.75 * (3500 - 200 - 289.5 * 1.2) - 100, 9);
    expect(r.fuelSensitivity!.down).toBeCloseTo(0.75 * (3500 - 200 - 289.5 * 0.8) - 100, 9);
  });
});

describe("partial, zero and invalid inputs", () => {
  it("returns what it can and lists what is missing", () => {
    const r = computeTrip({ crew: { kind: "afterExpenses" } });
    expect(r.netUsd).toBeNull();
    expect(r.sensitivity).toBeNull();
    expect(r.missing.sort()).toEqual(["catch", "crewShare", "fuel", "price"]);
  });
  it("prices the trip without a catch", () => {
    const r = computeTrip(trip({ catch: [{ species: "Dungeness crab", lb: null, usdPerLb: 3.5 }] }));
    expect(r.tripCostUsd).toBeCloseTo(589.5, 9);
    expect(r.netUsd).toBeNull();
    expect(r.missing).toEqual(["catch"]);
    expect(r.breakEvenLb.value).toBeCloseTo((489.5 + 100 / 0.75) / 3.5, 9);
    expect(r.breakEvenUsdPerLb.note).toBe("unknown");
  });
  it("asks for the fuel price only when fuel is used", () => {
    expect(computeTrip(trip({ fuelUsdPerGal: null })).missing).toEqual(["fuelPrice"]);
    expect(computeTrip(trip({ fuelUsdPerGal: null, fuelGal: 0 })).missing).toEqual([]);
  });
  it("handles a zero catch", () => {
    const r = computeTrip(trip({ catch: [{ species: "Dungeness crab", lb: 0, usdPerLb: 3.5 }] }));
    expect(r.grossUsd).toBe(0);
    expect(r.netUsd).toBeCloseTo(-489.5 - 100, 9);
    expect(r.breakEvenUsdPerLb.note).toBe("none");
  });
  it("rejects NaN and negatives, clamps absurd values", () => {
    const s = sanitizeInputs({ roundTripNm: -5, cruiseKt: Number.NaN, fuelUsdPerGal: 1e9, crew: { kind: "afterExpenses", pct: 140 } });
    expect(s.roundTripNm).toBeNull();
    expect(s.cruiseKt).toBeNull();
    expect(s.fuelUsdPerGal).toBe(50);
    expect(s.crew!.pct).toBe(100);
    expect(computeTrip(trip({ fuelUsdPerGal: -3 })).missing).toContain("fuelPrice");
  });
  it("parses what people type", () => {
    expect(parseField("$1,234.5", "usd")).toEqual({ value: 1234.5, error: null, clamped: false });
    expect(parseField(" 28.1 % ", "pct").value).toBe(28.1);
    expect(parseField("", "lb")).toEqual({ value: null, error: null, clamped: false });
    expect(parseField("-2", "lb").error).toMatch(/negative/);
    expect(parseField("abc", "lb").error).toMatch(/number/);
    expect(parseField("1e3", "lb").error).toMatch(/number/);
    expect(parseField("250", "pct")).toEqual({ value: 100, error: null, clamped: true });
  });
});

describe("units", () => {
  it("converts lb ⇄ kg and $/ton ⇄ $/lb", () => {
    expect(lbToKg(kgToLb(12.5))).toBeCloseTo(12.5, 12);
    expect(lbToKg(1000)).toBeCloseTo(453.59237, 5);
    expect(usdPerLbToPerKg(1)).toBeCloseTo(2.2046226218, 9);
    expect(usdPerTonToPerLb(1073)).toBeCloseTo(0.5365, 9);
    expect(usdPerLbToPerTon(usdPerTonToPerLb(900))).toBeCloseTo(900, 9);
  });
  it("formats signed dollars", () => {
    expect(usd(1234.4)).toBe("$1,234");
    expect(usd(-56.2, { signed: true })).toBe("−$56");
    expect(usd(12, { signed: true })).toBe("+$12");
    expect(usd(null)).toBe("—");
  });
});

describe("references and presets", () => {
  it("computes CDFW 2024 statewide $/lb exactly as listed", () => {
    const r = (k: string) => Math.round(PRICE_REFS[k].usdPerLb * 100) / 100;
    expect([r("dungeness"), r("squid"), r("urchin"), r("halibut"), r("sablefish"), r("petrale"), r("spot-prawn"), r("lobster")]).toEqual([3.51, 0.54, 3.51, 5.67, 1.64, 1.2, 22.28, 19.35]);
    expect(Math.round(usdPerLbToPerTon(PRICE_REFS.squid.usdPerLb))).toBe(1073);
  });
  it("has no reference for species CDFW does not list exactly", () => {
    expect(priceRef("albacore")).toBeNull();
    expect(priceRef("salmon")).toBeNull();
    expect(priceRef("groundfish-live")).toBeNull();
  });
  it("dates the diesel reference and marks it stale after 14 days", () => {
    expect(DIESEL_REF.usdPerGal).toBe(8.242);
    expect(dieselAge(new Date("2026-10-20T12:00:00Z")).stale).toBe(false);
    // the week of 2026-10-19: 14 days later is still current, 15 is stale
    expect(dieselAge(new Date("2026-11-02T12:00:00Z")).stale).toBe(false);
    expect(dieselAge(new Date("2026-11-03T12:00:00Z")).stale).toBe(true);
  });
  it("gives every preset default a source and never a price or catch", () => {
    for (const p of FISHERY_PRESETS) {
      for (const v of [p.fishingGph, p.steamGph, p.crewPct, p.crewPctAlt, p.feesPct]) if (v) expect(SOURCES[v.source]).toBeTruthy();
      expect(JSON.stringify(p)).not.toMatch(/usdPerLb|"lb":/);
    }
    expect(presetForSpecies("dungeness")!.fishingGph!.value).toBe(4.8);
    expect(presetForSpecies("dungeness")!.potTiers!.tiers).toEqual([500, 450, 400, 350, 300, 250, 175]);
    expect(presetForSpecies("squid")!.fishingGph).toBeNull();
    expect(FISHERY_PRESETS.find((p) => p.id === "groundfish-trawl")!.feesPct!.value).toBe(2.25);
  });
});

describe("copy", () => {
  it("uses none of the banned words in the economics UI", async () => {
    const { readdirSync, readFileSync } = await import("node:fs");
    const path = await import("node:path");
    const dir = path.resolve(__dirname, "../../src/v4/coast/economics");
    const text = readdirSync(dir).map((f) => readFileSync(path.join(dir, f), "utf8")).join("\n").replace(/safe-area-inset/g, "");
    expect(text).not.toMatch(/\b(safe|go\/no-go|good day|hotspot|expected catch|price forecast)\b/i);
    expect(text).toContain("Scenario from your inputs, not a forecast.");
  });
});
