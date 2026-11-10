import { describe, expect, it } from "vitest";
import { LB_PER_KG, LB_PER_TON } from "@/lib/tripEconomics";
import { emptyDraft, decodeShare, encodeShare, sanitizeDraft } from "@/lib/tripEconomicsStore";
import { evaluateTrip, type Output, type TripScenario } from "@/lib/tripEvaluation";
import { SENSITIVITY_HELD, fuelBreakEvenWords, waterfallDomain } from "@/v4/coast/economics/model";

/**
 * The v12 ledger audit (docs/coastwatch/v12/research/E-science-and-perf.md §3, E1 to E6), every figure
 * worked by hand:
 *   gross 1,000 lb × $2 = 2,000; fees 5 % = 100; ice 100 shared; gear 50 boat-paid; crew 30 % after expenses.
 *   steaming 24 nm ÷ 8 kt × 7.8 gal/h = 23.4 gal; fishing 10 h × 4.8 gal/h = 48 gal; fuel 71.4 gal × $5 = 357.
 *   N = 2,000 − 100 − 357 − 100 = 1,443; crew 30 % = 432.90; net = 1,443 − 432.90 − 50 = 960.10.
 */
const trip = (over: Partial<TripScenario> = {}): TripScenario => ({
  roundTripNm: 24,
  cruiseKt: 8,
  steamGph: 7.8,
  fishingHours: 10,
  fishingGph: 4.8,
  fuelUsdPerGal: 5,
  fuelShared: true,
  feesPctOfGross: 5,
  costs: [
    { label: "Ice", usd: 100, shared: true },
    { label: "Gear", usd: 50, shared: false },
  ],
  catch: [{ species: "Salmon", lb: 1000, usdPerLb: 2 }],
  crew: { kind: "afterExpenses", pct: 30 },
  ...over,
});

const val = (o: Output) => {
  if (o.status !== "value") throw new Error(`expected a value, got ${JSON.stringify(o)}`);
  return o;
};
const miss = (o: Output) => {
  if (o.status !== "missing") throw new Error(`expected missing, got ${JSON.stringify(o)}`);
  return o;
};

describe("E1: a blank fuel leg never makes the trip cheaper", () => {
  it("both legs give the full figure", () => {
    const e = evaluateTrip(trip());
    expect(val(e.fuelGallons).value).toBeCloseTo(71.4, 9);
    expect(val(e.fuelCost).value).toBeCloseTo(357, 9);
    expect(val(e.crewPay).value).toBeCloseTo(432.9, 9);
    expect(val(e.netToBoat).value).toBeCloseTo(960.1, 9);
  });

  it("steaming entered, hours blank: the leg stays, net waits", () => {
    const e = evaluateTrip(trip({ fishingHours: null }));
    expect(val(e.steamingGallons).value).toBeCloseTo(23.4, 9);
    expect(miss(e.fuelGallons).missing).toEqual(["fishingHours"]);
    expect(miss(e.fuelGallons).alternatives).toEqual([["fuelGal"]]);
    expect(miss(e.fuelGallons).reason).toMatch(/steaming leg is worked out/);
    expect(miss(e.netToBoat).missing).toEqual(["fishingHours"]);
    expect(miss(e.tripCost).missing).toEqual(["fishingHours"]);
    expect(miss(e.breakEvenCatch).missing).toEqual(["fishingHours"]);
    expect(e.missingFields).toEqual(["fishingHours"]);
  });

  it("an untouched fishing leg says that 0 hours is the way to rule it out", () => {
    const e = evaluateTrip(trip({ fishingHours: null, fishingGph: null }));
    expect(miss(e.fuelGallons).missing).toEqual(["fishingHours", "fishingGph"]);
    expect(miss(e.fuelGallons).reason).toMatch(/Enter 0 hours/);
  });

  it("0 hours is the user's own statement: fuel 23.4 gal × $5 = 117; net 1,683 × 70 % − 50 = 1,128.10", () => {
    const e = evaluateTrip(trip({ fishingHours: 0, fishingGph: null }));
    expect(val(e.fuelGallons).value).toBeCloseTo(23.4, 9);
    expect(val(e.fuelGallons).assumptions).toEqual([]);
    expect(val(e.netToBoat).value).toBeCloseTo(1128.1, 9);
  });

  it("gallons entered directly replace the legs: 40 gal × $5 = 200; net 1,600 × 70 % − 50 = 1,070", () => {
    const e = evaluateTrip(trip({ fuelGal: 40, fishingHours: null, roundTripNm: null }));
    expect(val(e.netToBoat).value).toBeCloseTo(1070, 9);
  });

  it("a recreational trip waits the same way (the cost is not understated)", () => {
    const e = evaluateTrip(trip({ fishingHours: null }), { audience: "recreational" });
    expect(miss(e.tripCost).missing).toEqual(["fishingHours"]);
  });
});

describe("E2: who pays for fuel is never silently assumed", () => {
  const note = "fuelShared";
  const withField = (o: Output) => val(o).assumptions.map((a) => a.field);

  it("unset: the fuel comes off before shares (the same net) and that is said", () => {
    const e = evaluateTrip(trip({ fuelShared: undefined }));
    expect(val(e.netToBoat).value).toBeCloseTo(960.1, 9);
    expect(withField(e.crewPay)).toContain(note);
    expect(e.assumptions.map((a) => a.field)).toContain(note);
    expect(withField(e.netToBoat)).toContain(note);
  });

  it("chosen either way: no assumption, and boat-paid moves the crew pay (N 1,800; crew 540; net 853)", () => {
    expect(evaluateTrip(trip()).assumptions.map((a) => a.field)).not.toContain(note);
    const e = evaluateTrip(trip({ fuelShared: false }));
    expect(e.assumptions.map((a) => a.field)).not.toContain(note);
    expect(val(e.crewPay).value).toBeCloseTo(540, 9);
    expect(val(e.netToBoat).value).toBeCloseTo(853, 9);
  });

  it("not mentioned where it cannot move the money", () => {
    const has = (o: Partial<TripScenario>) => evaluateTrip(trip({ fuelShared: undefined, ...o })).assumptions.map((a) => a.field).includes(note);
    expect(has({ crew: { kind: "ofGross", pct: 30 } })).toBe(false);
    expect(has({ crew: { kind: "dayRate", crew: 2, usdPerDay: 200, days: 1 } })).toBe(false);
    expect(has({ crew: { kind: "none" } })).toBe(false);
    expect(has({ crew: { kind: "afterExpenses", pct: 0 } })).toBe(false);
  });

  it("a draft keeps the choice unmade through storage and a link", () => {
    expect("fuelShared" in emptyDraft().inputs).toBe(false);
    expect("fuelShared" in sanitizeDraft({ inputs: {} }).inputs).toBe(false);
    expect("fuelShared" in sanitizeDraft({ inputs: { fuelShared: "yes" } }).inputs).toBe(false);
    expect(sanitizeDraft({ inputs: { fuelShared: false } }).inputs.fuelShared).toBe(false);
    expect(sanitizeDraft({ inputs: { fuelShared: true } }).inputs.fuelShared).toBe(true);
    const unset = emptyDraft();
    expect("fuelShared" in decodeShare(encodeShare(unset))!.inputs).toBe(false);
    unset.inputs.fuelShared = false;
    expect(decodeShare(encodeShare(unset))!.inputs.fuelShared).toBe(false);
    unset.inputs.fuelShared = true;
    expect(decodeShare(encodeShare(unset))!.inputs.fuelShared).toBe(true);
  });
});

describe("E3: the waterfall's range holds every bar", () => {
  it("costs past gross with net still waiting reach below zero", () => {
    expect(waterfallDomain(100, [300, null], null)).toEqual({ hi: 300, lo: -200 });
  });
  it("uses the lowest running point, not only the net", () => {
    expect(waterfallDomain(100, [300, 50], -250)).toEqual({ hi: 350, lo: -250 });
  });
  it("a term after an unknown one has no place, so it cannot lower the range", () => {
    expect(waterfallDomain(100, [null, 300], null)).toEqual({ hi: 300, lo: 0 });
    expect(waterfallDomain(null, [200, 100], null)).toEqual({ hi: 300, lo: 0 });
  });
  it("an ordinary trip keeps its zero baseline", () => {
    expect(waterfallDomain(1000, [200, 300], 500)).toEqual({ hi: 1000, lo: 0 });
  });
});

describe("E4: the sensitivity grid says what it holds still", () => {
  it("catch −30 %: R 1,400, fees 70, N 873, crew 261.90, net 561.10 with fuel, ice and gear unchanged", () => {
    const e = evaluateTrip(trip());
    const cell = val(e.sensitivity as Output).value as unknown as { priceDelta: number; catchDelta: number; netUsd: number }[];
    expect(cell.find((c) => c.priceDelta === 0 && c.catchDelta === -0.3)!.netUsd).toBeCloseTo(561.1, 9);
    expect(val(e.sensitivity as Output).formula).toMatch(/fuel, other costs and day-rate pay held as entered/);
    expect(SENSITIVITY_HELD).toMatch(/Fuel, bait, other costs and day-rate pay/);
  });
});

describe("E5: worked sums read in the worksheet's unit", () => {
  const ton = { unit: "ton", w: 1 / LB_PER_TON, p: LB_PER_TON } as const;
  const kg = { unit: "kg", w: 1 / LB_PER_KG, p: LB_PER_KG } as const;
  const squid = trip({ catch: [{ species: "Squid", lb: 2000, usdPerLb: 1 }], feesPctOfGross: null, feesUsdPerLb: 0.01, costs: [], fuelGal: 40, crew: { kind: "none" } });

  it("short tons: 1 ton × $2,000.00; $20.00/ton × 1 ton; the money is unchanged", () => {
    const e = evaluateTrip(squid, { weight: ton });
    expect(val(e.gross).formula).toBe("Gross $2,000 = 1 ton × $2,000.00");
    expect(val(e.fees).formula).toBe("Fees $20 = $20.00/ton × 1 ton");
    // net 2,000 − 20 − 200 = 1,780; break-even q at $1 − $0.01: 200 ÷ 0.99 = 202.02 lb = 0.10 ton
    expect(val(e.netToBoat).value).toBeCloseTo(1780, 9);
    expect(val(e.breakEvenCatch).value).toBeCloseTo(200 / 0.99, 9);
    expect(val(e.breakEvenCatch).formula).toBe("Break-even catch 0.1 ton of Squid: net to boat reaches $0 at $2,000.00/ton");
    expect(val(e.breakEvenPrice).formula).toBe("Break-even price $220.00/ton of Squid: net to boat reaches $0 at 1 ton");
  });

  it("kilograms", () => {
    const e = evaluateTrip(trip(), { weight: kg });
    expect(val(e.gross).formula).toBe("Gross $2,000 = 453.59 kg × $4.409");
  });

  it("pounds stay the default, word for word", () => {
    const e = evaluateTrip(squid);
    expect(val(e.gross).formula).toBe("Gross $2,000 = 2,000 lb × $1.00");
    expect(val(e.fees).formula).toBe("Fees $20 = $0.01/lb × 2,000 lb");
  });

  it("the unit never changes the arithmetic", () => {
    const lb = evaluateTrip(squid);
    const t = evaluateTrip(squid, { weight: ton });
    expect(val(t.netToBoat).value).toBe(val(lb.netToBoat).value);
    expect(val(t.breakEvenCatch).value).toBe(val(lb.breakEvenCatch).value);
  });
});

describe("E6: a break-even fuel price past the worksheet's limit", () => {
  // 10 gal; 0.7 × (1,800 − 10 f) = 50, so f = 172.857… (and the plain price is kept for the arithmetic)
  it("says only that it is above the limit", () => {
    const e = evaluateTrip(trip({ fuelGal: 10 }));
    expect(val(e.breakEvenFuelPrice).value).toBeCloseTo((1800 - 50 / 0.7) / 10, 9);
    expect(val(e.breakEvenFuelPrice).formula).toMatch(/^Break-even fuel price above \$50\/gal:/);
    expect(fuelBreakEvenWords(53.88)).toBe("above $50/gal");
  });
  it("within the limit, the price itself: 0.7 × (1,800 − 71.4 f) = 50, f = 24.21", () => {
    const e = evaluateTrip(trip());
    const v = val(e.breakEvenFuelPrice).value;
    expect(v).toBeCloseTo((1800 - 50 / 0.7) / 71.4, 9);
    expect(fuelBreakEvenWords(v)).toBe("$24.21/gal");
    expect(fuelBreakEvenWords(50)).toBe("$50.00/gal");
  });
});
