import { describe, expect, it } from "vitest";
import { computeTrip, type TripInputs } from "@/lib/tripEconomics";
import { OUTPUT_KEYS, compareScenarios, evaluateTrip, fieldLabel, type Missing, type Output, type TripEvaluation, type Value, type TripScenario } from "@/lib/tripEvaluation";
import { FISHERY_PRESETS, applyReference, presetById, presetReferences } from "@/content/fisheryPresets";
import { STARTER_COSTS } from "@/v4/coast/economics/draft";

const salmon = presetById("salmon-troll")!;
const ref = (field: string) => presetReferences(salmon).find((r) => r.field === field)!;

/** The owner's Santa Cruz state: 400 lb, $9.00/lb, EIA diesel tapped, burn references not applied, crew not chosen. */
const santaCruz = (): TripScenario => ({
  fuelUsdPerGal: 8.242,
  costs: STARTER_COSTS.map((c) => ({ ...c })),
  catch: [{ species: "Salmon", ref: "salmon", lb: 400, usdPerLb: 9 }],
  crew: { kind: null },
});

/** The decision-workspace "partial" state: round trip, speed, both burn references tapped, 10 h fishing. */
const partial = (): TripScenario => {
  let s: TripScenario = { ...santaCruz(), roundTripNm: 40, cruiseKt: 8, fishingHours: 10 };
  s = applyReference(s, ref("steamGph"));
  s = applyReference(s, ref("fishingGph"));
  return s;
};

/** "Complete": + ice $60, food $40 (shared), crew share after expenses 30 %. */
const complete = (): TripScenario => {
  const s = partial();
  s.costs = [
    { label: "Ice", usd: 60, shared: true },
    { label: "Food and supplies", usd: 40, shared: true },
    { label: "Gear and repairs (this trip)", usd: null, shared: false },
  ];
  s.crew = { kind: "afterExpenses", pct: 30 };
  return s;
};

const v = <T,>(o: Output<T>): Value<T> => {
  if (o.status !== "value") throw new Error(`expected a value, got ${JSON.stringify(o)}`);
  return o;
};
const m = <T,>(o: Output<T>): Missing => {
  if (o.status !== "missing") throw new Error(`expected missing, got ${JSON.stringify(o)}`);
  return o;
};

const BANNED = /\b(safe|go\/no-go|good day|hotspot|expected catch|price forecast|recommend\w*|worth|permitted|allowed|you may|you should|open season)\b/i;

/** Every output is one of the four states, with its reason, formula or fields. */
function wellFormed(e: TripEvaluation) {
  for (const k of OUTPUT_KEYS) {
    const o = e[k] as Output<unknown>;
    expect(o, k).toBeDefined();
    expect(["value", "missing", "unreachable", "notApplicable"], k).toContain(o.status);
    if (o.status === "value") {
      expect(o.value, k).not.toBeUndefined();
      if (typeof o.value === "number") expect(Number.isFinite(o.value), k).toBe(true);
      expect(o.formula.length, k).toBeGreaterThan(0);
      expect(o.formula, k).not.toMatch(BANNED);
      expect(o.formula, k).not.toMatch(/NaN|undefined|null|Infinity/);
      for (const a of o.assumptions) expect(a.note).not.toMatch(BANNED);
    } else if (o.status === "missing") {
      expect(o.missing.length, k).toBeGreaterThan(0);
      expect(o.reason, k).not.toMatch(BANNED);
      for (const f of o.missing) expect(fieldLabel(f)).not.toBe("");
    } else {
      expect(o.reason.length, k).toBeGreaterThan(0);
      expect(o.reason, k).not.toMatch(BANNED);
    }
  }
}

describe("the Santa Cruz salmon state (catch, price, fuel price; burn references not applied; crew not chosen)", () => {
  const e = evaluateTrip(santaCruz());

  it("shows gross $3,600 with its worked sum", () => {
    expect(v(e.gross).value).toBe(3600);
    expect(v(e.gross).formula).toBe("Gross $3,600 = 400 lb × $9.00");
    expect(v(e.costCeilingBeforeCrew).value).toBe(3600);
  });

  it("fuel quantity is missing, by field, with the trip route as an alternative", () => {
    const f = m(e.fuelGallons);
    expect(f.missing).toEqual(["fuelGal"]);
    expect(f.alternatives).toEqual([
      ["roundTripNm", "cruiseKt", "steamGph"],
      ["fishingHours", "fishingGph"],
    ]);
    expect(m(e.fuelCost).missing).toEqual(["fuelGal"]);
    expect(m(e.tripCost).missing).toEqual(["fuelGal"]);
  });

  it("the crew model is missing and nothing is assumed for it", () => {
    expect(m(e.crewPay).missing).toEqual(["crew.kind"]);
    expect(m(e.netToBoat).missing).toEqual(["fuelGal", "crew.kind"]);
    expect(e.missingFields).toEqual(["fuelGal", "crew.kind"]);
  });

  it("break-even catch waits for fuel and crew, not for the catch", () => {
    expect(m(e.breakEvenCatch).missing).toEqual(["fuelGal", "crew.kind"]);
    expect(m(e.breakEvenPrice).missing).toEqual(["fuelGal", "crew.kind"]);
    expect(m(e.sensitivity).missing).toEqual(["fuelGal", "crew.kind"]);
  });

  it("blank cost lines and fees are stated as assumptions", () => {
    expect(v(e.otherCosts).value).toBe(0);
    expect(v(e.otherCosts).assumptions.map((a) => a.field)).toEqual(["costs.0.usd", "costs.1.usd", "costs.2.usd"]);
    expect(e.assumptions.map((a) => a.field)).toEqual(expect.arrayContaining(["feesPctOfGross", "feesUsdPerLb", "costs.0.usd"]));
  });

  it("is well formed", () => wellFormed(e));
});

describe("partial states", () => {
  it("hours alone count fishing fuel and say that steaming is not counted (audit step 4)", () => {
    const s: TripScenario = { ...santaCruz(), fishingHours: 10, fishingGph: 3.6 };
    const e = evaluateTrip(s);
    expect(v(e.fuelGallons).value).toBeCloseTo(36);
    expect(v(e.fuelGallons).assumptions).toEqual([{ field: "roundTripNm", note: expect.stringMatching(/Steaming not counted/) }]);
    expect(v(e.tripCost).value).toBeCloseTo(296.712);
    expect(v(e.tripCost).assumptions.map((a) => a.field)).toContain("roundTripNm");
    // the arithmetic of record agrees
    expect(computeTrip(s as TripInputs).fuelGal).toBeCloseTo(36);
    wellFormed(e);
  });

  it("distance before speed names the cruising speed and keeps the fishing leg (audit step 5)", () => {
    let s: TripScenario = { ...santaCruz(), fishingHours: 10, roundTripNm: 40 };
    s = applyReference(applyReference(s, ref("steamGph")), ref("fishingGph"));
    const e = evaluateTrip(s);
    const f = m(e.fuelGallons);
    expect(f.missing).toEqual(["cruiseKt"]);
    expect(f.reason).toMatch(/cruising speed/);
    expect(f.reason).toMatch(/fishing leg is worked out/);
    expect(v(e.fishingGallons).value).toBeCloseTo(36);
    expect(m(e.steamingGallons).missing).toEqual(["cruiseKt"]);
    expect(m(e.tripCost).missing).toEqual(["cruiseKt"]);
    wellFormed(e);
  });

  it("a cruising speed of 0 is named, not taken", () => {
    const e = evaluateTrip({ ...santaCruz(), roundTripNm: 40, cruiseKt: 0, steamGph: 7.8 });
    expect(m(e.fuelGallons).missing).toEqual(["cruiseKt"]);
    expect(m(e.steamingGallons).reason).toMatch(/above 0/);
  });

  it("a tapped burn reference alone asks for the rest of its leg, with the other leg as an alternative", () => {
    const e = evaluateTrip(applyReference(santaCruz(), ref("steamGph")));
    const f = m(e.fuelGallons);
    expect(f.missing).toEqual(["roundTripNm", "cruiseKt"]);
    expect(f.alternatives).toEqual([["fishingHours", "fishingGph"], ["fuelGal"]]);
  });

  it("the partial workspace state: fuel 75 gal, $618, before crew $2,982; net waits only for crew", () => {
    const e = evaluateTrip(partial());
    expect(v(e.fuelGallons).value).toBeCloseTo(75);
    expect(v(e.fuelGallons).formula).toBe("Fuel 75 gal = 40 nm ÷ 8 kt × 7.8 gal/h + 10 h × 3.6 gal/h");
    expect(v(e.fuelGallons).assumptions).toEqual([]);
    expect(v(e.fuelCost).formula).toBe("Fuel cost $618 = 75 gal × $8.242/gal");
    expect(v(e.tripCost).value).toBeCloseTo(618.15);
    expect(v(e.tripCost).formula).toBe("Trip cost $618 = fuel $618 + other costs $0");
    expect(v(e.poolBeforeCrew).value).toBeCloseTo(2981.85);
    expect(m(e.netToBoat).missing).toEqual(["crew.kind"]);
    expect(m(e.breakEvenCatch).missing).toEqual(["crew.kind"]);
    expect(e.missingFields).toEqual(["crew.kind"]);
    wellFormed(e);
  });

  it("break-even catch needs the price but not the catch; break-even price needs the catch but not the price", () => {
    const s = complete();
    s.catch = [{ species: "Salmon", ref: "salmon", lb: null, usdPerLb: 9 }];
    const a = evaluateTrip(s);
    expect(v(a.breakEvenCatch).value).toBeCloseTo(718.15 / 9);
    expect(m(a.gross).missing).toEqual(["catch.0.lb"]);
    expect(m(a.breakEvenPrice).missing).toEqual(["catch.0.lb"]);
    s.catch = [{ species: "Salmon", ref: "salmon", lb: 400, usdPerLb: null }];
    const b = evaluateTrip(s);
    expect(v(b.breakEvenPrice).value).toBeCloseTo(718.15 / 400);
    expect(m(b.breakEvenCatch).missing).toEqual(["catch.0.usdPerLb"]);
  });

  it("crew pay under a day rate does not need the catch or fuel", () => {
    const e = evaluateTrip({ crew: { kind: "dayRate", crew: 2, usdPerDay: 250, days: 1 } });
    expect(v(e.crewPay).value).toBe(500);
    expect(v(e.crewPay).formula).toBe("Crew pay $500 = 2 crew × $250/day × 1 days");
    expect(v(e.perCrew).value).toBe(250);
    expect(m(e.gross).missing).toEqual(["catch.0.lb", "catch.0.usdPerLb"]);
    wellFormed(e);
  });

  it("a day rate names each blank parameter", () => {
    const e = evaluateTrip({ ...santaCruz(), fuelGal: 60, crew: { kind: "dayRate", crew: 2 } });
    expect(m(e.crewPay).missing).toEqual(["crew.usdPerDay", "crew.days"]);
    expect(m(e.netToBoat).missing).toEqual(["crew.usdPerDay", "crew.days"]);
  });

  it("share after expenses with boat-paid fuel needs no fuel for crew pay", () => {
    const e = evaluateTrip({ ...santaCruz(), fuelShared: false, crew: { kind: "afterExpenses", pct: 30 } });
    expect(v(e.crewPay).value).toBeCloseTo(1080);
    expect(m(e.netToBoat).missing).toEqual(["fuelGal"]);
  });

  it("gallons entered directly replace the trip estimate, and say so", () => {
    const s = { ...partial(), fuelGal: 60 };
    const e = evaluateTrip(s);
    expect(v(e.fuelGallons).value).toBe(60);
    expect(v(e.fuelGallons).assumptions[0].field).toBe("fuelGal");
    expect(v(e.tripCost).value).toBeCloseTo(494.52);
  });

  it("multi-line catch names the exact line that is blank", () => {
    const e = evaluateTrip({ ...santaCruz(), catch: [{ species: "A", lb: 100, usdPerLb: 2 }, { species: "B", lb: 50, usdPerLb: null }] });
    expect(m(e.gross).missing).toEqual(["catch.1.usdPerLb"]);
    expect(m(e.breakEvenCatch).missing).toEqual(["catch.1.usdPerLb", "fuelGal", "crew.kind"]);
  });

  it("no crew chosen: crew pay share parameters typed earlier are not used", () => {
    const e = evaluateTrip({ ...complete(), crew: { kind: null, pct: 30 } });
    expect(m(e.netToBoat).missing).toEqual(["crew.kind"]);
  });

  it("fully blank: everything says what it needs, nothing is undefined", () => {
    const e = evaluateTrip({});
    wellFormed(e);
    expect(e.missingFields).toEqual(["catch.0.lb", "catch.0.usdPerLb", "fuelGal", "fuelUsdPerGal", "crew.kind"]);
  });
});

describe("a complete hypothetical scenario", () => {
  const s = complete();
  const e = evaluateTrip(s);

  it("matches the worked numbers in the decision workspace", () => {
    expect(v(e.tripCost).value).toBeCloseTo(718.15);
    expect(v(e.crewPay).value).toBeCloseTo(0.3 * 2881.85);
    expect(v(e.crewPay).formula).toBe("Crew pay $865 = 30 % × $2,882 (gross − fees − shared costs)");
    expect(v(e.netToBoat).value).toBeCloseTo(2017.295);
    expect(v(e.netToBoat).formula).toBe("Net to boat $2,017 = gross $3,600 − fees $0 − trip cost $718 − crew $865");
    expect(Math.round(v(e.breakEvenCatch).value)).toBe(80);
    expect(v(e.breakEvenPrice).value).toBeCloseTo(1.795);
    expect(v(e.breakEvenFuelPrice).value).toBeCloseTo(46.667, 2);
    expect(v(e.sensitivity).value).toHaveLength(9);
    expect(v(e.fuelSensitivity).value.measure).toBe("netToBoat");
    expect(e.missingFields).toEqual([]);
    expect(m(e.perCrew).missing).toEqual(["crew.crew"]);
    wellFormed(e);
  });

  it("gives exactly computeTrip's numbers", () => {
    const r = computeTrip({ ...(s as TripInputs), crew: { kind: "afterExpenses", pct: 30 } });
    expect(v(e.netToBoat).value).toBe(r.netUsd);
    expect(v(e.crewPay).value).toBe(r.crewUsd);
    expect(v(e.tripCost).value).toBe(r.tripCostUsd);
    expect(v(e.gross).value).toBe(r.grossUsd);
    expect(v(e.fuelGallons).value).toBe(r.fuelGal);
    expect(v(e.sensitivity).value).toEqual(r.sensitivity);
  });

  it("no hired crew is a real choice: crew $0, net = gross − fees − trip cost", () => {
    const n = evaluateTrip({ ...s, crew: { kind: "none" } });
    expect(v(n.crewPay).value).toBe(0);
    expect(v(n.netToBoat).value).toBeCloseTo(3600 - 718.15);
    expect(n.perCrew.status).toBe("notApplicable");
  });
});

describe("identical to computeTrip for complete inputs (every crew model)", () => {
  const base = (over: Partial<TripInputs> = {}): TripInputs => ({
    roundTripNm: 20,
    cruiseKt: 8,
    steamGph: 7.8,
    fishingHours: 8,
    fishingGph: 4.8,
    fuelUsdPerGal: 5,
    feesPctOfGross: 2.25,
    feesUsdPerLb: 0.1,
    costs: [
      { label: "Bait", usd: 200, shared: true },
      { label: "Insurance (day)", usd: 100, shared: false },
    ],
    catch: [
      { species: "Dungeness crab", ref: "dungeness", lb: 1000, usdPerLb: 3.5 },
      { species: "Rock crab", lb: 120, usdPerLb: 2 },
    ],
    crew: { kind: "afterExpenses", pct: 25 },
    ...over,
  });
  const cases: [string, TripInputs][] = [
    ["after expenses", base()],
    ["of gross", base({ crew: { kind: "ofGross", pct: 30 } })],
    ["day rate", base({ crew: { kind: "dayRate", crew: 2, usdPerDay: 250, days: 2 } })],
    ["boat-paid fuel", base({ fuelShared: false })],
    ["a losing trip", base({ catch: [{ species: "x", lb: 20, usdPerLb: 1 }] })],
    ["gallons direct", base({ fuelGal: 60 })],
    ["fees take all", base({ feesPctOfGross: 80, crew: { kind: "ofGross", pct: 30 } })],
    ["zero steaming", base({ roundTripNm: 0 })],
  ];
  for (const [name, inp] of cases) {
    it(name, () => {
      const r = computeTrip(inp);
      const e = evaluateTrip(inp);
      const val = (o: Output<unknown>) => (o.status === "value" ? o.value : null);
      expect(val(e.fuelGallons)).toBe(r.fuelGal);
      expect(val(e.fuelCost)).toBe(r.fuelUsd);
      expect(val(e.tripCost)).toBe(r.tripCostUsd);
      expect(val(e.gross)).toBe(r.grossUsd);
      expect(val(e.fees)).toBe(r.feesUsd);
      expect(val(e.crewPay)).toBeCloseTo(r.crewUsd!, 9);
      expect(val(e.netToBoat)).toBe(r.netUsd);
      expect(val(e.breakEvenCatch)).toBe(r.breakEvenLb.value ?? null);
      expect(val(e.breakEvenPrice)).toBe(r.breakEvenUsdPerLb.value ?? null);
      expect(val(e.breakEvenFuelPrice)).toBe(r.breakEvenFuelUsdPerGal.value ?? null);
      expect(val(e.sensitivity)).toEqual(r.sensitivity);
      expect(e.warnings).toEqual(r.warnings);
      if (r.breakEvenLb.note === "none") expect(e.breakEvenCatch.status).toBe("unreachable");
      wellFormed(e);
    });
  }
});

describe("recreational: cost only", () => {
  const s: TripScenario = { ...partial(), anglers: 3 };
  const e = evaluateTrip(s, { audience: "recreational" });

  it("works out fuel, trip cost and a per-person split", () => {
    expect(v(e.fuelGallons).value).toBeCloseTo(75);
    expect(v(e.tripCost).value).toBeCloseTo(618.15);
    expect(v(e.perPerson).value).toBeCloseTo(206.05);
    expect(v(e.perPerson).formula).toBe("Per person $206 = trip cost $618 ÷ 3");
    expect(v(e.fuelSensitivity).value).toMatchObject({ measure: "tripCost" });
    expect(v(e.fuelSensitivity).value.up).toBeCloseTo(618.15 * 1.2);
  });

  it("has no price, gross, crew, net or break-even, and asks for none of them", () => {
    for (const k of ["gross", "fees", "crewPay", "perCrew", "netToBoat", "breakEvenCatch", "breakEvenPrice", "breakEvenFuelPrice", "sensitivity", "costCeilingBeforeCrew", "poolBeforeCrew"] as const) {
      expect(e[k].status, k).toBe("notApplicable");
    }
    expect(e.missingFields).toEqual([]);
    expect(e.missingFields.some((f) => f.startsWith("catch") || f.startsWith("crew"))).toBe(false);
    wellFormed(e);
  });

  it("without fuel, asks only for fuel and its price; the split is optional", () => {
    const r = evaluateTrip({ anglers: null }, { audience: "recreational" });
    expect(r.missingFields).toEqual(["fuelGal", "fuelUsdPerGal"]);
    expect(m(r.perPerson).missing).toEqual(["fuelGal", "fuelUsdPerGal", "anglers"]);
  });
});

describe("presets offer references; nothing is pre-filled (D1, D2)", () => {
  it("salmon troll offers its two burn rates with their source, and no crew share", () => {
    const refs = presetReferences(salmon);
    expect(refs.map((r) => [r.field, r.value])).toEqual([
      ["steamGph", 7.8],
      ["fishingGph", 3.6],
    ]);
    expect(refs[0].sourceShort).toMatch(/NWFSC-121, Table 31/);
    expect(refs[0].note).toBe("Fleet average then, not a norm");
  });

  it("a crew share reference is offered only for the model it describes", () => {
    const crab = presetById("dungeness")!;
    expect(presetReferences(crab).filter((r) => r.field === "crew.pct")).toHaveLength(2);
    expect(presetReferences(crab, { crewKind: null }).filter((r) => r.field === "crew.pct")).toHaveLength(0);
    expect(presetReferences(crab, { crewKind: "ofGross" }).filter((r) => r.field === "crew.pct")).toHaveLength(0);
    expect(presetReferences(crab, { crewKind: "afterExpenses" }).filter((r) => r.field === "crew.pct")).toHaveLength(2);
  });

  it("applying a reference changes that one field only and does not choose a crew model", () => {
    const crab = presetById("dungeness")!;
    const pct = presetReferences(crab).find((r) => r.field === "crew.pct")!;
    const before: TripScenario = { fuelUsdPerGal: 5 };
    const after = applyReference(before, pct);
    expect(after.crew).toEqual({ kind: null, pct: 28.1 });
    expect(before.crew).toBeUndefined();
    expect(evaluateTrip(after).crewPay).toMatchObject({ status: "missing", missing: ["crew.kind"] });
  });

  it("no preset holds a catch or a price", () => {
    for (const p of FISHERY_PRESETS) for (const r of presetReferences(p)) expect(["steamGph", "fishingGph", "crew.pct", "feesPctOfGross"]).toContain(r.field);
  });
});

describe("comparing two scenarios", () => {
  it("lists only the inputs that differ and the result deltas", () => {
    const a = complete();
    const b = complete();
    b.catch = [{ species: "Salmon", ref: "salmon", lb: 250, usdPerLb: 9 }];
    const c = compareScenarios(a, b);
    expect(c.inputs).toEqual([{ field: "catch.0.lb", a: 400, b: 250 }]);
    const d = Object.fromEntries(c.outputs.map((o) => [o.output, o]));
    expect(d.gross).toMatchObject({ a: 3600, b: 2250, delta: -1350 });
    expect(d.tripCost.delta).toBe(0);
    expect(d.netToBoat.delta).toBeCloseTo(-1350 * 0.7);
    expect(d.breakEvenCatch.delta).toBe(0);
    expect(d.breakEvenPrice.delta).toBeGreaterThan(0);
  });

  it("a result one side cannot compute has no delta", () => {
    const c = compareScenarios(santaCruz(), complete());
    const d = Object.fromEntries(c.outputs.map((o) => [o.output, o]));
    expect(d.netToBoat).toMatchObject({ a: null, delta: null });
    expect(d.gross.delta).toBe(0);
    expect(c.inputs.map((i) => i.field)).toEqual(
      expect.arrayContaining(["roundTripNm", "cruiseKt", "steamGph", "fishingHours", "fishingGph", "costs.0.usd", "costs.1.usd", "crew.kind", "crew.pct"]),
    );
    expect(c.inputs.map((i) => i.field)).not.toContain("fuelUsdPerGal");
  });

  it("recreational comparison leaves out what does not apply", () => {
    const c = compareScenarios({ fuelGal: 50, fuelUsdPerGal: 5 }, { fuelGal: 60, fuelUsdPerGal: 5 }, { audience: "recreational" });
    expect(c.outputs.map((o) => o.output)).not.toContain("netToBoat");
    expect(c.outputs.find((o) => o.output === "tripCost")!.delta).toBe(50);
  });
});

describe("robustness", () => {
  it("never an undefined output across many partial scenarios", () => {
    const fields: (keyof TripScenario)[] = ["roundTripNm", "cruiseKt", "steamGph", "fishingHours", "fishingGph", "fuelGal", "fuelUsdPerGal", "feesPctOfGross", "feesUsdPerLb"];
    const vals = [null, 0, 7.5, 40];
    const kinds = [null, "none", "afterExpenses", "ofGross", "dayRate"] as const;
    let seed = 7;
    const rnd = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
    const pick = <T,>(xs: readonly T[]) => xs[Math.floor(rnd() * xs.length)];
    for (let t = 0; t < 400; t++) {
      const s: TripScenario = { catch: [], costs: [] };
      for (const f of fields) (s as Record<string, unknown>)[f] = pick(vals);
      s.catch = Array.from({ length: Math.floor(rnd() * 3) }, (_, i) => ({ species: `s${i}`, lb: pick(vals), usdPerLb: pick(vals) }));
      s.costs = [{ label: "c", usd: pick(vals), shared: rnd() > 0.5 }];
      s.fuelShared = rnd() > 0.3;
      s.crew = { kind: pick(kinds), pct: pick(vals), crew: pick([null, 0, 2]), usdPerDay: pick(vals), days: pick([null, 1]) };
      s.anglers = pick([null, 0, 3]);
      for (const audience of ["commercial", "recreational"] as const) {
        const e = evaluateTrip(s, { audience });
        wellFormed(e);
        // a value is never shown while the page-wide list says its inputs are still needed
        if (e.netToBoat.status === "value") expect(e.missingFields).toEqual([]);
        // net never computes while computeTrip with the same crew would refuse it
        if (e.netToBoat.status === "value" && s.crew.kind !== "none") expect(computeTrip(s as TripInputs).netUsd).toBe(e.netToBoat.value);
      }
    }
  });

  it("garbage values are cleaned like computeTrip does", () => {
    const e = evaluateTrip({ fuelGal: -5, fuelUsdPerGal: Number.NaN, catch: [{ species: "x", lb: Infinity, usdPerLb: 3 }], crew: { kind: "bogus" as never } });
    expect(m(e.fuelGallons).missing).toEqual(["fuelGal"]);
    expect(m(e.gross).missing).toEqual(["catch.0.lb"]);
    expect(m(e.crewPay).missing).toEqual(["crew.kind"]);
    wellFormed(e);
  });
});
