import { describe, expect, it } from "vitest";
import { breakEvenCatchText, evaluateWorkspace, fromDraft, money, NET_IS_NOT } from "@/nx/coast/workspace";
import { emptyWorkspace, toDraft } from "@/nx/coast/workspace";
import { NOT_RECORDED, planVsActual, signedMoney } from "@/nx/trips/outcome";
import type { Outcome } from "@/nx/trips/types";
import { crab, ownerTrip, sport } from "./tripsFixtures";

const outcome = (o: Partial<Outcome> = {}): Outcome => ({
  departedOn: "2026-11-20",
  returnedOn: "2026-11-20",
  landed: [{ label: "Dungeness crab", lb: 350, usdPerLb: 10 }],
  fuelGal: 35,
  fuelUsdPerGal: 8,
  costs: [
    { label: "Bait", usd: 80 },
    { label: "Ice", usd: 40 },
  ],
  crewPaidUsd: 600,
  notes: "",
  conditions: "",
  landingRef: null,
  recordedAt: "2026-11-21T02:00:00.000Z",
  updatedAt: "2026-11-21T02:00:00.000Z",
  ...o,
});

const v = (f: { value: number | null }) => f.value;

describe("the plan, through the engine (the owner's regression)", () => {
  it("$3,600 gross, $347.20 operating, $650.56 crew, $2,602.24 net, 39 lb break-even", () => {
    const pv = planVsActual(ownerTrip());
    expect(money(v(pv.planned.gross)!)).toBe("$3,600.00");
    expect(money(v(pv.planned.operating)!)).toBe("$347.20");
    expect(money(v(pv.planned.crew)!)).toBe("$650.56");
    expect(money(v(pv.planned.result)!)).toBe("$2,602.24");
    // the same draft read back by the worksheet gives the same break-even
    const ws = fromDraft(ownerTrip().economics!, "dungeness_commercial");
    const be = evaluateWorkspace(ws, crab).out.breakEvenCatch;
    expect(be.status === "value" && breakEvenCatchText(be.value, "lb")).toBe("39 lb");
    expect(pv.measure).toBe("netToBoat");
    expect(pv.notes).toContain(NET_IS_NOT);
  });

  it("no outcome: actual is 'not recorded', never 0, and no parts are drawn", () => {
    const pv = planVsActual(ownerTrip());
    expect(pv.actual.gross).toEqual({ value: null, why: NOT_RECORDED });
    expect(pv.actual.result.value).toBeNull();
    expect(pv.difference).toBeNull();
    expect(pv.parts).toEqual([]);
  });
});

describe("actual against planned", () => {
  const pv = planVsActual(ownerTrip({ state: "completed", outcome: outcome() }));

  it("works the actual figures from what was recorded", () => {
    // gross 350 × 10 = 3500; fees 0 % (the plan's terms); fuel 35 × 8 = 280; supplies 120; crew 600
    expect(v(pv.actual.gross)).toBe(3500);
    expect(v(pv.actual.fees)).toBe(0);
    expect(v(pv.actual.fuel)).toBe(280);
    expect(v(pv.actual.supplies)).toBe(120);
    expect(v(pv.actual.operating)).toBe(400);
    expect(v(pv.actual.crew)).toBe(600);
    expect(v(pv.actual.result)).toBeCloseTo(2500, 9);
    expect(pv.difference).toBeCloseTo(2500 - 2602.24, 9);
  });

  it("attributes the difference to catch, price, fuel, supplies, crew and fees, and the parts add up to it", () => {
    const by = Object.fromEntries(pv.parts.map((p) => [p.id, p.effect]));
    // volume at the planned price: (350 − 400) × 9 = −450; price at the actual volume: (10 − 9) × 350 = +350
    expect(by.catch).toBe(-450);
    expect(by.price).toBe(350);
    expect(by.fuel).toBeCloseTo(-(280 - 247.2), 9);
    expect(by.supplies).toBe(-20);
    expect(by.crew).toBeCloseTo(50.56, 9);
    expect(by.fees).toBe(0);
    const total = pv.parts.reduce((a, p) => a + p.effect!, 0);
    expect(total).toBeCloseTo(pv.difference!, 2);
    expect(pv.explained).toBe(true);
  });

  it("the parts add up for many outcomes, multi-line catches and fees included", () => {
    const withFees = toDraft({ ...fromDraft(ownerTrip().economics!, "dungeness_commercial"), feesPct: "3", feesPerUnit: "0.1" }, crab);
    for (const [lb1, p1, lb2, p2, gal, gp, crew] of [
      [100, 5, 50, 12, 10, 6, 0],
      [800, 11.5, 0, 0, 60, 7.25, 1400],
      [400, 9, 20, 3, 30, 8.24, 650.56],
    ]) {
      const t = ownerTrip({ economics: withFees, state: "completed", outcome: outcome({ landed: [{ label: "a", lb: lb1, usdPerLb: p1 }, { label: "b", lb: lb2, usdPerLb: p2 }], fuelGal: gal, fuelUsdPerGal: gp, crewPaidUsd: crew }) });
      const r = planVsActual(t);
      expect(r.explained, JSON.stringify([lb1, p1])).toBe(true);
      expect(r.parts.reduce((a, p) => a + p.effect!, 0)).toBeCloseTo(r.difference!, 1);
    }
  });

  it("a missing figure leaves its part 'not recorded' and the total unexplained, never a 0", () => {
    const r = planVsActual(ownerTrip({ state: "completed", outcome: outcome({ fuelUsdPerGal: null, crewPaidUsd: null }) }));
    const by = Object.fromEntries(r.parts.map((p) => [p.id, p]));
    expect(by.fuel.effect).toBeNull();
    expect(by.fuel.why).toBe(NOT_RECORDED);
    expect(by.crew.effect).toBeNull();
    expect(by.catch.effect).toBe(-450);
    expect(r.actual.result.value).toBeNull();
    expect(r.difference).toBeNull();
    expect(r.explained).toBe(false);
  });

  it("no costs recorded is 'not recorded', not $0 of costs", () => {
    const r = planVsActual(ownerTrip({ state: "completed", outcome: outcome({ costs: [] }) }));
    expect(r.actual.supplies).toEqual({ value: null, why: NOT_RECORDED });
  });

  it("never says profit", () => {
    expect(JSON.stringify(pv)).not.toMatch(/profit(?!\s+or take-home)/i);
    expect(signedMoney(-102.24)).toBe("−$102.24");
    expect(signedMoney(5)).toBe("+$5.00");
  });
});

describe("recreational: costs only", () => {
  const ws = { ...emptyWorkspace(sport), fuelGal: "10", fuelPrice: "6", costs: [{ label: "Bait and ice", usd: "30", payer: null }], anglers: "2" };
  const trip = ownerTrip({ activity: "recreational", fishery: "dungeness_recreational", economics: toDraft(ws, sport), state: "completed", outcome: outcome({ landed: [{ label: "Dungeness crab", lb: 20, usdPerLb: null }], fuelGal: 12, fuelUsdPerGal: 6, costs: [{ label: "Bait", usd: 25 }], crewPaidUsd: null }) });
  const pv = planVsActual(trip);

  it("compares the trip cost; no catch value, gross, crew or net anywhere", () => {
    expect(pv.measure).toBe("tripCost");
    expect(v(pv.planned.result)).toBe(90);
    expect(v(pv.actual.result)).toBe(97);
    expect(pv.difference).toBe(7);
    expect(pv.planned.gross.value).toBeNull();
    expect(pv.actual.gross.value).toBeNull();
    expect(pv.actual.crew.value).toBeNull();
    expect(pv.parts.map((p) => p.id)).toEqual(["fuel", "supplies"]);
    expect(pv.parts.map((p) => p.effect)).toEqual([12, -5]);
    expect(pv.explained).toBe(true);
    expect(pv.notes.join(" ")).toMatch(/not sold/);
  });
});

describe("the outcome form's model", async () => {
  const { formFor, readForm } = await import("@/nx/trips/ui/OutcomeForm");
  it("a new form names the plan's lines but fills in no number", () => {
    const f = formFor(ownerTrip());
    expect(f.landed).toEqual([{ label: "Dungeness crab", lb: "", price: "" }]);
    expect(f.costs).toEqual([{ label: "Bait and ice", usd: "" }]);
    expect([f.fuelGal, f.fuelPrice, f.crew]).toEqual(["", "", ""]);
  });
  it("blanks read back as not recorded (null), never 0; a typo is an error, not a figure", () => {
    const r = readForm(formFor(ownerTrip()), false);
    expect(r.outcome).toMatchObject({ landed: [], costs: [], fuelGal: null, fuelUsdPerGal: null, crewPaidUsd: null });
    const bad = readForm({ ...formFor(ownerTrip()), fuelGal: "lots", landed: [{ label: "Crab", lb: "350", price: "10" }] }, false);
    expect(bad.errors.fuelGal).toBe("Enter a number");
    expect(bad.outcome.landed).toEqual([{ label: "Crab", lb: 350, usdPerLb: 10 }]);
  });
  it("recreational: no price and no crew pay are kept", () => {
    const r = readForm({ ...formFor(ownerTrip()), landed: [{ label: "Crab", lb: "20", price: "9" }], crew: "100" }, true);
    expect(r.outcome.landed[0].usdPerLb).toBeNull();
    expect(r.outcome.crewPaidUsd).toBeNull();
  });
  it("a recorded outcome opens again with its own figures", () => {
    const o = formFor(ownerTrip({ outcome: outcome() }));
    expect(o.landed[0]).toEqual({ label: "Dungeness crab", lb: "350", price: "10" });
  });
});
