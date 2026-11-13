import { draftInputs } from "@/lib/tripEconomicsStore";
import { evaluateTrip, type Output, type TripEvaluation, type TripScenario } from "@/lib/tripEvaluation";
import { NET_IS_NOT } from "@/nx/coast/workspace";
import type { Outcome, Trip } from "./types";

/**
 * Planned against actual (v15): what the reader planned (their saved economics) beside what they recorded after the
 * trip, and where the difference in net to boat came from. Every figure goes through the engine of record
 * (src/lib/tripEvaluation.ts) except crew pay, which the reader records as one paid total (no crew model applies to
 * money already paid): it is subtracted as recorded.
 *
 * Honesty rules:
 * - a figure the reader did not record is "not recorded", never 0, and nothing that rests on it is worked out;
 * - fees are never recorded after the trip: the actual fees are the planned fee terms (percent of gross and per pound)
 *   applied to what was actually landed and sold, and say so;
 * - a recreational trip is cost only: fuel and supplies, no catch value, gross, crew or net;
 * - net to boat is never called profit; every view that shows it carries `NET_IS_NOT`.
 */

export const NOT_RECORDED = "not recorded";
export const NOT_PLANNED = "not in the plan";

export type Fig = { value: number } | { value: null; why: string };
const val = (v: number): Fig => ({ value: v });
const gap = (why: string): Fig => ({ value: null, why });
const known = (f: Fig): f is { value: number } => f.value !== null;
const whyOf = (f: Fig): string | null => (f.value === null ? f.why : null);
const fromOutput = (o: Output, why: string): Fig => (o.status === "value" ? val(o.value) : gap(o.status === "notApplicable" ? "does not apply" : why));

export type Side = {
  /** pounds landed (planned: the catch lines), and the average price per pound over them */
  lb: Fig;
  usdPerLb: Fig;
  gross: Fig;
  fees: Fig;
  fuelGal: Fig;
  fuel: Fig;
  /** bait, ice and other costs */
  supplies: Fig;
  /** fuel + supplies */
  operating: Fig;
  crew: Fig;
  /** commercial: net to boat; recreational: trip cost */
  result: Fig;
};

export type PartId = "catch" | "price" | "fuel" | "supplies" | "crew" | "fees";
export type Part = {
  id: PartId;
  label: string;
  /** dollars this part moved the result by (commercial: + raised net to boat; recreational: + raised the trip cost) */
  effect: number | null;
  why: string | null;
};

export type PlanVsActual = {
  audience: "commercial" | "recreational";
  /** what `result` is */
  measure: "netToBoat" | "tripCost";
  planned: Side;
  actual: Side;
  /** actual − planned result; null while either side has a gap */
  difference: number | null;
  /** in the order a trip's money runs; parts only when the trip has a recorded outcome and a plan */
  parts: Part[];
  /** every part has an effect and they add up to `difference` (to the cent) */
  explained: boolean;
  /** planned economics exist / an outcome was recorded */
  hasPlan: boolean;
  hasOutcome: boolean;
  /** what each figure is and is not, said once */
  notes: string[];
};

const LABEL: Record<PartId, string> = { catch: "Catch landed", price: "Price received", fuel: "Fuel", supplies: "Bait, ice and other", crew: "Crew pay", fees: "Fees" };

const sum = (xs: Array<number | null>): number | null => (xs.length && xs.every((x) => x !== null) ? (xs as number[]).reduce((a, b) => a + b, 0) : null);

function plannedSide(trip: Trip, rec: boolean): { side: Side; ev: TripEvaluation | null } {
  if (!trip.economics) {
    const g = gap(NOT_PLANNED);
    return { side: { lb: g, usdPerLb: g, gross: g, fees: g, fuelGal: g, fuel: g, supplies: g, operating: g, crew: g, result: g }, ev: null };
  }
  const inputs = draftInputs(trip.economics);
  const ev = evaluateTrip(inputs, { audience: rec ? "recreational" : "commercial" });
  const lb = rec ? null : sum((inputs.catch ?? []).map((c) => c.lb ?? null));
  const gross = fromOutput(ev.gross, NOT_PLANNED);
  const na = gap("does not apply");
  return {
    ev,
    side: {
      lb: rec ? na : lb === null ? gap(NOT_PLANNED) : val(lb),
      usdPerLb: rec ? na : known(gross) && lb ? val(gross.value / lb) : gap(NOT_PLANNED),
      gross: rec ? na : gross,
      fees: rec ? na : fromOutput(ev.fees, NOT_PLANNED),
      fuelGal: fromOutput(ev.fuelGallons, NOT_PLANNED),
      fuel: fromOutput(ev.fuelCost, NOT_PLANNED),
      supplies: fromOutput(ev.otherCosts, NOT_PLANNED),
      operating: fromOutput(ev.tripCost, NOT_PLANNED),
      crew: rec ? na : fromOutput(ev.crewPay, NOT_PLANNED),
      result: rec ? fromOutput(ev.tripCost, NOT_PLANNED) : fromOutput(ev.netToBoat, NOT_PLANNED),
    },
  };
}

/**
 * The recorded outcome through the engine: the landed lines as the catch, the fuel bought, every cost as the boat's,
 * the planned fee terms, and no crew model (crew pay is the recorded total, subtracted after).
 */
function actualSide(trip: Trip, o: Outcome | null, rec: boolean): Side {
  const nr = gap(NOT_RECORDED);
  const na = gap("does not apply");
  if (!o) return { lb: rec ? na : nr, usdPerLb: rec ? na : nr, gross: rec ? na : nr, fees: rec ? na : nr, fuelGal: nr, fuel: nr, supplies: nr, operating: nr, crew: rec ? na : nr, result: nr };
  const plan = trip.economics?.inputs;
  const scenario: TripScenario = {
    catch: rec ? [] : o.landed.map((l) => ({ species: l.label, lb: l.lb, usdPerLb: l.usdPerLb })),
    fuelGal: o.fuelGal,
    fuelUsdPerGal: o.fuelUsdPerGal,
    fuelShared: false,
    costs: o.costs.map((c) => ({ label: c.label, usd: c.usd, shared: false })),
    feesPctOfGross: plan?.feesPctOfGross ?? null,
    feesUsdPerLb: plan?.feesUsdPerLb ?? null,
    crew: { kind: "none" },
  };
  const ev = evaluateTrip(scenario, { audience: rec ? "recreational" : "commercial" });
  const lb = sum(o.landed.map((l) => l.lb));
  const fuel = o.fuelGal === null || o.fuelUsdPerGal === null ? nr : fromOutput(ev.fuelCost, NOT_RECORDED);
  const supplies = o.costs.length ? fromOutput(ev.otherCosts, NOT_RECORDED) : nr;
  const operating = known(fuel) && known(supplies) ? val(fuel.value + supplies.value) : nr;
  if (rec) return { lb: na, usdPerLb: na, gross: na, fees: na, fuelGal: o.fuelGal === null ? nr : val(o.fuelGal), fuel, supplies, operating, crew: na, result: operating };
  const gross = o.landed.length ? fromOutput(ev.gross, NOT_RECORDED) : nr;
  const fees = !known(gross) ? nr : plan?.feesPctOfGross == null ? gap("no fee terms in the plan") : fromOutput(ev.fees, NOT_RECORDED);
  const crew = o.crewPaidUsd === null ? nr : val(o.crewPaidUsd);
  // net to boat: the engine's gross − fees − trip cost (no crew model), then the crew pay as recorded
  const result = known(gross) && known(fees) && known(operating) && known(crew) && ev.netToBoat.status === "value" ? val(ev.netToBoat.value - crew.value) : nr;
  return {
    lb: o.landed.length && lb !== null ? val(lb) : nr,
    usdPerLb: known(gross) && lb ? val(gross.value / lb) : nr,
    gross,
    fees,
    fuelGal: o.fuelGal === null ? nr : val(o.fuelGal),
    fuel,
    supplies,
    operating,
    crew,
    result,
  };
}

const round2 = (v: number) => Math.round(v * 100) / 100 || 0;

/**
 * Where the difference came from. A standard price-volume decomposition of gross:
 *   catch (volume) effect = (actual lb − planned lb) × planned average price
 *   price effect          = (actual average price − planned average price) × actual lb
 * which add up exactly to actual gross − planned gross (average prices are gross ÷ pounds over all lines). Every cost
 * line then moves the result by its own difference: fuel, supplies (bait, ice, other), crew pay and fees, each
 * subtracted. Crew pay and fees that are a share of gross move with the catch; they are shown as their own lines, not
 * folded into catch or price. For a recreational trip the result is the trip cost, so a part is what that cost line
 * added.
 */
function attribute(p: Side, a: Side, rec: boolean): Part[] {
  const part = (id: PartId, effect: number | null, why: string | null = null): Part => ({ id, label: LABEL[id], effect: effect === null ? null : round2(effect), why: effect === null ? (why ?? NOT_RECORDED) : null });
  const cost = (id: PartId, pl: Fig, ac: Fig, sign: 1 | -1) => (known(pl) && known(ac) ? part(id, sign * (ac.value - pl.value)) : part(id, null, whyOf(ac) ?? whyOf(pl)));
  if (rec) return [cost("fuel", p.fuel, a.fuel, 1), cost("supplies", p.supplies, a.supplies, 1)];
  const volume = known(a.lb) && known(p.lb) && known(p.usdPerLb) ? (a.lb.value - p.lb.value) * p.usdPerLb.value : null;
  const price = known(a.usdPerLb) && known(p.usdPerLb) && known(a.lb) ? (a.usdPerLb.value - p.usdPerLb.value) * a.lb.value : null;
  return [
    part("catch", volume, !known(a.lb) ? a.lb.why : !known(p.lb) || !known(p.usdPerLb) ? NOT_PLANNED : null),
    part("price", price, !known(a.usdPerLb) ? a.usdPerLb.why : NOT_PLANNED),
    cost("fuel", p.fuel, a.fuel, -1),
    cost("supplies", p.supplies, a.supplies, -1),
    cost("crew", p.crew, a.crew, -1),
    cost("fees", p.fees, a.fees, -1),
  ];
}

/** The trip's plan against its recorded outcome, through the engine. Never a verdict, never "profit". */
export function planVsActual(trip: Trip): PlanVsActual {
  const rec = trip.activity === "recreational";
  const { side: planned } = plannedSide(trip, rec);
  const actual = actualSide(trip, trip.outcome, rec);
  const difference = known(planned.result) && known(actual.result) ? round2(actual.result.value - planned.result.value) : null;
  const parts = trip.outcome && trip.economics ? attribute(planned, actual, rec) : [];
  const total = sum(parts.map((x) => x.effect));
  const explained = difference !== null && total !== null && Math.abs(total - difference) < 0.015;
  const notes = rec
    ? ["A recreational catch is not sold: this compares what the trip cost, fuel and supplies, with no catch value, gross or net."]
    : [NET_IS_NOT, "Fees after the trip are your planned fee terms applied to what you landed and sold.", "Crew pay is what you recorded paying."];
  return { audience: rec ? "recreational" : "commercial", measure: rec ? "tripCost" : "netToBoat", planned, actual, difference, parts, explained, hasPlan: !!trip.economics, hasOutcome: !!trip.outcome, notes };
}

/** "+$120.00" / "−$80.50": an effect as a signed amount. */
export function signedMoney(v: number): string {
  const r = Math.abs(v) < 0.005 ? 0 : v;
  return `${r > 0 ? "+" : r < 0 ? "−" : ""}$${Math.abs(r).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

const CSV_HEAD = ["name", "state", "fishery", "port", "departure", "planned_lb", "landed_lb", "planned_gross", "actual_gross", "planned_trip_costs", "actual_trip_costs", "planned_crew", "actual_crew", "planned_result", "actual_result", "result_is", "outcome_updated"];

/** One CSV row per trip, planned against actual. Blank cells are figures not recorded (never 0). */
export function tripsCsv(trips: readonly Trip[], portName: (key: string) => string | null = () => null, cell: (v: string | number | null) => string): string {
  const rows = trips.map((t) => {
    const pv = planVsActual(t);
    const f = (x: Fig) => x.value;
    return [t.name, t.state, t.fishery, portName(t.placeKey) ?? t.placeKey, t.depart?.date ?? null, f(pv.planned.lb), f(pv.actual.lb), f(pv.planned.gross), f(pv.actual.gross), f(pv.planned.operating), f(pv.actual.operating), f(pv.planned.crew), f(pv.actual.crew), f(pv.planned.result), f(pv.actual.result), pv.measure === "netToBoat" ? "net to boat (not profit)" : "trip cost", t.outcome?.updatedAt ?? null].map(cell).join(",");
  });
  return [CSV_HEAD.join(","), ...rows].join("\r\n") + "\r\n";
}
