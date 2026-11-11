import { parseField, type LimitKind, type SensitivityCell } from "@/lib/tripEconomics";
import { evaluateTrip, type FieldKey, type Output, type TripEvaluation, type TripScenario, type WeightUnit } from "@/lib/tripEvaluation";
import type { Draft } from "@/lib/tripEconomicsStore";
import { layoutWaterfall, type WaterfallStep } from "@/nx/ds/worksheet/waterfallLayout";
import { coastFishery, LB_PER_UNIT, type CoastFishery } from "./fisheries";

/**
 * My Coast's trip worksheet: the reader's typed text in, the outputs of `evaluateTrip` (the engine of record,
 * src/lib/tripEvaluation.ts) out. No arithmetic of its own beyond units (a squid boat types tons). What this
 * layer adds is the page's honesty rules (research E, T45–T47):
 *
 * - nothing is assumed: a blank field is a gap, never 0. The engine counts a blank fee or cost line as none
 *   (and says so as an assumption); here those are fields the outputs wait for, so "enter 0" is the reader's
 *   statement, not the page's;
 * - who pays fuel is a required choice wherever it moves the money (crew paid a share after costs, E2);
 * - every output either has a value or names the fields it waits for, in the reader's words.
 */

export type CrewKind = "afterExpenses" | "ofGross" | "dayRate";
export type FuelPayer = "shared" | "boat";

export type TextField = "catch" | "price" | "fuelGal" | "fuelPrice" | "supplies" | "fees" | "crew" | "share" | "dayRate" | "days";
export type FieldId = TextField | "crewKind" | "fuelPayer";

export type CoastForm = Record<TextField, string> & {
  crewKind: CrewKind | null;
  fuelPayer: FuelPayer | null;
  /** fields whose value the reader took from a sourced reference (they stay labelled) */
  taken: Partial<Record<"price" | "fuelPrice" | "share", true>>;
};

export const EMPTY_FORM: CoastForm = { catch: "", price: "", fuelGal: "", fuelPrice: "", supplies: "", fees: "", crew: "", share: "", dayRate: "", days: "", crewKind: null, fuelPayer: null, taken: {} };

/** Field order on the page; gaps are always listed in it. */
export const FIELD_ORDER: FieldId[] = ["catch", "price", "fuelGal", "fuelPrice", "supplies", "fees", "crew", "crewKind", "share", "dayRate", "days", "fuelPayer"];

/** The words a gap is said with: "waiting for price". */
export function fieldWords(id: FieldId, f: CoastFishery | null): string {
  switch (id) {
    case "catch":
      return "catch";
    case "price":
      return "price";
    case "fuelGal":
      return "fuel used";
    case "fuelPrice":
      return "fuel price";
    case "supplies":
      return (f?.supplies ?? "Bait and ice").toLowerCase();
    case "fees":
      return "fees";
    case "crew":
      return "crew";
    case "crewKind":
      return "how crew is paid";
    case "share":
      return "crew share";
    case "dayRate":
      return "day rate";
    case "days":
      return "days at sea";
    case "fuelPayer":
      return "who pays fuel";
  }
}

const LIMIT: Record<TextField, LimitKind> = { catch: "lb", price: "usdPerLb", fuelGal: "gal", fuelPrice: "usdPerGal", supplies: "usd", fees: "pct", crew: "people", share: "pct", dayRate: "usdPerDay", days: "days" };

export type Parsed = { value: number | null; error: string | null; clamped: boolean };

/** One typed field read as a number in the worksheet's own unit. A squid price per ton is read against the money limit, not $/lb. */
export function readField(form: CoastForm, id: TextField, f: CoastFishery | null): Parsed {
  const kind: LimitKind = id === "price" && f?.unit === "ton" ? "usd" : LIMIT[id];
  return parseField(form[id], kind);
}

export type Numbers = Record<TextField, number | null>;
export function numbers(form: CoastForm, f: CoastFishery | null): { n: Numbers; errors: Partial<Record<TextField, string>>; clamped: TextField[] } {
  const n = {} as Numbers;
  const errors: Partial<Record<TextField, string>> = {};
  const clamped: TextField[] = [];
  for (const id of Object.keys(LIMIT) as TextField[]) {
    const p = readField(form, id, f);
    n[id] = p.value;
    if (p.error) errors[id] = p.error;
    if (p.clamped) clamped.push(id);
  }
  return { n, errors, clamped };
}

/** Catch and price moved by the sensitivity grid's factors (1 = as entered). */
export type Scale = { catch: number; price: number };
const AS_ENTERED: Scale = { catch: 1, price: 1 };

/** The engine's scenario for this form. Weight is typed in the fishery's unit and handed over in pounds. */
export function toScenario(form: CoastForm, f: CoastFishery | null, scale: Scale = AS_ENTERED): { scenario: TripScenario; weight: WeightUnit } {
  const { n } = numbers(form, f);
  const per = LB_PER_UNIT[f?.unit ?? "lb"];
  const none = n.crew === 0;
  const fuelShared = form.fuelPayer === null ? undefined : form.fuelPayer === "shared";
  const scenario: TripScenario = {
    fuelGal: n.fuelGal,
    fuelUsdPerGal: n.fuelPrice,
    ...(fuelShared === undefined ? {} : { fuelShared }),
    feesPctOfGross: n.fees,
    feesUsdPerLb: null,
    // bait and ice are paid the way fuel is (one choice, stated in the assumptions)
    costs: [{ label: f?.supplies ?? "Bait and ice", usd: n.supplies, shared: fuelShared ?? true }],
    catch: [{ species: f?.label ?? "Catch", ref: f?.priceKey ?? null, lb: n.catch === null ? null : n.catch * per * scale.catch, usdPerLb: n.price === null ? null : (n.price / per) * scale.price }],
    crew: none ? { kind: "none" } : { kind: form.crewKind, pct: n.share, crew: n.crew, usdPerDay: n.dayRate, days: n.days },
  };
  const weight: WeightUnit = f?.unit === "ton" ? { unit: "ton", w: 1 / 2000, p: 2000 } : { unit: "lb", w: 1, p: 1 };
  return { scenario, weight };
}

// ---------------------------------------------------------------- what each output waits for

const uniq = <T>(xs: T[]) => [...new Set(xs)];
const ordered = (xs: FieldId[]) => uniq(xs).sort((a, b) => FIELD_ORDER.indexOf(a) - FIELD_ORDER.indexOf(b));

/** The fields crew pay rests on, by how crew is paid. 0 crew: nobody else is paid. */
export function crewDeps(form: CoastForm, n: Numbers): FieldId[] {
  if (n.crew === 0) return [];
  if (form.crewKind === null) return [form.crew.trim() === "" ? "crew" : "crewKind"];
  if (form.crewKind === "afterExpenses") return ["catch", "price", "fees", ...fuelDeps(n), "supplies", "share", "fuelPayer"];
  if (form.crewKind === "ofGross") return ["catch", "price", "share"];
  return ["crew", "dayRate", "days"];
}

/** 0 gallons needs no fuel price. */
const fuelDeps = (n: Numbers): FieldId[] => (n.fuelGal === 0 ? ["fuelGal"] : ["fuelGal", "fuelPrice"]);

export type OutputId = "gross" | "fees" | "fuel" | "supplies" | "tripCost" | "crewPay" | "net" | "perCrew" | "breakEvenPrice" | "breakEvenCatch";

export function outputDeps(form: CoastForm, n: Numbers): Record<OutputId, FieldId[] | null> {
  const crew = crewDeps(form, n);
  const net = ordered(["catch", "price", "fees", ...fuelDeps(n), "supplies", ...crew]);
  return {
    gross: ["catch", "price"],
    fees: ["catch", "price", "fees"],
    fuel: fuelDeps(n),
    supplies: ["supplies"],
    tripCost: ordered([...fuelDeps(n), "supplies"]),
    crewPay: ordered(crew),
    net,
    perCrew: n.crew === 0 ? null : ordered([...crew, "crew"]),
    breakEvenPrice: net.filter((x) => x !== "price"),
    breakEvenCatch: net.filter((x) => x !== "catch"),
  };
}

// ---------------------------------------------------------------- results

export type Result =
  | { status: "value"; value: number }
  /** fields still blank or not a number */
  | { status: "waiting"; fields: FieldId[] }
  /** the inputs are complete and no value exists (e.g. no price covers the costs) */
  | { status: "none"; reason: string }
  /** does not apply (per crew with no crew) */
  | { status: "na" };

export type CoastResult = {
  out: Record<OutputId, Result>;
  /** price −20 / 0 / +20 % × catch −30 / 0 / +30 % (the engine's grid); null while net waits */
  grid: SensitivityCell[] | null;
  /** every field any output still waits for, in page order */
  waiting: FieldId[];
  errors: Partial<Record<TextField, string>>;
  clamped: TextField[];
  steps: WaterfallStep[];
  evaluation: TripEvaluation;
};

const MODEL_FIELD: Partial<Record<string, FieldId>> = {
  "catch.0.lb": "catch",
  "catch.0.usdPerLb": "price",
  fuelGal: "fuelGal",
  roundTripNm: "fuelGal",
  cruiseKt: "fuelGal",
  steamGph: "fuelGal",
  fishingHours: "fuelGal",
  fishingGph: "fuelGal",
  fuelUsdPerGal: "fuelPrice",
  feesPctOfGross: "fees",
  "costs.0.usd": "supplies",
  "crew.kind": "crewKind",
  "crew.pct": "share",
  "crew.crew": "crew",
  "crew.usdPerDay": "dayRate",
  "crew.days": "days",
};

function result(o: Output, deps: FieldId[] | null, gaps: Set<FieldId>, convert: (v: number) => number = (v) => v): Result {
  if (deps === null) return { status: "na" };
  const waiting = deps.filter((d) => gaps.has(d));
  if (waiting.length) return { status: "waiting", fields: ordered(waiting) };
  if (o.status === "value") return { status: "value", value: convert(o.value) };
  if (o.status === "unreachable") return { status: "none", reason: o.reason };
  if (o.status === "notApplicable") return { status: "na" };
  return { status: "waiting", fields: ordered(o.missing.map((k: FieldKey) => MODEL_FIELD[k] ?? "fuelGal")) };
}

/** The fields that are blank or not a number. */
export function gapsOf(form: CoastForm, f: CoastFishery | null): Set<FieldId> {
  const { n } = numbers(form, f);
  const g = new Set<FieldId>();
  for (const id of Object.keys(LIMIT) as TextField[]) if (n[id] === null) g.add(id);
  if (form.crewKind === null) g.add("crewKind");
  if (form.fuelPayer === null) g.add("fuelPayer");
  return g;
}

/** Every output for this form, through the engine. `scale` previews one cell of the grid. */
export function evaluateForm(form: CoastForm, f: CoastFishery | null, scale: Scale = AS_ENTERED): CoastResult {
  const { n, errors, clamped } = numbers(form, f);
  const { scenario, weight } = toScenario(form, f, scale);
  const ev = evaluateTrip(scenario, { audience: "commercial", weight });
  const deps = outputDeps(form, n);
  const gaps = gapsOf(form, f);
  const per = LB_PER_UNIT[f?.unit ?? "lb"];
  const out: Record<OutputId, Result> = {
    gross: result(ev.gross, deps.gross, gaps),
    fees: result(ev.fees, deps.fees, gaps),
    fuel: result(ev.fuelCost, deps.fuel, gaps),
    supplies: result(ev.otherCosts, deps.supplies, gaps),
    tripCost: result(ev.tripCost, deps.tripCost, gaps),
    crewPay: result(ev.crewPay, deps.crewPay, gaps),
    net: result(ev.netToBoat, deps.net, gaps),
    perCrew: result(ev.perCrew, deps.perCrew, gaps),
    // the engine answers in pounds; the page in the fishery's unit
    breakEvenPrice: result(ev.breakEvenPrice, deps.breakEvenPrice, gaps, (v) => v * per),
    breakEvenCatch: result(ev.breakEvenCatch, deps.breakEvenCatch, gaps, (v) => v / per),
  };
  const grid = out.net.status === "value" && ev.sensitivity.status === "value" ? ev.sensitivity.value : null;
  const waiting = ordered(Object.values(out).flatMap((r) => (r.status === "waiting" ? r.fields : [])));
  return { out, grid, waiting, errors, clamped, steps: waterfallSteps(out, f), evaluation: ev };
}

const say = (fields: FieldId[], f: CoastFishery | null) => {
  const w = fields.map((x) => fieldWords(x, f));
  return `waiting for ${w.length > 3 ? `${w.slice(0, 2).join(", ")} and ${w.length - 2} more` : w.length === 3 ? `${w[0]}, ${w[1]} and ${w[2]}` : w.join(" and ")}`;
};

/** "waiting for price", "waiting for catch and price", "waiting for fees, fuel used and 3 more". */
export const waitingWords = (r: Result, f: CoastFishery | null) => (r.status === "waiting" ? say(r.fields, f) : null);

/** Gross to net, as the ledger draws it. */
export function waterfallSteps(out: Record<OutputId, Result>, f: CoastFishery | null): WaterfallStep[] {
  const step = (id: string, label: string, kind: WaterfallStep["kind"], r: Result): WaterfallStep => ({ id, label, kind, amount: r.status === "value" ? r.value : null, waiting: waitingWords(r, f) ?? (r.status === "none" ? "no value" : undefined) });
  return [
    step("gross", "Gross", "gross", out.gross),
    step("fees", "Fees", "cost", out.fees),
    step("fuel", "Fuel", "cost", out.fuel),
    step("supplies", f?.supplies ?? "Bait and ice", "cost", out.supplies),
    step("crew", "Crew pay", "cost", out.crewPay),
    step("net", "Net to boat", "net", out.net),
  ];
}

/** The grid's factors, as the ds SensitivityTable takes them: catch down the side, price across. */
export const GRID_ROWS = [
  { label: "+30 %", factor: 1.3 },
  { label: "As entered", factor: 1 },
  { label: "−30 %", factor: 0.7 },
];
export const GRID_COLS = [
  { label: "−20 %", factor: 0.8 },
  { label: "As entered", factor: 1 },
  { label: "+20 %", factor: 1.2 },
];

/** A grid cell's net, from the engine's own grid. */
export function gridValue(grid: SensitivityCell[] | null, catchFactor: number, priceFactor: number): number | null {
  if (!grid) return null;
  const c = grid.find((x) => Math.abs(x.catchDelta - (catchFactor - 1)) < 1e-9 && Math.abs(x.priceDelta - (priceFactor - 1)) < 1e-9);
  return c ? c.netUsd : null;
}

/**
 * One money scale for the waterfall across the reader's numbers and every grid scenario, so previewing a
 * cell moves the bars against a fixed axis (gross visibly grows or shrinks) instead of rescaling to it.
 */
export function scenarioDomain(form: CoastForm, f: CoastFishery | null): [number, number] {
  let lo = 0;
  let hi = 0;
  const take = (steps: WaterfallStep[]) => {
    const d = layoutWaterfall(steps).domain;
    if (steps.some((s) => s.amount !== null)) {
      lo = Math.min(lo, d[0]);
      hi = Math.max(hi, d[1]);
    }
  };
  const base = evaluateForm(form, f);
  take(base.steps);
  if (base.grid) for (const r of GRID_ROWS) for (const c of GRID_COLS) take(evaluateForm(form, f, { catch: r.factor, price: c.factor }).steps);
  if (hi === lo) hi = lo + 1;
  return [lo, hi];
}

// ---------------------------------------------------------------- assumptions, in plain words

/** What the numbers leave out or take as given, from this form (never a number repeated from the ledger). */
export function assumptions(form: CoastForm, f: CoastFishery | null): string[] {
  const { n } = numbers(form, f);
  const supplies = (f?.supplies ?? "Bait and ice").toLowerCase();
  const out = ["Net to boat comes before insurance, permits, maintenance, gear loss, loan payments and the skipper's own pay. It is not profit or take-home pay."];
  if (n.crew === 0) out.push("Nobody else is paid from this trip.");
  else if (form.crewKind === "afterExpenses") {
    if (form.fuelPayer === "shared") out.push(`${cap(withFuel(f))} come off the top: the crew's share is worked out on what is left after fees, ${withFuel(f)}.`);
    else if (form.fuelPayer === "boat") out.push(`The crew's share is worked out on gross less fees; the boat pays ${withFuel(f)} from its own share.`);
    else out.push(`Who pays fuel changes the crew's share, so nothing is worked out until you choose. ${cap(supplies)} are paid the same way as fuel.`);
  } else if (form.crewKind === "ofGross") out.push(`The crew's share comes off gross before any cost; the boat pays fees, ${withFuel(f)}.`);
  else if (form.crewKind === "dayRate") out.push("Each crew member is paid the day rate for every day at sea, whatever the catch.");
  out.push("Fees are the share of gross you enter: landing fees and buyer deductions count only if they are in it.");
  out.push("One trip as entered: no weather days, soak time, lost gear or second trip.");
  return out;
}
const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

/** "fuel, bait and ice", "fuel and ice": fuel with the trip's consumables, as one list. */
export function withFuel(f: CoastFishery | null): string {
  const sup = (f?.supplies ?? "Bait and ice").toLowerCase();
  return / and /.test(sup) ? `fuel, ${sup}` : `fuel and ${sup}`;
}

// ---------------------------------------------------------------- saving on this device

const text = (v: number | null | undefined): string => (typeof v === "number" && Number.isFinite(v) ? String(Number(v.toPrecision(10))) : "");

/** The form as the trip store keeps it (lib/tripEconomicsStore `Draft`, schema 2): pounds, references as their basis. */
export function toDraft(form: CoastForm, f: CoastFishery | null): Draft {
  const { scenario } = toScenario(form, f);
  const basis: Draft["basis"] = {};
  if (form.taken.price) basis["catch.0.usdPerLb"] = "cdfw";
  if (form.taken.fuelPrice) basis.fuelUsdPerGal = "eia";
  if (form.taken.share) basis["crew.pct"] = "preset";
  return { preset: f?.preset ?? null, inputs: scenario, pots: null, baitUsdPerPot: null, basis };
}

/** A saved draft back into the form, in the fishery's unit. */
export function fromDraft(d: Draft, fisheryId: string | null): CoastForm {
  const f = coastFishery(fisheryId);
  const per = LB_PER_UNIT[f?.unit ?? "lb"];
  const i = d.inputs;
  const line = i.catch?.[0];
  const kind = i.crew?.kind ?? null;
  const none = kind === "none";
  return {
    catch: line?.lb != null ? text(line.lb / per) : "",
    price: line?.usdPerLb != null ? text(line.usdPerLb * per) : "",
    fuelGal: text(i.fuelGal),
    fuelPrice: text(i.fuelUsdPerGal),
    supplies: text(i.costs?.[0]?.usd),
    fees: text(i.feesPctOfGross),
    crew: none ? "0" : text(i.crew?.crew),
    crewKind: kind === "afterExpenses" || kind === "ofGross" || kind === "dayRate" ? kind : null,
    share: none ? "" : text(i.crew?.pct),
    dayRate: text(i.crew?.usdPerDay),
    days: text(i.crew?.days),
    fuelPayer: typeof i.fuelShared === "boolean" ? (i.fuelShared ? "shared" : "boat") : null,
    taken: { ...(d.basis["catch.0.usdPerLb"] === "cdfw" ? { price: true as const } : {}), ...(d.basis.fuelUsdPerGal === "eia" ? { fuelPrice: true as const } : {}), ...(d.basis["crew.pct"] === "preset" ? { share: true as const } : {}) },
  };
}

/** Anything typed or chosen at all. */
export const hasEntry = (form: CoastForm) => (Object.keys(LIMIT) as TextField[]).some((k) => form[k].trim() !== "") || form.crewKind !== null || form.fuelPayer !== null;

/** A saved trip's name: what it was, in the reader's numbers. */
export function tripName(form: CoastForm, f: CoastFishery | null): string {
  const unit = f?.unit ?? "lb";
  const what = f?.label ?? "Trip";
  const c = form.catch.trim();
  const p = form.price.trim();
  return c && p ? `${what}, ${c} ${unit} at $${p.replace(/^\$/, "")}/${unit}` : what;
}
