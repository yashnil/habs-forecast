import { parseField, type LimitKind, type SensitivityCell } from "@/lib/tripEconomics";
import { evaluateTrip, type Output, type TripEvaluation, type TripScenario, type WeightUnit } from "@/lib/tripEvaluation";
import type { Draft, DraftView } from "@/lib/tripEconomicsStore";
import { audienceOf, LB_PER_UNIT, tripFishery, type CoastFishery, type TripAudience } from "./fisheries";

/**
 * The trip workspace's model (v13): what the reader typed in, every output of `evaluateTrip` (the engine of
 * record, src/lib/tripEvaluation.ts) out. Headless and pure: no React, no storage, no arithmetic of its own
 * beyond units (a squid boat types tons), the per-person share (typed "each", handed to the engine as the
 * crew's total) and rounding for display.
 *
 * The page's honesty rules (v12 research E, T45 to T47) hold for every line the reader can add:
 * - nothing is assumed: a blank field is a gap, never 0. The outputs that rest on it say what they wait for;
 * - who pays each cost (shared off the top, or the boat) is a required choice wherever it moves the money,
 *   which is only when crew is paid a share after costs. It is chosen per line, fuel included;
 * - a break-even is rounded up, never to the nearest: one pound less than the figure shown must lose money;
 * - a recreational trip is cost only. Its catch is not sold: no price, gross, crew pay or net exists for it.
 */

export type CrewKind = "afterExpenses" | "ofGross" | "dayRate";
export type Payer = "shared" | "boat";
/** How the reader typed the crew share: the crew's total, or what each person gets. */
export type ShareBasis = "total" | "each";
export type FuelRoute = "gallons" | "distance";

export type CatchRow = { label: string; ref: string | null; amount: string; price: string };
export type CostRow = { label: string; usd: string; payer: Payer | null };
export type Taken = Partial<Record<"price" | "fuelPrice" | "share", true>>;

export type Workspace = {
  /** the primary line first; break-evens are for it, the other lines held as entered */
  catch: CatchRow[];
  fuelRoute: FuelRoute;
  fuelGal: string;
  roundTripNm: string;
  cruiseKt: string;
  steamGph: string;
  fishingHours: string;
  fishingGph: string;
  fuelPrice: string;
  fuelPayer: Payer | null;
  costs: CostRow[];
  feesPct: string;
  /** fees per pound (or ton): null until the reader adds that line; then a blank is a gap like any other */
  feesPerUnit: string | null;
  crew: string;
  crewKind: CrewKind | null;
  share: string;
  shareBasis: ShareBasis;
  dayRate: string;
  days: string;
  /** recreational: the people sharing the trip's cost */
  anglers: string;
  /** fields whose value the reader took from a sourced reference (they stay labelled) */
  taken: Taken;
};

export const MAX_CATCH_ROWS = 10;
export const MAX_COST_ROWS = 20;

export function emptyWorkspace(f: CoastFishery | null): Workspace {
  return {
    catch: [{ label: f?.label ?? "Catch", ref: f?.priceKey ?? null, amount: "", price: "" }],
    fuelRoute: "gallons",
    fuelGal: "",
    roundTripNm: "",
    cruiseKt: "",
    steamGph: "",
    fishingHours: "",
    fishingGph: "",
    fuelPrice: "",
    fuelPayer: null,
    costs: [{ label: f?.supplies ?? "Bait and ice", usd: "", payer: null }],
    feesPct: "",
    feesPerUnit: null,
    crew: "",
    crewKind: null,
    share: "",
    shareBasis: "total",
    dayRate: "",
    days: "",
    anglers: "",
    taken: {},
  };
}

// ---------------------------------------------------------------- fields

type Scalar = "fuelGal" | "roundTripNm" | "cruiseKt" | "steamGph" | "fishingHours" | "fishingGph" | "fuelPrice" | "feesPct" | "feesPerUnit" | "crew" | "share" | "dayRate" | "days" | "anglers";
export type FieldId = Scalar | "crewKind" | "fuelPayer" | `catch.${number}.amount` | `catch.${number}.price` | `cost.${number}.usd` | `cost.${number}.payer`;

const SCALAR_LIMIT: Record<Scalar, LimitKind> = {
  fuelGal: "gal",
  roundTripNm: "nm",
  cruiseKt: "kt",
  steamGph: "gph",
  fishingHours: "hours",
  fishingGph: "gph",
  fuelPrice: "usdPerGal",
  feesPct: "pct",
  feesPerUnit: "usdPerLb",
  crew: "people",
  share: "pct",
  dayRate: "usdPerDay",
  days: "days",
  anglers: "people",
};

const SCALAR_WORDS: Record<Scalar | "crewKind" | "fuelPayer", string> = {
  fuelGal: "fuel used",
  roundTripNm: "round trip distance",
  cruiseKt: "cruising speed",
  steamGph: "steaming burn",
  fishingHours: "hours fishing",
  fishingGph: "fishing burn",
  fuelPrice: "fuel price",
  feesPct: "fees",
  feesPerUnit: "fees per pound",
  crew: "crew",
  share: "crew share",
  dayRate: "day rate",
  days: "days at sea",
  anglers: "people sharing the cost",
  crewKind: "how crew is paid",
  fuelPayer: "who pays fuel",
};

/** The words a gap is said with: "waiting for price", "waiting for who pays bait and ice". */
export function fieldWords(id: FieldId, ws: Workspace, f: CoastFishery | null): string {
  const m = /^(catch|cost)\.(\d+)\.(\w+)$/.exec(id);
  if (!m) return id === "feesPerUnit" && f?.unit === "ton" ? "fees per ton" : SCALAR_WORDS[id as Scalar];
  const i = Number(m[2]);
  if (m[1] === "catch") {
    const what = m[3] === "amount" ? "catch" : "price";
    return ws.catch.length > 1 ? `${what} of ${(ws.catch[i]?.label || `line ${i + 1}`).toLowerCase()}` : what;
  }
  const label = (ws.costs[i]?.label || `cost ${i + 1}`).toLowerCase();
  return m[3] === "payer" ? `who pays ${label}` : label;
}

/** Page order: catch lines, fuel, costs, fees, crew. Gaps are always listed in it. */
function fieldOrder(id: FieldId): number {
  const m = /^(catch|cost)\.(\d+)\.(\w+)$/.exec(id);
  if (m) return (m[1] === "catch" ? 0 : 2000) + Number(m[2]) * 2 + (m[3] === "amount" || m[3] === "usd" ? 0 : 1);
  const fixed: FieldId[] = ["fuelGal", "roundTripNm", "cruiseKt", "steamGph", "fishingHours", "fishingGph", "fuelPrice", "fuelPayer"];
  const late: FieldId[] = ["feesPct", "feesPerUnit", "crew", "crewKind", "share", "dayRate", "days", "anglers"];
  const i = fixed.indexOf(id);
  return i >= 0 ? 1000 + i : 3000 + late.indexOf(id);
}
const ordered = (xs: FieldId[]) => [...new Set(xs)].sort((a, b) => fieldOrder(a) - fieldOrder(b));

// ---------------------------------------------------------------- reading what was typed

export type Numbers = Record<Scalar, number | null> & {
  catch: Array<{ amount: number | null; price: number | null }>;
  costs: Array<number | null>;
  /** the crew's total share in percent, whichever way it was typed */
  totalPct: number | null;
  /** gallons for the trip when the fuel fields give them; null while a fuel field is a gap */
  gallons: number | null;
};

export type Read = { n: Numbers; errors: Partial<Record<FieldId, string>>; clamped: FieldId[] };

/** Every typed field as a number in the worksheet's own unit, with what could not be read. */
export function read(ws: Workspace, f: CoastFishery | null): Read {
  const errors: Partial<Record<FieldId, string>> = {};
  const clamped: FieldId[] = [];
  const take = (id: FieldId, raw: string, kind: LimitKind): number | null => {
    const p = parseField(raw, kind);
    if (p.error) errors[id] = p.error;
    if (p.clamped) clamped.push(id);
    return p.value;
  };
  const n = {} as Numbers;
  for (const id of Object.keys(SCALAR_LIMIT) as Scalar[]) {
    if (id === "feesPerUnit") n[id] = ws.feesPerUnit === null ? null : take(id, ws.feesPerUnit, f?.unit === "ton" ? "usd" : "usdPerLb");
    else n[id] = take(id, ws[id], SCALAR_LIMIT[id]);
  }
  // people come whole
  for (const id of ["crew", "anglers"] as const) {
    if (n[id] !== null && !Number.isInteger(n[id])) {
      errors[id] = "Whole people only";
      n[id] = null;
    }
  }
  if (n.cruiseKt === 0 && n.roundTripNm !== 0) {
    errors.cruiseKt = "Above 0";
    n.cruiseKt = null;
  }
  // a squid price per ton is read against the money limit, not $/lb
  n.catch = ws.catch.map((r, i) => ({ amount: take(`catch.${i}.amount`, r.amount, "lb"), price: take(`catch.${i}.price`, r.price, f?.unit === "ton" ? "usd" : "usdPerLb") }));
  n.costs = ws.costs.map((r, i) => take(`cost.${i}.usd`, r.usd, "usd"));

  n.totalPct = n.share;
  if (ws.shareBasis === "each" && n.share !== null) {
    if (n.crew === null || n.crew === 0) n.totalPct = null;
    else if (n.share * n.crew > 100) {
      errors.share = `${trim(n.share)} % each is more than 100 % between ${n.crew} crew`;
      n.totalPct = null;
    } else n.totalPct = n.share * n.crew;
  }

  if (ws.fuelRoute === "gallons") n.gallons = n.fuelGal;
  else {
    const steam = n.roundTripNm === 0 ? 0 : n.roundTripNm !== null && n.cruiseKt !== null && n.steamGph !== null ? (n.roundTripNm / n.cruiseKt) * n.steamGph : null;
    const fish = n.fishingHours === 0 ? 0 : n.fishingHours !== null && n.fishingGph !== null ? n.fishingHours * n.fishingGph : null;
    n.gallons = steam === null || fish === null ? null : steam + fish;
  }
  return { n, errors, clamped };
}

const trim = (v: number) => String(Number(v.toPrecision(10)));

// ---------------------------------------------------------------- the engine's scenario

/** Catch and price moved by the sensitivity grid's factors (1 = as entered), every line together. */
export type Scale = { catch: number; price: number };
const AS_ENTERED: Scale = { catch: 1, price: 1 };

const someoneElsePaid = (n: Numbers) => n.crew !== null && n.crew > 0;
/** Who pays a cost moves the money only when crew is paid a share of what is left after costs. */
export const payersMatter = (ws: Workspace, n: Numbers, audience: TripAudience) => audience === "commercial" && someoneElsePaid(n) && ws.crewKind === "afterExpenses";

/** The engine's scenario for this workspace. Weight is typed in the fishery's unit and handed over in pounds. */
export function toScenario(ws: Workspace, f: CoastFishery | null, scale: Scale = AS_ENTERED): { scenario: TripScenario; weight: WeightUnit; audience: TripAudience } {
  const { n } = read(ws, f);
  const audience = audienceOf(f);
  const per = LB_PER_UNIT[f?.unit ?? "lb"];
  const byGallons = ws.fuelRoute === "gallons";
  const scenario: TripScenario = {
    fuelGal: byGallons ? n.fuelGal : null,
    roundTripNm: byGallons ? null : n.roundTripNm,
    cruiseKt: byGallons ? null : n.cruiseKt,
    steamGph: byGallons ? null : n.steamGph,
    fishingHours: byGallons ? null : n.fishingHours,
    fishingGph: byGallons ? null : n.fishingGph,
    fuelUsdPerGal: n.fuelPrice,
    ...(ws.fuelPayer === null ? {} : { fuelShared: ws.fuelPayer === "shared" }),
    costs: ws.costs.map((r, i) => ({ label: r.label || `Cost ${i + 1}`, usd: n.costs[i], shared: r.payer !== "boat" })),
    feesPctOfGross: audience === "recreational" ? null : n.feesPct,
    feesUsdPerLb: audience === "recreational" || n.feesPerUnit === null ? null : n.feesPerUnit / per,
    catch:
      audience === "recreational"
        ? []
        : ws.catch.map((r, i) => ({
            species: r.label || `Catch ${i + 1}`,
            ref: r.ref,
            lb: n.catch[i].amount === null ? null : n.catch[i].amount! * per * scale.catch,
            usdPerLb: n.catch[i].price === null ? null : (n.catch[i].price! / per) * scale.price,
          })),
    crew: audience === "recreational" || n.crew === 0 ? { kind: "none" } : { kind: ws.crewKind, pct: n.totalPct, crew: n.crew, usdPerDay: n.dayRate, days: n.days },
    ...(audience === "recreational" ? { anglers: n.anglers } : {}),
  };
  const weight: WeightUnit = f?.unit === "ton" ? { unit: "ton", w: 1 / 2000, p: 2000 } : { unit: "lb", w: 1, p: 1 };
  return { scenario, weight, audience };
}

// ---------------------------------------------------------------- gaps and what each output rests on

/** The fields that are blank, not a number, or not chosen, among those this trip uses. */
export function gapsOf(ws: Workspace, f: CoastFishery | null): Set<FieldId> {
  const { n } = read(ws, f);
  const audience = audienceOf(f);
  const g = new Set<FieldId>();
  for (const id of fuelFields(ws, n)) if (n[id] === null) g.add(id);
  n.costs.forEach((v, i) => v === null && g.add(`cost.${i}.usd`));
  if (audience === "recreational") {
    if (n.anglers === null || n.anglers === 0) g.add("anglers");
    return g;
  }
  n.catch.forEach((c, i) => {
    if (c.amount === null) g.add(`catch.${i}.amount`);
    if (c.price === null) g.add(`catch.${i}.price`);
  });
  if (n.feesPct === null) g.add("feesPct");
  if (ws.feesPerUnit !== null && n.feesPerUnit === null) g.add("feesPerUnit");
  if (n.crew === null) g.add("crew");
  else if (n.crew > 0) {
    if (ws.crewKind === null) g.add("crewKind");
    else if (ws.crewKind === "dayRate") {
      if (n.dayRate === null) g.add("dayRate");
      if (n.days === null) g.add("days");
    } else if (n.totalPct === null) g.add("share");
    if (payersMatter(ws, n, audience)) for (const id of payerFields(ws, n)) if (id === "fuelPayer" ? ws.fuelPayer === null : ws.costs[Number(id.split(".")[1])].payer === null) g.add(id);
  }
  return g;
}

/** The fuel fields this route uses. A leg set to 0 (0 nm, 0 h) needs nothing more; 0 gallons needs no price. */
function fuelFields(ws: Workspace, n: Numbers): Scalar[] {
  const out: Scalar[] = ws.fuelRoute === "gallons" ? ["fuelGal"] : ["roundTripNm", ...(n.roundTripNm === 0 ? [] : (["cruiseKt", "steamGph"] as Scalar[])), "fishingHours", ...(n.fishingHours === 0 ? [] : (["fishingGph"] as Scalar[]))];
  if (n.gallons !== 0) out.push("fuelPrice");
  return out;
}

/** The payer choices that move the money: fuel unless none is burned, each cost line unless it is $0. */
function payerFields(ws: Workspace, n: Numbers): FieldId[] {
  return [...(n.gallons === 0 ? [] : (["fuelPayer"] as FieldId[])), ...ws.costs.flatMap((_, i) => (n.costs[i] === 0 ? [] : ([`cost.${i}.payer`] as FieldId[])))];
}

export type OutputId = "gross" | "fees" | "gallons" | "fuel" | "otherCosts" | "operating" | "crewPay" | "perCrew" | "net" | "breakEvenCatch" | "breakEvenPrice" | "breakEvenFuel" | "perPerson";

function outputDeps(ws: Workspace, n: Numbers, audience: TripAudience): Record<OutputId, FieldId[] | null> {
  const fuelAll = fuelFields(ws, n);
  const gallons: FieldId[] = fuelAll.filter((x) => x !== "fuelPrice");
  const costs = ws.costs.map((_, i): FieldId => `cost.${i}.usd`);
  const operating = [...fuelAll, ...costs];
  if (audience === "recreational") {
    return { gross: null, fees: null, gallons, fuel: fuelAll, otherCosts: costs, operating, crewPay: null, perCrew: null, net: null, breakEvenCatch: null, breakEvenPrice: null, breakEvenFuel: null, perPerson: [...operating, "anglers"] };
  }
  const gross = ws.catch.flatMap((_, i): FieldId[] => [`catch.${i}.amount`, `catch.${i}.price`]);
  const fees: FieldId[] = [...gross, "feesPct", ...(ws.feesPerUnit !== null ? (["feesPerUnit"] as FieldId[]) : [])];
  let crew: FieldId[];
  if (n.crew === null) crew = ["crew"];
  else if (n.crew === 0) crew = [];
  else if (ws.crewKind === null) crew = ["crewKind"];
  else if (ws.crewKind === "afterExpenses") crew = [...fees, ...operating, "share", ...payerFields(ws, n)];
  else if (ws.crewKind === "ofGross") crew = [...gross, "share"];
  else crew = ["dayRate", "days"];
  const net = [...fees, ...operating, ...crew];
  return {
    gross,
    fees,
    gallons,
    fuel: fuelAll,
    otherCosts: costs,
    operating,
    crewPay: crew,
    perCrew: n.crew === 0 ? null : crew,
    net,
    breakEvenCatch: net.filter((x) => x !== "catch.0.amount"),
    breakEvenPrice: net.filter((x) => x !== "catch.0.price"),
    breakEvenFuel: net,
    perPerson: null,
  };
}

// ---------------------------------------------------------------- results

export type Result =
  /** `formula`: the engine's worked sum for this figure, in the worksheet's unit */
  | { status: "value"; value: number; formula: string }
  /** fields still blank, not a number or not chosen */
  | { status: "waiting"; fields: FieldId[] }
  /** the inputs are complete and no value exists (e.g. no price covers the costs) */
  | { status: "none"; reason: string }
  /** does not apply (per crew with no crew; anything about a sale on a recreational trip) */
  | { status: "na" };

export type Step = { id: string; label: string; kind: "gross" | "cost" | "net"; amount: number | null; waiting?: string };

export type ShareNote = {
  basis: ShareBasis;
  crew: number;
  totalPct: number;
  eachPct: number;
  /** what the share is a share of */
  of: "after" | "gross";
  totalUsd: number | null;
  eachUsd: number | null;
  /** "20 % in total, split 2 ways: 10 % each." */
  words: string;
};

export type TripResult = {
  audience: TripAudience;
  out: Record<OutputId, Result>;
  /** price −20 / 0 / +20 % × catch −30 / 0 / +30 % (the engine's grid); null while net waits */
  grid: SensitivityCell[] | null;
  /** net to boat with fuel 20 % cheaper and dearer (recreational: the trip cost) */
  fuelSwing: { down: number; up: number; measure: "netToBoat" | "tripCost" } | null;
  /** every field any output still waits for, in page order */
  waiting: FieldId[];
  errors: Partial<Record<FieldId, string>>;
  clamped: FieldId[];
  /** the engine's own warnings (fees and share taking all of the gross), never dropped */
  warnings: string[];
  /** gross down to net to boat, one step per cost line; recreational: the costs up to the trip cost */
  steps: Step[];
  /** the crew share said both ways, once it can be */
  share: ShareNote | null;
  evaluation: TripEvaluation;
};

const MODEL_FIELD: Record<string, FieldId> = {
  fuelGal: "fuelGal",
  roundTripNm: "roundTripNm",
  cruiseKt: "cruiseKt",
  steamGph: "steamGph",
  fishingHours: "fishingHours",
  fishingGph: "fishingGph",
  fuelUsdPerGal: "fuelPrice",
  fuelShared: "fuelPayer",
  feesPctOfGross: "feesPct",
  feesUsdPerLb: "feesPerUnit",
  "crew.kind": "crewKind",
  "crew.pct": "share",
  "crew.crew": "crew",
  "crew.usdPerDay": "dayRate",
  "crew.days": "days",
  anglers: "anglers",
};
const modelField = (k: string): FieldId => {
  const m = /^(catch|costs)\.(\d+)\.(\w+)$/.exec(k);
  if (!m) return MODEL_FIELD[k] ?? "fuelGal";
  if (m[1] === "catch") return m[3] === "lb" ? `catch.${Number(m[2])}.amount` : `catch.${Number(m[2])}.price`;
  return m[3] === "shared" ? `cost.${Number(m[2])}.payer` : `cost.${Number(m[2])}.usd`;
};

function result(o: Output, deps: FieldId[] | null, gaps: Set<FieldId>, convert: (v: number) => number = (v) => v): Result {
  if (deps === null) return { status: "na" };
  const waiting = deps.filter((d) => gaps.has(d));
  if (waiting.length) return { status: "waiting", fields: ordered(waiting) };
  if (o.status === "value") return { status: "value", value: convert(o.value), formula: o.formula };
  if (o.status === "unreachable") return { status: "none", reason: o.reason };
  if (o.status === "notApplicable") return { status: "na" };
  return { status: "waiting", fields: ordered(o.missing.map(modelField)) };
}

/** Every output for this workspace, through the engine. `scale` previews one cell of the grid. */
export function evaluateWorkspace(ws: Workspace, f: CoastFishery | null, scale: Scale = AS_ENTERED): TripResult {
  const { n, errors, clamped } = read(ws, f);
  const { scenario, weight, audience } = toScenario(ws, f, scale);
  const ev = evaluateTrip(scenario, { audience, weight });
  const deps = outputDeps(ws, n, audience);
  const gaps = gapsOf(ws, f);
  const per = LB_PER_UNIT[f?.unit ?? "lb"];
  const out: Record<OutputId, Result> = {
    gross: result(ev.gross, deps.gross, gaps),
    fees: result(ev.fees, deps.fees, gaps),
    gallons: result(ev.fuelGallons, deps.gallons, gaps),
    fuel: result(ev.fuelCost, deps.fuel, gaps),
    otherCosts: result(ev.otherCosts, deps.otherCosts, gaps),
    operating: result(ev.tripCost, deps.operating, gaps),
    crewPay: result(ev.crewPay, deps.crewPay, gaps),
    perCrew: result(ev.perCrew, deps.perCrew, gaps),
    net: result(ev.netToBoat, deps.net, gaps),
    // the engine answers in pounds; the page in the fishery's unit
    breakEvenCatch: result(ev.breakEvenCatch, deps.breakEvenCatch, gaps, (v) => v / per),
    breakEvenPrice: result(ev.breakEvenPrice, deps.breakEvenPrice, gaps, (v) => v * per),
    breakEvenFuel: result(ev.breakEvenFuelPrice, deps.breakEvenFuel, gaps),
    perPerson: result(ev.perPerson, deps.perPerson, gaps),
  };
  const done = audience === "recreational" ? out.operating : out.net;
  const grid = out.net.status === "value" && ev.sensitivity.status === "value" ? ev.sensitivity.value : null;
  const fuelSwing = done.status === "value" && ev.fuelSensitivity.status === "value" ? ev.fuelSensitivity.value : null;
  const waiting = ordered(Object.values(out).flatMap((r) => (r.status === "waiting" ? r.fields : [])));
  return { audience, out, grid, fuelSwing, waiting, errors, clamped, warnings: ev.warnings, steps: steps(ws, f, n, out, audience), share: shareNote(ws, n, out, audience), evaluation: ev };
}

const say = (fields: FieldId[], ws: Workspace, f: CoastFishery | null) => {
  const w = fields.map((x) => fieldWords(x, ws, f));
  return `waiting for ${w.length > 3 ? `${w.slice(0, 2).join(", ")} and ${w.length - 2} more` : w.length === 3 ? `${w[0]}, ${w[1]} and ${w[2]}` : w.join(" and ")}`;
};

/** "waiting for price", "waiting for catch and price", "waiting for fees, fuel used and 3 more". */
export const waitingWords = (r: Result, ws: Workspace, f: CoastFishery | null) => (r.status === "waiting" ? say(r.fields, ws, f) : null);

function steps(ws: Workspace, f: CoastFishery | null, n: Numbers, out: Record<OutputId, Result>, audience: TripAudience): Step[] {
  const step = (id: string, label: string, kind: Step["kind"], r: Result): Step => ({ id, label, kind, amount: r.status === "value" ? r.value : null, waiting: waitingWords(r, ws, f) ?? (r.status === "none" ? "no value" : undefined) });
  const cost = ws.costs.map((c, i): Step => ({ id: `cost-${i}`, label: c.label || `Cost ${i + 1}`, kind: "cost", amount: n.costs[i], waiting: n.costs[i] === null ? `waiting for ${fieldWords(`cost.${i}.usd`, ws, f)}` : undefined }));
  if (audience === "recreational") return [step("fuel", "Fuel", "cost", out.fuel), ...cost, step("net", "Trip cost", "net", out.operating)];
  return [step("gross", "Gross", "gross", out.gross), step("fees", "Fees", "cost", out.fees), step("fuel", "Fuel", "cost", out.fuel), ...cost, step("crew", "Crew pay", "cost", out.crewPay), step("net", "Net to boat", "net", out.net)];
}

/** The crew share said both ways ("20 % in total, split 2 ways: 10 % each"), with the dollars once they exist. */
function shareNote(ws: Workspace, n: Numbers, out: Record<OutputId, Result>, audience: TripAudience): ShareNote | null {
  if (audience === "recreational" || !someoneElsePaid(n) || n.totalPct === null || (ws.crewKind !== "afterExpenses" && ws.crewKind !== "ofGross")) return null;
  const crew = n.crew!;
  const totalPct = n.totalPct;
  const eachPct = totalPct / crew;
  const totalUsd = out.crewPay.status === "value" ? out.crewPay.value : null;
  const eachUsd = totalUsd === null ? null : totalUsd / crew;
  const p = (v: number) => `${trim(Math.round(v * 100) / 100)} %`;
  const words = crew === 1 ? `${p(totalPct)} to your 1 crew.` : ws.shareBasis === "each" ? `${p(eachPct)} each for ${crew} crew: ${p(totalPct)} in total.` : `${p(totalPct)} in total, split ${crew} ways: ${p(eachPct)} each.`;
  return { basis: ws.shareBasis, crew, totalPct, eachPct, of: ws.crewKind === "afterExpenses" ? "after" : "gross", totalUsd, eachUsd, words };
}

// ---------------------------------------------------------------- sensitivity

/** The grid's factors: catch down the side, price across. */
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

/** The money range every grid scenario's waterfall fits in, so a preview moves bars against one fixed axis. */
export function moneyExtent(ws: Workspace, f: CoastFishery | null): [number, number] {
  let lo = 0;
  let hi = 0;
  const take = (r: TripResult) => {
    for (const k of ["gross", "net", "operating"] as const) {
      const o = r.out[k];
      if (o.status === "value") {
        lo = Math.min(lo, o.value);
        hi = Math.max(hi, o.value);
      }
    }
  };
  const base = evaluateWorkspace(ws, f);
  take(base);
  if (base.grid) for (const r of GRID_ROWS) for (const c of GRID_COLS) take(evaluateWorkspace(ws, f, { catch: r.factor, price: c.factor }));
  return hi === lo ? [lo, lo + 1] : [lo, hi];
}

export type SweepVariable = "catch" | "price" | "fuelPrice";
export type Sweep = {
  variable: SweepVariable;
  /** the unit of x: "lb", "$/lb", "$/gal" (tons for a fishery landed by the ton) */
  unit: string;
  /** what the reader entered */
  entered: number;
  /** net to boat at each x, from the engine, everything else held as entered */
  points: Array<{ x: number; net: number }>;
  /** where net to boat crosses $0 within the range, exact (not rounded) */
  breakEven: number | null;
};

/**
 * One thing changes: net to boat as the primary line's catch, its price or the fuel price runs from 0 to
 * twice what was entered, every other input held. Each point is the engine's own answer for that input, so
 * the line bends where the engine's does (a crew share after costs stops at $0 pay). Null until net exists.
 */
export function sweep(ws: Workspace, f: CoastFishery | null, variable: SweepVariable, count = 24): Sweep | null {
  const base = evaluateWorkspace(ws, f);
  if (base.audience !== "commercial" || base.out.net.status !== "value") return null;
  const { n } = read(ws, f);
  const entered = variable === "catch" ? n.catch[0]?.amount : variable === "price" ? n.catch[0]?.price : n.fuelPrice;
  if (entered == null || entered <= 0) return null;
  const { scenario, weight, audience } = toScenario(ws, f);
  const per = LB_PER_UNIT[f?.unit ?? "lb"];
  const at = (x: number): number | null => {
    const s: TripScenario = { ...scenario, catch: scenario.catch?.map((c) => ({ ...c })) };
    if (variable === "catch") s.catch![0].lb = x * per;
    else if (variable === "price") s.catch![0].usdPerLb = x / per;
    else s.fuelUsdPerGal = x;
    const o = evaluateTrip(s, { audience, weight }).netToBoat;
    return o.status === "value" ? o.value : null;
  };
  const points: Sweep["points"] = [];
  for (let i = 0; i <= count; i++) {
    const x = (entered * 2 * i) / count;
    const net = at(x);
    if (net !== null) points.push({ x, net });
  }
  const be = variable === "catch" ? base.out.breakEvenCatch : variable === "price" ? base.out.breakEvenPrice : base.out.breakEvenFuel;
  const breakEven = be.status === "value" && be.value <= entered * 2 ? be.value : null;
  const u = f?.unit ?? "lb";
  return { variable, unit: variable === "catch" ? u : variable === "price" ? `$/${u}` : "$/gal", entered, points, breakEven };
}

// ---------------------------------------------------------------- figures as the page writes them

/** Money to the cent: "$2,602.24", "−$167.20". */
export function money(v: number): string {
  const r = Math.abs(v) < 0.005 ? 0 : v;
  const s = Math.abs(r).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  return `${r < 0 ? "−" : ""}$${s}`;
}

/** Up to the next step, forgiving float dust (38.999999999 is 39, not 40). */
export const roundUp = (v: number, step: number) => Math.ceil(v / step - 1e-9) * step;
const roundDown = (v: number, step: number) => Math.floor(v / step + 1e-9) * step;

/** The catch at which net to boat reaches $0, rounded up: whole pounds, hundredths of a ton. */
export function breakEvenCatchText(v: number, unit: "lb" | "ton"): string {
  return unit === "ton" ? `${roundUp(v, 0.01).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })} ton` : `${roundUp(v, 1).toLocaleString("en-US")} lb`;
}

/** The price at which net to boat reaches $0, rounded up: cents per pound, whole dollars per ton. */
export function breakEvenPriceText(v: number, unit: "lb" | "ton"): string {
  return unit === "ton" ? `$${roundUp(v, 1).toLocaleString("en-US")}/ton` : `$${roundUp(v, 0.01).toFixed(2)}/lb`;
}

/** The most fuel can cost before net to boat falls to $0, rounded down: a cent more would lose money. */
export function breakEvenFuelText(v: number): string {
  return v > 50 ? "over $50/gal" : `$${roundDown(v, 0.01).toFixed(2)}/gal`;
}

// ---------------------------------------------------------------- assumptions, in plain words

const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);
const list = (xs: string[]) => (xs.length <= 1 ? xs.join("") : `${xs.slice(0, -1).join(", ")} and ${xs[xs.length - 1]}`);

/** What net to boat is not. Carried wherever the figure is shown. */
export const NET_IS_NOT = "Net to boat comes before insurance, permits, maintenance, gear loss, loan payments and the skipper's own pay. It is not profit or take-home pay.";

/** What the numbers leave out or take as given, from this workspace (never a figure repeated from the ledger). */
export function assumptions(ws: Workspace, f: CoastFishery | null): string[] {
  const { n } = read(ws, f);
  const audience = audienceOf(f);
  if (audience === "recreational") {
    return ["A recreational catch is not sold: this is what the trip costs, with no price, gross or net.", "The cost is split evenly between the people you enter.", "One trip as entered: no licence, gear, launch fee or boat upkeep unless you add them as costs."];
  }
  const out = [NET_IS_NOT];
  if (n.crew === 0) out.push("Nobody else is paid from this trip.");
  else if (ws.crewKind === "afterExpenses") {
    const lines: Array<{ label: string; payer: Payer | null }> = [...(n.gallons === 0 ? [] : [{ label: "fuel", payer: ws.fuelPayer }]), ...ws.costs.map((c, i) => ({ label: (c.label || `cost ${i + 1}`).toLowerCase(), payer: c.payer })).filter((_, i) => n.costs[i] !== 0)];
    const shared = lines.filter((l) => l.payer === "shared").map((l) => l.label);
    const boat = lines.filter((l) => l.payer === "boat").map((l) => l.label);
    const open = lines.filter((l) => l.payer === null).map((l) => l.label);
    if (shared.length) out.push(`${cap(list(shared))} come${shared.length === 1 && !/ and /.test(shared[0]) ? "s" : ""} off the top: the crew's share is worked out on what is left after fees and those costs.`);
    if (boat.length) out.push(`The boat pays ${list(boat)} from its own share${shared.length ? "" : ": the crew's share is worked out on gross less fees"}.`);
    if (open.length) out.push(`Who pays ${list(open)} changes the crew's share, so nothing is worked out until you choose.`);
  } else if (ws.crewKind === "ofGross") out.push("The crew's share comes off gross before any cost; the boat pays fees and every trip cost.");
  else if (ws.crewKind === "dayRate") out.push("Each crew member is paid the day rate for every day at sea, whatever the catch.");
  if (someoneElsePaid(n) && (ws.crewKind === "afterExpenses" || ws.crewKind === "ofGross")) out.push(ws.shareBasis === "each" ? "The crew share is what you entered for each person, added up across the crew." : "The crew share is the crew's total, split evenly between them.");
  out.push(ws.feesPerUnit !== null ? `Fees are the share of gross and the amount per ${f?.unit === "ton" ? "ton" : "pound"} you enter: landing fees and buyer deductions count only if they are in them.` : "Fees are the share of gross you enter: landing fees and buyer deductions count only if they are in it.");
  if (ws.catch.length > 1) out.push("Break-evens are for the first catch line, the others held as entered.");
  out.push("One trip as entered: no weather days, soak time, lost gear or second trip.");
  return out;
}

// ---------------------------------------------------------------- the store's draft

const text = (v: number | null | undefined): string => (typeof v === "number" && Number.isFinite(v) ? trim(v) : "");

/** The workspace as the trip store keeps it (lib/tripEconomicsStore `Draft`): pounds, the share as the crew's total. */
export function toDraft(ws: Workspace, f: CoastFishery | null): Draft {
  const { scenario } = toScenario(ws, f);
  const { n } = read(ws, f);
  // a per-person share typed before the crew count is known is kept as typed until it can be totalled
  if (scenario.crew && scenario.crew.kind !== "none" && ws.shareBasis === "each" && n.totalPct === null) scenario.crew = { ...scenario.crew, pct: n.share };
  // a recreational trip pays no crew: nothing about one is stored
  if (audienceOf(f) === "recreational") scenario.crew = { kind: null };
  const basis: Draft["basis"] = {};
  if (ws.taken.price) basis["catch.0.usdPerLb"] = "cdfw";
  if (ws.taken.fuelPrice) basis.fuelUsdPerGal = "eia";
  if (ws.taken.share) basis["crew.pct"] = "preset";
  const view: DraftView = {};
  if (ws.shareBasis === "each") view.shareBasis = "each";
  if (ws.fuelRoute === "distance") view.fuelRoute = "distance";
  const unset = ws.costs.flatMap((c, i) => (c.payer === null ? [i] : []));
  if (unset.length) view.payerUnset = unset;
  return { preset: f?.preset ?? null, inputs: scenario, pots: null, baitUsdPerPot: null, basis, ...(Object.keys(view).length ? { view } : {}) };
}

/** A saved draft back into the workspace, in the fishery's unit. */
export function fromDraft(d: Draft, fisheryId: string | null): Workspace {
  const f = tripFishery(fisheryId);
  const per = LB_PER_UNIT[f?.unit ?? "lb"];
  const i = d.inputs;
  const empty = emptyWorkspace(f);
  const kind = i.crew?.kind ?? null;
  const none = kind === "none";
  const crew = none ? 0 : (i.crew?.crew ?? null);
  const each = d.view?.shareBasis === "each";
  const pct = none ? null : (i.crew?.pct ?? null);
  const byDistance = d.view?.fuelRoute === "distance" || (i.fuelGal == null && [i.roundTripNm, i.cruiseKt, i.steamGph, i.fishingHours, i.fishingGph].some((v) => v != null));
  const unset = new Set(d.view?.payerUnset ?? []);
  const lines = (i.catch ?? []).slice(0, MAX_CATCH_ROWS);
  const costs = (i.costs ?? []).slice(0, MAX_COST_ROWS);
  return {
    catch: lines.length ? lines.map((l, k) => ({ label: l.species || (k === 0 ? empty.catch[0].label : `Catch ${k + 1}`), ref: l.ref ?? null, amount: l.lb != null ? text(l.lb / per) : "", price: l.usdPerLb != null ? text(l.usdPerLb * per) : "" })) : empty.catch,
    fuelRoute: byDistance ? "distance" : "gallons",
    fuelGal: text(i.fuelGal),
    roundTripNm: text(i.roundTripNm),
    cruiseKt: text(i.cruiseKt),
    steamGph: text(i.steamGph),
    fishingHours: text(i.fishingHours),
    fishingGph: text(i.fishingGph),
    fuelPrice: text(i.fuelUsdPerGal),
    fuelPayer: typeof i.fuelShared === "boolean" ? (i.fuelShared ? "shared" : "boat") : null,
    costs: costs.length ? costs.map((c, k) => ({ label: c.label, usd: text(c.usd), payer: unset.has(k) ? null : c.shared ? "shared" : "boat" })) : empty.costs,
    feesPct: text(i.feesPctOfGross),
    feesPerUnit: i.feesUsdPerLb != null ? text(i.feesUsdPerLb * per) : null,
    crew: text(crew),
    crewKind: kind === "afterExpenses" || kind === "ofGross" || kind === "dayRate" ? kind : null,
    share: pct === null ? "" : each && crew !== null && crew > 0 ? text(pct / crew) : text(pct),
    shareBasis: each ? "each" : "total",
    dayRate: text(i.crew?.usdPerDay),
    days: text(i.crew?.days),
    anglers: text(i.anglers),
    taken: { ...(d.basis["catch.0.usdPerLb"] === "cdfw" ? { price: true as const } : {}), ...(d.basis.fuelUsdPerGal === "eia" ? { fuelPrice: true as const } : {}), ...(d.basis["crew.pct"] === "preset" ? { share: true as const } : {}) },
  };
}

/** Anything typed or chosen at all. */
export function hasEntry(ws: Workspace): boolean {
  const typed = [ws.fuelGal, ws.roundTripNm, ws.cruiseKt, ws.steamGph, ws.fishingHours, ws.fishingGph, ws.fuelPrice, ws.feesPct, ws.feesPerUnit ?? "", ws.crew, ws.share, ws.dayRate, ws.days, ws.anglers, ...ws.catch.flatMap((c) => [c.amount, c.price]), ...ws.costs.map((c) => c.usd)];
  return typed.some((t) => t.trim() !== "") || ws.crewKind !== null || ws.fuelPayer !== null || ws.costs.some((c) => c.payer !== null) || ws.catch.length > 1 || ws.costs.length > 1;
}

// ---------------------------------------------------------------- editing (each returns a new workspace)

/**
 * A typed scalar field. Typing never labels a value as a reference's, even when the figures happen to match
 * (a dock price of $8.24 is the reader's own); only `takeReference` does. Typing over a taken value drops the label.
 */
export function setField(ws: Workspace, id: Exclude<Scalar, "feesPerUnit">, value: string): Workspace {
  const next: Workspace = { ...ws, [id]: value, taken: { ...ws.taken } };
  if (id === "fuelPrice" || id === "share") delete next.taken[id];
  return next;
}

/** The reader tapped a sourced reference: its value enters the field and stays labelled as the source's. */
export function takeReference(ws: Workspace, id: "price" | "fuelPrice" | "share", value: string): Workspace {
  const taken = { ...ws.taken, [id]: true as const };
  if (id === "price") return { ...ws, catch: ws.catch.map((c, i) => (i === 0 ? { ...c, price: value } : c)), taken };
  return { ...ws, [id]: value, taken };
}

export function setCatch(ws: Workspace, i: number, patch: Partial<CatchRow>): Workspace {
  const taken = { ...ws.taken };
  if (i === 0 && patch.price !== undefined) delete taken.price;
  return { ...ws, catch: ws.catch.map((c, k) => (k === i ? { ...c, ...patch } : c)), taken };
}
export const addCatch = (ws: Workspace): Workspace => (ws.catch.length >= MAX_CATCH_ROWS ? ws : { ...ws, catch: [...ws.catch, { label: "", ref: null, amount: "", price: "" }] });
/** The primary line stays: a trip has at least one. */
export const removeCatch = (ws: Workspace, i: number): Workspace => (i === 0 || ws.catch.length <= 1 ? ws : { ...ws, catch: ws.catch.filter((_, k) => k !== i) });

export const setCost = (ws: Workspace, i: number, patch: Partial<CostRow>): Workspace => ({ ...ws, costs: ws.costs.map((c, k) => (k === i ? { ...c, ...patch } : c)) });
export const addCost = (ws: Workspace): Workspace => (ws.costs.length >= MAX_COST_ROWS ? ws : { ...ws, costs: [...ws.costs, { label: "", usd: "", payer: null }] });
export const removeCost = (ws: Workspace, i: number): Workspace => ({ ...ws, costs: ws.costs.filter((_, k) => k !== i) });

/**
 * Total ↔ each, keeping the money where it was: 20 % in total for 2 crew becomes 10 % each. With the crew
 * count still unknown the typed figure stays and only its meaning changes (the share then waits for the crew).
 */
export function setShareBasis(ws: Workspace, f: CoastFishery | null, basis: ShareBasis): Workspace {
  if (basis === ws.shareBasis) return ws;
  const { n } = read(ws, f);
  const taken = { ...ws.taken };
  let share = ws.share;
  if (n.share !== null && n.crew !== null && n.crew > 1) {
    share = text(basis === "each" ? n.share / n.crew : n.share * n.crew);
    delete taken.share;
  }
  return { ...ws, shareBasis: basis, share, taken };
}

/**
 * Another fishery: a value taken from the last one's reference described that fishery, and a change of unit
 * (squid is sold by the ton) makes the catch, price and per-unit fee mean something else. Those are cleared and
 * named, so the page can say what it dropped; everything else the reader typed stays.
 */
export function switchFishery(ws: Workspace, from: CoastFishery | null, to: CoastFishery | null): { workspace: Workspace; cleared: string[] } {
  if (from?.id === to?.id) return { workspace: ws, cleared: [] };
  const cleared: string[] = [];
  const unitChanged = (from?.unit ?? "lb") !== (to?.unit ?? "lb");
  const next: Workspace = { ...ws, catch: ws.catch.map((c) => ({ ...c })), taken: {} };
  const first = next.catch[0];
  if (first.label === (from?.label ?? "Catch")) first.label = to?.label ?? "Catch";
  first.ref = to?.priceKey ?? null;
  if ((ws.taken.price || unitChanged) && first.price !== "") {
    first.price = "";
    cleared.push("price");
  }
  if (unitChanged) {
    if (next.catch.some((c) => c.amount !== "")) cleared.push("catch");
    next.catch = next.catch.map((c, i) => ({ ...c, amount: "", price: i === 0 ? c.price : "" }));
    if (next.feesPerUnit) {
      next.feesPerUnit = "";
      cleared.push(`fees per ${from?.unit === "ton" ? "ton" : "pound"}`);
    }
  }
  if (ws.taken.share && next.share !== "") {
    next.share = "";
    cleared.push("crew share");
  }
  if (ws.taken.fuelPrice) next.taken.fuelPrice = true;
  if (next.costs.length && next.costs[0].label === (from?.supplies ?? "Bait and ice")) next.costs = next.costs.map((c, i) => (i === 0 ? { ...c, label: to?.supplies ?? "Bait and ice" } : c));
  return { workspace: next, cleared };
}
