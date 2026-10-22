/**
 * Trip evaluation (v10): every output of the trip worksheet, each either worked out from the user's own
 * numbers with the sum that produced it, or listing the exact input fields it still waits for.
 *
 * Pure and additive. The arithmetic of record stays in `tripEconomics.ts`: net, break-evens and the
 * sensitivity grid are read from `computeTrip`, so a complete scenario gives the same numbers here as
 * there. What this module adds:
 * - partial results: each output appears as soon as its own inputs exist (gross before fuel, fuel
 *   cost before crew, break-even catch before the catch itself);
 * - field-level gaps (`missing: FieldKey[]`), never a dash without a reason;
 * - the module's implicit zeros returned as data (`assumptions`), e.g. "steaming not counted";
 * - a crew model that may be unset (no default: net waits for it) or "none" (no hired crew);
 * - a cost-only mode for recreational trips (no price, gross, crew or net);
 * - a comparison of two scenarios (differing inputs and result deltas).
 *
 * Nothing here is a forecast, a recommendation or a statement that a trip may be made: the outputs
 * are the user's own inputs worked through.
 */

import { computeTrip, fuelGallons as fuelGallons_, sanitizeInputs, clampNum, type CrewKind, type CrewModel, type SensitivityCell, type TripInputs } from "./tripEconomics";

// ---------------------------------------------------------------- inputs

/** How crew is paid. `null` / absent = not chosen yet (there is no default). */
export type CrewChoice = CrewKind | "none";

export type ScenarioCrew = Omit<CrewModel, "kind"> & { kind: CrewChoice | null };

/** The calculator's inputs, with a crew model that may be unset and an optional head count for a cost split. */
export type TripScenario = Omit<TripInputs, "crew"> & {
  crew?: ScenarioCrew | null;
  /** people sharing the cost of a recreational trip (optional split) */
  anglers?: number | null;
};

export type Audience = "commercial" | "recreational";

/** An input field the UI can name and focus. Catch and cost indexes are positions in the scenario's own arrays. */
export type FieldKey =
  | "roundTripNm"
  | "cruiseKt"
  | "steamGph"
  | "fishingHours"
  | "fishingGph"
  | "fuelGal"
  | "fuelUsdPerGal"
  | "fuelShared"
  | "feesPctOfGross"
  | "feesUsdPerLb"
  | "crew.kind"
  | "crew.pct"
  | "crew.crew"
  | "crew.usdPerDay"
  | "crew.days"
  | "anglers"
  | `catch.${number}.species`
  | `catch.${number}.lb`
  | `catch.${number}.usdPerLb`
  | `costs.${number}.label`
  | `costs.${number}.usd`
  | `costs.${number}.shared`;

export const FIELD_LABEL: Record<string, string> = {
  roundTripNm: "round trip distance",
  cruiseKt: "cruising speed",
  steamGph: "steaming burn",
  fishingHours: "hours fishing",
  fishingGph: "fishing burn",
  fuelGal: "fuel used (gallons)",
  fuelUsdPerGal: "fuel price",
  fuelShared: "who pays fuel",
  feesPctOfGross: "fees, % of gross",
  feesUsdPerLb: "fees, $ per lb",
  "crew.kind": "how crew is paid",
  "crew.pct": "crew share",
  "crew.crew": "number of crew",
  "crew.usdPerDay": "crew day rate",
  "crew.days": "days paid",
  anglers: "people sharing the cost",
};

/** A plain label for a field key ("cruising speed", "pounds landed, line 1"). */
export function fieldLabel(k: FieldKey): string {
  if (FIELD_LABEL[k]) return FIELD_LABEL[k];
  const m = /^(catch|costs)\.(\d+)\.(\w+)$/.exec(k);
  if (!m) return k;
  const n = Number(m[2]) + 1;
  if (m[1] === "catch") return `${m[3] === "lb" ? "pounds landed" : m[3] === "usdPerLb" ? "price per pound" : "species"}, line ${n}`;
  return `${m[3] === "usd" ? "cost" : m[3] === "label" ? "cost label" : "who pays the cost"}, line ${n}`;
}

// ---------------------------------------------------------------- outputs

/** An implicit value the evaluation used, stated as data so the UI never has to restate it by hand. */
export type Assumption = { field: FieldKey; note: string };

export type Value<T> = { status: "value"; value: T; formula: string; assumptions: Assumption[] };
/** `missing`: the fields still needed (one route). `alternatives`: other sets of fields that would also do. */
export type Missing = { status: "missing"; missing: FieldKey[]; alternatives?: FieldKey[][]; reason: string };
/** Inputs are complete but no value exists (e.g. no catch covers the costs). */
export type Unreachable = { status: "unreachable"; reason: string };
/** The output does not apply (recreational trips have no gross, crew or net; no hired crew has no per-crew pay). */
export type NotApplicable = { status: "notApplicable"; reason: string };

export type Output<T = number> = Value<T> | Missing | Unreachable | NotApplicable;

export type FuelSensitivity = { down: number; up: number; measure: "netToBoat" | "tripCost" };

export type TripEvaluation = {
  audience: Audience;
  gross: Output;
  fees: Output;
  /** steaming and fishing legs, so a known leg stays visible while the other waits */
  steamingGallons: Output;
  fishingGallons: Output;
  fuelGallons: Output;
  fuelCost: Output;
  otherCosts: Output;
  tripCost: Output;
  /** gross − fees: the most fuel and other costs the catch can carry before anything is paid to crew */
  costCeilingBeforeCrew: Output;
  /** gross − fees − trip cost: what is left before crew pay */
  poolBeforeCrew: Output;
  crewPay: Output;
  perCrew: Output;
  /** recreational: trip cost ÷ people sharing it */
  perPerson: Output;
  netToBoat: Output;
  breakEvenCatch: Output;
  breakEvenPrice: Output;
  breakEvenFuelPrice: Output;
  /** price −20 / 0 / +20 % × catch −30 / 0 / +30 %; the centre equals net to boat */
  sensitivity: Output<SensitivityCell[]>;
  fuelSensitivity: Output<FuelSensitivity>;
  /** every field any output still waits for, deduplicated, in input order */
  missingFields: FieldKey[];
  /** every assumption any output used, deduplicated */
  assumptions: Assumption[];
  warnings: string[];
};

export const OUTPUT_KEYS = [
  "gross",
  "fees",
  "steamingGallons",
  "fishingGallons",
  "fuelGallons",
  "fuelCost",
  "otherCosts",
  "tripCost",
  "costCeilingBeforeCrew",
  "poolBeforeCrew",
  "crewPay",
  "perCrew",
  "perPerson",
  "netToBoat",
  "breakEvenCatch",
  "breakEvenPrice",
  "breakEvenFuelPrice",
  "sensitivity",
  "fuelSensitivity",
] as const satisfies readonly (keyof TripEvaluation)[];

export type OutputKey = (typeof OUTPUT_KEYS)[number];
export type ScalarOutputKey = Exclude<OutputKey, "sensitivity" | "fuelSensitivity">;

// ---------------------------------------------------------------- formatting for worked sums

function fmt(v: number, min: number, max: number): string {
  const r = Math.abs(v) < 0.5 * 10 ** -max ? 0 : v;
  const s = Math.abs(r).toLocaleString("en-US", { minimumFractionDigits: min, maximumFractionDigits: max });
  return r < 0 ? `−${s}` : s;
}
/** whole dollars: "$3,600", "−$56" */
const $ = (v: number) => (v < 0 && Math.abs(v) >= 0.5 ? `−$${fmt(-v, 0, 0)}` : `$${fmt(Math.abs(v) < 0.5 ? 0 : v, 0, 0)}`);
/** a unit price: "$9.00", "$8.242" */
const $p = (v: number) => `$${fmt(v, 2, 3)}`;
const n = (v: number, d = 1) => fmt(v, 0, d);

// ---------------------------------------------------------------- helpers

const known = (v: number | null | undefined): v is number => typeof v === "number" && Number.isFinite(v);

const value = <T>(v: T, formula: string, assumptions: Assumption[] = []): Value<T> => ({ status: "value", value: v, formula, assumptions });

function uniq<T>(xs: T[]): T[] {
  return [...new Set(xs)];
}

function missing(fields: FieldKey[], reason: string, alternatives?: FieldKey[][]): Missing {
  const m: Missing = { status: "missing", missing: uniq(fields), reason };
  if (alternatives && alternatives.length) m.alternatives = alternatives;
  return m;
}

/** The fields an output waits for: the union of its inputs' gaps. */
function gaps(...outs: Output<unknown>[]): FieldKey[] {
  return uniq(outs.flatMap((o) => (o.status === "missing" ? o.missing : [])));
}

const assumptionsOf = (...outs: Output<unknown>[]): Assumption[] => outs.flatMap((o) => (o.status === "value" ? o.assumptions : []));

function dedupe(as: Assumption[]): Assumption[] {
  const seen = new Set<string>();
  return as.filter((a) => {
    const k = `${a.field}|${a.note}`;
    if (seen.has(k)) return false;
    seen.add(k);
    return true;
  });
}

const isVal = <T>(o: Output<T>): o is Value<T> => o.status === "value";

const NOT_REC = "Recreational trip: cost only (no price, gross, crew pay or net).";

/** The order fields appear in the worksheet: revenue, fuel, other costs, fees, crew. */
function fieldOrder(k: FieldKey): number {
  const fixed: FieldKey[] = ["fuelGal", "roundTripNm", "cruiseKt", "steamGph", "fishingHours", "fishingGph", "fuelUsdPerGal", "fuelShared"];
  const late: FieldKey[] = ["feesPctOfGross", "feesUsdPerLb", "crew.kind", "crew.pct", "crew.crew", "crew.usdPerDay", "crew.days", "anglers"];
  const m = /^(catch|costs)\.(\d+)\.(\w+)$/.exec(k);
  if (m) {
    const sub = ["species", "lb", "usdPerLb", "label", "usd", "shared"].indexOf(m[3]);
    return (m[1] === "catch" ? 0 : 2000) + Number(m[2]) * 10 + sub;
  }
  const f = fixed.indexOf(k);
  if (f >= 0) return 1000 + f;
  return 3000 + late.indexOf(k);
}

// ---------------------------------------------------------------- sanitizing

type Clean = { inputs: TripInputs; crewKind: CrewChoice | null; anglers: number | null };

function clean(s: TripScenario): Clean {
  const c = s.crew ?? null;
  const k = c?.kind;
  const crewKind: CrewChoice | null = k === "none" || k === "afterExpenses" || k === "ofGross" || k === "dayRate" ? k : null;
  // the arithmetic of record: "none" is a share of 0 %; unset keeps its parameters for the gaps below
  const crew: CrewModel = { ...(c ?? {}), kind: crewKind === "none" || crewKind === null ? "afterExpenses" : crewKind };
  if (crewKind === "none") crew.pct = 0;
  const inputs = sanitizeInputs({ ...s, crew } as TripInputs);
  return { inputs, crewKind, anglers: clampNum(s.anglers, "people") };
}

// ---------------------------------------------------------------- the evaluation

export function evaluateTrip(scenario: TripScenario, opts: { audience?: Audience } = {}): TripEvaluation {
  const audience: Audience = opts.audience === "recreational" ? "recreational" : "commercial";
  const rec = audience === "recreational";
  const { inputs: i, crewKind, anglers } = clean(scenario);
  const crew = i.crew!;

  // ---- fuel, leg by leg. The total follows `fuelGallons` exactly; what is added is the reason and the legs.
  const nm = i.roundTripNm ?? null;
  const kt = i.cruiseKt ?? null;
  const sg = i.steamGph ?? null;
  const hrs = i.fishingHours ?? null;
  const fg = i.fishingGph ?? null;
  const steamKnown = known(nm) && known(kt) && kt > 0 && known(sg);
  const fishKnown = known(hrs) && known(fg);
  const steamGaps: FieldKey[] = [...(known(nm) ? [] : (["roundTripNm"] as FieldKey[])), ...(known(kt) && kt > 0 ? [] : (["cruiseKt"] as FieldKey[])), ...(known(sg) ? [] : (["steamGph"] as FieldKey[]))];
  const fishGaps: FieldKey[] = [...(known(hrs) ? [] : (["fishingHours"] as FieldKey[])), ...(known(fg) ? [] : (["fishingGph"] as FieldKey[]))];
  const steamBody = steamKnown ? `${n(nm!)} nm ÷ ${n(kt!)} kt × ${n(sg!)} gal/h` : "";
  const fishBody = fishKnown ? `${n(hrs!)} h × ${n(fg!)} gal/h` : "";

  const steamingGallons: Output = steamKnown
    ? value((nm! / kt!) * sg!, `Steaming ${n((nm! / kt!) * sg!)} gal = ${steamBody}`)
    : nm === 0
      ? value(0, "Steaming 0 gal (round trip 0 nm)")
      : missing(steamGaps, kt === 0 ? "Steaming needs a cruising speed above 0." : "Steaming needs the round trip, cruising speed and steaming burn.");
  const fishingGallons: Output = fishKnown
    ? value(hrs! * fg!, `Fishing ${n(hrs! * fg!)} gal = ${fishBody}`)
    : hrs === 0
      ? value(0, "Fishing 0 gal (0 h fishing)")
      : missing(fishGaps, "Fishing fuel needs hours fishing and the fishing burn.");

  let fuelGallons: Output;
  const gal = fuelGallons_(i);
  const steamStarted = (known(nm) && nm > 0) || known(kt) || known(sg);
  const fishStarted = (known(hrs) && hrs > 0) || known(fg);
  if (known(i.fuelGal)) {
    const a: Assumption[] = steamStarted || fishStarted ? [{ field: "fuelGal", note: "Gallons entered directly: distance, speed and burn are not used." }] : [];
    fuelGallons = value(i.fuelGal, `Fuel ${n(i.fuelGal)} gal (entered)`, a);
  } else if (gal !== null) {
    const a: Assumption[] = [];
    const parts: string[] = [];
    if (steamKnown) parts.push(steamBody);
    else if (nm === 0) parts.push("no steaming (0 nm)");
    else a.push({ field: "roundTripNm", note: "Steaming not counted: no round trip entered. Add it to include steaming." });
    if (fishKnown) parts.push(fishBody);
    else if (hrs === 0) parts.push("no fishing time (0 h)");
    else a.push({ field: "fishingHours", note: "Fishing time not counted: no hours fishing entered. Add them to include fishing." });
    fuelGallons = value(gal, `Fuel ${n(gal)} gal = ${parts.join(" + ")}`, a);
  } else if (!steamStarted && !fishStarted) {
    fuelGallons = missing(["fuelGal"], "Fuel used: enter the gallons, or the trip's distance, speed and burn.", [
      ["roundTripNm", "cruiseKt", "steamGph"],
      ["fishingHours", "fishingGph"],
    ]);
  } else {
    // a started leg that is incomplete is what blocks; with nothing blocking, any one leg completes it
    const steamBlocks = !steamKnown && known(nm) && nm > 0;
    const fishBlocks = !fishKnown && known(hrs) && hrs > 0;
    let need: FieldKey[];
    let alternatives: FieldKey[][];
    if (steamBlocks || fishBlocks) {
      need = [...(steamBlocks ? steamGaps : []), ...(fishBlocks ? fishGaps : [])];
      alternatives = [["fuelGal"]];
    } else {
      const first = steamStarted ? steamGaps : fishGaps;
      const second = steamStarted ? fishGaps : steamGaps;
      need = first;
      alternatives = [second, ["fuelGal"]];
    }
    const kept = steamKnown ? "steaming" : fishKnown ? "fishing" : null;
    const say = need.map(fieldLabel).join(", ");
    fuelGallons = missing(need, kept ? `Fuel needs ${say}; the ${kept} leg is worked out and kept.` : `Fuel needs ${say}.`, alternatives);
  }
  // ---- fuel cost
  let fuelCost: Output;
  const price = i.fuelUsdPerGal ?? null;
  if (isVal(fuelGallons)) {
    const g = fuelGallons.value;
    if (g === 0) fuelCost = value(0, "Fuel cost $0 = 0 gal", fuelGallons.assumptions);
    else if (price === null) fuelCost = missing(["fuelUsdPerGal"], "Fuel cost needs the fuel price.");
    else fuelCost = value(g * price, `Fuel cost ${$(g * price)} = ${n(g)} gal × ${$p(price)}/gal`, fuelGallons.assumptions);
  } else {
    fuelCost = missing([...gaps(fuelGallons), ...(price === null ? (["fuelUsdPerGal"] as FieldKey[]) : [])], "Fuel cost needs fuel used and the fuel price.", (fuelGallons as Missing).alternatives);
  }

  // ---- other costs (a blank line is not counted, and said so)
  const rawCosts = i.costs ?? [];
  let sharedOther = 0;
  let boatOther = 0;
  const costAssumptions: Assumption[] = [];
  const costParts: string[] = [];
  rawCosts.forEach((c, idx) => {
    if (!known(c.usd)) {
      costAssumptions.push({ field: `costs.${idx}.usd`, note: `Blank: ${c.label || "cost line"} not counted.` });
      return;
    }
    if (c.shared) sharedOther += c.usd;
    else boatOther += c.usd;
    costParts.push(`${c.label || `line ${idx + 1}`} ${$(c.usd)}`);
  });
  const otherTotal = sharedOther + boatOther;
  const otherCosts: Output = value(otherTotal, costParts.length ? `Other costs ${$(otherTotal)} = ${costParts.join(" + ")}` : "Other costs $0 (none entered)", costAssumptions);

  // ---- trip cost
  const tripCost: Output = isVal(fuelCost)
    ? value(fuelCost.value + otherTotal, `Trip cost ${$(fuelCost.value + otherTotal)} = fuel ${$(fuelCost.value)} + other costs ${$(otherTotal)}`, [...fuelCost.assumptions, ...costAssumptions])
    : missing(gaps(fuelCost), "Trip cost needs the fuel cost.", (fuelCost as Missing).alternatives);

  // ---- per person (recreational split)
  let perPerson: Output;
  if (!rec) perPerson = { status: "notApplicable", reason: "Per-person split is for recreational trips." };
  else if (!known(anglers) || anglers === 0) {
    const g = isVal(tripCost) ? [] : gaps(tripCost);
    perPerson = missing([...g, "anglers"], "Split needs the number of people sharing the cost.");
  } else if (!isVal(tripCost)) perPerson = missing(gaps(tripCost), "Split needs the trip cost.");
  else perPerson = value(tripCost.value / anglers, `Per person ${$(tripCost.value / anglers)} = trip cost ${$(tripCost.value)} ÷ ${n(anglers, 0)}`, tripCost.assumptions);

  // ---- fuel sensitivity for a cost-only trip
  const ref = computeTrip(i);

  if (rec) {
    const na: NotApplicable = { status: "notApplicable", reason: NOT_REC };
    const fuelSens: Output<FuelSensitivity> = isVal(fuelCost) && isVal(tripCost)
      ? value({ down: fuelCost.value * 0.8 + otherTotal, up: fuelCost.value * 1.2 + otherTotal, measure: "tripCost" as const }, `Trip cost with fuel −20 % ${$(fuelCost.value * 0.8 + otherTotal)}, +20 % ${$(fuelCost.value * 1.2 + otherTotal)}`, tripCost.assumptions)
      : missing(gaps(tripCost), "Fuel ±20 % needs the trip cost.");
    return finish({
      audience,
      gross: na,
      fees: na,
      steamingGallons,
      fishingGallons,
      fuelGallons,
      fuelCost,
      otherCosts,
      tripCost,
      costCeilingBeforeCrew: na,
      poolBeforeCrew: na,
      crewPay: na,
      perCrew: na,
      perPerson,
      netToBoat: na,
      breakEvenCatch: na,
      breakEvenPrice: na,
      breakEvenFuelPrice: na,
      sensitivity: na,
      fuelSensitivity: fuelSens,
      missingFields: [],
      assumptions: [],
      warnings: [],
    });
  }

  // ---- catch and gross (indexes kept as in the scenario)
  const lines = (i.catch ?? []).map((l, idx) => ({ ...l, idx })).filter((l) => l.species !== "" || known(l.lb) || known(l.usdPerLb));
  const catchGaps: FieldKey[] = [];
  if (lines.length === 0) catchGaps.push("catch.0.lb", "catch.0.usdPerLb");
  for (const l of lines) {
    if (!known(l.lb)) catchGaps.push(`catch.${l.idx}.lb`);
    if (!known(l.usdPerLb)) catchGaps.push(`catch.${l.idx}.usdPerLb`);
  }
  const R = catchGaps.length === 0 ? lines.reduce((a, l) => a + l.lb! * l.usdPerLb!, 0) : null;
  const Q = catchGaps.length === 0 ? lines.reduce((a, l) => a + l.lb!, 0) : null;
  const gross: Output = R !== null ? value(R, `Gross ${$(R)} = ${lines.map((l) => `${n(l.lb!, 2)} lb × ${$p(l.usdPerLb!)}`).join(" + ")}`) : missing(catchGaps, "Gross needs pounds landed and the price per pound.");

  // ---- fees (blank = none, said so)
  const feeAssumptions: Assumption[] = [];
  if (!known(i.feesPctOfGross)) feeAssumptions.push({ field: "feesPctOfGross", note: "Blank: no fees as a % of gross counted." });
  if (!known(i.feesUsdPerLb)) feeAssumptions.push({ field: "feesUsdPerLb", note: "Blank: no fees per lb counted." });
  const tau = (i.feesPctOfGross ?? 0) / 100;
  const u = i.feesUsdPerLb ?? 0;
  let fees: Output;
  if (R === null || Q === null) fees = missing(catchGaps, "Fees need the gross and pounds landed.");
  else {
    const f = R * tau + u * Q;
    const parts = [known(i.feesPctOfGross) && i.feesPctOfGross > 0 ? `${n(i.feesPctOfGross, 2)} % × ${$(R)}` : null, known(i.feesUsdPerLb) && i.feesUsdPerLb > 0 ? `${$p(i.feesUsdPerLb)}/lb × ${n(Q, 2)} lb` : null].filter(Boolean);
    fees = value(f, parts.length ? `Fees ${$(f)} = ${parts.join(" + ")}` : "Fees $0 (none entered)", feeAssumptions);
  }

  const costCeilingBeforeCrew: Output = isVal(gross) && isVal(fees)
    ? value(R! - fees.value, `Cost ceiling before crew ${$(R! - fees.value)} = gross ${$(R!)} − fees ${$(fees.value)}`, fees.assumptions)
    : missing(catchGaps, "The cost ceiling needs the gross.");

  const poolBeforeCrew: Output = isVal(fees) && isVal(tripCost)
    ? value(R! - fees.value - tripCost.value, `Before crew ${$(R! - fees.value - tripCost.value)} = gross ${$(R!)} − fees ${$(fees.value)} − trip cost ${$(tripCost.value)}`, [...fees.assumptions, ...tripCost.assumptions])
    : missing([...catchGaps, ...gaps(tripCost)], "What is left before crew needs the gross and the trip cost.", catchGaps.length ? undefined : (tripCost as Missing).alternatives);

  // ---- crew
  const fuelShared = i.fuelShared !== false;
  const crewParamGaps: FieldKey[] = [];
  if (crewKind === null) crewParamGaps.push("crew.kind");
  else if (crewKind === "afterExpenses" || crewKind === "ofGross") {
    if (!known(crew.pct)) crewParamGaps.push("crew.pct");
  } else if (crewKind === "dayRate" && crew.crew !== 0) {
    if (!known(crew.crew)) crewParamGaps.push("crew.crew");
    if (!known(crew.usdPerDay)) crewParamGaps.push("crew.usdPerDay");
    if (!known(crew.days)) crewParamGaps.push("crew.days");
  }
  const s = known(crew.pct) ? crew.pct / 100 : 0;
  let crewPay: Output;
  if (crewKind === null) crewPay = missing(["crew.kind"], "Choose how crew is paid: needed for crew pay and net. Nothing is assumed.");
  else if (crewKind === "none") crewPay = value(0, "Crew pay $0 (no hired crew)");
  else if (crewParamGaps.length) crewPay = missing(crewParamGaps, `Crew pay needs ${crewParamGaps.map(fieldLabel).join(", ")}.`);
  else if (crewKind === "dayRate") {
    const d = crew.crew === 0 ? 0 : crew.crew! * crew.usdPerDay! * crew.days!;
    crewPay = value(d, crew.crew === 0 ? "Crew pay $0 (0 crew)" : `Crew pay ${$(d)} = ${n(crew.crew!, 0)} crew × ${$(crew.usdPerDay!)}/day × ${n(crew.days!)} days`);
  } else if (crewKind === "ofGross") {
    crewPay = R === null ? missing(catchGaps, "Crew pay (share of gross) needs the gross.") : value(s * R, `Crew pay ${$(s * R)} = ${n(crew.pct!, 2)} % × gross ${$(R)}`);
  } else {
    // share after expenses: s × (gross − fees − shared costs), nothing when that is ≤ 0
    const fuelNeeded = fuelShared && !isVal(fuelCost);
    if (R === null || Q === null || fuelNeeded) {
      crewPay = missing([...catchGaps, ...(fuelNeeded ? gaps(fuelCost) : [])], "Crew pay (share after expenses) needs the gross and the shared costs.");
    } else {
      const fu = isVal(fuelCost) ? fuelCost.value : 0;
      const D = sharedOther + (fuelShared ? fu : 0);
      const N = R * (1 - tau) - u * Q - D;
      const pay = N > 0 ? s * N : 0;
      const a = [...(isVal(fees) ? fees.assumptions : []), ...costAssumptions, ...(fuelShared && isVal(fuelCost) ? fuelCost.assumptions : [])];
      crewPay = value(pay, N > 0 ? `Crew pay ${$(pay)} = ${n(crew.pct!, 2)} % × ${$(N)} (gross − fees − shared costs)` : `Crew pay $0: gross − fees − shared costs is ${$(N)}, nothing to share`, a);
    }
  }

  let perCrew: Output;
  if (crewKind === "none" || (crewKind === "dayRate" && crew.crew === 0)) perCrew = { status: "notApplicable", reason: "No hired crew." };
  else if (!isVal(crewPay)) perCrew = missing([...gaps(crewPay), ...(known(crew.crew) ? [] : (["crew.crew"] as FieldKey[]))], "Per-crew pay needs crew pay and the number of crew.");
  else if (crewKind === "dayRate") perCrew = value(crew.usdPerDay! * crew.days!, `Per crew ${$(crew.usdPerDay! * crew.days!)} = ${$(crew.usdPerDay!)}/day × ${n(crew.days!)} days`);
  else if (!known(crew.crew) || crew.crew === 0) perCrew = missing(["crew.crew"], "Per-crew pay needs the number of crew sharing it.");
  else perCrew = value(crewPay.value / crew.crew, `Per crew ${$(crewPay.value / crew.crew)} = crew pay ${$(crewPay.value)} ÷ ${n(crew.crew, 0)}`, crewPay.assumptions);

  // ---- net, break-evens and sensitivity: the arithmetic of record (computeTrip)
  const crewGap: FieldKey[] = crewKind === null ? ["crew.kind"] : crewParamGaps;
  const costGap = gaps(fuelCost);
  const costAlt = (fuelCost as Missing).alternatives;
  const allA = dedupe(assumptionsOf(gross, fees, tripCost, crewPay));

  const netNeed = [...catchGaps, ...costGap, ...crewGap];
  const netToBoat: Output = netNeed.length || ref.netUsd === null
    ? missing(netNeed, "Net to boat needs the gross, the trip cost and how crew is paid.", catchGaps.length || crewGap.length ? undefined : costAlt)
    : value(ref.netUsd, `Net to boat ${$(ref.netUsd)} = gross ${$(R!)} − fees ${$(isVal(fees) ? fees.value : 0)} − trip cost ${$(isVal(tripCost) ? tripCost.value : 0)} − crew ${$(isVal(crewPay) ? crewPay.value : 0)}`, allA);

  const primary = lines[0];
  const others = lines.slice(1);
  const otherGaps: FieldKey[] = others.flatMap((l) => [...(known(l.lb) ? [] : [`catch.${l.idx}.lb` as FieldKey]), ...(known(l.usdPerLb) ? [] : [`catch.${l.idx}.usdPerLb` as FieldKey])]);
  const pIdx = primary?.idx ?? 0;
  const beAssumptions = assumptionsOf(fees, tripCost, crewPay);
  if (!isVal(fees)) beAssumptions.push(...feeAssumptions);
  const beA = dedupe(beAssumptions);
  const unit = primary?.species ? ` of ${primary.species}` : "";

  const beLbNeed: FieldKey[] = [...(primary && known(primary.usdPerLb) ? [] : [`catch.${pIdx}.usdPerLb` as FieldKey]), ...otherGaps, ...costGap, ...crewGap];
  let breakEvenCatch: Output;
  if (beLbNeed.length) breakEvenCatch = missing(beLbNeed, "Break-even catch needs the price per pound, the trip cost and how crew is paid (not the catch itself).", costAlt && !otherGaps.length && !crewGap.length && primary && known(primary.usdPerLb) ? costAlt : undefined);
  else breakEvenCatch = fromBreakEven(ref.breakEvenLb, (v) => `Break-even catch ${n(v)} lb${unit}: net to boat reaches $0 at ${$p(primary!.usdPerLb!)}/lb`, beA, "No catch covers the costs at this price: fees and crew pay take all of each pound's value.");

  const beUsdNeed: FieldKey[] = [...(primary && known(primary.lb) ? [] : [`catch.${pIdx}.lb` as FieldKey]), ...otherGaps, ...costGap, ...crewGap];
  let breakEvenPrice: Output;
  if (beUsdNeed.length) breakEvenPrice = missing(beUsdNeed, "Break-even price needs the pounds landed, the trip cost and how crew is paid (not the price itself).", costAlt && !otherGaps.length && !crewGap.length && primary && known(primary.lb) ? costAlt : undefined);
  else breakEvenPrice = fromBreakEven(ref.breakEvenUsdPerLb, (v) => `Break-even price ${$p(v)}/lb${unit}: net to boat reaches $0 at ${n(primary!.lb!, 2)} lb`, beA, primary!.lb === 0 ? "No price covers the costs with 0 lb landed." : "No price covers the costs: fees and crew share take all of the gross.");

  const beFuelNeed: FieldKey[] = [...netNeed, ...(isVal(fuelGallons) ? [] : gaps(fuelGallons))];
  let breakEvenFuelPrice: Output;
  if (beFuelNeed.length) breakEvenFuelPrice = missing(uniq(beFuelNeed), "Break-even fuel price needs everything net to boat needs.", catchGaps.length || crewGap.length ? undefined : costAlt);
  else breakEvenFuelPrice = fromBreakEven(ref.breakEvenFuelUsdPerGal, (v) => `Break-even fuel price ${$p(v)}/gal: net to boat reaches $0 at ${n((fuelGallons as Value<number>).value)} gal`, allA, (fuelGallons as Value<number>).value === 0 ? "No fuel is used, so the fuel price does not move net." : "Net to boat is below $0 even with free fuel.");

  const sensitivity: Output<SensitivityCell[]> = isVal(netToBoat) && ref.sensitivity
    ? value(ref.sensitivity, `Net to boat at price −20 / 0 / +20 % × catch −30 / 0 / +30 %; the centre is your ${$(netToBoat.value)}`, allA)
    : missing(netNeed, "The price × catch grid needs everything net to boat needs.", (netToBoat as Missing).alternatives);
  const fuelSensitivity: Output<FuelSensitivity> = isVal(netToBoat) && ref.fuelSensitivity
    ? value({ ...ref.fuelSensitivity, measure: "netToBoat" as const }, `Net to boat with fuel −20 % ${$(ref.fuelSensitivity.down)}, +20 % ${$(ref.fuelSensitivity.up)}`, allA)
    : missing(netNeed, "Fuel ±20 % needs everything net to boat needs.", (netToBoat as Missing).alternatives);

  return finish({
    audience,
    gross,
    fees,
    steamingGallons,
    fishingGallons,
    fuelGallons,
    fuelCost,
    otherCosts,
    tripCost,
    costCeilingBeforeCrew,
    poolBeforeCrew,
    crewPay,
    perCrew,
    perPerson,
    netToBoat,
    breakEvenCatch,
    breakEvenPrice,
    breakEvenFuelPrice,
    sensitivity,
    fuelSensitivity,
    missingFields: [],
    assumptions: [],
    warnings: ref.warnings,
  });
}

function fromBreakEven(b: { value: number | null; note: string }, formula: (v: number) => string, a: Assumption[], none: string): Output {
  if (b.note === "covered") return value(0, `${formula(0).split(":")[0]}: the other catch lines already cover the costs`, a);
  if (b.note === "ok" && b.value !== null) return value(b.value, formula(b.value), a);
  return { status: "unreachable", reason: none };
}

/** The legs of fuel are details; the page-wide gap list names the route the user is on. */
const LEG_KEYS: OutputKey[] = ["steamingGallons", "fishingGallons"];
/** Optional splits: their gaps (head counts) are not things the trip still needs. */
const OPTIONAL_KEYS: OutputKey[] = ["perCrew", "perPerson"];

function finish(e: TripEvaluation): TripEvaluation {
  const outs = OUTPUT_KEYS.filter((k) => !LEG_KEYS.includes(k) && !OPTIONAL_KEYS.includes(k)).map((k) => e[k] as Output<unknown>);
  e.missingFields = uniq(outs.flatMap((o) => (o.status === "missing" ? o.missing : []))).sort((a, b) => fieldOrder(a) - fieldOrder(b));
  e.assumptions = dedupe(OUTPUT_KEYS.flatMap((k) => assumptionsOf(e[k] as Output<unknown>)));
  return e;
}

// ---------------------------------------------------------------- comparison

type FieldValue = number | string | boolean | null;

/** Every input of a scenario as field → value (blank = null). */
export function scenarioFields(sc: TripScenario): Partial<Record<FieldKey, FieldValue>> {
  const { inputs: i, crewKind, anglers } = clean(sc);
  const out: Partial<Record<FieldKey, FieldValue>> = {
    roundTripNm: i.roundTripNm ?? null,
    cruiseKt: i.cruiseKt ?? null,
    steamGph: i.steamGph ?? null,
    fishingHours: i.fishingHours ?? null,
    fishingGph: i.fishingGph ?? null,
    fuelGal: i.fuelGal ?? null,
    fuelUsdPerGal: i.fuelUsdPerGal ?? null,
    fuelShared: i.fuelShared !== false,
    feesPctOfGross: i.feesPctOfGross ?? null,
    feesUsdPerLb: i.feesUsdPerLb ?? null,
    "crew.kind": crewKind,
    "crew.pct": crewKind === "none" ? null : (i.crew?.pct ?? null),
    "crew.crew": i.crew?.crew ?? null,
    "crew.usdPerDay": i.crew?.usdPerDay ?? null,
    "crew.days": i.crew?.days ?? null,
    anglers,
  };
  (i.catch ?? []).forEach((c, idx) => {
    out[`catch.${idx}.species`] = c.species;
    out[`catch.${idx}.lb`] = c.lb;
    out[`catch.${idx}.usdPerLb`] = c.usdPerLb;
  });
  (i.costs ?? []).forEach((c, idx) => {
    out[`costs.${idx}.label`] = c.label;
    out[`costs.${idx}.usd`] = c.usd;
    out[`costs.${idx}.shared`] = c.shared;
  });
  return out;
}

export type InputDiff = { field: FieldKey; a: FieldValue; b: FieldValue };
export type OutputDelta = { output: ScalarOutputKey; a: number | null; b: number | null; /** b − a, when both have a value */ delta: number | null };
export type ScenarioComparison = { inputs: InputDiff[]; outputs: OutputDelta[]; a: TripEvaluation; b: TripEvaluation };

const SCALAR_KEYS = OUTPUT_KEYS.filter((k): k is ScalarOutputKey => k !== "sensitivity" && k !== "fuelSensitivity");

/** Two scenarios side by side: only the inputs that differ, and each result for both with b − a. */
export function compareScenarios(a: TripScenario, b: TripScenario, opts: { audience?: Audience } = {}): ScenarioComparison {
  const fa = scenarioFields(a);
  const fb = scenarioFields(b);
  const keys = uniq([...Object.keys(fa), ...Object.keys(fb)] as FieldKey[]).sort((x, y) => fieldOrder(x) - fieldOrder(y));
  const inputs: InputDiff[] = keys
    .filter((k) => (fa[k] ?? null) !== (fb[k] ?? null))
    .map((k) => ({ field: k, a: fa[k] ?? null, b: fb[k] ?? null }));
  const ea = evaluateTrip(a, opts);
  const eb = evaluateTrip(b, opts);
  const val = (o: Output) => (o.status === "value" ? o.value : null);
  const outputs: OutputDelta[] = SCALAR_KEYS.filter((k) => ea[k].status !== "notApplicable" || eb[k].status !== "notApplicable").map((k) => {
    const va = val(ea[k]);
    const vb = val(eb[k]);
    return { output: k, a: va, b: vb, delta: va !== null && vb !== null ? vb - va : null };
  });
  return { inputs, outputs, a: ea, b: eb };
}
