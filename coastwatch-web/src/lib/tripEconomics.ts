/**
 * Trip economics: the arithmetic of one fishing trip from the fisherman's own numbers.
 *
 * Pure (no React, no window). Floats inside, rounding only at display. Every input may be missing:
 * the calculator returns what it can and lists what it still needs. It never supplies a catch or a
 * price; a scenario is the user's inputs worked through, not a forecast.
 *
 * Notation (docs/coastwatch/v6/plan.md, "Trip economics specification"):
 *   R  gross = Σ lb × $/lb over the catch lines
 *   τ  fees as a fraction of gross;  u  fees in $ per lb;  Q  total lb landed
 *   D  shared costs (deducted before crew shares);  B  boat-paid costs
 *   s  crew share as a fraction
 * Model A, share after expenses: N = R(1−τ) − uQ − D; crew = s·N (0 if N ≤ 0); owner net = N − crew − B.
 * Model B, share of gross:        crew = s·R; owner net = R(1−s−τ) − uQ − D − B.
 * Model C, day rate:              crew = people × $/day × days (boat-paid); owner net = R(1−τ) − uQ − D − B − crew.
 */

export type CrewKind = "afterExpenses" | "ofGross" | "dayRate";

export type CrewModel = {
  kind: CrewKind;
  /** total crew share, percent (A and B) */
  pct?: number | null;
  /** people paid a day rate (C) */
  crew?: number | null;
  usdPerDay?: number | null;
  days?: number | null;
};

export type CostLine = { label: string; usd: number | null; shared: boolean };
/** `species` is the line's label ("Dungeness crab", "Red urchin, grade 1"); `ref` keys its CDFW reference, if any. */
export type CatchLine = { species: string; ref?: string | null; lb: number | null; usdPerLb: number | null };

export type TripInputs = {
  roundTripNm?: number | null;
  cruiseKt?: number | null;
  steamGph?: number | null;
  fishingHours?: number | null;
  fishingGph?: number | null;
  /** gallons for the trip, entered directly; replaces the steaming + fishing estimate when set */
  fuelGal?: number | null;
  fuelUsdPerGal?: number | null;
  /** fuel is deducted before crew shares (the usual "deductions"); false = the boat pays it */
  fuelShared?: boolean;
  costs?: CostLine[];
  feesPctOfGross?: number | null;
  feesUsdPerLb?: number | null;
  /** the primary line first */
  catch?: CatchLine[];
  crew?: CrewModel;
};

export type MissingKey = "fuel" | "fuelPrice" | "catch" | "price" | "crewShare" | "dayRate";

export const MISSING_LABEL: Record<MissingKey, string> = {
  fuel: "fuel use (gallons, or distance, speed and burn)",
  fuelPrice: "fuel price",
  catch: "pounds landed",
  price: "price per pound",
  crewShare: "crew share",
  dayRate: "crew day rate",
};

export type BreakEven = { value: number | null; note: "ok" | "covered" | "none" | "unknown" };

export type SensitivityCell = { priceDelta: number; catchDelta: number; netUsd: number };

export type TripResult = {
  fuelGal: number | null;
  fuelUsd: number | null;
  /** fuel + every cost line (shared and boat-paid); crew pay and fees are separate rows */
  tripCostUsd: number | null;
  grossUsd: number | null;
  feesUsd: number | null;
  crewUsd: number | null;
  netUsd: number | null;
  totalLb: number | null;
  /** lb of the primary line at which owner net is 0, other lines held fixed */
  breakEvenLb: BreakEven;
  /** $/lb for the primary line at which owner net is 0, at its entered catch */
  breakEvenUsdPerLb: BreakEven;
  /** $/gal at which owner net is 0 */
  breakEvenFuelUsdPerGal: BreakEven;
  /** price −20 / 0 / +20 % × catch −30 / 0 / +30 % (all lines); the centre equals netUsd */
  sensitivity: SensitivityCell[] | null;
  /** owner net with fuel price −20 % and +20 % */
  fuelSensitivity: { down: number; up: number } | null;
  missing: MissingKey[];
  warnings: string[];
};

// ---------------------------------------------------------------- parse and clamp

/** The largest value each kind of field accepts; larger entries are clamped (and said so by the UI). */
export const LIMITS = {
  nm: 2000,
  kt: 60,
  gph: 500,
  hours: 24 * 21,
  gal: 50_000,
  usdPerGal: 50,
  usd: 1_000_000,
  pct: 100,
  usdPerLb: 1000,
  lb: 2_000_000,
  people: 50,
  usdPerDay: 10_000,
  days: 60,
  pots: 1000,
} as const;
export type LimitKind = keyof typeof LIMITS;

/**
 * Read what someone typed: "1,234.5", "$3.51", " 28 % ". Empty → null. NaN, infinities and
 * negatives are rejected (error); values over the field's limit are clamped (note).
 */
export function parseField(raw: string, kind: LimitKind): { value: number | null; error: string | null; clamped: boolean } {
  const t = raw.replace(/[$,%\s]/g, "");
  if (t === "") return { value: null, error: null, clamped: false };
  if (!/^[-+]?(\d+\.?\d*|\.\d+)$/.test(t)) return { value: null, error: "Enter a number", clamped: false };
  const n = Number(t);
  if (!Number.isFinite(n)) return { value: null, error: "Enter a number", clamped: false };
  if (n < 0) return { value: null, error: "Can't be negative", clamped: false };
  const max = LIMITS[kind];
  return n > max ? { value: max, error: null, clamped: true } : { value: n, error: null, clamped: false };
}

/** A stored or linked number made safe: finite, ≥ 0, within its limit; anything else is null. */
export function clampNum(v: unknown, kind: LimitKind): number | null {
  if (typeof v !== "number" || !Number.isFinite(v) || v < 0) return null;
  return Math.min(v, LIMITS[kind]);
}

/** Every number in a set of inputs made safe (used on anything read back from storage or a link). */
export function sanitizeInputs(i: TripInputs): TripInputs {
  const crew = i.crew ?? { kind: "afterExpenses" };
  const kind: CrewKind = crew.kind === "ofGross" || crew.kind === "dayRate" ? crew.kind : "afterExpenses";
  return {
    roundTripNm: clampNum(i.roundTripNm, "nm"),
    cruiseKt: clampNum(i.cruiseKt, "kt"),
    steamGph: clampNum(i.steamGph, "gph"),
    fishingHours: clampNum(i.fishingHours, "hours"),
    fishingGph: clampNum(i.fishingGph, "gph"),
    fuelGal: clampNum(i.fuelGal, "gal"),
    fuelUsdPerGal: clampNum(i.fuelUsdPerGal, "usdPerGal"),
    fuelShared: i.fuelShared !== false,
    costs: (Array.isArray(i.costs) ? i.costs : []).slice(0, 20).map((c) => ({ label: typeof c?.label === "string" ? c.label.slice(0, 60) : "", usd: clampNum(c?.usd, "usd"), shared: c?.shared !== false })),
    feesPctOfGross: clampNum(i.feesPctOfGross, "pct"),
    feesUsdPerLb: clampNum(i.feesUsdPerLb, "usdPerLb"),
    catch: (Array.isArray(i.catch) ? i.catch : []).slice(0, 10).map((c) => ({ species: typeof c?.species === "string" ? c.species.slice(0, 60) : "", ref: typeof c?.ref === "string" ? c.ref.slice(0, 30) : null, lb: clampNum(c?.lb, "lb"), usdPerLb: clampNum(c?.usdPerLb, "usdPerLb") })),
    crew: { kind, pct: clampNum(crew.pct, "pct"), crew: clampNum(crew.crew, "people"), usdPerDay: clampNum(crew.usdPerDay, "usdPerDay"), days: clampNum(crew.days, "days") },
  };
}

// ---------------------------------------------------------------- units

export const LB_PER_KG = 2.2046226218;
/** a US short ton, the unit CDFW and buyers use for market squid */
export const LB_PER_TON = 2000;
export const lbToKg = (lb: number) => lb / LB_PER_KG;
export const kgToLb = (kg: number) => kg * LB_PER_KG;
export const usdPerLbToPerKg = (p: number) => p * LB_PER_KG;
export const usdPerKgToPerLb = (p: number) => p / LB_PER_KG;
export const usdPerTonToPerLb = (p: number) => p / LB_PER_TON;
export const usdPerLbToPerTon = (p: number) => p * LB_PER_TON;

// ---------------------------------------------------------------- the calculator

const known = (v: number | null | undefined): v is number => typeof v === "number" && Number.isFinite(v);

/** Gallons for the trip: entered directly, else steaming (nm ÷ kt × gph) + fishing (h × gph). */
export function fuelGallons(i: TripInputs): number | null {
  if (known(i.fuelGal)) return i.fuelGal;
  const steamKnown = known(i.roundTripNm) && known(i.cruiseKt) && i.cruiseKt > 0 && known(i.steamGph);
  const steamNone = !known(i.roundTripNm) || i.roundTripNm === 0;
  const fishKnown = known(i.fishingHours) && known(i.fishingGph);
  const fishNone = !known(i.fishingHours) || i.fishingHours === 0;
  if (!steamKnown && !fishKnown) return null;
  if (!steamKnown && !steamNone) return null;
  if (!fishKnown && !fishNone) return null;
  const steam = steamKnown ? (i.roundTripNm! / i.cruiseKt!) * i.steamGph! : 0;
  const fish = fishKnown ? i.fishingHours! * i.fishingGph! : 0;
  return steam + fish;
}

type Core = {
  R: number;
  Q: number;
  tau: number;
  u: number;
  D: number;
  B: number;
  kind: CrewKind;
  s: number;
  dayCrew: number;
};

/** Owner net and crew pay for fully known terms (crew clamped to 0 when model A's N ≤ 0). */
function evaluate(c: Core): { net: number; crew: number; fees: number } {
  const fees = c.R * c.tau + c.u * c.Q;
  if (c.kind === "afterExpenses") {
    const N = c.R * (1 - c.tau) - c.u * c.Q - c.D;
    const crew = N > 0 ? c.s * N : 0;
    return { net: N - crew - c.B, crew, fees };
  }
  if (c.kind === "ofGross") {
    const crew = c.s * c.R;
    return { net: c.R * (1 - c.s - c.tau) - c.u * c.Q - c.D - c.B, crew, fees };
  }
  return { net: c.R * (1 - c.tau) - c.u * c.Q - c.D - c.B - c.dayCrew, crew: c.dayCrew, fees };
}

/** The same without model A's clamp: linear in catch, price and fuel price, so its root is the break-even. */
function linearNet(c: Core): number {
  if (c.kind === "afterExpenses") return (1 - c.s) * (c.R * (1 - c.tau) - c.u * c.Q - c.D) - c.B;
  return evaluate(c).net;
}

/** x where a linear f(x) crosses 0, from f(0) and f(1); slope must have the given sign. */
function root(f0: number, f1: number, sign: 1 | -1): number | null {
  const slope = f1 - f0;
  if (!(slope * sign > 1e-12)) return null;
  return -f0 / slope;
}

export function computeTrip(raw: TripInputs): TripResult {
  const i = sanitizeInputs(raw);
  const missing = new Set<MissingKey>();
  const warnings: string[] = [];

  // fuel
  const fuelGal = fuelGallons(i);
  if (fuelGal === null) missing.add("fuel");
  const fuelPrice = i.fuelUsdPerGal ?? null;
  if (fuelGal !== null && fuelGal > 0 && fuelPrice === null) missing.add("fuelPrice");
  const fuelUsd = fuelGal === null ? null : fuelGal === 0 ? 0 : fuelPrice === null ? null : fuelGal * fuelPrice;

  // other costs (a blank line is simply not counted)
  let sharedOther = 0;
  let boatOther = 0;
  for (const c of i.costs ?? []) {
    if (!known(c.usd)) continue;
    if (c.shared) sharedOther += c.usd;
    else boatOther += c.usd;
  }
  const fuelShared = i.fuelShared !== false;
  const tripCostUsd = fuelUsd === null ? null : fuelUsd + sharedOther + boatOther;

  // catch
  const lines = (i.catch ?? []).filter((l) => l.species !== "" || known(l.lb) || known(l.usdPerLb));
  if (lines.length === 0) {
    missing.add("catch");
    missing.add("price");
  }
  for (const l of lines) {
    if (!known(l.lb)) missing.add("catch");
    if (!known(l.usdPerLb)) missing.add("price");
  }
  const grossKnown = lines.length > 0 && lines.every((l) => known(l.lb) && known(l.usdPerLb));
  const R = grossKnown ? lines.reduce((a, l) => a + l.lb! * l.usdPerLb!, 0) : null;
  const Q = lines.length > 0 && lines.every((l) => known(l.lb)) ? lines.reduce((a, l) => a + l.lb!, 0) : null;

  // fees (blank = none)
  const tau = (i.feesPctOfGross ?? 0) / 100;
  const u = i.feesUsdPerLb ?? 0;

  // crew
  const crew = i.crew!;
  let s = 0;
  let dayCrew = 0;
  if (crew.kind === "dayRate") {
    if (!known(crew.crew) || !known(crew.usdPerDay) || !known(crew.days)) {
      if (crew.crew !== 0) missing.add("dayRate");
    } else dayCrew = crew.crew * crew.usdPerDay * crew.days;
    if (crew.crew === 0) dayCrew = 0;
  } else if (!known(crew.pct)) missing.add("crewShare");
  else s = crew.pct / 100;

  if (crew.kind !== "dayRate" && tau + s >= 1) warnings.push("Fees and crew share together take all of the gross: no catch covers the costs.");
  else if (crew.kind === "dayRate" && tau >= 1) warnings.push("Fees take all of the gross: no catch covers the costs.");

  const crewKnown = !missing.has("crewShare") && !missing.has("dayRate");
  const costKnown = fuelUsd !== null;
  const D0 = sharedOther + (fuelShared ? fuelUsd ?? 0 : 0);
  const B0 = boatOther + (fuelShared ? 0 : fuelUsd ?? 0);

  const base: Core | null = R !== null && Q !== null && crewKnown && costKnown ? { R, Q, tau, u, D: D0, B: B0, kind: crew.kind, s, dayCrew } : null;
  const ev = base ? evaluate(base) : null;
  const feesUsd = R !== null && Q !== null ? R * tau + u * Q : null;
  const crewUsd = ev ? ev.crew : crew.kind === "dayRate" && crewKnown ? dayCrew : crew.kind === "ofGross" && R !== null && crewKnown ? s * R : null;

  // break-evens
  const unknownBE: BreakEven = { value: null, note: "unknown" };
  let breakEvenLb = unknownBE;
  let breakEvenUsdPerLb = unknownBE;
  let breakEvenFuel = unknownBE;
  const primary = lines[0];
  const others = lines.slice(1);
  const othersKnown = others.every((l) => known(l.lb) && known(l.usdPerLb));
  if (primary && crewKnown && costKnown && othersKnown) {
    const Ro = others.reduce((a, l) => a + l.lb! * l.usdPerLb!, 0);
    const Qo = others.reduce((a, l) => a + l.lb!, 0);
    const at = (q: number, p: number): Core => ({ R: Ro + q * p, Q: Qo + q, tau, u, D: D0, B: B0, kind: crew.kind, s, dayCrew });
    if (known(primary.usdPerLb)) {
      const p = primary.usdPerLb;
      const q = root(linearNet(at(0, p)), linearNet(at(1, p)), 1);
      breakEvenLb = q === null ? { value: null, note: "none" } : q <= 0 ? { value: 0, note: "covered" } : { value: q, note: "ok" };
    }
    if (known(primary.lb)) {
      const q = primary.lb;
      const p = root(linearNet(at(q, 0)), linearNet(at(q, 1)), 1);
      breakEvenUsdPerLb = p === null ? { value: null, note: "none" } : p <= 0 ? { value: 0, note: "covered" } : { value: p, note: "ok" };
    }
  }
  if (base && fuelGal !== null) {
    const withFuel = (f: number): Core => {
      const fu = fuelGal * f;
      return { ...base, D: sharedOther + (fuelShared ? fu : 0), B: boatOther + (fuelShared ? 0 : fu) };
    };
    const f = root(linearNet(withFuel(0)), linearNet(withFuel(1)), -1);
    // f is where net falls to 0 as fuel rises; none when fuel does not move net (no gallons) or net < 0 even with free fuel
    breakEvenFuel = f === null ? { value: null, note: "none" } : f < 0 ? { value: null, note: "none" } : { value: f, note: "ok" };
  }

  // sensitivity
  let sensitivity: SensitivityCell[] | null = null;
  let fuelSensitivity: TripResult["fuelSensitivity"] = null;
  if (base && ev) {
    sensitivity = [];
    for (const pd of [-0.2, 0, 0.2]) {
      for (const cd of [-0.3, 0, 0.3]) {
        if (pd === 0 && cd === 0) {
          sensitivity.push({ priceDelta: 0, catchDelta: 0, netUsd: ev.net });
          continue;
        }
        const Rx = lines.reduce((a, l) => a + l.lb! * (1 + cd) * l.usdPerLb! * (1 + pd), 0);
        sensitivity.push({ priceDelta: pd, catchDelta: cd, netUsd: evaluate({ ...base, R: Rx, Q: Q! * (1 + cd) }).net });
      }
    }
    if (fuelUsd !== null) {
      const at = (k: number) => {
        const fu = fuelUsd * k;
        return evaluate({ ...base, D: sharedOther + (fuelShared ? fu : 0), B: boatOther + (fuelShared ? 0 : fu) }).net;
      };
      fuelSensitivity = { down: at(0.8), up: at(1.2) };
    }
  }

  return {
    fuelGal,
    fuelUsd,
    tripCostUsd,
    grossUsd: R,
    feesUsd,
    crewUsd,
    netUsd: ev ? ev.net : null,
    totalLb: Q,
    breakEvenLb,
    breakEvenUsdPerLb,
    breakEvenFuelUsdPerGal: breakEvenFuel,
    sensitivity,
    fuelSensitivity,
    missing: [...missing],
    warnings,
  };
}

// ---------------------------------------------------------------- display

/** "$1,234" (whole dollars), "−$56"; signed adds "+" to positives. */
export function usd(v: number | null, opts: { signed?: boolean; cents?: boolean } = {}): string {
  if (v === null || !Number.isFinite(v)) return "—";
  const d = opts.cents ? 2 : 0;
  const r = Math.abs(v) < (opts.cents ? 0.005 : 0.5) ? 0 : v;
  const s = Math.abs(r).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
  const sign = r < 0 ? "−" : opts.signed && r > 0 ? "+" : "";
  return `${sign}$${s}`;
}

/** "1,234" with up to `d` decimals. */
export function num(v: number | null, d = 0): string {
  if (v === null || !Number.isFinite(v)) return "—";
  return v.toLocaleString("en-US", { maximumFractionDigits: d, minimumFractionDigits: 0 });
}
