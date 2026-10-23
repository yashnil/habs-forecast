import { usd } from "@/lib/tripEconomics";
import { evaluateTrip, fieldLabel, type Audience as EconAudience, type FieldKey, type Output, type TripEvaluation } from "@/lib/tripEvaluation";
import { draftInputs, type Draft } from "@/lib/tripEconomicsStore";
import type { Audience } from "@/lib/coast";

/**
 * The worksheet's adapter over `evaluateTrip` (the engine of record): which audience a trip is worked
 * for, money as the page writes it, and the words for the fields a result waits for. No arithmetic here.
 */

/** Recreational and shellfish trips are cost-only (nothing is sold); commercial and "any" get the full worksheet. */
export const econAudience = (a: Audience): EconAudience => (a === "recreational" || a === "shellfish" ? "recreational" : "commercial");

export function evaluateDraft(d: Draft, a: Audience): TripEvaluation {
  return evaluateTrip(draftInputs(d), { audience: econAudience(a) });
}

export const money = (v: number | null, signed = false) => usd(v, { signed });

/** What a "Needs …" link says: the field in plain words ("fuel used" for the gallons-or-trip route). */
export function needLabel(k: FieldKey, d?: Draft | null): string {
  if (k === "fuelGal") return "fuel used";
  const m = /^catch\.(\d+)\.(lb|usdPerLb)$/.exec(k);
  if (m) {
    const lines = d?.inputs.catch ?? [];
    const what = m[2] === "lb" ? "pounds landed" : "price per pound";
    return lines.length > 1 ? `${what} (${lines[Number(m[1])]?.species || `line ${Number(m[1]) + 1}`})` : what;
  }
  return fieldLabel(k);
}

export const valueOf = (o: Output): number | null => (o.status === "value" ? o.value : null);
export const gapsOf = (o: Output<unknown>): FieldKey[] => (o.status === "missing" ? o.missing : []);

/** The fuel-route keys (the trip's distance, speed and burn). */
export const TRIP_FUEL_KEYS: FieldKey[] = ["roundTripNm", "cruiseKt", "steamGph", "fishingHours", "fishingGph"];

/** Anything at all typed in a draft (so an untouched worksheet can say "nothing entered yet"). */
export function hasInput(d: Draft): boolean {
  const i = d.inputs;
  const nums = [i.roundTripNm, i.cruiseKt, i.steamGph, i.fishingHours, i.fishingGph, i.fuelGal, i.fuelUsdPerGal, i.feesPctOfGross, i.feesUsdPerLb, i.anglers, i.crew?.pct, i.crew?.crew, i.crew?.usdPerDay, i.crew?.days, d.pots, d.baitUsdPerPot];
  return nums.some((v) => v != null) || !!i.crew?.kind || (i.catch ?? []).some((c) => c.lb != null || c.usdPerLb != null) || (i.costs ?? []).some((c) => c.usd != null);
}
