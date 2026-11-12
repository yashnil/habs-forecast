import { presetById, presetReferences, type FisheryPresetId, type PresetReference } from "@/content/fisheryPresets";
import { DIESEL_REF, dieselAge, priceRef, type PriceRef } from "@/content/referenceValues";
import type { FisheryId } from "@/pilot/contracts";

/**
 * The commercial fisheries My Coast plans a trip for, each tied to what Neritic already holds about it: its
 * official status record (pilot/contracts `FisheryId`), the worksheet preset whose sourced references sit
 * beside the fields (content/fisheryPresets), a CDFW price reference when one is species-exact
 * (content/referenceValues), the unit it is landed in, and whether CDFW's whale measures (RAMP) apply.
 * Recreational and shellfish harvest are not trips with a gross, so they are not listed.
 */
export type CoastFishery = {
  id: FisheryId;
  /** as a fisherman says it: "Dungeness crab" */
  label: string;
  /** the worksheet preset: catch line label, crew share references (NWFSC-121) */
  preset: FisheryPresetId | null;
  /** key into the CDFW price references, only when the reference is for this exact species */
  priceKey: string | null;
  /** market squid is landed and sold by the short ton */
  unit: "lb" | "ton";
  /** CDFW's Risk Assessment and Mitigation Program covers Dungeness crab trap gear only */
  ramp: boolean;
  /** the trip's consumables, named as this fishery uses them */
  supplies: string;
};

export const COAST_FISHERIES: CoastFishery[] = [
  { id: "dungeness_commercial", label: "Dungeness crab", preset: "dungeness", priceKey: "dungeness", unit: "lb", ramp: true, supplies: "Bait and ice" },
  { id: "rock_crab", label: "Rock crab", preset: null, priceKey: "rock-crab", unit: "lb", ramp: false, supplies: "Bait and ice" },
  { id: "spiny_lobster", label: "Spiny lobster", preset: null, priceKey: "lobster", unit: "lb", ramp: false, supplies: "Bait and ice" },
  { id: "market_squid", label: "Market squid", preset: "squid", priceKey: "squid", unit: "ton", ramp: false, supplies: "Ice" },
  { id: "salmon_commercial", label: "Salmon", preset: "salmon-troll", priceKey: null, unit: "lb", ramp: false, supplies: "Ice and bait" },
  { id: "groundfish", label: "Groundfish, hook-and-line", preset: "groundfish-hl", priceKey: null, unit: "lb", ramp: false, supplies: "Ice and bait" },
  { id: "anchovy_sardine", label: "Anchovy and sardine", preset: null, priceKey: null, unit: "lb", ramp: false, supplies: "Ice" },
];

export const coastFishery = (id: string | null | undefined): CoastFishery | null => COAST_FISHERIES.find((f) => f.id === id) ?? null;

/**
 * Recreational trips My Coast can plan: a cost-only worksheet (fuel, supplies, a split between the people
 * aboard). A recreational catch is not sold, so there is no price, gross, crew pay or net anywhere on it.
 * Listed apart from the commercial fisheries so nothing that reads `COAST_FISHERIES` gains a trip with no gross.
 */
export const RECREATIONAL_FISHERIES: CoastFishery[] = [
  { id: "dungeness_recreational", label: "Dungeness crab, recreational", preset: null, priceKey: null, unit: "lb", ramp: false, supplies: "Bait and ice" },
];

export type TripAudience = "commercial" | "recreational";
export const TRIP_FISHERIES: CoastFishery[] = [...COAST_FISHERIES, ...RECREATIONAL_FISHERIES];
/** Any fishery a trip can be planned for, commercial or recreational. */
export const tripFishery = (id: string | null | undefined): CoastFishery | null => TRIP_FISHERIES.find((f) => f.id === id) ?? null;
export const audienceOf = (f: CoastFishery | null | undefined): TripAudience => (f && RECREATIONAL_FISHERIES.some((r) => r.id === f.id) ? "recreational" : "commercial");

/** The first of the reader's followed fisheries that My Coast plans for (Today's list, nx-port-v1); none is assumed. */
export function defaultFishery(followed: readonly string[]): FisheryId | null {
  return followed.map(coastFishery).find((f): f is CoastFishery => !!f)?.id ?? null;
}

/** pounds per unit of weight, and the price factor: a $/ton is 2,000 × a $/lb */
export const LB_PER_UNIT: Record<CoastFishery["unit"], number> = { lb: 1, ton: 2000 };

/** A sourced value offered beside a field (T47): it enters only on a tap, then stays labelled. */
export type CoastReference = {
  /** what the field receives, in the field's own unit, as typed text */
  value: string;
  /** the offer: "$3.51/lb, 2024 statewide average" */
  words: string;
  /** who: "CDFW" */
  source: string;
  /** once taken, the field says "Your entry, from …" */
  taken: string;
  url: string;
  /** past its own stale limit (the diesel week) */
  stale?: boolean;
};

const money = (v: number, d = 2) => v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });

/** The CDFW past-average price, converted to the fishery's unit. Never a neighbouring species. */
export function priceReference(f: CoastFishery | null): CoastReference | null {
  const r: PriceRef | null = f ? priceRef(f.priceKey) : null;
  if (!f || !r) return null;
  const per = r.usdPerLb * LB_PER_UNIT[f.unit];
  const unit = f.unit === "ton" ? "ton" : "lb";
  const value = f.unit === "ton" ? String(Math.round(per)) : per.toFixed(2);
  const year = r.period.match(/\d{4}/)?.[0] ?? "";
  return {
    value,
    words: `$${f.unit === "ton" ? money(per, 0) : money(per)}/${unit}, a ${year} statewide average`,
    source: "CDFW",
    taken: `CDFW's ${year} statewide average${r.caveat ? ` (${r.caveat})` : ""}, a past average and not today's dock price`,
    url: r.url,
  };
}

/**
 * EIA's California on-road diesel week: a dated static value, old after 14 days, never a marine dock price.
 * Not offered on a page whose clock is before that week (a reference cannot be read before it was published).
 */
export function fuelReference(now: Date | null): CoastReference | null {
  const { days, stale } = dieselAge(now);
  if (days != null && days < 0) return null;
  const week = new Date(`${DIESEL_REF.week}T12:00:00Z`).toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "UTC" });
  return {
    value: DIESEL_REF.usdPerGal.toFixed(2),
    words: `$${DIESEL_REF.usdPerGal.toFixed(2)}/gal, road diesel, week of ${week}${stale ? " (old)" : ""}`,
    source: "EIA",
    taken: `EIA's California on-road diesel price for the week of ${week}, with road taxes; a fuel dock's price differs`,
    url: DIESEL_REF.url,
    stale,
  };
}

/** NOAA's 2008 fleet survey share, offered only when crew is paid a share after costs (the survey's own basis). */
export function shareReference(f: CoastFishery | null): CoastReference | null {
  const p = presetById(f?.preset);
  const r: PresetReference | undefined = presetReferences(p, { crewKind: "afterExpenses" }).find((x) => x.field === "crew.pct");
  if (!r) return null;
  return {
    value: String(r.value),
    words: `${r.value} %, ${r.label}`,
    source: "NOAA 2008 fleet survey",
    taken: `NOAA's 2008 fleet survey (${r.label}), a fleet average then and not a norm now`,
    url: r.sourceUrl,
  };
}
