import type { SeasonGate, WhaleGate } from "./brief";
import type { CoastFishery } from "./fisheries";
import { breakEvenCatchText, breakEvenPriceText, evaluateWorkspace, fieldWords, money, read, setCatch, type FieldId, type Step, type TripResult, type Workspace } from "./workspace";

/**
 * The trip pass's own arithmetic and words (v14 synthesis, art/concept-4-synthesis): the order the waterfall reads
 * in, the what-if ends, the one action that walks the blanks, and the season line in the record's words. Pure; every
 * figure comes from the engine through `evaluateWorkspace` (a what-if end is the trip worked again with that one
 * input set to the end's value, exactly what tapping it does).
 */

// ---------------------------------------------------------------- what if

/** Catch moves a quarter either way, in steps of 5 (never less than 5). */
export const catchStep = (amount: number) => Math.max(5, Math.round((amount * 0.25) / 5) * 5);

/**
 * Price moves $1 a pound between $4 and $20 a pound; otherwise, or in another unit, a tenth (to the cent, never less
 * than 5¢), since a dollar is too small or too large a step to mean anything there ($99 → $89.10 / $108.90).
 */
export const priceStep = (price: number, unit: "lb" | "ton" = "lb") => (unit === "lb" && price >= 4 && price <= 20 ? 1 : Math.max(0.05, Math.round(price * 10) / 100));

export type WhatIfEnd = { v: number; label: string; net: number };
export type WhatIfRow = { k: "catch" | "price"; label: string; lo: WhatIfEnd; hi: WhatIfEnd };

const round2 = (v: number) => Math.round(v * 100) / 100;
const amountWords = (v: number, unit: string) => `${v.toLocaleString("en-US", { maximumFractionDigits: 1 })} ${unit}`;

/**
 * The two what-if rows (catch and price), each end the trip worked again with only that input changed. Null while net
 * to boat has no value, for a recreational trip (no sale), or when the trip has more than one catch line (an end sets
 * one line's input; with several lines it would not be the trip the row names).
 */
export function whatIfRows(ws: Workspace, f: CoastFishery | null, result: TripResult): WhatIfRow[] | null {
  if (result.audience !== "commercial" || result.out.net.status !== "value" || ws.catch.length !== 1) return null;
  const { n } = read(ws, f);
  const c = n.catch[0];
  if (!c || c.amount === null || c.price === null) return null;
  const unit = f?.unit ?? "lb";
  const netWith = (patch: { amount?: string; price?: string }): number | null => {
    const r = evaluateWorkspace(setCatch(ws, 0, patch), f).out.net;
    return r.status === "value" ? r.value : null;
  };
  const rows: WhatIfRow[] = [];
  const dc = catchStep(c.amount);
  const c0 = Math.max(0, c.amount - dc);
  const c1 = c.amount + dc;
  const n0 = netWith({ amount: String(c0) });
  const n1 = netWith({ amount: String(c1) });
  if (n0 !== null && n1 !== null) rows.push({ k: "catch", label: "Catch", lo: { v: c0, label: amountWords(c0, unit), net: n0 }, hi: { v: c1, label: amountWords(c1, unit), net: n1 } });
  const dp = priceStep(c.price, unit);
  const p0 = Math.max(0, round2(c.price - dp));
  const p1 = round2(c.price + dp);
  const m0 = netWith({ price: String(p0) });
  const m1 = netWith({ price: String(p1) });
  if (m0 !== null && m1 !== null) rows.push({ k: "price", label: "Price", lo: { v: p0, label: `${money(p0)}/${unit}`, net: m0 }, hi: { v: p1, label: `${money(p1)}/${unit}`, net: m1 } });
  return rows.length ? rows : null;
}

// ---------------------------------------------------------------- the waterfall

/** The waterfall in the equation's order: gross, fuel, the cost lines, fees, crew, net (recreational: fuel, costs, trip cost). */
export function passSteps(steps: Step[]): Step[] {
  const rank = (s: Step) => (s.id === "gross" ? 0 : s.id === "fuel" ? 1 : s.id.startsWith("cost-") ? 2 : s.id === "fees" ? 3 : s.id === "crew" ? 4 : 5);
  return [...steps].sort((a, b) => rank(a) - rank(b));
}

/** The gross that just covers the costs (the break-even catch at the price entered), for the caliper on the gross column. */
export function breakEvenGross(ws: Workspace, f: CoastFishery | null, result: TripResult): { gross: number; words: string; beyond: boolean } | null {
  const be = result.out.breakEvenCatch;
  const gross = result.out.gross;
  if (result.audience !== "commercial" || be.status !== "value" || gross.status !== "value" || ws.catch.length !== 1) return null;
  const { n } = read(ws, f);
  const price = n.catch[0]?.price;
  if (price == null || price <= 0) return null;
  const unit = f?.unit ?? "lb";
  const at = be.value * price;
  const bp = result.out.breakEvenPrice;
  const beyond = at > gross.value + 1e-9;
  return { gross: at, beyond, words: `${breakEvenCatchText(be.value, unit)}${beyond ? ", more than this catch" : bp.status === "value" ? `, or ${breakEvenPriceText(bp.value, unit)}` : ""}` };
}

// ---------------------------------------------------------------- the one action

/** What the pass's filled button says while a blank is left: "Add fuel", "Choose how crew is paid". */
export function nextWords(id: FieldId, ws: Workspace, f: CoastFishery | null): string {
  if (id === "crewKind" || id === "fuelPayer" || /^cost\.\d+\.payer$/.test(id)) return `Choose ${fieldWords(id, ws, f)}`;
  if (id === "fuelGal") return "Add fuel";
  if (/^catch\.0\.amount$/.test(id) && ws.catch.length === 1) return "Add catch";
  if (/^catch\.0\.price$/.test(id) && ws.catch.length === 1) return "Add price";
  return `Add ${fieldWords(id, ws, f)}`;
}

/** "Needs fuel, bait and ice, fees and crew. A blank is a gap, never 0." */
export function needsWords(fields: FieldId[], ws: Workspace, f: CoastFishery | null): string {
  const w = [...new Set(fields.map((x) => (x === "fuelGal" ? "fuel" : fieldWords(x, ws, f))))];
  if (!w.length) return "A blank is a gap, never 0.";
  const list = w.length > 4 ? `${w.slice(0, 3).join(", ")} and ${w.length - 3} more` : w.length === 1 ? w[0] : `${w.slice(0, -1).join(", ")} and ${w[w.length - 1]}`;
  return `Needs ${list}. A blank is a gap, never 0.`;
}

/** Each field's element id in the pass, so the button can take the reader to it. */
export const fieldDomId = (id: FieldId) => `trip-f-${id.replace(/\./g, "-")}`;

// ---------------------------------------------------------------- the record's words

/**
 * The season line on the pass, in the record's own words with the cause left to the gates and the sector to the strip
 * ("Season delayed to at least Jan 1, 2027: a plan, not a trip you can make now"). Only a delayed or closed record (not one in conflict).
 */
export function seasonWarning(season: SeasonGate | null): string | null {
  if (!season || season.conflict || (season.level !== "delayed" && season.level !== "closed")) return null;
  // the fishery (commercial or recreational) is named beside it on the strip
  const said = season.text.replace(/\s*\([^)]*\)/g, "").replace(/[.\s]+$/, "").replace(/^(commercial|recreational)\s+(\w)/i, (_, _k, c: string) => c.toUpperCase());
  return `${said}: a plan, not a trip you can make now`;
}

/** The whale zone tag's second line in the records' words: "Season delayed · Fleet Advisory". */
export function zoneCaption(season: SeasonGate | null, whale: WhaleGate | null): string | null {
  const parts: string[] = [];
  if (season && !season.conflict && season.level === "delayed") parts.push("Season delayed");
  else if (season && !season.conflict && season.level === "closed" && !season.seasonal) parts.push("Season closed");
  const first = whale?.measures.split(/(?<=\.)\s/)[0]?.replace(/\.$/, "").trim();
  if (first && !/^no\b/i.test(first) && first.length <= 28) parts.push(first);
  return parts.length ? parts.join(" · ") : null;
}
