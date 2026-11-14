import type { Draft } from "@/lib/tripEconomicsStore";
import type { RefSnapshot } from "@/content/referenceValues";
import type { FisheryId } from "@/pilot/contracts";
import { TRIP_SCHEMA, type Activity, type Change, type Departure, type Lifecycle, type OperatingLimits, type Outcome, type Recheck, type Trip } from "./types";

/**
 * The trip's lifecycle (v15): what a trip can become, what blocks it, and the one thing worth doing next.
 *
 * Draft → Planned → Completed, and Cancelled from Draft or Planned (reopened to Planned, or to Draft when it no
 * longer has what a plan needs). A recheck is an action, never a state: nothing here, and no field of a trip, says
 * a trip is safe, legal, approved or profitable. Pure: every function takes the trip and the time and returns a new
 * trip; nothing is written anywhere (the store is ./store.ts).
 */

export type Result = { ok: true; trip: Trip } | { ok: false; reason: Refusal; words: string };
export type Refusal = "missing" | "state" | "gone";
export type PlanNeeds = "port" | "fishery" | "date";

/** How many earlier versions of the economics a trip keeps (newest first). */
export const MAX_HISTORY = 12;

export const STATE_WORDS: Record<Lifecycle, string> = { draft: "Draft", planned: "Planned", completed: "Completed", cancelled: "Cancelled" };

const iso = (now: Date) => now.toISOString();
const DATE = /^\d{4}-\d{2}-\d{2}$/;
const TIME = /^\d{2}:\d{2}$/;

// ---------------------------------------------------------------- creating

export type NewTrip = {
  id: string;
  activity: Activity;
  fishery: FisheryId;
  placeKey: string;
  depart?: Departure | null;
  departKey?: string | null;
  area?: string | null;
  limits?: Partial<OperatingLimits>;
  economics?: Draft | null;
  refs?: RefSnapshot[];
  notes?: string;
  vesselId?: string | null;
  /** the reader's own name; blank: Neritic writes one from the details (and may rewrite it while they change) */
  name?: string | null;
  /** names for the automatic name */
  words?: NameWords;
};

export type NameWords = { port: string | null; fishery: string | null };

/** "Dungeness crab from Moss Landing, Nov 14" — what the trip is, never a verdict. */
export function autoName(t: Pick<Trip, "depart">, w: NameWords): string {
  const what = w.fishery ?? "Trip";
  const where = w.port ? ` from ${w.port}` : "";
  const when = t.depart ? `, ${shortDate(t.depart.date)}` : "";
  return `${what}${where}${when}`.slice(0, 90);
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
/** "2026-11-14" → "Nov 14" (a calendar date, no time zone involved). */
export function shortDate(d: string): string {
  if (!DATE.test(d)) return d;
  const [, m, day] = d.split("-").map(Number);
  return `${MONTHS[m - 1]} ${day}`;
}

/** A departure as the reader gave it, or null when the date is not a calendar date. */
export function departure(date: string | null | undefined, time: string | null | undefined, timeZone = "America/Los_Angeles"): Departure | null {
  if (!date || !DATE.test(date)) return null;
  return { date, time: time && TIME.test(time) ? time : null, timeZone };
}

/** Blank, zero-or-less or not a number is "none set", never 0 (a limit of 0 kt is not a limit anyone means). */
export function limitValue(v: unknown): number | null {
  const n = typeof v === "string" ? Number(v.trim() === "" ? NaN : v) : typeof v === "number" ? v : NaN;
  return Number.isFinite(n) && n > 0 ? Math.round(n * 10) / 10 : null;
}

/** A new trip, saved as a draft (the store keeps it). Plan it with `plan`. */
export function draftTrip(n: NewTrip, now: Date): Trip {
  const at = iso(now);
  const depart = n.depart ?? null;
  const own = n.name?.trim();
  const t: Trip = {
    schema: TRIP_SCHEMA,
    id: n.id,
    name: "",
    autoName: !own,
    activity: n.activity,
    fishery: n.fishery,
    placeKey: n.placeKey,
    departKey: n.departKey ?? null,
    depart,
    area: n.area?.trim() ? n.area.trim().slice(0, 120) : null,
    vesselId: n.vesselId ?? null,
    limits: { windKt: limitValue(n.limits?.windKt), seasFt: limitValue(n.limits?.seasFt) },
    economics: n.economics ?? null,
    refs: n.refs ?? [],
    history: [],
    notes: n.notes ?? "",
    state: "draft",
    createdAt: at,
    updatedAt: at,
    snapshots: { planning: null, rechecks: [], superseded: [] },
    outcome: null,
  };
  t.name = own ? own.slice(0, 90) : autoName(t, n.words ?? { port: null, fishery: null });
  return t;
}

// ---------------------------------------------------------------- guards

/** What a plan still needs: a port, a fishery and a calendar date. */
export function planNeeds(t: Pick<Trip, "placeKey" | "fishery" | "depart">): PlanNeeds[] {
  const out: PlanNeeds[] = [];
  if (!t.placeKey) out.push("port");
  if (!t.fishery) out.push("fishery");
  if (!t.depart || !DATE.test(t.depart.date)) out.push("date");
  return out;
}

const NEED_WORDS: Record<PlanNeeds, string> = { port: "a port", fishery: "a fishery", date: "a date" };
export const needsWords = (n: PlanNeeds[]) => (n.length ? `Needs ${list(n.map((x) => NEED_WORDS[x]))}.` : "");
const list = (a: string[]) => (a.length <= 1 ? (a[0] ?? "") : `${a.slice(0, -1).join(", ")} and ${a[a.length - 1]}`);

export const canPlan = (t: Trip) => t.state === "draft" && planNeeds(t).length === 0;
export const canComplete = (t: Trip) => t.state === "planned";
export const canCancel = (t: Trip) => t.state === "draft" || t.state === "planned";
export const canReopen = (t: Trip) => t.state === "cancelled";
/** assumptions (date, limits, economics, area) change only while the trip is ahead */
export const canEditAssumptions = (t: Trip) => t.state === "draft" || t.state === "planned";

const refuse = (reason: Refusal, words: string): Result => ({ ok: false, reason, words });
const STATE_REFUSAL: Record<Lifecycle, string> = {
  draft: "This trip is a draft.",
  planned: "This trip is already planned.",
  completed: "This trip is completed.",
  cancelled: "This trip is cancelled. Reopen it first.",
};

// ---------------------------------------------------------------- transitions

/** Draft (or a reopened trip) → Planned: needs a port, a fishery and a date. */
export function plan(t: Trip, now: Date): Result {
  if (t.state !== "draft") return refuse("state", STATE_REFUSAL[t.state]);
  const needs = planNeeds(t);
  if (needs.length) return refuse("missing", needsWords(needs));
  return { ok: true, trip: { ...t, state: "planned", updatedAt: iso(now) } };
}

/** Planned → Completed, with what happened (or with the outcome still to record). */
export function complete(t: Trip, now: Date, outcome: Outcome | null = t.outcome): Result {
  if (t.state !== "planned") return refuse("state", t.state === "draft" ? "Plan the trip before marking it completed." : STATE_REFUSAL[t.state]);
  return { ok: true, trip: { ...t, state: "completed", outcome, updatedAt: iso(now) } };
}

/** Draft or Planned → Cancelled. Nothing is deleted: the plan and its snapshots stay, and it can be reopened. */
export function cancel(t: Trip, now: Date): Result {
  if (!canCancel(t)) return refuse("state", t.state === "cancelled" ? "This trip is already cancelled." : STATE_REFUSAL[t.state]);
  return { ok: true, trip: { ...t, state: "cancelled", updatedAt: iso(now) } };
}

/** Cancelled → Planned when it still has what a plan needs, else → Draft. */
export function reopen(t: Trip, now: Date): Result {
  if (t.state !== "cancelled") return refuse("state", STATE_REFUSAL[t.state]);
  return { ok: true, trip: { ...t, state: planNeeds(t).length ? "draft" : "planned", updatedAt: iso(now) } };
}

/** Record or correct what happened on a completed trip (each edit stamps updatedAt). */
export function recordOutcome(t: Trip, outcome: Omit<Outcome, "recordedAt" | "updatedAt">, now: Date): Result {
  if (t.state !== "completed" && t.state !== "planned") return refuse("state", STATE_REFUSAL[t.state]);
  const at = iso(now);
  const next: Outcome = { ...outcome, recordedAt: t.outcome?.recordedAt ?? at, updatedAt: at };
  return { ok: true, trip: { ...t, state: "completed", outcome: next, updatedAt: at } };
}

export type TripPatch = Partial<Pick<Trip, "name" | "depart" | "departKey" | "area" | "limits" | "economics" | "refs" | "notes" | "vesselId" | "fishery" | "activity" | "placeKey">>;

const ASSUMPTIONS: (keyof TripPatch)[] = ["depart", "departKey", "area", "limits", "economics", "refs", "vesselId", "fishery", "activity", "placeKey"];

/**
 * Edit a trip. Its assumptions change only while it is ahead (Draft, Planned); a name and notes change in any state.
 * Changed economics keep the earlier version in `history` (newest first, at most MAX_HISTORY). A planned trip that
 * loses its date goes back to Draft. An automatic name follows the details until the reader writes their own.
 */
export function edit(t: Trip, patch: TripPatch, now: Date, words?: NameWords): Result {
  const touches = ASSUMPTIONS.filter((k) => k in patch);
  if (touches.length && !canEditAssumptions(t)) return refuse("state", t.state === "completed" ? "A completed trip keeps its plan. Its outcome and notes can still change." : STATE_REFUSAL[t.state]);
  const at = iso(now);
  let next: Trip = { ...t, ...patch, updatedAt: at };
  if ("limits" in patch && patch.limits) next.limits = { windKt: limitValue(patch.limits.windKt), seasFt: limitValue(patch.limits.seasFt) };
  if ("area" in patch) next.area = patch.area?.trim() ? patch.area.trim().slice(0, 120) : null;
  if ("economics" in patch && t.economics && JSON.stringify(t.economics) !== JSON.stringify(patch.economics ?? null)) {
    next.history = [{ at: t.updatedAt, economics: t.economics, refs: t.refs }, ...t.history].slice(0, MAX_HISTORY);
  }
  if ("name" in patch) {
    const own = (patch.name ?? "").trim();
    next = own ? { ...next, name: own.slice(0, 90), autoName: false } : { ...next, autoName: true, name: autoName(next, words ?? { port: null, fishery: null }) };
  } else if (next.autoName && words) next.name = autoName(next, words);
  if (next.state === "planned" && planNeeds(next).length) next.state = "draft";
  return { ok: true, trip: next };
}

// ---------------------------------------------------------------- what's next

export type ActionId = "review-changes" | "compare-departures" | "recheck" | "edit-assumptions" | "record-outcome" | "edit-outcome" | "plan" | "set-date" | "reopen";

export type NextAction = {
  id: ActionId;
  /** the button's words */
  label: string;
  /** one short line under it: why this, now (descriptive, never advice about going) */
  why: string;
};

export type NextCtx = {
  /** changes since planning, as last found (null: not looked at yet) */
  changes: Change[] | null;
  /** the latest recheck (null: none yet) */
  recheck: Recheck | null;
  now: Date;
  /** how many days ahead the saved forecast reaches (NWS zone forecasts run about 5); default 5 */
  forecastDays?: number;
  /** the reader has already opened the changes since they were found */
  reviewed?: boolean;
  /** the forecast the device holds (knownView): from the latest check, from an earlier one, or none */
  forecast?: "latest" | "older" | "none";
};

/** The local calendar date of `now` in `timeZone` ("2026-11-13"). */
export function localDate(now: Date, timeZone: string): string {
  try {
    const p = new Intl.DateTimeFormat("en-CA", { timeZone, year: "numeric", month: "2-digit", day: "2-digit" }).formatToParts(now);
    const g = (k: string) => p.find((x) => x.type === k)?.value ?? "";
    return `${g("year")}-${g("month")}-${g("day")}`;
  } catch {
    return now.toISOString().slice(0, 10);
  }
}

/** Whole calendar days from today (in the trip's zone) to its departure date: 0 today, negative once past; null without a date. */
export function daysUntil(t: Pick<Trip, "depart">, now: Date): number | null {
  if (!t.depart || !DATE.test(t.depart.date)) return null;
  const today = localDate(now, t.depart.timeZone);
  return Math.round((Date.parse(`${t.depart.date}T00:00:00Z`) - Date.parse(`${today}T00:00:00Z`)) / 86_400_000);
}

const RECHECK_FRESH_H = 12;

/**
 * The one primary action for a trip (null only for a state with none). In order: a cancelled trip can be reopened; a
 * completed trip records its outcome, then edits it; a draft needs its date or its plan; a departure that has passed asks for the
 * outcome; changes since planning are reviewed before anything else; close to departure (today or tomorrow) a recheck
 * (again when the last one was incomplete or is more than 12 hours old); when the last check could not read NWS, a
 * recheck of the forecast before any comparison; without limits of their own the reader is
 * asked for them; within the forecast's reach, departures are compared; further out, the assumptions are edited.
 */
export function nextAction(t: Trip, ctx: NextCtx): NextAction | null {
  if (t.state === "cancelled") return { id: "reopen", label: "Reopen this trip", why: "Cancelled. Its plan and snapshots are kept." };
  if (t.state === "completed") return t.outcome ? { id: "edit-outcome", label: "Edit outcome", why: "Correct a figure or add one that is missing." } : { id: "record-outcome", label: "Record outcome", why: "Catch, price, fuel and crew, against the plan." };
  if (t.state === "draft") {
    const needs = planNeeds(t);
    if (needs.includes("date")) return { id: "set-date", label: "Plan this trip", why: "Set a date to compare departures." };
    return { id: "plan", label: "Plan this trip", why: needs.length ? needsWords(needs) : "Port, fishery and date are set." };
  }
  const days = daysUntil(t, ctx.now);
  if (days !== null && days < 0) return { id: "record-outcome", label: "Record outcome", why: "The departure date has passed." };
  const changes = (ctx.changes ?? []).filter((c) => c.priority <= 2);
  if (changes.length && !ctx.reviewed) return { id: "review-changes", label: "Review changes", why: `${changes.length} since you planned.` };
  const r = ctx.recheck;
  const ageH = r ? (ctx.now.getTime() - Date.parse(r.at)) / 3_600_000 : Infinity;
  if (days !== null && days <= 1) {
    if (!r || ageH > RECHECK_FRESH_H) return { id: "recheck", label: "Recheck before departure", why: days === 0 ? "Departure is today." : "Departure is tomorrow." };
    if (!r.complete) return { id: "recheck", label: "Recheck before departure", why: "The last recheck was incomplete." };
  }
  // the forecast the timeline would compare is not the latest: NWS did not answer the last check
  const reachDays = ctx.forecastDays ?? 5;
  const inReach = days === null || days <= reachDays;
  const older = ctx.forecast ? ctx.forecast === "older" : !!r && !r.complete && r.unchecked.some((u) => u.source === "nwsForecast");
  if (older && inReach) return { id: "recheck", label: "Recheck the forecast", why: "The latest check could not read NWS." };
  if (ctx.forecast === "none" && inReach && days !== null) return { id: "recheck", label: "Recheck the forecast", why: "No forecast is saved for this trip." };
  if (t.limits.windKt === null && t.limits.seasFt === null) return { id: "edit-assumptions", label: "Edit assumptions", why: "Add your own wind and sea limits to read the forecast against." };
  if (days !== null && days <= reachDays) return { id: "compare-departures", label: "Compare departures", why: "Your departure is inside the forecast." };
  return { id: "edit-assumptions", label: "Edit assumptions", why: days !== null ? `The forecast does not reach ${shortDate(t.depart!.date)} yet.` : "Check the plan." };
}

/** Every transition a trip offers now, for a menu (the primary action is `nextAction`). */
export function available(t: Trip): ("plan" | "complete" | "cancel" | "reopen")[] {
  const out: ("plan" | "complete" | "cancel" | "reopen")[] = [];
  if (t.state === "draft" && planNeeds(t).length === 0) out.push("plan");
  if (canComplete(t)) out.push("complete");
  if (canCancel(t)) out.push("cancel");
  if (canReopen(t)) out.push("reopen");
  return out;
}
