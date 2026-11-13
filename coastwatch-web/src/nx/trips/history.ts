import { tripFishery } from "@/nx/coast/fisheries";
import { planVsActual } from "./outcome";
import { MAX_TRIP_NAME, type Result, type TripsStore } from "./store";
import type { Trip } from "./types";

/**
 * The reader's own trip history (v15): grouped by where each trip stands, searchable by port and fishery, and, once
 * there are enough completed trips with recorded figures, a few plain comparisons of their own plans with their own
 * outcomes. Nothing here predicts a catch, a price or a result, and nothing is compared across other people.
 */

export type GroupId = "upcoming" | "pastDue" | "drafts" | "completed" | "cancelled";
export type Group = { id: GroupId; label: string; trips: Trip[] };

export const GROUP_LABEL: Record<GroupId, string> = { upcoming: "Upcoming", pastDue: "Planned, date passed", drafts: "Drafts", completed: "Completed", cancelled: "Cancelled" };

/** At least this many completed trips with recorded figures before any comparison is shown. */
export const MIN_FOR_COMPARISON = 3;

/** "YYYY-MM-DD" in a time zone. */
export function dayIn(now: Date, timeZone: string = "America/Los_Angeles"): string {
  try {
    return new Intl.DateTimeFormat("en-CA", { timeZone, year: "numeric", month: "2-digit", day: "2-digit" }).format(now);
  } catch {
    return now.toISOString().slice(0, 10);
  }
}

/**
 * Upcoming: planned, departing today or later (soonest first). Planned, date passed: planned with a departure before
 * today and no outcome yet (most recent first). Drafts, completed and cancelled: most recently changed first. A planned
 * trip with no date is upcoming (it has not passed).
 */
export function groupTrips(trips: readonly Trip[], now: Date): Group[] {
  const g: Record<GroupId, Trip[]> = { upcoming: [], pastDue: [], drafts: [], completed: [], cancelled: [] };
  for (const t of trips) {
    if (t.state === "completed") g.completed.push(t);
    else if (t.state === "cancelled") g.cancelled.push(t);
    else if (t.state === "draft") g.drafts.push(t);
    else if (t.depart && t.depart.date < dayIn(now, t.depart.timeZone)) g.pastDue.push(t);
    else g.upcoming.push(t);
  }
  const byChange = (a: Trip, b: Trip) => b.updatedAt.localeCompare(a.updatedAt);
  // a planned trip with no date sorts after every dated one ("~" is after every digit)
  const when = (t: Trip) => (t.depart ? `${t.depart.date}T${t.depart.time ?? "99:99"}` : "~");
  const cmp = (x: string, y: string) => (x < y ? -1 : x > y ? 1 : 0);
  g.upcoming.sort((a, b) => cmp(when(a), when(b)));
  g.pastDue.sort((a, b) => cmp(when(b), when(a)));
  const done = (t: Trip) => t.outcome?.returnedOn ?? t.outcome?.departedOn ?? t.depart?.date ?? t.updatedAt.slice(0, 10);
  g.completed.sort((a, b) => done(b).localeCompare(done(a)) || byChange(a, b));
  g.drafts.sort(byChange);
  g.cancelled.sort(byChange);
  return (Object.keys(g) as GroupId[]).filter((id) => g[id].length).map((id) => ({ id, label: GROUP_LABEL[id], trips: g[id] }));
}

const fold = (s: string) => s.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();

/** Trips whose name, port (home or departure) or fishery match every word typed. */
export function searchTrips(trips: readonly Trip[], query: string, portName: (key: string) => string | null): Trip[] {
  const words = fold(query).split(/\s+/).filter(Boolean);
  if (!words.length) return [...trips];
  return trips.filter((t) => {
    const hay = fold([t.name, portName(t.placeKey) ?? "", t.departKey ? (portName(t.departKey) ?? "") : "", tripFishery(t.fishery)?.label ?? t.fishery, t.fishery.replace(/_/g, " ")].join(" "));
    return words.every((w) => hay.includes(w));
  });
}

/** "Name (copy)", "Name (copy 2)", …: never a name already used. */
export function copyName(name: string, taken: readonly string[]): string {
  const base = name.replace(/ \(copy(?: \d+)?\)$/, "");
  const fit = (suffix: string) => `${base.slice(0, MAX_TRIP_NAME - suffix.length)}${suffix}`;
  let out = fit(" (copy)");
  for (let n = 2; taken.includes(out) && n < 100; n++) out = fit(` (copy ${n})`);
  return out;
}

/** A new draft from a trip: same port, fishery, limits and economics; no date, no outcome, no snapshots, no history. */
export function duplicateTrip(store: TripsStore, id: string, now: Date): Result<{ trip: Trip }> {
  const d = store.load();
  const src = d.trips.find((t) => t.id === id);
  if (!src) return { ok: false, reason: "missing" };
  const at = now.toISOString();
  const { legacyId: _legacy, ...rest } = src;
  void _legacy;
  return store.put({ ...rest, id: store.newId(), name: copyName(src.name, d.trips.map((t) => t.name)), autoName: false, state: "draft", depart: null, outcome: null, history: [], snapshots: { planning: null, rechecks: [], superseded: [] }, createdAt: at, updatedAt: at });
}

/** A name the reader typed is theirs: it never changes on its own again. A blank name keeps the old one. */
export function renameTrip(store: TripsStore, id: string, name: string): Result<{ trip: Trip }> {
  const t = store.get(id);
  if (!t) return { ok: false, reason: "missing" };
  const clean = name.trim().slice(0, MAX_TRIP_NAME);
  return store.put({ ...t, name: clean || t.name, autoName: clean ? false : t.autoName });
}

// ---------------------------------------------------------------- the reader's own records, compared

export type OwnComparison = {
  /** completed trips with a plan and recorded figures for this comparison */
  count: number;
  /** the middle value of actual − planned */
  median: number;
  /** how many came in above / below / at the plan */
  above: number;
  below: number;
  even: number;
  /** "Across your 4 completed trips with both figures recorded, the middle actual net to boat was $120.00 above plan." */
  words: string;
};

export type OwnRecords = { completed: number; net: OwnComparison | null; catch: OwnComparison | null; minimum: number; enough: boolean };

const median = (xs: number[]) => {
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

function compare(diffs: number[], what: string, unit: (v: number) => string): OwnComparison | null {
  if (diffs.length < MIN_FOR_COMPARISON) return null;
  const m = median(diffs);
  const eps = 0.005;
  const above = diffs.filter((d) => d > eps).length;
  const below = diffs.filter((d) => d < -eps).length;
  const dir = Math.abs(m) < eps ? "the same as planned" : `${unit(Math.abs(m))} ${m > 0 ? "above" : "below"} plan`;
  return { count: diffs.length, median: m, above, below, even: diffs.length - above - below, words: `Across your ${diffs.length} completed trips with both figures recorded, the middle ${what} was ${dir}.` };
}

const usd = (v: number) => `$${v.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
const lb = (v: number) => `${Math.round(v).toLocaleString("en-US")} lb`;

/**
 * Plain comparisons of the reader's own plans with their own outcomes, only once at least three completed trips have
 * the figures recorded, each said with its count. Descriptive of the past only: no trend, forecast or advice.
 */
export function ownRecords(trips: readonly Trip[]): OwnRecords {
  const done = trips.filter((t) => t.state === "completed" && t.outcome && t.economics && t.activity === "commercial");
  const nets: number[] = [];
  const catches: number[] = [];
  for (const t of done) {
    const pv = planVsActual(t);
    if (pv.difference !== null) nets.push(pv.difference);
    if (pv.planned.lb.value !== null && pv.actual.lb.value !== null) catches.push(pv.actual.lb.value - pv.planned.lb.value);
  }
  const completed = trips.filter((t) => t.state === "completed").length;
  const net = compare(nets, "actual net to boat (not profit)", usd);
  const c = compare(catches, "catch landed", lb);
  return { completed, net, catch: c, minimum: MIN_FOR_COMPARISON, enough: !!(net || c) };
}
