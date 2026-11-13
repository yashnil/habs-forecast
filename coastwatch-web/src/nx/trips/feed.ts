import { age, day, dayTime, time } from "@/nx/ds/format";
import { diffSnapshots } from "./diff";
import { readOk, REQUIRED_SOURCES, SOURCE_KEYS, SOURCE_LABEL } from "./snapshot";
import type { Change, Recheck, Snapshot, SourceKey } from "./types";

/**
 * The words and states of "Since you planned this trip" (ChangeFeed, RecheckPanel), pure so their honesty rules are
 * unit-tested: "No changes found" only after a complete live check; an incomplete check names what it could not see;
 * a saved-copy comparison says its age and never that a source was read.
 */

export type FeedState = "none" | "unchecked" | "checking" | "complete" | "incomplete" | "saved";

export type FeedView = {
  state: FeedState;
  headline: string;
  /** one short line under the headline, or null */
  sub: string | null;
  changes: Change[];
  unchecked: { source: SourceKey; label: string; why: string; required: boolean }[];
  /** official records or tests that answered but are old */
  stale: { key: string; title: string; asOf: string | null }[];
};

export const FEED_SHOWN = 5;
const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;

export function feedView(o: { planning: Snapshot | null; result: Recheck | null; checking: boolean; now: string }): FeedView {
  const { planning, result, checking, now } = o;
  const unchecked = (result?.unchecked ?? []).map((u) => ({ ...u, label: SOURCE_LABEL[u.source] }));
  const stale = result?.stale ?? [];
  if (!planning) return { state: "none", headline: "No planning snapshot yet", sub: "Saving the trip keeps a dated copy of what Neritic shows now, to compare against later.", changes: [], unchecked: [], stale: [] };
  if (checking) return { state: "checking", headline: "Checking the sources again", sub: "CDFW, CDPH, NWS, NDBC and CO-OPS, through Neritic", changes: result?.changed ?? [], unchecked: [], stale: [] };
  if (!result) return { state: "unchecked", headline: "Not rechecked since you planned", sub: `Planned with the sources as read ${dayTime(planning.takenAt, { ref: now })}.`, changes: [], unchecked: [], stale: [] };
  const n = result.changed.length;
  if (result.mode === "saved") {
    const { oldest, newest, missing } = savedCopies(result);
    const ago = (t: string) => (age(t, now) === "today" ? `at ${time(t)}` : age(t, now));
    return {
      state: "saved",
      headline: n
        ? `${plural(n, "difference")} in the copies saved on this device`
        : !missing && newest
          ? `No changes in the copies saved on this device (newest ${dayTime(newest, { ref: now })}); nothing was checked now.`
          : "Offline: nothing was checked now",
      sub: missing
        ? `Not a live check. ${plural(missing, "source")} ${missing === 1 ? "has" : "have"} no copy on this device; ages of the rest are listed below.`
        : `Not a live check. Compared against copies saved on this device${oldest && newest && oldest !== newest ? `, the oldest read ${ago(oldest)} and the newest ${ago(newest)}` : oldest ? ` ${ago(oldest)}` : ""}.`,
      changes: result.changed,
      unchecked,
      stale: [],
    };
  }
  const optional = unchecked.filter((u) => !u.required);
  const optionalWords = optional.length ? ` ${optional.map((u) => u.label).join(" and ")} could not be read (optional context, listed below).` : "";
  const staleWords = stale.length ? ` ${plural(stale.length, "record")} answered but ${stale.length === 1 ? "is" : "are"} old.` : "";
  if (result.complete) {
    return {
      state: "complete",
      headline: n ? `${plural(n, "change")} since you planned` : `No changes found in the sources checked at ${time(result.at)}`,
      sub: (n ? `Every required source answered at ${time(result.at)}.` : `Every required source answered; compared with what you planned with on ${day(planning.takenAt, { ref: now })}.`) + optionalWords + staleWords,
      changes: result.changed,
      unchecked: optional,
      stale,
    };
  }
  const k = unchecked.filter((u) => u.required).length;
  return {
    state: "incomplete",
    headline: n ? `${plural(n, "change")} found; ${plural(k, "required source")} could not be checked` : `Incomplete: ${plural(k, "required source")} could not be checked`,
    sub: `Checked ${time(result.at)}. What could not be checked is listed below; nothing about it is known from this check.${staleWords}`,
    changes: result.changed,
    unchecked,
    stale,
  };
}

/** The original time of the copy a saved recheck compared against (RecheckResult carries its snapshot). */
/** The copies an offline recheck compared (RecheckResult carries its snapshot and knownNewest). */
function savedCopies(r: Recheck): { oldest: string | null; newest: string | null; missing: number } {
  const x = r as Recheck & { snapshot?: Snapshot; knownNewest?: string | null };
  const missing = x.snapshot ? SOURCE_KEYS.filter((k) => !readOk(x.snapshot!.sources[k])).length : SOURCE_KEYS.length;
  const oldest = x.snapshot?.savedFrom ?? null;
  return { oldest, newest: x.knownNewest ?? oldest, missing };
}

// ---------------------------------------------------------------- the change history

export type TimelineStop = {
  id: string;
  kind: "planning" | "superseded" | "recheck";
  /** when the evidence was read (a saved copy: its original time) */
  at: string;
  mode: Snapshot["mode"];
  complete: boolean;
  /** changes against the planning snapshot (0 for the planning one) */
  changes: number;
};

export const snapshotComplete = (s: Snapshot) => s.mode === "live" && REQUIRED_SOURCES.every((k) => readOk(s.sources[k]));

/** Planning → rechecks along time; superseded planning snapshots stay on the line, quieter. */
export function timelineOf(planning: Snapshot | null, rechecks: Snapshot[], superseded: Snapshot[], now: string): TimelineStop[] {
  const out: TimelineStop[] = [];
  for (const s of superseded) out.push({ id: s.id, kind: "superseded", at: s.takenAt, mode: s.mode, complete: snapshotComplete(s), changes: 0 });
  if (planning) out.push({ id: planning.id, kind: "planning", at: planning.takenAt, mode: planning.mode, complete: snapshotComplete(planning), changes: 0 });
  for (const s of rechecks) {
    out.push({ id: s.id, kind: "recheck", at: s.mode === "saved" ? (s.savedFrom ?? s.takenAt) : s.takenAt, mode: s.mode, complete: snapshotComplete(s), changes: planning ? diffSnapshots(planning, s, now).length : 0 });
  }
  return out.sort((a, b) => Date.parse(a.at) - Date.parse(b.at));
}

/** A stop's place on the axis, 0–1, from the first stop to now. */
export function stopPosition(at: string, first: string, now: string): number {
  const a = Date.parse(first);
  const b = Math.max(Date.parse(now), a + 1);
  return Math.min(1, Math.max(0, (Date.parse(at) - a) / (b - a)));
}

/** Sources that could not be checked for the same reason, said once: "CDFW season record, NDBC buoy · <why>". */
/** Sources that failed for the same reason, said once; required and optional never share a line (optional is marked). */
export function groupUnchecked(u: { source: SourceKey; why: string; required?: boolean }[]): { sources: SourceKey[]; labels: string; why: string; required: boolean }[] {
  const by = new Map<string, { sources: SourceKey[]; why: string; required: boolean }>();
  for (const x of u) {
    const required = x.required !== false;
    const k = `${required}|${x.why}`;
    const g = by.get(k) ?? { sources: [], why: x.why, required };
    g.sources.push(x.source);
    by.set(k, g);
  }
  return [...by.values()]
    .sort((a, b) => Number(b.required) - Number(a.required))
    .map((g) => ({ ...g, labels: (g.sources.length === SOURCE_KEYS.length ? "Every source" : g.sources.map((k) => SOURCE_LABEL[k]).join(", ")) + (g.required ? "" : " (optional)") }));
}

/** A change's detail when it is words for a reader (not a list of period keys). */
export const readableDetail = (d: string | null): string | null => (d && !/\d{4}-\d{2}-\d{2}\|(day|night)/.test(d) ? d : null);
