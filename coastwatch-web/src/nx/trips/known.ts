import { dayTime } from "@/nx/ds/format";
import { readOk, SOURCE_KEYS, SOURCE_LABEL, SOURCE_OF } from "./snapshot";
import { SNAPSHOT_SCHEMA, type EvidenceItem, type ForecastSnap, type Snapshot, type SourceKey, type SourceRead, type Trip } from "./types";

/**
 * What this device knows, across a trip's snapshots (planning + rechecks). The rule: a source that failed to read never
 * replaces evidence the device already holds. Each source's evidence comes from the newest snapshot in which that
 * source answered (ok, or "nothing applies"); a record is dropped only when a later answering read no longer lists it.
 * A failed or offline read leaves the last known copy in place, with its own time, so a page never falls back to an
 * older plan and never shows a lost forecast as gone.
 */

export type Known<T> = {
  value: T;
  snapshotId: string;
  /** that snapshot's evidence time (a saved copy: when it was originally read) */
  readAt: string;
  /** read in the newest snapshot */
  latest: boolean;
};

export type KnownView = {
  forecast: Known<ForecastSnap> | null;
  tides: Known<NonNullable<Snapshot["tides"]>> | null;
  /** per key, from the newest snapshot whose source for that kind answered */
  evidence: Known<EvidenceItem>[];
  sources: Record<SourceKey, { last: SourceRead; knownAt: string | null }>;
};

/** When a snapshot's evidence was read: a saved copy keeps its original time. */
export const evidenceTime = (s: Snapshot): string => (s.mode === "saved" ? (s.savedFrom ?? s.takenAt) : s.takenAt);

const NOT_READ: SourceRead = { state: "notChecked", at: null, detail: "No snapshot on this device." };

/** Snapshots that carry reads, oldest first (a saved copy with no original time carries none of its own). */
function ordered(snaps: Snapshot[]): Snapshot[] {
  const seen = new Set<string>();
  return snaps
    .filter((s) => s && !(s.mode === "saved" && !s.savedFrom) && !seen.has(s.id) && (seen.add(s.id), true))
    .sort((a, b) => Date.parse(evidenceTime(a)) - Date.parse(evidenceTime(b)) || Date.parse(a.takenAt) - Date.parse(b.takenAt));
}

/** The newest answering read of each source, with the snapshot it came from (internal; knownView reports times). */
function lastGood(list: Snapshot[]): Partial<Record<SourceKey, { read: SourceRead; snap: Snapshot }>> {
  const out: Partial<Record<SourceKey, { read: SourceRead; snap: Snapshot }>> = {};
  for (const s of list) for (const k of SOURCE_KEYS) if (readOk(s.sources?.[k])) out[k] = { read: s.sources[k], snap: s };
  return out;
}

export function knownView(snaps: Snapshot[]): KnownView {
  const list = ordered(snaps);
  const newest = list.at(-1) ?? null;
  const known = (s: Snapshot) => ({ snapshotId: s.id, readAt: evidenceTime(s), latest: s.id === newest?.id });

  let forecast: KnownView["forecast"] = null;
  let tides: KnownView["tides"] = null;
  const evidence = new Map<string, Known<EvidenceItem>>();
  for (const s of list) {
    if (readOk(s.sources?.nwsForecast)) forecast = s.forecast && s.forecast.periods.length ? { value: s.forecast, ...known(s) } : null;
    if (readOk(s.sources?.tides)) tides = s.tides ? { value: s.tides, ...known(s) } : null;
    // each source that answered in this snapshot replaces what it said before; one that did not leaves it
    const answered = new Set(SOURCE_KEYS.filter((k) => readOk(s.sources?.[k])));
    for (const [key, k] of evidence) if (answered.has(SOURCE_OF[k.value.kind])) evidence.delete(key);
    for (const e of s.evidence) if (answered.has(SOURCE_OF[e.kind])) evidence.set(e.key, { value: e, ...known(s) });
  }
  const good = lastGood(list);
  const sources = Object.fromEntries(SOURCE_KEYS.map((k) => [k, { last: newest?.sources?.[k] ?? NOT_READ, knownAt: good[k] ? evidenceTime(good[k]!.snap) : null }])) as KnownView["sources"];
  return { forecast, tides, evidence: [...evidence.values()], sources };
}

/** What the device still holds for a source that did not answer now, in words; null when it holds nothing. */
export function holdingWords(v: KnownView, k: SourceKey, ref?: string): string | null {
  const at = v.sources[k].knownAt;
  if (!at) return null;
  const when = dayTime(at, ref ? { ref } : {});
  if (k === "nwsForecast") return v.forecast ? `Showing the forecast from ${when}.` : `The last answer, ${when}, listed no forecast periods.`;
  if (k === "tides") return `Showing the tide predictions from ${when}.`;
  const n = v.evidence.filter((e) => SOURCE_OF[e.value.kind] === k).length;
  return n ? `Showing the ${SOURCE_LABEL[k]} copy from ${when}.` : `The last answer, ${when}, listed nothing that applies here.`;
}

/**
 * The newest known copy of every source as one "saved" snapshot, for comparing offline: nothing in it was fetched.
 * Each source keeps the read it was known by (ok/absent, with its time) or, with no copy, its last failed attempt.
 * `savedFrom` is the oldest read used; `newest` the newest.
 */
export function knownSnapshot(trip: Trip, snaps: Snapshot[], now: string, id: string): { snapshot: Snapshot; newest: string | null; view: KnownView } {
  const list = ordered(snaps);
  const view = knownView(list);
  const good = lastGood(list);
  const sources = Object.fromEntries(SOURCE_KEYS.map((k) => [k, good[k] ? good[k]!.read : view.sources[k].last])) as Snapshot["sources"];
  const times = SOURCE_KEYS.map((k) => view.sources[k].knownAt).filter((x): x is string => !!x).sort();
  const latest = list.at(-1) ?? null;
  const limitations = [
    ...new Set([
      ...(latest?.limitations ?? []).filter((l) => !l.startsWith("Rebuilt from") && !/could not be read; this snapshot says nothing/.test(l)),
      ...SOURCE_KEYS.map((k) => (view.sources[k].knownAt ? `${SOURCE_LABEL[k]}: the copy on this device is from ${dayTime(view.sources[k].knownAt!)}.` : `${SOURCE_LABEL[k]}: no copy on this device.`)),
      "Built from copies saved on this device. Nothing was fetched for it.",
    ]),
  ];
  const snapshot: Snapshot = {
    schema: SNAPSHOT_SCHEMA,
    id,
    tripId: trip.id,
    kind: "recheck",
    takenAt: now,
    serverNow: null,
    mode: "saved",
    savedFrom: times[0] ?? null,
    sources,
    evidence: view.evidence.map((e) => e.value),
    forecast: view.forecast?.value ?? null,
    tides: view.tides?.value ?? null,
    assumptions: { limits: { ...trip.limits }, depart: trip.depart ? { ...trip.depart } : null, economics: trip.economics },
    limitations,
  };
  return { snapshot, newest: times.at(-1) ?? null, view };
}
