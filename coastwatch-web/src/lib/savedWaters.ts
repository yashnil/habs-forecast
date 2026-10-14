import type { Snapshot } from "@/lib/briefingSnapshot";
import { SNAPSHOT_VERSION } from "@/lib/briefingSnapshot";

/**
 * Waters, activity and species remembered on this device (no account, nothing sent anywhere),
 * with the snapshot seen at the last visit. Every read and write tolerates blocked or cleared
 * storage: callers then behave as on a first visit and claim nothing.
 */
export const WATERS_KEY = "cw-waters-v1";
export const SESSION_KEY = "cw-changes-session-v1";
export const MAX_WATERS = 6;

export type SavedWaters = {
  /** "port-201|commercial|dungeness" */
  key: string;
  place: string;
  placeName: string;
  audience: string;
  species: string;
  speciesLabel: string;
  /** when this device last showed these waters (ISO) */
  lastSeenAt: string;
  snapshot: Snapshot;
};

/** The snapshot this browser session compares against: the one remembered before the session began. */
export type SessionBaseline = { key: string; since: string; baseline: Snapshot };

export const watersKey = (place: string, audience: string, species: string) => `${place}|${audience}|${species}`;

function store(kind: "local" | "session"): Storage | null {
  try {
    const s = kind === "local" ? window.localStorage : window.sessionStorage;
    const probe = "cw-probe";
    s.setItem(probe, "1");
    s.removeItem(probe);
    return s;
  } catch {
    return null;
  }
}

export function storageAvailable(): boolean {
  return store("local") !== null;
}

export function readWaters(): SavedWaters[] {
  const s = store("local");
  if (!s) return [];
  try {
    const v = JSON.parse(s.getItem(WATERS_KEY) ?? "[]");
    return Array.isArray(v) ? v.filter((w) => w && typeof w.key === "string" && w.snapshot?.v === SNAPSHOT_VERSION) : [];
  } catch {
    return [];
  }
}

function writeWaters(list: SavedWaters[]): boolean {
  const s = store("local");
  if (!s) return false;
  try {
    s.setItem(WATERS_KEY, JSON.stringify(list.slice(0, MAX_WATERS)));
    return true;
  } catch {
    return false;
  }
}

/**
 * Record that these waters were shown now. Returns what was remembered before (null on a first visit
 * or when storage is unavailable), so the caller can compare.
 */
export function recordVisit(snap: Snapshot, now: Date): { previous: SavedWaters | null; saved: boolean } {
  const key = watersKey(snap.place.key, snap.audience, snap.species.id);
  const list = readWaters();
  const previous = list.find((w) => w.key === key) ?? null;
  const entry: SavedWaters = { key, place: snap.place.key, placeName: snap.place.name, audience: snap.audience, species: snap.species.id, speciesLabel: snap.species.label, lastSeenAt: now.toISOString(), snapshot: snap };
  const saved = writeWaters([entry, ...list.filter((w) => w.key !== key)]);
  return { previous, saved };
}

/** The most recently seen activity and species for a place, if they were more specific than "any". */
export function rememberedFor(place: string): SavedWaters | null {
  return readWaters().find((w) => w.place === place && (w.audience !== "all" || w.species !== "any")) ?? null;
}

export function forgetAll(): void {
  const s = store("local");
  try {
    s?.removeItem(WATERS_KEY);
    store("session")?.removeItem(SESSION_KEY);
  } catch {
    // nothing remembered
  }
}

/**
 * The baseline for this browser session. A reload (or another view of the same waters) compares
 * against the same earlier visit, so changes stay listed until the session ends, and anything that
 * arrives meanwhile is added.
 */
export function sessionBaseline(key: string): SessionBaseline | null {
  const s = store("session");
  if (!s) return null;
  try {
    const v = JSON.parse(s.getItem(SESSION_KEY) ?? "null");
    return v && v.key === key && v.baseline?.v === SNAPSHOT_VERSION ? (v as SessionBaseline) : null;
  } catch {
    return null;
  }
}

export function keepSessionBaseline(v: SessionBaseline): void {
  try {
    store("session")?.setItem(SESSION_KEY, JSON.stringify(v));
  } catch {
    // shown once only
  }
}
