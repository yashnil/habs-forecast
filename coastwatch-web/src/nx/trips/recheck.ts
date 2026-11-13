import type { CoastError, CoastGates, CoastSea } from "@/nx/coast/brief";
import { gatesUrl, seaUrl } from "@/nx/coast/useCoast";
import { dayTime } from "@/nx/ds/format";
import { FIELD_CACHE, FIELD_RECORD } from "@/nx/offline/pack";
import { diffSnapshots } from "./diff";
import { holdingWords, knownSnapshot, knownView, type KnownView } from "./known";
import { readOk, SOURCE_KEYS, SOURCE_LABEL, takeSnapshot, type Answer } from "./snapshot";
import type { Change, ForecastSnap, Recheck, Snapshot, Trip } from "./types";

/**
 * Recheck before departure (v15 §9). Reads the two My Coast endpoints again (each on its own 12 s clock, so a slow
 * sea never holds the official records back), takes a "recheck" snapshot, and compares it with the planning
 * snapshot (and with the previous recheck). The answer is in four parts: changed, same, couldn't check, verify
 * yourself. It is `complete` only when every source answered (ok or "nothing applies"); an incomplete recheck is
 * never an all-clear.
 *
 * Offline (the browser says so, or both requests fail at the network), nothing is fetched: the comparison is made
 * against the newest copy on this device (a live snapshot of this trip, or a saved field pack), its original time
 * is kept as `savedFrom`, and the result is marked "saved" and incomplete. No wording says an agency was read.
 */

export const RECHECK_TIMEOUT_MS = 12_000;
export const MAX_RECHECKS = 8;

/** One request. Throws on a network failure or abort; resolves with the HTTP status and parsed body otherwise. */
export type FetchJson = (url: string, init: { signal: AbortSignal }) => Promise<{ status: number; body: unknown }>;

/** A copy of the two answers kept on this device (the field pack saves exactly these addresses). */
export type SavedCopy = { gates: CoastGates | null; sea: CoastSea | null; savedAt: string };

/**
 * The trips store (store.ts `createTripsStore` fits as it is): `putSnapshot` writes the snapshot and attaches it to its
 * trip (a planning one supersedes the old, a recheck is appended and the oldest beyond 8 pruned), answering
 * `{ ok, trip }`. A store that answers anything else is taken to have written only the snapshot; the result's `trip`
 * then carries the new ids for the caller to save.
 */
export type RecheckStore = {
  getSnapshot(id: string): Snapshot | null;
  putSnapshot(s: Snapshot): unknown;
  /** the field pack's copy of this trip's two answers, when one is on this device */
  savedCopy?(trip: Trip): Promise<SavedCopy | null>;
};

export type RecheckDeps = {
  fetchJson?: FetchJson;
  store: RecheckStore;
  /** device clock */
  now?: () => Date;
  online?: boolean;
  /** Agent B's forecastFromSea, when merged; snapshot.ts falls back to its local adapter */
  forecastFromSea?: (sea: CoastSea) => ForecastSnap;
  timeoutMs?: number;
};

export type RecheckResult = Recheck & {
  /** offline: the newest read among the copies compared (savedFrom on the snapshot is the oldest); null live */
  knownNewest: string | null;
  snapshot: Snapshot;
  /** false: the device would not keep the snapshot (storage full or blocked); the result is shown, not kept */
  persisted: boolean;
  /** changes since the previous recheck (null when this is the first, or offline) */
  sinceLast: Change[] | null;
  /** the trip with this recheck's id recorded (unchanged offline: a saved-copy comparison is not a new snapshot) */
  trip: Trip;
};

type Got<T> = { answer: Answer<T>; network: boolean };

export const defaultFetchJson: FetchJson = async (url, init) => {
  const r = await fetch(url, { cache: "no-store", signal: init.signal, credentials: "same-origin" });
  const body = await r.json().catch(() => null);
  return { status: r.status, body };
};

async function get<T extends { ok: true }>(fetchJson: FetchJson, url: string, ms: number): Promise<Got<T>> {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), ms);
  try {
    const { status, body } = await fetchJson(url, { signal: ctl.signal });
    const b = body as T | CoastError | null;
    if (b && typeof b === "object" && "ok" in b) {
      // an error answer without its words is named by its HTTP status, never "(undefined)"
      if (b.ok === false && (typeof b.error !== "string" || !b.error.trim())) return { answer: { ok: false, error: `HTTP ${status}` }, network: false };
      return { answer: b, network: false };
    }
    return { answer: { ok: false, error: `HTTP ${status}` }, network: false };
  } catch (e) {
    const aborted = ctl.signal.aborted || (e as Error)?.name === "AbortError";
    return { answer: { ok: false, error: aborted ? `no answer within ${Math.round(ms / 1000)} s` : "the network did not answer" }, network: true };
  } finally {
    clearTimeout(timer);
  }
}

/** Snapshots of a trip on this device, newest first (the planning one included). */
function snapshotsOf(trip: Trip, store: RecheckStore): Snapshot[] {
  const ids = [trip.snapshots.planning, ...trip.snapshots.rechecks].filter((x): x is string => !!x);
  return ids
    .map((id) => store.getSnapshot(id))
    .filter((s): s is Snapshot => !!s)
    .sort((a, b) => Date.parse(b.takenAt) - Date.parse(a.takenAt));
}

/** When a copy was really read: a saved snapshot carries its original time. */
const readAt = (s: Snapshot) => (s.mode === "saved" ? (s.savedFrom ?? s.takenAt) : s.takenAt);

/** Official pages to read directly, from what the snapshots cite. */
export function verifyLinks(...snaps: (Snapshot | null)[]): { label: string; url: string }[] {
  const out: { label: string; url: string }[] = [];
  const add = (label: string, url: string | null | undefined) => {
    if (url && !out.some((x) => x.url === url)) out.push({ label, url });
  };
  for (const s of snaps) {
    if (!s) continue;
    for (const e of s.evidence) {
      if (e.kind === "season") add(`${e.source} page for ${e.scope ?? "this fishery"}`, e.url);
      if (e.kind === "whale") add(`CDFW whale measures, ${e.scope ?? "your zone"}`, e.url);
      if (e.kind === "cdphTest") add(`CDPH test results, ${e.scope ?? "your site"}`, e.url);
    }
    if (s.forecast?.url) add(`NWS zone forecast${s.forecast.zone ? `, ${s.forecast.zone}` : ""}`, s.forecast.url);
    if (s.tides?.url) add(`CO-OPS tide station${s.tides.station ? `, ${s.tides.station}` : ""}`, s.tides.url);
  }
  return out;
}

/**
 * The four-part answer. `view` (knownView over the trip's snapshots, this one included) lets "couldn't check" say what
 * the device still holds for a source that did not answer, and, offline, how old each source's copy is.
 */
export function summarize(base: Snapshot, next: Snapshot, changed: Change[], at: string, view?: KnownView): Recheck {
  const changedKeys = new Set(changed.map((c) => c.evidenceKey).filter(Boolean));
  const A = new Map(base.evidence.map((e) => [e.key, e]));
  const same = next.evidence.filter((e) => A.get(e.key)?.version === e.version && !changedKeys.has(e.key)).map((e) => ({ key: e.key, title: e.title }));
  // a source that could not be read is told once, under "couldn't check", with what it hides from the plan
  const lost = new Map(changed.filter((c) => c.category === "source" && c.kind === "unavailable").map((c) => [c.id.split(":")[1], c]));
  const unchecked = SOURCE_KEYS.filter((k) => next.mode === "saved" || !readOk(next.sources[k])).map((k) => {
    const hides = lost.get(k)?.detail?.match(/Shown when you planned: .*$/)?.[0] ?? null;
    if (next.mode === "saved") {
      const known = view?.sources[k].knownAt ?? (readOk(next.sources[k]) ? readAt(next) : null);
      return { source: k, why: known ? `Offline: not checked now. The copy on this device is from ${dayTime(known, { ref: at })}.` : "Offline: not checked now, and there is no copy of it on this device." };
    }
    const why = next.sources[k].detail ?? `${SOURCE_LABEL[k]} did not answer.`;
    const holding = view ? holdingWords(view, k, at) : null;
    return { source: k, why: [why, holding, hides].filter(Boolean).join(" ") };
  });
  return {
    at,
    mode: next.mode,
    complete: next.mode === "live" && unchecked.length === 0,
    changed: changed.filter((c) => !(c.category === "source" && c.kind === "unavailable")),
    same,
    unchecked,
    verify: verifyLinks(next, base),
    baseId: base.id,
    snapshotId: next.id,
  };
}

async function fetchBoth(trip: Trip, deps: RecheckDeps) {
  const fetchJson = deps.fetchJson ?? defaultFetchJson;
  const ms = deps.timeoutMs ?? RECHECK_TIMEOUT_MS;
  const [g, s] = await Promise.all([get<CoastGates>(fetchJson, gatesUrl(trip.placeKey, trip.fishery), ms), get<CoastSea>(fetchJson, seaUrl(trip.placeKey), ms)]);
  return { g, s, offline: g.network && s.network };
}

const isOnline = (deps: RecheckDeps) => deps.online ?? (typeof navigator === "undefined" ? true : navigator.onLine !== false);

export async function recheck(trip: Trip, deps: RecheckDeps): Promise<RecheckResult> {
  const now = (deps.now ?? (() => new Date()))().toISOString();
  const store = deps.store;
  const planning = trip.snapshots.planning ? store.getSnapshot(trip.snapshots.planning) : null;
  if (!planning) throw new Error("This trip has no planning snapshot to compare against.");
  const mine = snapshotsOf(trip, store);
  const lastRecheck = mine.find((s) => s.kind === "recheck" && s.mode === "live" && s.id !== planning.id) ?? null;

  const fetched = isOnline(deps) ? await fetchBoth(trip, deps) : null;
  if (!fetched || fetched.offline) {
    const { snapshot: snap, newest, view } = await savedSnapshot(trip, deps, mine, now);
    // the same evidence clock a live recheck uses: the server's, carried forward from the newest live answer by
    // the device's own elapsed time (offline the device's clock is all there is; a skew learned online still holds)
    const changed = diffSnapshots(planning, snap, evidenceNow(mine, now));
    return { ...summarize(planning, snap, changed, now, view), snapshot: snap, persisted: false, sinceLast: null, trip, knownNewest: newest };
  }

  const fsea = fetched.s.answer && typeof fetched.s.answer === "object" && fetched.s.answer.ok ? fetched.s.answer : null;
  const snap = takeSnapshot({
    trip,
    gates: fetched.g.answer,
    sea: fetched.s.answer,
    forecast: fsea && deps.forecastFromSea ? deps.forecastFromSea(fsea) : null,
    now,
    mode: "live",
    savedFrom: null,
    kind: "recheck",
  });
  const rechecks = [...trip.snapshots.rechecks, snap.id].slice(-MAX_RECHECKS);
  const kept = keep(store, snap, { ...trip, snapshots: { ...trip.snapshots, rechecks } });
  const next = kept.trip;
  // ages (a forecast's issuance, a period's end) are read on the evidence's own clock: the server's, as its answers
  // carry it (a fixture pins it); the device's when the answers give none
  const at = snap.serverNow ?? now;
  const changed = diffSnapshots(planning, snap, at);
  const sinceLast = lastRecheck ? diffSnapshots(lastRecheck, snap, at) : null;
  const view = knownView([...mine, snap]);
  return { ...summarize(planning, snap, changed, now, view), snapshot: snap, persisted: kept.persisted, sinceLast, trip: next, knownNewest: null };
}

/**
 * Offline: the newest known copy of every source on this device (knownView over this trip's snapshots and the field
 * pack's answers, when one is here), as one "saved" snapshot. A source that failed in a later recheck never hides the
 * copy an earlier one kept. With nothing on the device, a snapshot that checked nothing.
 */
/** Device time shifted by the newest live snapshot's server-minus-device offset; the device time when none has one. */
export function evidenceNow(snaps: Snapshot[], now: string): string {
  const live = snaps.filter((s) => s.mode === "live" && s.serverNow).sort((a, b) => Date.parse(b.takenAt) - Date.parse(a.takenAt))[0];
  if (!live?.serverNow) return now;
  const offset = Date.parse(live.serverNow) - Date.parse(live.takenAt);
  return Number.isFinite(offset) ? new Date(Date.parse(now) + offset).toISOString() : now;
}

async function savedSnapshot(trip: Trip, deps: RecheckDeps, mine: Snapshot[], now: string): Promise<{ snapshot: Snapshot; newest: string | null; view: KnownView }> {
  const pack = deps.store.savedCopy ? await deps.store.savedCopy(trip).catch(() => null) : null;
  const id = `saved-${trip.id}-${Date.parse(now).toString(36)}`;
  const packSnap = pack ? takeSnapshot({ trip, gates: pack.gates, sea: pack.sea, forecast: pack.sea && deps.forecastFromSea ? deps.forecastFromSea(pack.sea) : null, now, mode: "saved", savedFrom: pack.savedAt, kind: "recheck", id: `${id}-pack` }) : null;
  const all = packSnap ? [...mine, packSnap] : mine;
  if (!all.length) {
    const snapshot = takeSnapshot({ trip, gates: null, sea: null, forecast: null, now, mode: "saved", savedFrom: null, kind: "recheck", id });
    return { snapshot, newest: null, view: knownView([]) };
  }
  return knownSnapshot(trip, all, now, id);
}

export type RefreshResult = { ok: true; trip: Trip; snapshot: Snapshot; persisted: boolean } | { ok: false; reason: string };

/** Write a snapshot; the trip the store attached it to, or `fallback` (the ids added here) when the store did not answer one. */
function keep(store: RecheckStore, snap: Snapshot, fallback: Trip): { trip: Trip; persisted: boolean } {
  let r: unknown;
  try {
    r = store.putSnapshot(snap);
  } catch {
    return { trip: fallback, persisted: false };
  }
  if (r && typeof r === "object" && "ok" in r) {
    const res = r as { ok: boolean; trip?: Trip | null };
    return { trip: res.ok && res.trip ? res.trip : fallback, persisted: res.ok };
  }
  return { trip: fallback, persisted: true };
}

/**
 * A new planning snapshot (the first on save, or an explicit refresh). The old planning snapshot is kept and its id
 * moves to `superseded`, never overwritten. Needs a live answer: offline, or with both requests failing, nothing
 * changes (a saved copy never becomes what the reader "planned with").
 */
export async function refreshPlanning(trip: Trip, deps: RecheckDeps): Promise<RefreshResult> {
  if (!isOnline(deps)) return { ok: false, reason: "Offline: a planning snapshot needs a connection. The trip is saved on this device." };
  const now = (deps.now ?? (() => new Date()))().toISOString();
  const { g, s, offline } = await fetchBoth(trip, deps);
  if (offline) return { ok: false, reason: "Neither request answered; the planning snapshot was not changed." };
  const fsea = s.answer && typeof s.answer === "object" && s.answer.ok ? s.answer : null;
  const snap = takeSnapshot({ trip, gates: g.answer, sea: s.answer, forecast: fsea && deps.forecastFromSea ? deps.forecastFromSea(fsea) : null, now, mode: "live", savedFrom: null, kind: "planning" });
  const old = trip.snapshots.planning;
  const kept = keep(deps.store, snap, { ...trip, snapshots: { ...trip.snapshots, planning: snap.id, superseded: old ? [...trip.snapshots.superseded, old] : trip.snapshots.superseded } });
  return { ok: true, trip: kept.trip, snapshot: snap, persisted: kept.persisted };
}

/** The field pack's copy of a trip's two answers (src/nx/offline/pack.ts saves these addresses), for `RecheckStore.savedCopy`. */
export async function fieldPackCopy(trip: Trip): Promise<SavedCopy | null> {
  if (typeof caches === "undefined") return null;
  try {
    if (!(await caches.has(FIELD_CACHE))) return null;
    const c = await caches.open(FIELD_CACHE);
    const [gr, sr, rec] = await Promise.all([c.match(gatesUrl(trip.placeKey, trip.fishery)), c.match(seaUrl(trip.placeKey)), c.match(FIELD_RECORD)]);
    if (!gr && !sr) return null;
    // the pack's own record dates its answers; a copy without its date (or of another port) is never used
    const meta = rec ? ((await rec.json()) as { savedAt?: string; port?: { key?: string } }) : null;
    const savedAt = meta?.port?.key === trip.placeKey && typeof meta.savedAt === "string" ? meta.savedAt : null;
    if (!savedAt) return null;
    const gates = gr ? ((await gr.json()) as CoastGates | CoastError) : null;
    const sea = sr ? ((await sr.json()) as CoastSea | CoastError) : null;
    return { gates: gates && gates.ok ? gates : null, sea: sea && sea.ok ? sea : null, savedAt };
  } catch {
    return null;
  }
}
