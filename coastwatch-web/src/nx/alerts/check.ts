import type { FactKind, PortToday } from "@/pilot/contracts";
import { diffFacts, snapshotOf, type PlaceSnapshot } from "@/pilot/alerts/diff";
import { FISHERY_OPTIONS, buildTodayModel, type PortTodayInputs } from "@/pilot/facts";
import { readFollows, type Follow } from "./follow";
import { mergeInbox, readBaselines, readInbox, readStamp, writeBaselines, writeInbox, writeStamp, type CheckStamp, type InboxItem, type PortBaseline } from "./inbox";

/**
 * The alert check. It runs in this browser when a page opens (AlertWatch.tsx), for the ports the reader
 * follows (follow.ts), and that is the only time anything is checked: nothing runs while Neritic is closed,
 * and nothing is sent to a phone, an inbox or a server.
 *
 * For each followed port it reads the records Today reads (`/today/data`), builds the same facts Today shows
 * (src/pilot/facts.ts `buildTodayModel`, the sea left out: no alert reads it), and compares them with what
 * this device saw at its last check, by the pilot's rules (src/pilot/alerts/diff.ts): one alert per changed
 * record version, never a second for the same change. The first look at a port keeps a baseline and alerts on
 * nothing: alerts begin from what the reader has already been shown.
 *
 * It never invents a change:
 *  - a part that could not be read (status, notices, tests, C-HARM) is left out of the comparison and its
 *    earlier baseline is kept, so a failed read is never "no longer listed" and its return is never "new";
 *  - a copy of the records older than the baseline (a saved offline copy, a stale cache) is not compared;
 *  - offline, nothing is fetched and the last check's time stands.
 */
export const CHECK_EVERY_MS = 30 * 60_000;
export const RETRY_AFTER_MS = 5 * 60_000;
/** how long a started check is given to finish before another may start */
export const CLAIM_MS = 20_000;

export type TodayPayload = { inputs: Omit<PortTodayInputs, "conditions" | "selected">; fixture: boolean };

/**
 * Which loaded parts each kind of fact is made from; a kind with a part missing is not compared. A fishery's
 * status is read from the status records AND the official notices (a notice can set it), so it needs both.
 */
type Part = "fisheries" | "notices" | "biotoxin" | "charm";
/** The kinds of fact an alert can come from (diff.ts triggerFor). */
const COMPARED_KINDS: FactKind[] = ["status", "notice", "test", "whale", "bloom_forecast", "data_age"];
const PARTS: Partial<Record<FactKind, Part[]>> = {
  status: ["fisheries", "notices"],
  whale: ["fisheries"],
  notice: ["notices"],
  test: ["biotoxin"],
  bloom_forecast: ["charm"],
};

/** A "data is getting old" fact is about staleness, never about a read that failed just now: it waits for the parts it describes. */
const ageWaitsFor = (id: string): Part[] => ["fisheries", "notices", "biotoxin", ...(/charm/.test(id) ? (["charm"] as Part[]) : [])];

/** The facts Today shows for a port, every fishery included, as the pilot's PortToday, and which facts could not be read. */
export function portToday(inputs: TodayPayload["inputs"]): { today: PortToday; unread: (kind: FactKind, id: string) => boolean } {
  const model = buildTodayModel({ ...inputs, conditions: null, selected: FISHERY_OPTIONS.map((f) => f.id) });
  const unread = (k: FactKind, id: string) => {
    const parts = k === "data_age" ? ageWaitsFor(id) : (PARTS[k] ?? []);
    return parts.some((part) => inputs[part] == null) || ((k === "status" || k === "whale") && model.statusError) || (k === "test" && model.testsError);
  };
  return { today: { ...model.today, facts: model.today.facts.filter((f) => !unread(f.kind, f.id)) }, unread };
}

export type PortCheck =
  | { state: "baselined"; baseline: PortBaseline; items: [] }
  | { state: "compared"; baseline: PortBaseline; items: InboxItem[] }
  /** the records read are older than the ones the baseline came from: nothing compared, baseline kept */
  | { state: "older_copy"; baseline: PortBaseline; items: [] };

/**
 * Compare one port. Pure: `now` is the device's clock (ISO), used only to date the alert and the baseline.
 */
export function checkPort(prev: PortBaseline | null, payload: TodayPayload, follow: Pick<Follow, "port" | "triggers">, now: string): PortCheck {
  const { inputs, fixture } = payload;
  const dataAt = inputs.dataGeneratedAt ?? null;
  if (prev && prev.data_at && dataAt && Date.parse(dataAt) < Date.parse(prev.data_at)) return { state: "older_copy", baseline: prev, items: [] };
  const { today, unread } = portToday(inputs);
  const fresh = snapshotOf(today, now);
  // the kinds this read could not see: when they first read later, what they list was never seen, so it is not "new"
  const blind = COMPARED_KINDS.filter((k) => unread(k, k === "data_age" ? "data_age:charm" : k));
  // no baseline, or one that holds nothing (damaged, or from before Neritic noted what it could not see): a first look
  if (!prev || (!Object.keys(prev.snap.facts).length && !prev.blind)) return { state: "baselined", baseline: { snap: fresh, data_at: dataAt, blind }, items: [] };

  // what was seen before of the kinds that could not be read now stays as it was: neither gone nor, later, new
  const held = Object.fromEntries(Object.entries(prev.snap.facts).filter(([id, f]) => unread(f.sig.kind, id)));
  const comparable: PlaceSnapshot = { ...prev.snap, facts: Object.fromEntries(Object.entries(prev.snap.facts).filter(([id]) => !(id in held))) };
  const kindOf = new Map(today.facts.map((x) => [x.id, x.kind]));
  const unseen = new Set<string>(prev.blind ?? []);
  const items = diffFacts(comparable, today, follow.triggers, now)
    // a record of a kind the baseline could not see is being seen for the first time, not newly listed
    .filter((a) => !(comparable.facts[a.fact_id] === undefined && !a.fact_version.startsWith("gone-") && unseen.has(kindOf.get(a.fact_id) ?? "")))
    .map((a): InboxItem => {
    const before = comparable.facts[a.fact_id]?.sig.headline ?? null;
    const change = a.fact_version.startsWith("gone-") ? "gone" : before === null ? "new" : "changed";
    return { ...a, read_at: null, change, was: before, ...(fixture ? { sample: true } : {}) };
  });
  // a kind never seen since the first look stays unseen while it is still unread; once read it is seen from here on
  const stillBlind = blind.filter((k) => unseen.has(k));
  return { state: "compared", baseline: { snap: { ...fresh, facts: { ...held, ...fresh.facts } }, data_at: dataAt ?? prev.data_at, blind: stillBlind }, items };
}

export type CheckReport =
  | { state: "nothing_followed" }
  | { state: "offline"; last: CheckStamp | null }
  | { state: "throttled"; last: CheckStamp }
  | { state: "checked"; stamp: CheckStamp; added: InboxItem[]; baselined: string[]; persisted: boolean };

export type CheckOptions = {
  now?: Date;
  /** false: the browser says there is no connection */
  online?: boolean;
  /** skip the 30-minute wait (the Alerts page's "Check now") */
  force?: boolean;
  /** how a port's records are read; null when they could not be */
  load?: (port: string) => Promise<TodayPayload | null>;
};

const sameSet = (a: string[], b: string[]) => a.length === b.length && a.every((x) => b.includes(x));

/** Is a check due? Not while the last one is under 30 minutes old and covered every followed port. */
export function due(stamp: CheckStamp | null, follows: string[], now: Date): boolean {
  if (!stamp) return true;
  // a check that started moments ago (another tab) is left to finish; one that never finished (the page was
  // closed or left mid-check) is not a check, and does not hold the next one back
  if (stamp.claim) {
    const started = now.getTime() - Date.parse(stamp.claim);
    if (Number.isFinite(started) && started >= 0 && started < CLAIM_MS) return false;
  }
  const age = now.getTime() - Date.parse(stamp.at);
  // a clock set back, or an unreadable time: check
  if (!Number.isFinite(age) || age < 0 || age >= CHECK_EVERY_MS) return true;
  if (!sameSet([...stamp.ports, ...stamp.failed], follows)) return true;
  // a port that could not be read is tried again sooner, but not on every page
  return stamp.failed.length > 0 && age >= RETRY_AFTER_MS;
}

async function loadFromServer(port: string): Promise<TodayPayload | null> {
  try {
    const r = await fetch(`/today/data?place=${encodeURIComponent(port)}`, { cache: "no-store" });
    if (!r.ok) return null;
    const j = (await r.json()) as { ok?: boolean; inputs?: TodayPayload["inputs"]; fixture?: boolean };
    return j.ok && j.inputs ? { inputs: j.inputs, fixture: j.fixture === true } : null;
  } catch {
    return null;
  }
}

let running: Promise<CheckReport> | null = null;

/** Run the check once; a second call while one is running gets the same answer. Never throws. */
export function runCheck(o: CheckOptions = {}): Promise<CheckReport> {
  if (running) return running;
  running = check(o).finally(() => {
    running = null;
  });
  return running;
}

/**
 * The page is going away with a check under way (a navigation, the tab closing): give its claim back, so the
 * next page opened checks at once instead of waiting for a check that will never finish.
 */
export function abandonCheck(): void {
  if (!running) return;
  const s = readStamp();
  if (s?.claim) writeStamp({ at: s.at, ports: s.ports, failed: s.failed, data_at: s.data_at });
}

async function check(o: CheckOptions): Promise<CheckReport> {
  const now = o.now ?? new Date();
  const iso = now.toISOString();
  const follows = readFollows();
  const stored = readBaselines();
  // only followed ports keep a baseline
  const baselines: Record<string, PortBaseline> = Object.fromEntries(Object.entries(stored).filter(([p]) => follows.some((f) => f.port === p)));
  if (Object.keys(baselines).length !== Object.keys(stored).length) writeBaselines(baselines);
  if (!follows.length) return { state: "nothing_followed" };

  const last = readStamp();
  const online = o.online ?? (typeof navigator === "undefined" ? true : navigator.onLine !== false);
  if (!online) return { state: "offline", last };
  const ports = follows.map((f) => f.port);
  if (!o.force && last && !due(last, ports, now)) return { state: "throttled", last };

  // claim the check before reading anything, so another tab opening now waits its turn. The last finished
  // check stays as it was: if this one is cut short (the page closed), nothing claims to have been checked.
  writeStamp({ at: last?.at ?? "", ports: last?.ports ?? [], failed: last?.failed ?? [], data_at: last?.data_at ?? null, claim: iso });

  const load = o.load ?? loadFromServer;
  const added: InboxItem[] = [];
  const baselined: string[] = [];
  const read: string[] = [];
  const failed: string[] = [];
  let dataAt: string | null = null;
  for (const f of follows) {
    let payload: TodayPayload | null = null;
    try {
      payload = await load(f.port);
    } catch {
      payload = null;
    }
    if (!payload) {
      failed.push(f.port);
      continue;
    }
    let res: PortCheck;
    try {
      res = checkPort(baselines[f.port] ?? null, payload, f, iso);
    } catch {
      failed.push(f.port);
      continue;
    }
    if (res.state === "older_copy") {
      failed.push(f.port);
      continue;
    }
    baselines[f.port] = res.baseline;
    if (res.state === "baselined") baselined.push(f.port);
    added.push(...res.items);
    read.push(f.port);
    const d = res.baseline.data_at;
    if (d && (!dataAt || d > dataAt)) dataAt = d;
  }

  // a port unfollowed while this ran keeps nothing
  const still = new Set(readFollows().map((f) => f.port));
  for (const p of Object.keys(baselines)) if (!still.has(p)) delete baselines[p];
  let persisted = writeBaselines(baselines);
  const mine = added.filter((i) => still.has(i.place_id));
  if (mine.length) persisted = writeInbox(mergeInbox(readInbox(), mine)) && persisted;
  const stamp: CheckStamp = { at: iso, ports: read.filter((p) => still.has(p)), failed: failed.filter((p) => still.has(p)), data_at: dataAt ?? last?.data_at ?? null };
  persisted = writeStamp(stamp) && persisted;
  return { state: "checked", stamp, added: mine, baselined, persisted };
}
