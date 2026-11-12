import type { AlertItem, PlaceSnapshot } from "@/pilot/alerts/diff";
import { readJSON, remove, writeJSON } from "@/nx/personal/store";

/**
 * The alerts this device made, and what it compares with (all in this browser, nothing sent):
 *   nx-inbox-v1        the alerts, newest first, at most fifty, with when each was read
 *   nx-alert-snap-v1   per followed port, what its records said at the last check (the baseline)
 *   nx-alert-check-v1  when the last check ran and which ports it could read
 * check.ts writes them when a page opens; the Alerts page and the bar's count read them (useInbox.ts).
 */
export const INBOX_KEY = "nx-inbox-v1";
export const SNAP_KEY = "nx-alert-snap-v1";
export const CHECK_KEY = "nx-alert-check-v1";
export const MAX_INBOX = 50;

export type InboxItem = AlertItem & {
  read_at: string | null;
  /** made from the test fixture, not published records: the item says so */
  sample?: boolean;
  /** what kind of change this is, and what the record said before (the page words it; older items lack them) */
  change?: "new" | "changed" | "gone";
  was?: string | null;
};

/** A port's baseline: the pilot's snapshot plus when the records it came from were published. */
export type PortBaseline = { snap: PlaceSnapshot; data_at: string | null };

/** The last check: when (device clock), the ports read, the ports that could not be read, and why it could not run. */
export type CheckStamp = { at: string; ports: string[]; failed: string[]; /** newest publish time among the ports read */ data_at: string | null };

const isList = (v: unknown): v is InboxItem[] => Array.isArray(v);
const isMap = (v: unknown): v is Record<string, PortBaseline> => !!v && typeof v === "object" && !Array.isArray(v);
const isStamp = (v: unknown): v is CheckStamp => !!v && typeof v === "object" && typeof (v as CheckStamp).at === "string" && Array.isArray((v as CheckStamp).ports);

export function readInbox(): InboxItem[] {
  return readJSON<InboxItem[]>(INBOX_KEY, [], isList).filter((i) => i && typeof i.key === "string" && typeof i.title === "string" && typeof i.place_id === "string");
}
export const writeInbox = (items: InboxItem[]) => writeJSON(INBOX_KEY, items.slice(0, MAX_INBOX));

export function readBaselines(): Record<string, PortBaseline> {
  const all = readJSON<Record<string, PortBaseline>>(SNAP_KEY, {}, isMap);
  return Object.fromEntries(Object.entries(all).filter(([, b]) => b && b.snap && typeof b.snap.seen_at === "string" && b.snap.facts && typeof b.snap.facts === "object"));
}
export const writeBaselines = (b: Record<string, PortBaseline>) => (Object.keys(b).length ? writeJSON(SNAP_KEY, b) : (remove(SNAP_KEY), true));

export function readStamp(): CheckStamp | null {
  const s = readJSON<CheckStamp | null>(CHECK_KEY, null, (v): v is CheckStamp | null => v === null || isStamp(v));
  return s ? { at: s.at, ports: s.ports.filter((p) => typeof p === "string"), failed: Array.isArray(s.failed) ? s.failed.filter((p) => typeof p === "string") : [], data_at: typeof s.data_at === "string" ? s.data_at : null } : null;
}
export const writeStamp = (s: CheckStamp) => writeJSON(CHECK_KEY, s);

/** Add items, newest first, never twice for the same key (port · record · version), however often a check runs. */
export function mergeInbox(current: InboxItem[], incoming: InboxItem[]): InboxItem[] {
  const seen = new Set(current.map((i) => i.key));
  const fresh = incoming.filter((i) => !seen.has(i.key) && (seen.add(i.key), true));
  return [...fresh, ...current].sort((a, b) => b.created_at.localeCompare(a.created_at)).slice(0, MAX_INBOX);
}

export const unreadCount = (items: InboxItem[] = readInbox()) => items.filter((i) => !i.read_at).length;

export function markRead(keys: string[] | "all", now: Date = new Date()): void {
  const iso = now.toISOString();
  writeInbox(readInbox().map((i) => (keys === "all" || keys.includes(i.key) ? { ...i, read_at: i.read_at ?? iso } : i)));
}

export function clearInbox(): void {
  remove(INBOX_KEY);
}

/** Everything kept about one port: its baseline and its alerts (follow.ts calls this on unfollow). */
export function dropPort(port: string): void {
  const base = readBaselines();
  if (port in base) {
    delete base[port];
    writeBaselines(base);
  }
  const inbox = readInbox();
  const kept = inbox.filter((i) => i.place_id !== port);
  if (kept.length !== inbox.length) {
    if (kept.length) writeInbox(kept);
    else remove(INBOX_KEY);
  }
  const stamp = readStamp();
  if (stamp && (stamp.ports.includes(port) || stamp.failed.includes(port))) writeStamp({ ...stamp, ports: stamp.ports.filter((p) => p !== port), failed: stamp.failed.filter((p) => p !== port) });
}
