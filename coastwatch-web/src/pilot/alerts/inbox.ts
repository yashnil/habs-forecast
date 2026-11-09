import type { PortToday } from "@/pilot/contracts";
import { readJSON, removeKeys, writeJSON } from "@/pilot/account/storage";
import type { LocalAccount } from "@/pilot/account/local";
import { diffFacts, snapshotOf, type AlertItem, type PlaceSnapshot } from "./diff";

/**
 * The in-app alert inbox kept on this device: the alerts the device produced by comparing the
 * current PortToday facts with the snapshot it saw last time, for the saved ports and triggers the
 * person chose. Signed in (live), server-written alerts are merged in by the inbox component.
 */
export const INBOX_KEY = "nrt-pilot-inbox-v1";
export const SNAP_KEY = "nrt-pilot-alert-snap-v1";
export const MAX_INBOX = 50;

export type InboxItem = AlertItem & {
  read_at: string | null;
  origin: "device" | "account";
  /** the alerts row id, for items from the account */
  account_id?: string;
  /** made from the test fixture, not published data: the inbox says so on the item */
  sample?: boolean;
};

const isList = (v: unknown): v is InboxItem[] => Array.isArray(v);
const isSnaps = (v: unknown): v is Record<string, PlaceSnapshot> => !!v && typeof v === "object" && !Array.isArray(v);

export function readInbox(): InboxItem[] {
  return readJSON<InboxItem[]>(INBOX_KEY, [], isList).filter((i) => i && typeof i.key === "string" && typeof i.title === "string");
}

export function writeInbox(items: InboxItem[]): boolean {
  return writeJSON(INBOX_KEY, items.slice(0, MAX_INBOX));
}

export function readSnapshots(): Record<string, PlaceSnapshot> {
  return readJSON<Record<string, PlaceSnapshot>>(SNAP_KEY, {}, isSnaps);
}

export function unreadCount(items: InboxItem[] = readInbox()): number {
  return items.filter((i) => !i.read_at).length;
}

/** Add items, newest first, never twice for the same key (place · fact · version). */
export function mergeInbox(current: InboxItem[], incoming: InboxItem[]): InboxItem[] {
  const seen = new Set(current.map((i) => i.key));
  const fresh = incoming.filter((i) => !seen.has(i.key) && (seen.add(i.key), true));
  return [...fresh, ...current].sort((a, b) => b.created_at.localeCompare(a.created_at)).slice(0, MAX_INBOX);
}

export type CheckResult = {
  added: InboxItem[];
  /** places compared for the first time on this device: a baseline was kept, nothing alerted */
  baselined: string[];
  persisted: boolean;
};

/**
 * Compare each saved port's facts with what this device saw last time and add inbox items for the
 * person's in-app triggers. Pure apart from storage; safe to call on every page view (idempotent:
 * the same change never produces a second item).
 */
export function runLocalCheck(account: LocalAccount, todays: PortToday[], now: Date = new Date(), opts: { sample?: boolean } = {}): CheckResult {
  const iso = now.toISOString();
  const stored = readSnapshots();
  // only saved ports keep a baseline: a port saved again later starts afresh, and forgetting all leaves nothing
  const snaps: Record<string, PlaceSnapshot> = Object.fromEntries(Object.entries(stored).filter(([p]) => account.saved_places.includes(p)));
  if (!account.saved_places.length) {
    if (Object.keys(stored).length) removeKeys([SNAP_KEY]);
    return { added: [], baselined: [], persisted: true };
  }
  const added: InboxItem[] = [];
  const baselined: string[] = [];
  for (const place of account.saved_places) {
    const today = todays.find((t) => t.place_id === place);
    if (!today) continue;
    const pref = account.alerts.find((p) => p.place_id === place);
    const prev = snaps[place] ?? null;
    if (!prev) baselined.push(place);
    else if (pref && pref.channels.includes("in_app") && pref.triggers.length) {
      for (const a of diffFacts(prev, today, pref.triggers, iso)) added.push({ ...a, read_at: null, origin: "device", ...(opts.sample ? { sample: true } : {}) });
    }
    snaps[place] = snapshotOf(today, iso);
  }
  let persisted = writeJSON(SNAP_KEY, snaps);
  if (added.length) persisted = writeInbox(mergeInbox(readInbox(), added)) && persisted;
  return { added, baselined, persisted };
}

export function markRead(keys: string[] | "all", now: Date = new Date()): void {
  const iso = now.toISOString();
  writeInbox(readInbox().map((i) => (keys === "all" || keys.includes(i.key) ? { ...i, read_at: i.read_at ?? iso } : i)));
}

export function clearInbox(): void {
  writeInbox([]);
}
