import { isPortKey } from "@/lib/portRecord";
import { TRIGGERS, type AlertTrigger } from "@/pilot/alerts/diff";
import { readJSON, writeJSON } from "@/nx/personal/store";
import { dropPort } from "./inbox";

/**
 * The ports the reader follows, and what should alert them at each (localStorage "nx-follow-v1"). Any CDFW
 * landing port can be followed, up to eight. It stays in this browser: nothing is sent anywhere, and nothing
 * checks these ports while Neritic is closed (check.ts runs when a page opens).
 *
 * The triggers are the pilot's own (src/pilot/alerts/diff.ts), so a followed port means the same thing here,
 * in the pilot account and in the server fan-out if that is ever switched on.
 */
export const FOLLOW_KEY = "nx-follow-v1";
export const MAX_FOLLOWS = 8;

/** A newly followed port alerts on official changes; model jumps and stale data are for the reader to turn on. */
export const DEFAULT_TRIGGERS: AlertTrigger[] = ["status_change", "new_toxin_result", "ramp_action"];

export type Follow = { port: string; triggers: AlertTrigger[]; /** when the reader followed it (device clock) */ since: string };
export type FollowRecord = { v: 1; ports: Follow[] };

const EMPTY: FollowRecord = { v: 1, ports: [] };
const isRecord = (v: unknown): v is FollowRecord => !!v && typeof v === "object" && (v as FollowRecord).v === 1 && Array.isArray((v as FollowRecord).ports);
const known = new Set<string>(TRIGGERS);
const ordered = (ts: readonly string[]): AlertTrigger[] => TRIGGERS.filter((t) => ts.includes(t));

/** Read tolerantly: anything that is not a port key or a known trigger is dropped, never trusted. */
export function readFollows(): Follow[] {
  const seen = new Set<string>();
  return readJSON<FollowRecord>(FOLLOW_KEY, EMPTY, isRecord)
    .ports.filter((p) => p && isPortKey(p.port) && !seen.has(p.port) && (seen.add(p.port), true))
    .slice(0, MAX_FOLLOWS)
    .map((p) => ({ port: p.port, triggers: ordered((Array.isArray(p.triggers) ? p.triggers : []).filter((t) => known.has(t))), since: typeof p.since === "string" ? p.since : "" }));
}

const write = (ports: Follow[]) => writeJSON(FOLLOW_KEY, { v: 1, ports } satisfies FollowRecord);

export type FollowResult = { ok: true; persisted: boolean } | { ok: false; why: "not_a_port" | "full" };

/** Follow a port (default triggers). Following it again changes nothing. */
export function follow(port: string, now: Date = new Date()): FollowResult {
  if (!isPortKey(port)) return { ok: false, why: "not_a_port" };
  const cur = readFollows();
  if (cur.some((p) => p.port === port)) return { ok: true, persisted: true };
  if (cur.length >= MAX_FOLLOWS) return { ok: false, why: "full" };
  return { ok: true, persisted: write([...cur, { port, triggers: DEFAULT_TRIGGERS, since: now.toISOString() }]) };
}

/**
 * Stop following a port. Its baseline and its alerts go with it (inbox.ts `dropPort`), so following it again
 * later starts afresh and nothing about it is left behind.
 */
export function unfollow(port: string): boolean {
  const cur = readFollows();
  dropPort(port);
  if (!cur.some((p) => p.port === port)) return true;
  return write(cur.filter((p) => p.port !== port));
}

export function setTrigger(port: string, trigger: AlertTrigger, on: boolean): boolean {
  const cur = readFollows();
  if (!cur.some((p) => p.port === port)) return false;
  return write(cur.map((p) => (p.port === port ? { ...p, triggers: ordered(on ? [...p.triggers, trigger] : p.triggers.filter((t) => t !== trigger)) } : p)));
}

export const isFollowed = (port: string | null | undefined) => !!port && readFollows().some((p) => p.port === port);
