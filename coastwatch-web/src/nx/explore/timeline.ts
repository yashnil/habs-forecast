/**
 * The time rail's arithmetic, DOM-free: a day axis in Pacific calendar days, where each frame sits on it, the
 * now line, and which frames play. Truth rules: frames are discrete and stamped (T26); a model run's leads never
 * play as if a field moved (T27); playback steps only through real frames, in time order.
 */
import { pacificYmd } from "@/nx/ds/format";
import type { CurrentFrame, ExploreFrame, ExploreProduct } from "./model";

const DAY = 86_400_000;
const utc = (d: string) => Date.parse(`${d}T00:00:00Z`);

/** Whole days from the axis start to a Pacific day. */
export const dayOffset = (axis: [string, string], d: string) => Math.round((utc(d) - utc(axis[0])) / DAY);
/** Number of day columns on the axis (both ends included). */
export const axisLength = (axis: [string, string]) => dayOffset(axis, axis[1]) + 1;

export function axisDays(axis: [string, string]): string[] {
  return Array.from({ length: axisLength(axis) }, (_, i) => new Date(utc(axis[0]) + i * DAY).toISOString().slice(0, 10));
}

/** A position on the axis, 0..1: a day's start (`frac` 0) to its end (`frac` 1). */
export function axisX(axis: [string, string], d: string, frac = 0): number {
  return (dayOffset(axis, d) + frac) / axisLength(axis);
}

/** The fraction of the Pacific day that has passed at an instant. */
export function dayFraction(iso: string): number {
  const t = new Date(iso);
  const p = new Intl.DateTimeFormat("en-US", { timeZone: "America/Los_Angeles", hour: "numeric", minute: "numeric", hourCycle: "h23" }).formatToParts(t);
  const h = Number(p.find((x) => x.type === "hour")?.value ?? 0), m = Number(p.find((x) => x.type === "minute")?.value ?? 0);
  return (h * 60 + m) / 1440;
}

/** Where the now line sits on the axis, 0..1 (clamped to the axis). */
export function nowX(axis: [string, string], nowIso: string): number {
  return Math.min(1, Math.max(0, axisX(axis, pacificYmd(nowIso), dayFraction(nowIso))));
}

/** An hourly frame's position on the day axis. */
export function hourX(axis: [string, string], f: Pick<CurrentFrame, "time">): number {
  return axisX(axis, pacificYmd(f.time), dayFraction(f.time));
}

/** The frame of a track nearest to an axis position (a frame whose span holds it wins). */
export function nearestFrame(axis: [string, string], frames: ExploreFrame[], x: number): ExploreFrame | null {
  let best: ExploreFrame | null = null;
  let bestD = Infinity;
  for (const f of frames) {
    const a = axisX(axis, f.start, 0), b = axisX(axis, f.end, 1);
    const d = x >= a && x <= b ? -1 / (b - a) : Math.min(Math.abs(x - a), Math.abs(x - b));
    if (d < bestD) {
      bestD = d;
      best = f;
    }
  }
  return best;
}

/** The hourly frame nearest a position on the hours strip (0..1 across the published hours). */
export function nearestHour(hours: CurrentFrame[], x: number): CurrentFrame | null {
  if (!hours.length) return null;
  const i = Math.round(x * (hours.length - 1));
  return hours[Math.max(0, Math.min(hours.length - 1, i))];
}

/**
 * The frames that play, in time order: a product's single-day frames (a latest view, which mixes days, does not
 * play), never a model run's leads (T27), and only when there are two or more.
 */
export function playFrames(p: ExploreProduct): ExploreFrame[] {
  if (p.kind === "model" || p.kind === "outlook") return [];
  const days = p.frames.filter((f) => f.start === f.end && !f.id.endsWith(":latest")).sort((a, b) => a.start.localeCompare(b.start));
  return days.length >= 2 ? days : [];
}

/** The hours that play: every published hour, in order (the 24 h mean is not an hour). */
export const playHours = (frames: CurrentFrame[]) => frames.filter((f) => !f.mean).sort((a, b) => a.time.localeCompare(b.time));

/** The next item after `id`, wrapping. */
export function nextOf<T extends { id: string }>(list: T[], id: string | null): T | null {
  if (!list.length) return null;
  const i = list.findIndex((x) => x.id === id);
  return list[(i + 1) % list.length];
}
