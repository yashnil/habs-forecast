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

/**
 * The day labels the axis can carry without touching each other or the now label: greedily from the left, each
 * label about 7.5 px a character at 14 px plus its tick's 4 px, a gap of 8 px. The first day and the first of a
 * month say the month ("Oct 1"); others the day number. `T` is the axis width in px; `nx` the now line's x.
 */
export function axisLabels(axis: [string, string], T: number, nx: number): Array<{ date: string; text: string; x: number }> {
  const w = (t: string) => t.length * 7.5 + 8;
  const nowHalf = w("Now") / 2 + 4;
  const out: Array<{ date: string; text: string; x: number }> = [];
  let right = -Infinity;
  axisDays(axis).forEach((d, i) => {
    const x = axisX(axis, d, 0) * T;
    const text = i === 0 || d.endsWith("-01") ? `${new Date(`${d}T12:00:00Z`).toLocaleString("en-US", { month: "short", timeZone: "UTC" })} ${Number(d.slice(8, 10))}` : String(Number(d.slice(8, 10)));
    const x1 = x + w(text);
    if (x < right + 8 || x1 > T) return;
    if (x1 > nx - nowHalf && x < nx + nowHalf) return;
    out.push({ date: d, text, x });
    right = x1;
  });
  return out;
}

/** A product's span on the rail and how much of it has data, for the rail's one availability line. */
export function productSpan(p: ExploreProduct): { first: string; last: string; days: number } | null {
  const dates = [...p.frames.flatMap((f) => [f.start, f.end]), ...p.tracks.flatMap((t) => t.days.filter((d) => d.volume > 0).map((d) => d.date))].filter(Boolean).sort();
  if (!dates.length) return null;
  const days = new Set(p.tracks.flatMap((t) => t.days.filter((d) => d.volume > 0).map((d) => d.date)));
  return { first: dates[0], last: dates[dates.length - 1], days: days.size };
}
