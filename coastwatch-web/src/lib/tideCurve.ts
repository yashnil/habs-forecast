import type { Tide } from "@/lib/conditions";

/**
 * A day's tide as a curve, from NOAA CO-OPS's predicted highs and lows (lib/conditions parseTides).
 * Between two turning points the height follows half a cosine, the usual way a tide table is drawn
 * between its extremes; it is a drawing of the predictions, not a prediction of its own. Before the
 * first turning point and after the last, one more half-cycle is mirrored from the next one so the
 * whole day is drawn; those stretches are flagged (`edge`) so they can be told apart.
 *
 * Times are CO-OPS local station time ("2026-10-12 18:42", lst_ldt), so minutes are minutes after the
 * Pacific midnight of `day` (YYYY-MM-DD), the day the curve is drawn for.
 */
export type TidePoint = { min: number; ft: number };
export type TideTurn = { min: number; ft: number; type: "H" | "L"; at: string };
export type TideDay = { points: TidePoint[]; turns: TideTurn[]; lo: number; hi: number; edgeBefore: number; edgeAfter: number };

const DAY_MIN = 1440;

/** Minutes after `day`'s midnight of a CO-OPS local time; NaN when unreadable. */
export function minutesOf(at: string, day: string): number {
  const m = /^(\d{4}-\d{2}-\d{2})[ T](\d{2}):(\d{2})/.exec(at);
  if (!m) return NaN;
  const days = Math.round((Date.parse(`${m[1]}T00:00:00Z`) - Date.parse(`${day}T00:00:00Z`)) / 86_400_000);
  return days * DAY_MIN + Number(m[2]) * 60 + Number(m[3]);
}

const cosStep = (t: number, t0: number, t1: number, h0: number, h1: number) => h0 + (h1 - h0) * (1 - Math.cos((Math.PI * (t - t0)) / (t1 - t0))) / 2;

/** The height at minute `t` from a list of turning points sorted by time (at least two). */
function heightAt(ex: { min: number; ft: number }[], t: number): number {
  let i = 0;
  while (i < ex.length - 2 && ex[i + 1].min < t) i++;
  return cosStep(t, ex[i].min, ex[i + 1].min, ex[i].ft, ex[i + 1].ft);
}

/** The curve for `day` sampled every `step` minutes, with the day's own turning points; null without two turning points to draw between. */
export function tideDay(tides: Tide[], day: string, step = 10): TideDay | null {
  const ex = tides
    .map((t) => ({ min: minutesOf(t.at, day), ft: t.ft, type: t.type, at: t.at }))
    .filter((t) => Number.isFinite(t.min) && Number.isFinite(t.ft))
    .sort((a, b) => a.min - b.min);
  // the day must hold at least one predicted turn: a curve made only by extending other days is not drawn
  if (ex.length < 2 || !ex.some((t) => t.min >= 0 && t.min <= DAY_MIN)) return null;
  // mirror one half-cycle at either end where the predictions do not reach past the day
  const all = [...ex];
  if (all[0].min > 0) all.unshift({ min: 2 * all[0].min - all[1].min, ft: all[1].ft, type: all[1].type, at: "" });
  const n = all.length;
  if (all[n - 1].min < DAY_MIN) all.push({ min: 2 * all[n - 1].min - all[n - 2].min, ft: all[n - 2].ft, type: all[n - 2].type, at: "" });
  const points: TidePoint[] = [];
  for (let t = 0; t <= DAY_MIN; t += step) points.push({ min: t, ft: heightAt(all, t) });
  const turns = ex.filter((t) => t.min >= 0 && t.min <= DAY_MIN);
  const fts = points.map((p) => p.ft);
  return { points, turns, lo: Math.min(...fts), hi: Math.max(...fts), edgeBefore: Math.max(0, ex[0].min), edgeAfter: Math.min(DAY_MIN, ex[ex.length - 1].min) };
}

/** Rising or falling at minute `t`, and the next turning point after it (from the predictions themselves). */
export function tideNow(tides: Tide[], day: string, t: number): { rising: boolean; next: TideTurn | null } | null {
  const ex = tides.map((x) => ({ min: minutesOf(x.at, day), ft: x.ft, type: x.type, at: x.at })).filter((x) => Number.isFinite(x.min)).sort((a, b) => a.min - b.min);
  const next = ex.find((x) => x.min > t) ?? null;
  if (!next) return null;
  // heading to a high is rising; heading to a low is falling
  return { rising: next.type === "H", next };
}

/** The width of one character of a turn label (11.5 px tabular figures), a little generous so a box never undercounts. */
export const LABEL_CHAR_W = 6.6;

/**
 * Where a turn's label goes: centred on its turn (or held inside the strip at either end), unless its box
 * would cross the now line; then wholly on the turn's own side of the line when it fits there. A label is
 * never moved across the line, away from its turn: where it cannot clear the line it stays put, and the
 * line passes behind its halo.
 */
export function placeLabel(tx: number, width: number, W: number, nowX: number | null): { lx: number; anchor: "start" | "middle" | "end" } {
  const edge = 2;
  const pref: { lx: number; anchor: "start" | "middle" | "end" } =
    tx - width / 2 < edge ? { lx: Math.max(edge, tx - 6), anchor: "start" } : tx + width / 2 > W - edge ? { lx: Math.min(W - edge, tx + 6), anchor: "end" } : { lx: tx, anchor: "middle" };
  if (nowX == null) return pref;
  const box = (c: { lx: number; anchor: string }) => (c.anchor === "start" ? [c.lx, c.lx + width] : c.anchor === "end" ? [c.lx - width, c.lx] : [c.lx - width / 2, c.lx + width / 2]);
  const clear = (c: { lx: number; anchor: string }) => {
    const [a, b] = box(c);
    return a >= edge && b <= W - edge && (b < nowX - 5 || a > nowX + 5);
  };
  if (clear(pref)) return pref;
  const side = tx < nowX ? { lx: nowX - 6, anchor: "end" as const } : { lx: nowX + 6, anchor: "start" as const };
  return clear(side) ? side : pref;
}

/** The now line's drawn stretches from y0 to y1, leaving a gap (2 px either side) wherever a label's box sits across it. */
export function nowLine(nx: number, y0: number, y1: number, labels: { box: { l: number; r: number; top: number; bottom: number } }[]): [number, number][] {
  const cuts = labels
    .map((x) => x.box)
    .filter((b) => b.l - 2 <= nx && nx <= b.r + 2)
    .map((b) => [b.top - 2, b.bottom + 2] as [number, number])
    .sort((a, b) => a[0] - b[0]);
  const out: [number, number][] = [];
  let from = y0;
  for (const [a, b] of cuts) {
    if (a > from) out.push([from, Math.min(a, y1)]);
    from = Math.max(from, b);
  }
  if (from < y1) out.push([from, y1]);
  return out;
}
