import type { Tide } from "@/lib/conditions";
import { pacificYmd, weekday } from "@/nx/ds/format";
import type { SeaPeriod } from "./brief";

/**
 * The sea window's days: NWS periods grouped by their own dates, the predicted tides of each day, and the
 * hours the tide is rising (for ports with a regulated bar). Pure and deterministic; nothing here ranks or
 * picks a day (the reader chooses, T36).
 */
export type SeaDay = { date: string; label: string; day: SeaPeriod | null; night: SeaPeriod | null };

/** "Today", then "Tue 13": the day a period belongs to, from its own date. */
export function dayLabel(date: string, now: string): string {
  if (date === pacificYmd(now)) return "Today";
  return `${weekday(date)} ${Number(date.slice(8, 10))}`;
}

export function seaDays(lanes: SeaPeriod[], now: string): SeaDay[] {
  const out: SeaDay[] = [];
  for (const p of lanes) {
    if (!p.date) continue;
    let d = out.find((x) => x.date === p.date);
    if (!d) {
      d = { date: p.date, label: dayLabel(p.date, now), day: null, night: null };
      out.push(d);
    }
    if (p.night) d.night ??= p;
    else d.day ??= p;
  }
  return out;
}

/** CO-OPS times are Pacific wall-clock text ("2026-10-13 12:03"): read as written, never through the browser's zone. */
export function tideClock(at: string): string {
  const m = /(\d{2}):(\d{2})/.exec(at);
  if (!m) return at;
  const h = Number(m[1]);
  return `${h % 12 === 0 ? 12 : h % 12}:${m[2]} ${h < 12 ? "AM" : "PM"}`;
}

export const tidesOn = (tides: Tide[], date: string) => [...tides].filter((t) => t.at.startsWith(date)).sort((a, b) => a.at.localeCompare(b.at));

/** Tide predictions are given to the thousandth of a foot; a tenth is what a tide table prints. */
export const tideFeet = (ft: number) => `${ft < 0 ? "−" : ""}${Math.abs(ft).toFixed(1)} ft`;

/** "High 5.4 ft 12:03 PM, low 0.1 ft 7:27 PM" */
export function tideWords(tides: Tide[], date: string): string | null {
  const day = tidesOn(tides, date);
  if (!day.length) return null;
  const s = day.map((t) => `${t.type === "H" ? "high" : "low"} ${tideFeet(t.ft)} ${tideClock(t.at)}`).join(", ");
  return s.charAt(0).toUpperCase() + s.slice(1);
}

/** The spans when the predicted tide rises (a low to the next high) that touch this day: "5:56 AM–12:03 PM". */
export function risingSpans(tides: Tide[], date: string): string[] {
  const t = [...tides].sort((a, b) => a.at.localeCompare(b.at));
  const out: string[] = [];
  for (let i = 0; i < t.length - 1; i++) {
    const a = t[i];
    const b = t[i + 1];
    if (a.type !== "L" || b.type !== "H") continue;
    if (!a.at.startsWith(date) && !b.at.startsWith(date)) continue;
    const later = b.at.slice(0, 10) !== a.at.slice(0, 10) ? ` ${weekday(b.at.slice(0, 10))}` : "";
    out.push(`${tideClock(a.at)}–${tideClock(b.at)}${later}`);
  }
  return out;
}

/** The forecast's largest stated seas, for the day strip's scale (never below 1 ft). */
export const seasScale = (days: SeaDay[]) => Math.max(1, ...days.flatMap((d) => [d.day?.seasHi ?? 0, d.night?.seasHi ?? 0]));

/** "3–4", "3": the seas NWS stated for a period, in feet. */
export const seasWords = (p: SeaPeriod | null) => (p?.seasHi == null ? null : p.seasLo != null && p.seasLo !== p.seasHi ? `${p.seasLo}–${p.seasHi}` : String(p.seasHi));
