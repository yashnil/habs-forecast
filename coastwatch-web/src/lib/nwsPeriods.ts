import type { MarinePeriod } from "@/lib/conditions";
import { pacificToday } from "@/lib/time";

// The NWS zone-forecast periods dated by their own start (shared by My Coast and the evidence graph,
// so neither ever matches a trip day by position).
const WEEKDAYS = ["sun", "mon", "tue", "wed", "thu", "fri", "sat"];
const addDays = (date: string, n: number) => new Date(Date.parse(`${date}T12:00:00Z`) + n * 86400000).toISOString().slice(0, 10);
const weekdayOf = (date: string) => new Date(`${date}T12:00:00Z`).getUTCDay();

/** A NWS zone-forecast period with the Pacific date and part of day it starts on, worked out from its own name and the forecast's issue time. */
export type DatedPeriod = MarinePeriod & { date: string | null; night: boolean; start: string | null };

/**
 * Each period's own start, from the forecast's issue time and the period's name: "Today", "This
 * Afternoon", "Tonight" and "Overnight" start on the day of issue; "Wed" and "Wednesday Night" on the next
 * Wednesday on or after the previous period's day. A period that cannot be placed has no date (null),
 * and is never matched to a trip day. `issued` is NWS's `updated` (ISO); without it nothing is dated.
 */
export function datePeriods(periods: MarinePeriod[], issued: string | null): DatedPeriod[] {
  const t = issued ? Date.parse(issued) : NaN;
  if (!Number.isFinite(t)) return periods.map((p) => ({ ...p, date: null, night: /night|tonight|overnight|evening/i.test(p.name), start: null }));
  const day0 = pacificToday(new Date(t));
  let off = 0;
  return periods.map((p, i) => {
    const name = p.name.trim().toLowerCase();
    const night = /night|tonight|overnight|evening/.test(name);
    let date: string | null = null;
    if (/^(today|tonight|this |rest of|overnight|late tonight)/.test(name)) {
      date = addDays(day0, off);
    } else {
      const w = WEEKDAYS.findIndex((d) => name.startsWith(d));
      if (w >= 0) {
        // the first period may be the day of issue itself; every later one is on or after the one before it
        let j = i === 0 ? 0 : off;
        while (j < off + 8 && weekdayOf(addDays(day0, j)) !== w) j++;
        if (j < off + 8) {
          off = j;
          date = addDays(day0, j);
        }
      }
    }
    return { ...p, date, night, start: date ? `${date}T${night ? "18" : "06"}:00` : null };
  });
}
