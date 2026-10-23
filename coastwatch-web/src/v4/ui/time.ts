/** Times in the reader's words, from the source's own timestamp (never re-dated). */

import { dtf, pacificToday } from "@/lib/time";

const TZ = "America/Los_Angeles";

export function relTime(iso: string, now: Date): string {
  if (iso.length === 10) {
    // a calendar date: whole days from the Pacific date, so the evening (already tomorrow in UTC) does not add a day
    const d = Math.round((Date.parse(`${pacificToday(now)}T00:00:00Z`) - Date.parse(`${iso}T00:00:00Z`)) / 86400000);
    if (!Number.isFinite(d)) return "";
    return d <= 0 ? "today" : d === 1 ? "yesterday" : `${d} days ago`;
  }
  const t = Date.parse(iso);
  if (!Number.isFinite(t)) return "";
  const m = Math.round((now.getTime() - t) / 60000);
  if (m < 1) return "just now";
  if (m < 60) return `${m} min ago`;
  // whole hours elapsed, as the map's source stamps count them (lib/currents hoursAgo), so the layer card
  // and the stamp never disagree by an hour
  const h = Math.floor((now.getTime() - t) / 3_600_000);
  if (h < 36) return `${h} h ago`;
  // older than a day and a half: Pacific calendar days, as for a date, so one observation never reads
  // "2 days ago" in one row and "3 days ago" in another
  return relTime(pacificDay(iso), now);
}

/** The Pacific calendar date (YYYY-MM-DD) of an instant. */
export function pacificDay(iso: string): string {
  const t = new Date(iso);
  return Number.isFinite(t.getTime()) ? dtf("en-CA", { timeZone: TZ }).format(t) : iso.slice(0, 10);
}

/** "Oct 16" for a date in the current (Pacific) year, "Aug 27, 2025" otherwise; "Thu, Oct 16" with the
 *  weekday. Dates are calendar dates, read at noon UTC; a full timestamp is read as its Pacific day. A date
 *  never reads as this year's when it is not. */
export function dayLabel(date: string, weekday = false, now: Date | null = null): string {
  // a timestamp is an instant: its day is the Pacific day (7 PM PDT on Oct 16 is 02:00 UTC on Oct 17)
  if (date.length > 10 && date.includes("T")) date = pacificDay(date);
  const day = date.slice(0, 10);
  const year = day.slice(0, 4);
  const other = /^\d{4}$/.test(year) && year !== pacificToday(now ?? new Date()).slice(0, 4);
  return dtf("en-US", { month: "short", day: "numeric", ...(weekday ? { weekday: "short" } : {}), ...(other ? { year: "numeric" } : {}), timeZone: "UTC" }).format(new Date(`${day}T12:00:00Z`));
}

/** "Oct 16, 9:24 AM PDT": a measured time, in Pacific time with the zone named. */
/** "Aug 27, 2025" when the date is not in the current (Pacific) year, else "Aug 27": a date never reads as this year's when it is not. */
export function dayYearLabel(date: string, now: Date | null = null): string {
  return dayLabel(date, false, now);
}

/** "last HAB sample Aug 27, 2025" (toxin or Pseudo-nitzschia, the date every view uses for a station) */
export const lastHabWords = (d: string | null, now: Date | null = null) => (d ? `last HAB sample ${dayYearLabel(d, now)}` : "no HAB sample published");

export function clockLabel(iso: string): string {
  const d = new Date(iso);
  // an unparseable stamp stays as it was published (toLocaleString would say "Invalid Date")
  return Number.isNaN(d.getTime()) ? "Invalid Date" : dtf("en-US", { month: "short", day: "numeric", hour: "numeric", minute: "2-digit", timeZone: TZ, timeZoneName: "short" }).format(d);
}

export function kmLabel(km: number): string {
  return km < 1 ? `${Math.round(km * 1000)} m` : km < 10 ? `${km.toFixed(1)} km` : `${Math.round(km)} km`;
}
