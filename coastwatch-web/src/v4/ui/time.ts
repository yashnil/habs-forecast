/** Times in the reader's words, from the source's own timestamp (never re-dated). */

import { pacificToday } from "@/lib/time";

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
  const h = Math.round(m / 60);
  if (h < 36) return `${h} h ago`;
  return `${Math.round(h / 24)} days ago`;
}

/** "Oct 16" for a date; "Thu, Oct 16" with the weekday. Dates are calendar dates, read at noon UTC. */
export function dayLabel(date: string, weekday = false): string {
  return new Date(`${date.slice(0, 10)}T12:00:00Z`).toLocaleDateString("en-US", { month: "short", day: "numeric", ...(weekday ? { weekday: "short" } : {}), timeZone: "UTC" });
}

/** "Oct 16, 9:24 AM PDT": a measured time, in Pacific time with the zone named. */
export function clockLabel(iso: string): string {
  return new Date(iso).toLocaleString("en-US", { month: "short", day: "numeric", hour: "numeric", minute: "2-digit", timeZone: TZ, timeZoneName: "short" });
}

export function kmLabel(km: number): string {
  return km < 1 ? `${Math.round(km * 1000)} m` : km < 10 ? `${km.toFixed(1)} km` : `${Math.round(km)} km`;
}
