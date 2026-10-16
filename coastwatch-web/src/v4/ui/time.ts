/** Times in the reader's words, from the source's own timestamp (never re-dated). */

const TZ = "America/Los_Angeles";

export function relTime(iso: string, now: Date): string {
  const t = Date.parse(iso.length === 10 ? `${iso}T12:00:00Z` : iso);
  if (!Number.isFinite(t)) return "";
  const m = Math.round((now.getTime() - t) / 60000);
  if (iso.length === 10) {
    const d = Math.round(m / 1440);
    return d <= 0 ? "today" : d === 1 ? "yesterday" : `${d} days ago`;
  }
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
