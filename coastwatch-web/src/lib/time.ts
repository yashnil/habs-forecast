const PT = "America/Los_Angeles";

/** "Thu, Oct 8" for a YYYY-MM-DD calendar date (no time-zone shifting). */
export function formatDate(date: string, opts: { year?: boolean } = {}): string {
  const d = new Date(`${date}T12:00:00Z`);
  return d.toLocaleDateString("en-US", {
    weekday: "short",
    month: "short",
    day: "numeric",
    ...(opts.year ? { year: "numeric" } : {}),
    timeZone: "UTC",
  });
}

/** "Oct 8, 10:04 AM PDT" in Pacific time. */
export function formatDateTimePT(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
    timeZone: PT,
    timeZoneName: "short",
  });
}

/** Today's calendar date in Pacific time, YYYY-MM-DD. */
export function pacificToday(now: Date): string {
  const parts = new Intl.DateTimeFormat("en-CA", { timeZone: PT, year: "numeric", month: "2-digit", day: "2-digit" }).format(now);
  return parts; // en-CA yields YYYY-MM-DD
}

/** "today", "tomorrow", "yesterday", "in 2 days", "3 days ago" relative to Pacific today. */
export function relativeDay(date: string, now: Date): string {
  const today = Date.parse(`${pacificToday(now)}T00:00:00Z`);
  const d = Date.parse(`${date}T00:00:00Z`);
  const diff = Math.round((d - today) / 86_400_000);
  if (diff === 0) return "today";
  if (diff === 1) return "tomorrow";
  if (diff === -1) return "yesterday";
  return diff > 0 ? `in ${diff} days` : `${-diff} days ago`;
}

export function formatAge(days: number | null): string {
  if (days === null) return "";
  if (days <= 0) return "today";
  if (days === 1) return "1 day ago";
  return `${days} days ago`;
}
