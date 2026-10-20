const PT = "America/Los_Angeles";

/**
 * One Intl.DateTimeFormat per locale and option set, built on first use and reused: building a
 * formatter costs tens of microseconds and the map re-labels dates on every render. The output is
 * identical to Date#toLocale*String with the same options.
 */
const FORMATTERS = new Map<string, Intl.DateTimeFormat>();
export function dtf(locale: string, opts: Intl.DateTimeFormatOptions): Intl.DateTimeFormat {
  const key = `${locale}|${JSON.stringify(opts)}`;
  let f = FORMATTERS.get(key);
  if (!f) FORMATTERS.set(key, (f = new Intl.DateTimeFormat(locale, opts)));
  return f;
}

/** "3:00 PM PDT": a measured instant's clock time in Pacific time, with the zone named. */
export function clockPT(iso: string): string {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : dtf("en-US", { hour: "numeric", minute: "2-digit", timeZone: PT, timeZoneName: "short" }).format(d);
}

/** "Thu, Oct 8" for a YYYY-MM-DD calendar date (no time-zone shifting). */
export function formatDate(date: string, opts: { year?: boolean } = {}): string {
  const d = new Date(`${date}T12:00:00Z`);
  return dtf("en-US", {
    weekday: "short",
    month: "short",
    day: "numeric",
    ...(opts.year ? { year: "numeric" } : {}),
    timeZone: "UTC",
  }).format(d);
}

/** "Oct 8, 10:04 AM PDT" in Pacific time. */
export function formatDateTimePT(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return dtf("en-US", {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
    timeZone: PT,
    timeZoneName: "short",
  }).format(d);
}

/** Today's calendar date in Pacific time, YYYY-MM-DD. */
// Building an Intl.DateTimeFormat is expensive (tens of microseconds); it is called once per sample
// when binning a station's record by Pacific date, so the formatter is built once and reused.
const PT_DAY = new Intl.DateTimeFormat("en-CA", { timeZone: PT, year: "numeric", month: "2-digit", day: "2-digit" });

export function pacificToday(now: Date): string {
  return PT_DAY.format(now); // en-CA yields YYYY-MM-DD
}

/**
 * The Pacific calendar date (YYYY-MM-DD) of a source's UTC timestamp. A sample taken at 6 PM PDT
 * carries the next day's UTC date; it is shown on its local day, and the original UTC timestamp
 * travels beside it (never re-dated). A date-only string is returned as it is.
 */
export function pacificDate(iso: string): string {
  if (iso.length <= 10) return iso;
  const t = new Date(iso);
  return Number.isNaN(t.getTime()) ? iso.slice(0, 10) : PT_DAY.format(t);
}

/** Whole days from Pacific today to a YYYY-MM-DD date (negative in the past). */
function dayDiff(date: string, now: Date): number {
  const today = Date.parse(`${pacificToday(now)}T00:00:00Z`);
  const d = Date.parse(`${date}T00:00:00Z`);
  return Math.round((d - today) / 86_400_000);
}

/** "today", "tomorrow", "yesterday", "in 2 days", "3 days ago" relative to Pacific today. */
export function relativeDay(date: string, now: Date): string {
  const diff = dayDiff(date, now);
  if (diff === 0) return "today";
  if (diff === 1) return "tomorrow";
  if (diff === -1) return "yesterday";
  return diff > 0 ? `in ${diff} days` : `${-diff} days ago`;
}

/**
 * The compact form for a chip a few characters wide: "today", "tomorrow", "yesterday", "in 2 d",
 * "3 d ago". Same days as `relativeDay`; only the plural words are shortened, so a chip never
 * clips "2 days ago" to "2 days a".
 */
export function relativeDayShort(date: string, now: Date): string {
  const diff = dayDiff(date, now);
  if (diff === 0) return "today";
  if (diff === 1) return "tomorrow";
  if (diff === -1) return "yesterday";
  return diff > 0 ? `in ${diff} d` : `${-diff} d ago`;
}

export function formatAge(days: number | null): string {
  if (days === null) return "";
  if (days <= 0) return "today";
  if (days === 1) return "1 day ago";
  return `${days} days ago`;
}
