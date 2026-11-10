/**
 * Words for dates, times and figures (spec section 10): absolute dates ("Dec 15–18"), Pacific time with "PT"
 * where a time is shown, units fishermen use. Components format; they never hold a date literal.
 * Deterministic on server and client: every format names its time zone.
 */
export const TZ = "America/Los_Angeles";

type DateIn = string | number | Date;

/** A plain calendar date ("2026-12-15") is a Pacific calendar day, not UTC midnight. */
export function toDate(d: DateIn): Date {
  if (d instanceof Date) return d;
  if (typeof d === "number") return new Date(d);
  if (/^\d{4}-\d{2}-\d{2}$/.test(d)) return new Date(`${d}T12:00:00Z`);
  return new Date(d.replace(" ", "T"));
}

const fmt = (opts: Intl.DateTimeFormatOptions) => new Intl.DateTimeFormat("en-US", { timeZone: TZ, ...opts });
const F = {
  md: fmt({ month: "short", day: "numeric" }),
  mdy: fmt({ month: "short", day: "numeric", year: "numeric" }),
  m: fmt({ month: "short" }),
  d: fmt({ day: "numeric" }),
  y: fmt({ year: "numeric" }),
  my: fmt({ month: "short", year: "numeric" }),
  t: fmt({ hour: "numeric", minute: "2-digit" }),
  wd: fmt({ weekday: "short" }),
  ymd: fmt({ year: "numeric", month: "2-digit", day: "2-digit" }),
};

/** "Nov 10", with the year only when it differs from `ref`'s year (or always with `year: true`). */
export function day(d: DateIn, o: { ref?: DateIn; year?: boolean } = {}): string {
  const x = toDate(d);
  const showYear = o.year ?? (o.ref !== undefined && F.y.format(x) !== F.y.format(toDate(o.ref)));
  return (showYear ? F.mdy : F.md).format(x);
}

/** "Dec 15–18", "Nov 28–Dec 2", "Dec 30, 2026–Jan 2, 2027". En dash, no spaces (Chicago style). */
export function range(a: DateIn, b?: DateIn | null, o: { ref?: DateIn; year?: boolean } = {}): string {
  if (b === undefined || b === null) return day(a, o);
  const x = toDate(a), y = toDate(b);
  if (pacificYmd(x) === pacificYmd(y)) return day(x, o);
  const sameYear = F.y.format(x) === F.y.format(y);
  if (!sameYear) return `${F.mdy.format(x)}–${F.mdy.format(y)}`;
  const yearTail = o.year || (o.ref !== undefined && F.y.format(y) !== F.y.format(toDate(o.ref))) ? `, ${F.y.format(y)}` : "";
  if (F.m.format(x) === F.m.format(y)) return `${F.md.format(x)}–${F.d.format(y)}${yearTail}`;
  return `${F.md.format(x)}–${F.md.format(y)}${yearTail}`;
}

/** "7:00 AM PT" */
export function time(d: DateIn): string {
  return `${F.t.format(toDate(d))} PT`;
}

/** "Nov 10, 7:00 AM PT" */
export function dayTime(d: DateIn, o: { ref?: DateIn } = {}): string {
  return `${day(d, o)}, ${time(d)}`;
}

export function weekday(d: DateIn): string {
  return F.wd.format(toDate(d));
}
export function monthYear(d: DateIn): string {
  return F.my.format(toDate(d));
}
export function year(d: DateIn): string {
  return F.y.format(toDate(d));
}

/** "2026-11-10" in Pacific time, for keys and comparisons. */
export function pacificYmd(d: DateIn): string {
  const p = F.ymd.formatToParts(toDate(d));
  const g = (t: string) => p.find((x) => x.type === t)?.value ?? "";
  return `${g("year")}-${g("month")}-${g("day")}`;
}

/** Whole Pacific calendar days from `a` to `b` (positive when `b` is later). */
export function daysBetween(a: DateIn, b: DateIn): number {
  const u = (s: string) => Date.UTC(+s.slice(0, 4), +s.slice(5, 7) - 1, +s.slice(8, 10));
  return Math.round((u(pacificYmd(b)) - u(pacificYmd(a))) / 86_400_000);
}

/** "43 days ago", "yesterday", "today". */
export function age(d: DateIn, now: DateIn): string {
  const n = daysBetween(d, now);
  if (n <= 0) return "today";
  if (n === 1) return "yesterday";
  return `${n} days ago`;
}

/** Significant figures without trailing noise: 0.69, 9.8, 0.0013, 1,058. Precision follows error (T15). */
export function sig(v: number, digits = 2): string {
  if (!Number.isFinite(v)) return "";
  if (v === 0) return "0";
  const abs = Math.abs(v);
  if (abs >= 10 ** digits) return Math.round(v).toLocaleString("en-US");
  const p = Number(v.toPrecision(digits));
  const decimals = Math.max(0, digits - 1 - Math.floor(Math.log10(Math.abs(p))));
  return p.toLocaleString("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
}

/** "$2,015", "−$190" (true minus sign). */
export function usd(v: number, o: { cents?: boolean; signed?: boolean } = {}): string {
  const s = Math.abs(v).toLocaleString("en-US", { minimumFractionDigits: o.cents ? 2 : 0, maximumFractionDigits: o.cents ? 2 : 0 });
  const sign = v < 0 ? "−" : o.signed && v > 0 ? "+" : "";
  return `${sign}$${s}`;
}

/** A figure at fixed decimals with a true minus: "−0.4", "5.7". */
export function fixed(v: number, decimals = 1): string {
  const s = Math.abs(v).toLocaleString("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
  return `${v < 0 && Number(s.replace(/,/g, "")) !== 0 ? "−" : ""}${s}`;
}

/** Whole percent, true minus, thin space before the sign: "−20 %", "+30 %". */
export function pct(v: number, o: { signed?: boolean } = {}): string {
  const r = Math.round(v);
  const sign = r < 0 ? "−" : o.signed && r > 0 ? "+" : "";
  return `${sign}${Math.abs(r)} %`;
}

/** Degrees and decimal minutes, as on the chart: 36°48.3′ N, 121°47.2′ W. */
export function latLon(lat: number, lon: number): string {
  const dm = (v: number) => {
    const a = Math.abs(v);
    const deg = Math.floor(a);
    const min = (a - deg) * 60;
    return `${deg}°${min.toFixed(1)}′`;
  };
  return `${dm(lat)} ${lat >= 0 ? "N" : "S"}, ${dm(lon)} ${lon >= 0 ? "E" : "W"}`;
}
