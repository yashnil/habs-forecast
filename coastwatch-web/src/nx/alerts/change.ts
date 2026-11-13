import type { InboxItem } from "./inbox";

/**
 * The change an alert reports, as the Alerts ticket draws it (v14 synthesis, SYSTEM.md "Alerts"): the date the record
 * gives now, the date it gave before, and the days between. Read from the record's own words (the alert's title and
 * what it said before), never made up: a record that names no date gives no date, and the ticket then shows only
 * the words.
 *
 *   "Commercial season delayed to at least Jan 1, 2027" → "Commercial season opens Jan 15, 2027"
 *   gives { was: Jan 1 2027, now: Jan 15 2027, days: 14 }
 */
export type AlertChange = {
  /** the date the record gives now (UTC noon), or null when it names none */
  now: Date | null;
  /** the date it gave before, or null (a new record, a record that named no date, or one no longer listed) */
  was: Date | null;
  /** now − was in whole days, when both are known */
  days: number | null;
};

const MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"];
const NAMED = /\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?\b/;
const ISO = /\b(\d{4})-(\d{2})-(\d{2})\b/;
const DAY = 864e5;

const noon = (y: number, m: number, d: number): Date | null => {
  const t = new Date(Date.UTC(y, m, d, 12));
  return t.getUTCMonth() === m && t.getUTCDate() === d ? t : null;
};

/**
 * The first date a record's words name. A date without a year takes the year that puts it nearest `ref` (a season
 * date in "Dec 15" read in January is last December's), so a record never jumps a year by being read late.
 */
export function firstDate(text: string | null | undefined, ref: Date = new Date()): Date | null {
  if (!text) return null;
  const a = NAMED.exec(text);
  const b = ISO.exec(text);
  // whichever comes first in the words
  if (b && (!a || b.index < a.index)) return noon(Number(b[1]), Number(b[2]) - 1, Number(b[3]));
  if (!a) return null;
  const m = MONTHS.indexOf(a[1].slice(0, 3).toLowerCase());
  const d = Number(a[2]);
  if (a[3]) return noon(Number(a[3]), m, d);
  const y = ref.getUTCFullYear();
  const options = [y - 1, y, y + 1].map((yy) => noon(yy, m, d)).filter((x): x is Date => !!x);
  options.sort((p, q) => Math.abs(p.getTime() - ref.getTime()) - Math.abs(q.getTime() - ref.getTime()));
  return options[0] ?? null;
}

/** What the alert's record said before and says now, as dates. */
export function alertChange(i: Pick<InboxItem, "title" | "was" | "change" | "created_at" | "as_of">): AlertChange {
  const ref = new Date(i.as_of ?? i.created_at);
  const r = Number.isFinite(ref.getTime()) ? ref : new Date();
  if (i.change === "gone") return { now: null, was: null, days: null };
  const now = firstDate(i.title, r);
  const was = i.change === "changed" ? firstDate(i.was, r) : null;
  const days = now && was ? Math.round((now.getTime() - was.getTime()) / DAY) : null;
  return { now, was, days };
}

/** "+14 days", "14 days earlier", "Same date": the gap in the reader's words. */
export function daysWords(days: number): { figure: string; note: string } {
  if (days === 0) return { figure: "Same date", note: "new words" };
  const n = Math.abs(days);
  const unit = n === 1 ? "day" : "days";
  return days > 0 ? { figure: `+${n} ${unit}`, note: "later" } : { figure: `−${n} ${unit}`, note: "earlier" };
}

