import type { MarinePeriod } from "@/lib/conditions";
import { datePeriods } from "@/lib/nwsPeriods";

/**
 * The NWS zone forecast as glanceable lanes (Today's sea and weather): each period's wind and seas
 * read from NWS's own words. Nothing is invented. A number the text does not state is null, and the
 * forecaster's full sentence is always kept beside it (`text`).
 */
export type PeriodNumbers = {
  /** first direction stated ("NW"), or "variable" / "light"; null when the text names none */
  dir: string | null;
  windLo: number | null;
  /** the highest wind speed stated anywhere in the period (a "becoming 15 to 20 kt" counts) */
  windHi: number | null;
  gust: number | null;
  seasLo: number | null;
  seasHi: number | null;
  /** the first swell stated: "W 5 ft at 11 s" */
  swell: { dir: string | null; ft: number; s: number | null } | null;
};

const DIR = "(N|NNE|NE|ENE|E|ESE|SE|SSE|S|SSW|SW|WSW|W|WNW|NW|NNW)";

export function periodNumbers(text: string): PeriodNumbers {
  const t = text.replace(/\s+/g, " ");
  const winds = [...t.matchAll(new RegExp(`(?:\\b${DIR}\\s+winds?|\\bwinds?\\s+${DIR})\\s+(?:around\\s+|up to\\s+|near\\s+)?(\\d+)(?:\\s+to\\s+(\\d+))?\\s*(?:kt|knots)`, "gi"))];
  let dir: string | null = null;
  let lo: number | null = null;
  let hi: number | null = null;
  for (const m of winds) {
    const d = (m[1] ?? m[2]).toUpperCase();
    const a = Number(m[3]);
    const b = m[4] ? Number(m[4]) : a;
    if (dir == null) {
      dir = d;
      lo = a;
    }
    hi = Math.max(hi ?? 0, b);
  }
  // a change later in the period ("backing to W 10 to 15 kt") raises the period's top speed
  for (const m of t.matchAll(new RegExp(`\\b(?:becoming|backing to|veering to|increasing to|rising to|shifting to)\\s+(?:${DIR}\\s+)?(?:around\\s+)?(\\d+)(?:\\s+to\\s+(\\d+))?\\s*kt`, "gi"))) {
    const b = Number(m[3] ?? m[2]);
    if (Number.isFinite(b)) hi = Math.max(hi ?? 0, b);
  }
  if (dir == null) {
    if (/\b(winds? light|light winds?)\b/i.test(t)) dir = "light";
    else if (/\bvariable winds?\b|\bwinds? variable\b/i.test(t)) dir = "variable";
    const lv = t.match(/\b(?:light|variable) winds?\s+(?:up to\s+)?(\d+)(?:\s+to\s+(\d+))?\s*kt/i) ?? t.match(/\bwinds?\s+(?:light|variable)[^.]*?(\d+)(?:\s+to\s+(\d+))?\s*kt/i);
    if (lv) {
      lo = Number(lv[1]);
      hi = lv[2] ? Number(lv[2]) : lo;
    }
  }
  const gusts = [...t.matchAll(/gusts?\s+(?:up\s+)?to\s+(\d+)\s*kt/gi)].map((m) => Number(m[1]));
  const seas = t.match(/\b(?:combined\s+)?seas?\s+(?:around\s+|up to\s+)?(\d+)(?:\s+to\s+(\d+))?\s*(?:ft|feet)/i);
  const wd = t.match(new RegExp(`wave detail:\\s*${DIR}\\s+(\\d+)\\s*ft\\s+at\\s+(\\d+)\\s*seconds`, "i"));
  const sw = t.match(new RegExp(`(?:\\b${DIR}\\s+)?swell\\s+(?:${DIR}\\s+)?(\\d+)(?:\\s+to\\s+(\\d+))?\\s*(?:ft|feet)(?:\\s+at\\s+(\\d+)\\s*(?:seconds|s)\\b)?`, "i"));
  return {
    dir,
    windLo: lo,
    windHi: hi,
    gust: gusts.length ? Math.max(...gusts) : null,
    seasLo: seas ? Number(seas[1]) : null,
    seasHi: seas ? Number(seas[2] ?? seas[1]) : null,
    swell: wd ? { dir: wd[1].toUpperCase(), ft: Number(wd[2]), s: Number(wd[3]) } : sw ? { dir: (sw[1] ?? sw[2] ?? null)?.toUpperCase() ?? null, ft: Number(sw[4] ?? sw[3]), s: sw[5] ? Number(sw[5]) : null } : null,
  };
}

export type LanePeriod = PeriodNumbers & { name: string; short: string; date: string | null; night: boolean; text: string };

const WD = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

/** "Tonight" → "Tnt", "Wednesday Night" → "Wed nt", "Today" → "Today" */
/** The column's day: "Today" for the day of issue, else the weekday ("Tue"); `night` says which half. */
export function shortPeriodName(name: string, date: string | null): string {
  const n = name.toLowerCase();
  if (/^(today|this afternoon|rest of today|tonight|overnight|late tonight|this evening)/.test(n)) return "Today";
  return date ? WD[new Date(`${date}T12:00:00Z`).getUTCDay()] : name.slice(0, 3);
}

/** The forecast's periods, dated by their own names (lib/nwsPeriods), with their numbers. */
export function lanePeriods(periods: MarinePeriod[], issued: string | null, max = 10): LanePeriod[] {
  return datePeriods(periods, issued)
    .slice(0, max)
    .map((p) => ({ ...periodNumbers(p.text), name: p.name, short: shortPeriodName(p.name, p.date), date: p.date, night: p.night, text: p.text }));
}

/** "NW 10–15 kt, gusts 20 · seas 4–6 ft": the period in one compact line (only what NWS stated). */
export function periodLine(p: PeriodNumbers): string | null {
  const wind =
    p.windHi != null
      ? `${p.dir && p.dir !== "variable" && p.dir !== "light" ? `${p.dir} ` : p.dir === "light" ? "light, " : p.dir === "variable" ? "variable " : ""}${p.windLo != null && p.windLo !== p.windHi ? `${p.windLo}–${p.windHi}` : p.windHi} kt${p.gust ? `, gusts ${p.gust}` : ""}`
      : p.dir === "light"
        ? "light winds"
        : null;
  const seas = p.seasHi != null ? `seas ${p.seasLo != null && p.seasLo !== p.seasHi ? `${p.seasLo}–${p.seasHi}` : p.seasHi} ft` : null;
  const s = [wind, seas].filter(Boolean).join(" · ");
  return s || null;
}
