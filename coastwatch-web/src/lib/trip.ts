import type { LayerArtifact } from "@/generated/schema";
import type { Audience } from "@/lib/coast";
import type { Step } from "@/lib/nextSteps";
import type { MarinePeriod } from "@/lib/conditions";
import { topicWords } from "@/lib/harborEvidence";
import { dtf, pacificToday } from "@/lib/time";

/**
 * The trip (My Coast, Coastal mode): the days a trip can be planned for and the checklist of what to
 * open before leaving. Every claim the briefing makes comes from lib/harborEvidence, never from here.
 */

// ---------------------------------------------------------------- the trip day

export type TripDay = {
  /** Pacific date, YYYY-MM-DD */
  date: string;
  /** "Today", "Tomorrow", "Thu" */
  label: string;
  /** "Oct 14" */
  day: string;
  /** the C-HARM lead index valid that day, or null when no model run covers it */
  lead: number | null;
};

/** Days a trip can be planned for: today and the next three (the span of NOAA's tide predictions the conditions read), each with the model day that covers it, if any. */
export function tripDays(today: string, leads: (LayerArtifact | null)[], span = 4): TripDay[] {
  const t0 = Date.parse(`${today}T12:00:00Z`);
  return Array.from({ length: span }, (_, i) => {
    const date = new Date(t0 + i * 86400000).toISOString().slice(0, 10);
    const d = new Date(`${date}T12:00:00Z`);
    const lead = leads.findIndex((l) => l?.time.valid_date === date);
    return {
      date,
      label: i === 0 ? "Today" : i === 1 ? "Tomorrow" : dtf("en-US", { weekday: "short", timeZone: "UTC" }).format(d),
      day: dtf("en-US", { month: "short", day: "numeric", timeZone: "UTC" }).format(d),
      lead: lead >= 0 ? lead : null,
    };
  });
}

/** A trip date from a link: kept when it is one of the plannable days, else today. */
export function parseTripDate(v: string | null, days: TripDay[]): string | null {
  if (!days.length) return null;
  return v && days.some((d) => d.date === v) ? v : days[0].date;
}

// ---------------------------------------------------------------- before you leave

const PRIORITY: [RegExp, number][] = [
  [/^dispute-/, 0],
  [/^statement-/, 1],
  [/^read-notices$/, 2],
  [/^cdph-map$/, 3],
  [/^confirm-/, 4],
  [/^closure-line$/, 5],
  [/^whale-safe$/, 6],
  [/^marine-alert$/, 7],
  [/^(marine|marine-link|marine-gap)$/, 8],
  [/^hab-sample$/, 9],
  [/^(species-rules|commercial|region-)/, 10],
  [/^toxin-gap$/, 11],
  [/^mpa$/, 12],
  [/^history$/, 13],
];
/** Steps that state a fact become the action that settles it. */
const ACTION_TITLE: Record<string, string> = {
  "toxin-gap": "Look at CDPH's latest toxin test results",
  "marine-gap": "Read the NWS marine forecast for your zone",
  marine: "Read NWS's full forecast for your zone",
};
const rank = (s: Step) => PRIORITY.find(([re]) => re.test(s.id))?.[1] ?? 20;

/**
 * "Before you leave": the steps as actions, most consequential first (the agency's own word on the
 * fishery, then its notices, then the safety lines and weather), at most `max`, the rest kept. A
 * quoted statement becomes the action of reading it at its source; buoy readings stay with the
 * conditions (they are facts, not actions) and the unverified-list reminder is the source line itself.
 */
export function checklist(steps: Step[], max = 5): { shown: Step[]; more: Step[] } {
  const seen = new Set<string>();
  const acts = steps
    .filter((s) => s.id !== "buoy" && s.id !== "verify")
    .map((s) =>
      s.id.startsWith("statement-")
        ? {
            ...s,
            title: /season status/i.test(s.reason) ? `Confirm the season status on ${s.source.label.replace(/ page$/, "")}'s page` : `Check ${s.source.label.replace(/ page$/, "")}'s page on ${topicWords(s.reason)}`,
            reason: `Neritic quotes it as “${s.title.replace(/^[^“]*“|”$/g, "")}” Agency pages can change between Neritic's checks.`,
          }
        : s,
    )
    .map((s) => (ACTION_TITLE[s.id] ? { ...s, title: ACTION_TITLE[s.id] } : s))
    .sort((a, b) => rank(a) - rank(b))
    .reduce<Step[]>((acc, s) => {
      // one step per source page: a second item on the same page is folded into the first
      const k = "url" in s.source ? s.source.url : s.source.tel;
      const kept = s.id.startsWith("dispute-") ? undefined : acc.find((y) => ("url" in y.source ? y.source.url : y.source.tel) === k);
      if (kept) {
        kept.reason = `${kept.reason} Also on this page: ${s.title.replace(/^(Check|Read|Confirm) /, (m) => m.toLowerCase())}.`;
        return acc;
      }
      if (seen.has(k)) return acc;
      seen.add(k);
      acc.push({ ...s });
      return acc;
    }, []);
  return { shown: acts.slice(0, max), more: acts.slice(max) };
}

/** A stable key for the checklist's ticks: one trip (waters, activity, species, day). */
export const checklistKey = (place: string, audience: Audience, species: string, date: string | null) => `cw-trip-checks-v1:${place}:${audience}:${species}:${date ?? "today"}`;

// ---------------------------------------------------------------- the reader's own sea limits

/**
 * The highest wind (kt, gusts included) and sea height (ft) a NWS marine period states in its own
 * words ("W wind 10 to 15 kt, rising to 15 to 20 kt with gusts up to 25 kt. Seas 4 to 6 ft."). Seas are
 * read from "Seas …" or, when a zone states no seas, from its swell and wind-wave heights. Null where
 * the text states none. This reads the forecaster's numbers; it never makes one.
 */
export function marineMaxima(text: string): { windKt: number | null; seasFt: number | null } {
  const sentences = text.split(/(?<=\.)\s+/);
  const allNums = (s: string, unit: RegExp) => [...s.matchAll(new RegExp(`(\\d+(?:\\.\\d+)?)\\s*(?:to\\s*(\\d+(?:\\.\\d+)?)\\s*)?${unit.source}`, "gi"))].flatMap((m) => [Number(m[1]), m[2] ? Number(m[2]) : NaN]).filter(Number.isFinite);
  const wind = sentences.filter((s) => /\b(wind|gusts?)\b/i.test(s) && /\bkt\b/i.test(s)).flatMap((s) => allNums(s, /kt\b/));
  let seas = sentences.filter((s) => /\bseas\b/i.test(s)).flatMap((s) => allNums(s.slice(s.search(/\bseas?\b/i)), /ft\b/));
  if (!seas.length) seas = sentences.filter((s) => /\b(swell|wind waves?)\b/i.test(s)).flatMap((s) => allNums(s, /ft\b/));
  return { windKt: wind.length ? Math.max(...wind) : null, seasFt: seas.length ? Math.max(...seas) : null };
}

/** The reader's own limits for going out, on this device only. Never a recommendation by Neritic. */
export type SeaLimits = { windKt: number | null; seasFt: number | null };
export const LIMITS_KEY = "cw-sea-limits-v1";

export function parseLimits(raw: string | null | undefined): SeaLimits {
  try {
    const o = JSON.parse(raw ?? "null") as Partial<SeaLimits> | null;
    const ok = (v: unknown, max: number) => (typeof v === "number" && Number.isFinite(v) && v > 0 && v <= max ? v : null);
    return { windKt: ok(o?.windKt, 80), seasFt: ok(o?.seasFt, 40) };
  } catch {
    return { windKt: null, seasFt: null };
  }
}

/** A period's stated maxima against the reader's limits: which limits the forecaster's own numbers go past. */
export function overLimits(m: { windKt: number | null; seasFt: number | null }, l: SeaLimits): { wind: boolean; seas: boolean } {
  return { wind: l.windKt != null && m.windKt != null && m.windKt > l.windKt, seas: l.seasFt != null && m.seasFt != null && m.seasFt > l.seasFt };
}

// ---------------------------------------------------------------- the NWS period for the trip day

export { datePeriods, type DatedPeriod } from "@/lib/nwsPeriods";
import { datePeriods, type DatedPeriod } from "@/lib/nwsPeriods";

/**
 * The NWS period for a trip day, matched by each period's own start date (never by position): the
 * day's daytime period, else its night period. Null when no period the forecast holds starts on that
 * day, which a saved copy read on a later day must say ("no NWS period saved for this day").
 */
export function periodFor(c: { marine: { status: string; fetchedAt?: string | null; data?: { periods: MarinePeriod[]; updated: string | null } } } | null, day: string | null, today: string | null): DatedPeriod | null {
  if (c?.marine.status !== "ok" || !c.marine.data) return null;
  const want = day ?? today;
  if (!want) return null;
  // NWS's own issue time, else when Neritic fetched it (the same day: a saved copy keeps its fetch time)
  const ps = datePeriods(c.marine.data.periods, c.marine.data.updated ?? c.marine.fetchedAt ?? null);
  return ps.find((p) => p.date === want && !p.night) ?? ps.find((p) => p.date === want) ?? null;
}
