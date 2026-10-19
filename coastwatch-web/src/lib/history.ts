/**
 * A shore station's newest HAB measurement, placed in that station's own record as Neritic holds it
 * (Neritic keeps CalHABMAP samples from the observations window start, 2014-01-01; several piers'
 * upstream series begin earlier, e.g. Monterey Wharf 2005, Scripps Pier 2008, Santa Cruz Wharf 2011). Pure, and computed from the published observations, so no
 * new pipeline field is needed.
 *
 * What it says, and only that:
 * - how the newest value compares with the same station's samples in the same season (the sample's
 *   month and the months either side) in earlier years: the share of those samples below it;
 * - the last earlier sample at least as high, with its date;
 * - the sample before it, with its date (the change between visits, said as two numbers);
 * - how long the record is and how many samples it holds.
 *
 * What it never says: that a value is "unusual", "dangerous", "safe" or "rising risk". A rank is a
 * statement about the station's own record, not about seafood, other waters or the future. A
 * record too short to compare (fewer than MIN_SEASON_SAMPLES same-season samples from at least
 * MIN_SEASON_YEARS earlier years) is said as too short.
 *
 * Counting rules: a sample is a sampling visit with a value for the variable. A value the pipeline
 * rejected in quality control is left out.
 *
 * Reported zeros are censored, not small numbers. The pipeline: "0 means the value was at or below the
 * laboratory's limit, not that toxin was absent" (pDA; no limit is published), and for cells "cells may
 * still be present below the counting limit". So a zero is never counted as "lower" than a positive
 * value: a positive value is ranked only against the quantified (positive) samples of the season, and
 * the zeros are counted and said beside the rank. When too few quantified samples remain to rank,
 * the comparison is said as counts. Pseudo-nitzschia is the sum of the two size classes CalHABMAP counts (seriata and
 * delicatissima) on the visits where either was counted, the same sum the briefing reports.
 */

import { pacificDate } from "@/lib/time";

export type RecordVariable = "pDA" | "pn";

export type RecordContext = {
  variable: RecordVariable;
  /** the newest sample: its Pacific date (YYYY-MM-DD), its original UTC timestamp, and value */
  date: string;
  at: string;
  /** the newest value is flagged for review (above the plausibility ceiling): said, never ranked */
  flagged: boolean;
  value: number;
  /** the first sample of this variable in Neritic's copy (not necessarily the station's first) */
  since: string;
  /** where Neritic's copy of the CalHABMAP series begins (observations window_start), when known */
  windowStart: string | null;
  /** samples of this variable in the whole record, the newest included */
  n: number;
  /** earlier samples in the whole record that were quantified (positive): "the highest" needs at least one */
  quantifiedEarlier: number;
  season: {
    /** calendar months (1–12) the comparison uses: the sample's month and its neighbours */
    months: [number, number, number];
    /** same-season samples from earlier years */
    n: number;
    /** distinct earlier years among them */
    years: number;
    firstYear: number | null;
    lastYear: number | null;
    /** same-season samples left out because they are flagged for review (above the plausibility ceiling) */
    flagged: number;
    /** of them, reported 0 (censored: at or below a limit, not absent) */
    zeros: number;
    /** of them, quantified (positive) */
    quantified: number;
    /** distinct earlier years among the quantified ones */
    quantifiedYears: number;
    /** quantified samples strictly below the newest value (counts, for the said-as-counts form) */
    quantifiedBelow: number;
    /**
     * newest value positive: share of the QUANTIFIED same-season samples strictly below it (0–1), null when
     * fewer than MIN_SEASON_SAMPLES quantified samples from MIN_SEASON_YEARS years; newest value 0: share of
     * all same-season samples below it (always 0), null when the record is too short.
     */
    below: number | null;
    /** share of the same-season samples equal to the newest value (used for a zero: "as in N% …") */
    equal: number | null;
    median: number | null;
  };
  /** the most recent earlier sample at least as high (null: the highest in the record) */
  lastAsHigh: { date: string; value: number } | null;
  /** the visit before the newest one that measured this variable */
  previous: { date: string; value: number; flagged: boolean } | null;
};

export const MIN_SEASON_SAMPLES = 12;
export const MIN_SEASON_YEARS = 3;
/** below this many samples a rank is said to rest on a small sample, with what one sample is worth (1/30 ≈ 3 points) */
export const SMALL_SAMPLE = 30;

/** "(a small sample: each of the 16 is 6 points of the rank)", or "" from SMALL_SAMPLE up */
export const smallWords = (n: number) => (n > 0 && n < SMALL_SAMPLE ? ` (a small sample: each of the ${n} is ${Math.round(100 / n)} points of the rank)` : "");

type SeriesIn = { variable: string; values: (number | null)[]; qualifiers?: Record<string, string> | null };
type StationIn = { sample_times: string[]; series: SeriesIn[] };

const VARS: Record<RecordVariable, string[]> = { pDA: ["pDA"], pn: ["pn_seriata", "pn_delicatissima"] };

/** The station's measured samples of a record variable, oldest first. */
export type RecordSample = {
  /** the Pacific calendar date of the sample (what the page shows) */
  date: string;
  /** the original UTC timestamp, as SCCOOS publishes it */
  at: string;
  value: number;
  /** above the pipeline's plausibility ceiling, kept and flagged for review (`flag_high`): never ranked */
  flagged: boolean;
};

export function recordSamples(st: StationIn, variable: RecordVariable): RecordSample[] {
  const series = VARS[variable].map((v) => st.series.find((s) => s.variable === v)).filter((s): s is SeriesIn => !!s);
  if (!series.length) return [];
  const out: RecordSample[] = [];
  st.sample_times.forEach((t, i) => {
    let sum: number | null = null;
    let flagged = false;
    for (const s of series) {
      const v = s.values[i];
      const q = s.qualifiers?.[String(i)];
      if (v == null || q === "rejected_negative") continue;
      if (q === "flag_high") flagged = true;
      sum = (sum ?? 0) + v;
    }
    if (sum != null) out.push({ date: pacificDate(t), at: t, value: sum, flagged });
  });
  return out;
}

const month = (d: string) => Number(d.slice(5, 7));
const year = (d: string) => Number(d.slice(0, 4));
const seasonOf = (m: number): [number, number, number] => [((m + 10) % 12) + 1, m, (m % 12) + 1];

function median(xs: number[]): number | null {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  const h = s.length >> 1;
  return s.length % 2 ? s[h] : (s[h - 1] + s[h]) / 2;
}

/** The newest sample of `variable` in the station's own record; null when the station never measured it. */
export function recordContext(st: StationIn, variable: RecordVariable, windowStart: string | null = null): RecordContext | null {
  const all = recordSamples(st, variable);
  const last = all.at(-1);
  if (!last) return null;
  const earlier = all.slice(0, -1);
  // a value flagged for review is never a comparison point: not ranked against, never "as high"
  const usable = earlier.filter((s) => !s.flagged);
  const months = seasonOf(month(last.date));
  const y = year(last.date);
  const inSeason = earlier.filter((s) => year(s.date) < y && months.includes(month(s.date)));
  const same = inSeason.filter((s) => !s.flagged);
  const years = new Set(same.map((s) => year(s.date)));
  const enough = same.length >= MIN_SEASON_SAMPLES && years.size >= MIN_SEASON_YEARS;
  const asHigh = [...usable].reverse().find((s) => s.value >= last.value) ?? null;
  const q = same.filter((s) => s.value > 0);
  const qYears = new Set(q.map((s) => year(s.date)));
  const qEnough = q.length >= MIN_SEASON_SAMPLES && qYears.size >= MIN_SEASON_YEARS;
  const qBelow = q.filter((s) => s.value < last.value).length;
  // the newest sample itself flagged for review: it is said, never ranked
  const rankable = !last.flagged;
  return {
    variable,
    date: last.date,
    at: last.at,
    flagged: last.flagged,
    value: last.value,
    since: all[0].date,
    windowStart,
    n: all.length,
    quantifiedEarlier: usable.filter((s) => s.value > 0).length,
    season: {
      months,
      n: same.length,
      years: years.size,
      firstYear: same.length ? Math.min(...years) : null,
      lastYear: same.length ? Math.max(...years) : null,
      zeros: same.length - q.length,
      flagged: inSeason.length - same.length,
      quantified: q.length,
      quantifiedYears: qYears.size,
      quantifiedBelow: qBelow,
      // a zero is never "lower" than a positive value: rank a positive value among quantified samples only
      below: !rankable ? null : last.value > 0 ? (qEnough ? qBelow / q.length : null) : enough ? same.filter((s) => s.value < last.value).length / same.length : null,
      equal: rankable && enough ? same.filter((s) => s.value === last.value).length / same.length : null,
      median: enough ? median(same.map((s) => s.value)) : null,
    },
    lastAsHigh: asHigh ? { date: asHigh.date, value: asHigh.value } : null,
    previous: earlier.length ? { date: earlier.at(-1)!.date, value: earlier.at(-1)!.value, flagged: earlier.at(-1)!.flagged } : null,
  };
}

const MONTH = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
/** "Sep–Nov" */
export const seasonWords = (m: [number, number, number]) => `${MONTH[m[0] - 1]}–${MONTH[m[2] - 1]}`;

/** how a reported 0 is said for the variable */
export const zeroWords = (v: RecordVariable, n: number) =>
  v === "pn" ? `${n === 1 ? "1 had" : `${n} had`} no cells counted (not necessarily none present)` : `${n === 1 ? "1 was" : `${n} were`} reported 0 (not quantified, not absent)`;

/**
 * The comparison as one phrase, or null when the season's record is too short to say anything:
 * "higher than 88% of the 142 quantified Aug–Oct samples there in 2014–2025; another 126 had no cells counted …"
 * A positive value is ranked only against quantified samples (a reported 0 is censored, not lower).
 * With too few quantified samples to rank, the counts are said instead:
 * "of the 28 Mar–May samples there in 2019–2024, 23 were reported 0 (…) and 1 of the 5 quantified was lower"
 */
export function rankWords(r: RecordContext): string | null {
  const s = r.season;
  // a value flagged for review is never ranked (historySentence says why)
  if (r.flagged) return null;
  const months = seasonWords(s.months);
  const span = `${months} samples there in ${s.firstYear}–${s.lastYear}`;
  // values above the plausibility ceiling, kept for review, are left out of every comparison and said so
  const left = s.flagged ? `; ${s.flagged} flagged for review (above the plausibility ceiling) ${s.flagged === 1 ? "is" : "are"} left out` : "";
  if (r.value === 0) {
    if (s.equal == null) return null;
    return `${r.variable === "pn" ? "no cells counted" : "not quantified (reported 0)"}, as in ${Math.round(s.equal * 100)}% of the ${s.n} ${span}${smallWords(s.n)}${left}`;
  }
  const enoughSeason = s.n >= MIN_SEASON_SAMPLES && s.years >= MIN_SEASON_YEARS;
  if (s.below == null) {
    // the season has enough samples but too few quantified ones to rank: say the counts, never a percentage
    if (!enoughSeason || s.zeros === 0) return null;
    const qb = s.quantifiedBelow;
    const qPart =
      s.quantified === 0 ? "none was quantified"
      : qb === s.quantified ? (s.quantified === 1 ? "the 1 quantified was lower" : `all ${s.quantified} quantified were lower`)
      : `${qb} of the ${s.quantified} quantified ${qb === 1 ? "was" : "were"} lower`;
    return `of the ${s.n} ${span}, ${zeroWords(r.variable, s.zeros)}, and ${qPart}${left}`;
  }
  // the sample size is always said; a small one is said as small, with what one sample is worth
  const qSpan = `${s.quantified}${s.zeros ? " quantified" : ""} ${span}${smallWords(s.quantified)}`;
  const rest = `${s.zeros ? `; another ${zeroWords(r.variable, s.zeros)} and ${s.zeros === 1 ? "is" : "are"} not ranked` : ""}${left}`;
  if (s.below === 1) return `higher than all ${qSpan}${rest}`;
  if (s.below === 0) return `none of the ${qSpan} was lower${rest}`;
  // never round to "0%" or "100%" when the record says otherwise
  const pct = Math.min(99, Math.max(1, Math.round(s.below * 100)));
  return `higher than ${pct}% of the ${qSpan}${rest}`;
}

/** "Neritic's copy of its record, which starts in 2014" */
export const copyWords = (r: RecordContext) => `Neritic's copy of its record, which starts in ${r.since.slice(0, 4)}`;
