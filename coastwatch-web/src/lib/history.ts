/**
 * A shore station's newest HAB measurement, placed in that station's own record (CalHABMAP series
 * since 2014 at the long-running piers). Pure, and computed from the published observations, so no
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
 * Counting rules: a sample is a sampling visit with a value for the variable. A reported zero
 * (below what the lab quantifies) counts as 0. A value the pipeline rejected in quality control is
 * left out. Pseudo-nitzschia is the sum of the two size classes CalHABMAP counts (seriata and
 * delicatissima) on the visits where either was counted, the same sum the briefing reports.
 */

export type RecordVariable = "pDA" | "pn";

export type RecordContext = {
  variable: RecordVariable;
  /** the newest sample: date (YYYY-MM-DD, UTC) and value */
  date: string;
  value: number;
  /** the first sample of this variable at the station */
  since: string;
  /** samples of this variable in the whole record, the newest included */
  n: number;
  season: {
    /** calendar months (1–12) the comparison uses: the sample's month and its neighbours */
    months: [number, number, number];
    /** same-season samples from earlier years */
    n: number;
    /** distinct earlier years among them */
    years: number;
    firstYear: number | null;
    lastYear: number | null;
    /** share of those samples strictly below the newest value (0–1); null when the record is too short */
    below: number | null;
    /** share of those samples equal to the newest value (meaningful for zeros) */
    equal: number | null;
    median: number | null;
  };
  /** the most recent earlier sample at least as high (null: the highest in the record) */
  lastAsHigh: { date: string; value: number } | null;
  /** the visit before the newest one that measured this variable */
  previous: { date: string; value: number } | null;
};

export const MIN_SEASON_SAMPLES = 12;
export const MIN_SEASON_YEARS = 3;

type SeriesIn = { variable: string; values: (number | null)[]; qualifiers?: Record<string, string> | null };
type StationIn = { sample_times: string[]; series: SeriesIn[] };

const VARS: Record<RecordVariable, string[]> = { pDA: ["pDA"], pn: ["pn_seriata", "pn_delicatissima"] };

/** The station's measured samples of a record variable, oldest first. */
export function recordSamples(st: StationIn, variable: RecordVariable): { date: string; value: number }[] {
  const series = VARS[variable].map((v) => st.series.find((s) => s.variable === v)).filter((s): s is SeriesIn => !!s);
  if (!series.length) return [];
  const out: { date: string; value: number }[] = [];
  st.sample_times.forEach((t, i) => {
    let sum: number | null = null;
    for (const s of series) {
      const v = s.values[i];
      if (v == null || s.qualifiers?.[String(i)] === "rejected_negative") continue;
      sum = (sum ?? 0) + v;
    }
    if (sum != null) out.push({ date: t.slice(0, 10), value: sum });
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
export function recordContext(st: StationIn, variable: RecordVariable): RecordContext | null {
  const all = recordSamples(st, variable);
  const last = all.at(-1);
  if (!last) return null;
  const earlier = all.slice(0, -1);
  const months = seasonOf(month(last.date));
  const y = year(last.date);
  const same = earlier.filter((s) => year(s.date) < y && months.includes(month(s.date)));
  const years = new Set(same.map((s) => year(s.date)));
  const enough = same.length >= MIN_SEASON_SAMPLES && years.size >= MIN_SEASON_YEARS;
  const asHigh = [...earlier].reverse().find((s) => s.value >= last.value) ?? null;
  return {
    variable,
    date: last.date,
    value: last.value,
    since: all[0].date,
    n: all.length,
    season: {
      months,
      n: same.length,
      years: years.size,
      firstYear: same.length ? Math.min(...years) : null,
      lastYear: same.length ? Math.max(...years) : null,
      below: enough ? same.filter((s) => s.value < last.value).length / same.length : null,
      equal: enough ? same.filter((s) => s.value === last.value).length / same.length : null,
      median: enough ? median(same.map((s) => s.value)) : null,
    },
    lastAsHigh: asHigh,
    previous: earlier.at(-1) ?? null,
  };
}

const MONTH = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
/** "Sep–Nov" */
export const seasonWords = (m: [number, number, number]) => `${MONTH[m[0] - 1]}–${MONTH[m[2] - 1]}`;

/**
 * The comparison as one phrase, or null when the record is too short:
 * "higher than 88% of the Sep–Nov samples there in 2014–2025" / "none counted, as in 41% of …"
 */
export function rankWords(r: RecordContext): string | null {
  const s = r.season;
  if (s.below == null || s.equal == null) return null;
  const span = `${seasonWords(s.months)} samples there in ${s.firstYear}–${s.lastYear}`;
  if (r.value === 0) return `${r.variable === "pn" ? "none counted" : "not quantified (reported 0)"}, as in ${Math.round(s.equal * 100)}% of the ${span}`;
  if (s.below === 1) return `higher than all ${s.n} ${span}`;
  if (s.below === 0) return `none of the ${s.n} ${span} was lower`;
  // never round to "0%" or "100%" when the record says otherwise
  const pct = Math.min(99, Math.max(1, Math.round(s.below * 100)));
  return `higher than ${pct}% of the ${span}`;
}
