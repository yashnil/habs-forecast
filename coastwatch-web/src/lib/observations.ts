import type { ObsStation, ObsVariable, ObsVariableSummary, ObservationDataset } from "@/generated/observations";
import type { FreshnessPolicy } from "@/generated/schema";
import { classifyDate, type Freshness } from "@/lib/freshness";

export type ObsVariableId = ObsVariable["id"];
export type Qualifier = NonNullable<ObsVariableSummary["last_qualifier"]>;

/** One sampling event for one variable. value null = sampled but this quantity not measured. */
export type Sample = { t: number; time: string; value: number | null; q: Qualifier | null; index: number };

export const DAY = 86_400_000;
export const DEFAULT_STATION = "HABs-SantaCruzWharf";
export const FLAGSHIP_REGION = "monterey_bay";

export const RANGES = [
  { id: "1y", label: "12 months", days: 365 },
  { id: "3y", label: "3 years", days: 3 * 365 },
  { id: "all", label: "Since 2014", days: null },
] as const;
export type RangeId = (typeof RANGES)[number]["id"];

export function samples(st: ObsStation, variable: ObsVariableId): Sample[] {
  const s = st.series.find((x) => x.variable === variable);
  if (!s) return [];
  return st.sample_times.map((time, i) => ({
    t: Date.parse(time),
    time,
    value: s.values[i] ?? null,
    q: (s.qualifiers?.[String(i)] as Qualifier | undefined) ?? null,
    index: i,
  }));
}

export function summaryOf(st: ObsStation, variable: ObsVariableId): ObsVariableSummary | null {
  return st.summaries.find((s) => s.variable === variable) ?? null;
}

export function lastSampleDate(st: ObsStation): string | null {
  return st.sample_times.at(-1)?.slice(0, 10) ?? null;
}

/** Freshness of a station's newest sample (any variable). */
export function stationFreshness(st: ObsStation, policy: FreshnessPolicy | null, now: Date | null): Freshness | null {
  if (!now || !policy) return null;
  if (st.status === "failed") return { state: "unavailable", basisDate: null, ageDays: null };
  return classifyDate(policy, lastSampleDate(st), now);
}

export function variableFreshness(sm: ObsVariableSummary | null, policy: FreshnessPolicy | null, now: Date | null): Freshness | null {
  if (!now || !policy) return null;
  return classifyDate(policy, sm?.last_date ?? null, now);
}

/** Readable value with units; a reported zero is never shown as a bare "0". */
export function formatObs(v: ObsVariable, value: number | null, q: Qualifier | null = null): string {
  if (q === "rejected_negative") return "rejected (QC)";
  if (value == null) return "not measured";
  if (q === "reported_zero") return "reported 0 (not quantified)";
  return `${formatNumber(v, value)} ${unitLabel(v.units)}`;
}

export function formatNumber(v: Pick<ObsVariable, "kind">, value: number): string {
  if (v.kind === "cell_abundance") return Math.round(value).toLocaleString("en-US");
  if (v.kind === "physical") return value.toFixed(1);
  if (value >= 100) return value.toFixed(0);
  if (value >= 1) return value.toPrecision(3);
  return value.toPrecision(2);
}

export function unitLabel(u: string): string {
  return u === "degree_C" ? "°C" : u === "mg/m3" ? "mg/m³" : u;
}

/** Compact log-axis tick labels: 0.001, 0.01 … 1k, 10k, 1M. */
export function formatTick(v: number): string {
  if (v >= 1e6) return `${v / 1e6}M`;
  if (v >= 1e3) return `${v / 1e3}k`;
  if (v >= 1) return String(v);
  return String(Number(v.toPrecision(1)));
}

export function logDomain(values: number[]): [number, number] | null {
  const pos = values.filter((v) => v > 0);
  if (!pos.length) return null;
  const lo = 10 ** Math.floor(Math.log10(Math.min(...pos)));
  let hi = 10 ** Math.ceil(Math.log10(Math.max(...pos)));
  if (hi <= lo) hi = lo * 10;
  return [lo, hi];
}

export function windowStart(range: RangeId, end: number, firstSample: number | null): number {
  const r = RANGES.find((x) => x.id === range)!;
  if (r.days == null) return firstSample ?? end - 365 * DAY;
  return end - r.days * DAY;
}

/** Sampling coverage of one variable inside a window: how often it was sampled vs measured. */
export function coverage(ss: Sample[], t0: number, t1: number) {
  const inWin = ss.filter((s) => s.t >= t0 && s.t <= t1);
  const measured = inWin.filter((s) => s.value != null);
  const zeros = measured.filter((s) => s.q === "reported_zero").length;
  const rejected = inWin.filter((s) => s.q === "rejected_negative").length;
  const gaps = measured.slice(1).map((s, i) => (s.t - measured[i].t) / DAY);
  const sorted = [...gaps].sort((a, b) => a - b);
  const median = sorted.length ? sorted[Math.floor(sorted.length / 2)] : null;
  const longest = sorted.length ? sorted[sorted.length - 1] : null;
  return { visits: inWin.length, measured: measured.length, zeros, rejected, medianGapDays: median, longestGapDays: longest };
}

export function cadenceLabel(days: number | null | undefined): string {
  if (days == null) return "no regular sampling in the last 12 months";
  if (days <= 8) return "about weekly";
  if (days <= 16) return "about every 2 weeks";
  if (days <= 35) return "about monthly";
  return `about every ${Math.round(days)} days`;
}

export function stationsByRegion(ds: ObservationDataset, regionOrder: string[]): { region: string; stations: ObsStation[] }[] {
  const groups = new Map<string, ObsStation[]>();
  for (const st of ds.stations) {
    const r = st.region ?? "other";
    groups.set(r, [...(groups.get(r) ?? []), st]);
  }
  const order = [FLAGSHIP_REGION, ...regionOrder.filter((r) => r !== FLAGSHIP_REGION), "other"];
  return order.filter((r) => groups.has(r)).map((r) => ({ region: r, stations: groups.get(r)! }));
}

/** Plain-language status of a toxin measurement for list rows; never implies absence. */
export function toxinStatus(v: ObsVariable, sm: ObsVariableSummary | null): string {
  if (!sm || sm.n_measured === 0) return `${v.id} not measured here since 2014`;
  if ((sm.days_since_last ?? 0) > 90) return `${v.id} last measured ${monthYear(sm.last_date!)}`;
  return `${v.id} ${formatObs(v, sm.last_value ?? null, sm.last_qualifier ?? null)}`;
}

export function monthYear(date: string): string {
  return new Date(`${date}T12:00:00Z`).toLocaleDateString("en-US", { month: "short", year: "numeric", timeZone: "UTC" });
}
