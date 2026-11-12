import type { ObsStation, ObsVariable, ObsVariableSummary } from "@/generated/observations";
import type { FreshnessPolicy } from "@/generated/schema";
import { classifyDate, type Freshness } from "@/lib/freshness";
import { dtf } from "@/lib/time";

export type ObsVariableId = ObsVariable["id"];
export type Qualifier = NonNullable<ObsVariableSummary["last_qualifier"]>;

/** One sampling event for one variable. value null = sampled but this quantity not measured. */
export type Sample = { t: number; time: string; value: number | null; q: Qualifier | null; index: number };

export const DAY = 86_400_000;
export const DEFAULT_STATION = "HABs-SantaCruzWharf";

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

export function cadenceLabel(days: number | null | undefined): string {
  if (days == null) return "no regular sampling in the last 12 months";
  if (days <= 8) return "about weekly";
  if (days <= 16) return "about every 2 weeks";
  if (days <= 35) return "about monthly";
  return `about every ${Math.round(days)} days`;
}

export function monthYear(date: string): string {
  return dtf("en-US", { month: "short", year: "numeric", timeZone: "UTC" }).format(new Date(`${date}T12:00:00Z`));
}

/** Map bounds per region, identical to data/curated/ports.json (checked by a unit test). */
export const REGION_BOUNDS: Record<string, [[number, number], [number, number]]> = {
  north_coast: [[-124.75, 39.9], [-123.6, 42.05]],
  mendocino_sonoma: [[-124.0, 38.2], [-122.8, 39.95]],
  sf_bay_farallones: [[-123.35, 37.25], [-122.2, 38.15]],
  monterey_bay: [[-122.45, 36.45], [-121.72, 37.15]],
  central_coast: [[-121.6, 34.4], [-120.4, 36.1]],
  southern_california: [[-120.6, 32.45], [-117.05, 34.55]],
};
export const CALIFORNIA_BOUNDS: [[number, number], [number, number]] = [[-124.6, 32.4], [-117.0, 42.0]];

const SOURCE_HOST = "erddap.sccoos.org";

/**
 * The station's original CalHABMAP data as CSV: the exact SCCOOS ERDDAP query the pipeline ran
 * (`request_url`), re-encoded so every constraint character is percent-encoded. Returns null unless
 * the query is an https tabledap CSV request on SCCOOS for this very station, so a malformed or
 * unexpected URL never becomes a download link.
 */
export function sourceCsvUrl(st: Pick<ObsStation, "station_id" | "request_url">): string | null {
  if (!st.request_url || !/^HABs-[A-Za-z0-9-]+$/.test(st.station_id)) return null;
  let u: URL;
  try {
    u = new URL(st.request_url);
  } catch {
    return null;
  }
  if (u.protocol !== "https:" || u.hostname !== SOURCE_HOST || u.pathname !== `/erddap/tabledap/${st.station_id}.csv`) return null;
  const raw = st.request_url.slice(st.request_url.indexOf("?") + 1);
  if (!raw || raw === st.request_url) return null;
  let parts: string[];
  try {
    parts = raw.split("&").map((p) => encodeURIComponent(decodeURIComponent(p)).replace(/%2C/g, ","));
  } catch {
    return null;
  }
  if (!parts[0].split(",").includes("time")) return null;
  return `https://${SOURCE_HOST}${u.pathname}?${parts.join("&")}`;
}

// ---------------------------------------------------------------- per-variable truth (v13)
// A station's newest VISIT says nothing about a quantity: Monterey Wharf is visited weekly and has
// published temperature only since 2025 (no pDA since 2022-08-31). Everything below is per variable.

/** One measured sample of one variable: a quantified value, or a reported 0 (never a rejected or unmeasured visit). */
export type Measured = { time: string; date: string; value: number; qualifier: Qualifier | null; index: number };

/** The newest sample of a variable that carries a value (a reported 0 counts; a rejected sample and a visit without it do not). */
export function lastMeasured(st: Pick<ObsStation, "sample_times" | "series">, variable: ObsVariableId): Measured | null {
  const s = st.series.find((x) => x.variable === variable);
  if (!s) return null;
  for (let i = st.sample_times.length - 1; i >= 0; i--) {
    const value = s.values[i];
    const q = (s.qualifiers?.[String(i)] as Qualifier | undefined) ?? null;
    if (value == null || q === "rejected_negative") continue;
    return { time: st.sample_times[i], date: st.sample_times[i].slice(0, 10), value, qualifier: q, index: i };
  }
  return null;
}

/** Whole days from a UTC date (YYYY-MM-DD…) to `now`. */
const daysSince = (date: string, now: Date | number) => Math.floor((+now - Date.parse(`${date.slice(0, 10)}T00:00:00Z`)) / DAY);

/**
 * What is known of one variable at a station, from either shape the app carries: the full station
 * (`summaries`, Research) or the slim one (`latest`, Explore and the briefings; it lists only pDA and the two
 * Pseudo-nitzschia groups). `known: false` means this shape cannot say (never read as "not measured").
 */
export type VariableRecord = { known: boolean; ever: boolean; lastDate: string | null; nLast365: number };
type WithSummaries = { summaries: Pick<ObsVariableSummary, "variable" | "n_measured" | "last_date" | "n_last_365d">[] };
type WithLatest = { latest: { variable: string; value: number | null; date: string; n_last_365d: number }[] };
export type MeasuringStation = WithSummaries | WithLatest;
/** The variables a slim station's `latest` covers (lib/data MAP_VARIABLES). */
const SLIM_VARIABLES: string[] = ["pDA", "pn_seriata", "pn_delicatissima"];

export function variableRecord(st: MeasuringStation, variable: ObsVariableId): VariableRecord {
  if ("summaries" in st) {
    const sm = st.summaries.find((x) => x.variable === variable);
    return { known: true, ever: (sm?.n_measured ?? 0) > 0, lastDate: sm?.last_date ?? null, nLast365: sm?.n_last_365d ?? 0 };
  }
  const l = st.latest.find((x) => x.variable === variable);
  if (!l) return { known: SLIM_VARIABLES.includes(variable), ever: false, lastDate: null, nLast365: 0 };
  return { known: true, ever: true, lastDate: l.date, nLast365: l.n_last_365d };
}

/**
 * The stations that measure a variable (the Explore → Research contract: link only to a station that has it).
 * With no options: every station that has ever published it. `withinDays` (with `now`) keeps only those whose
 * newest measured sample of THAT variable is that recent: a recent visit that measured something else does not
 * count. A slim station asked about a variable its shape does not carry is left out (unknown is not "measures").
 */
export function stationsMeasuring<S extends MeasuringStation>(stations: S[], variable: ObsVariableId, o: { withinDays?: number; now?: Date | number } = {}): S[] {
  return stations.filter((st) => {
    const r = variableRecord(st, variable);
    if (!r.known || !r.ever) return false;
    if (o.withinDays == null) return true;
    return r.lastDate != null && daysSince(r.lastDate, o.now ?? Date.now()) <= o.withinDays;
  });
}

/** The upstream column of each variable (pipeline sources/observations.py VARIABLES), for a source-row request. */
export const SOURCE_COLUMN: Record<ObsVariableId, string> = {
  pDA: "pDA",
  dDA: "dDA",
  tDA: "tDA",
  pn_seriata: "Pseudo_nitzschia_seriata_group",
  pn_delicatissima: "Pseudo_nitzschia_delicatissima_group",
  chl_extracted: "Avg_Chloro",
  temp: "Temp",
};

/**
 * One sample's own row at the source: the SCCOOS ERDDAP table for that station at that exact sample time, with
 * the sample id, the depth, the location code and every quantity Neritic reads (plus the filtered and settled
 * volumes the lab recorded). Null unless the station id and the time are what they should be.
 */
export function sourceRowUrl(st: Pick<ObsStation, "station_id">, time: string): string | null {
  if (!/^HABs-[A-Za-z0-9-]+$/.test(st.station_id) || !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$/.test(time)) return null;
  const cols = ["time", "latitude", "longitude", "depth", "Location_Code", "SampleID", ...Object.values(SOURCE_COLUMN), "DA_Volume_Filtered", "Chl_Volume_Filtered", "Volume_Settled_for_Counting"];
  return `https://${SOURCE_HOST}/erddap/tabledap/${st.station_id}.htmlTable?${cols.join(",")}&time=${encodeURIComponent(time)}`;
}
