import { displayManifest } from "@/lib/palette";
import { readFile } from "node:fs/promises";
import path from "node:path";
import Ajv2020 from "ajv/dist/2020";
import type { Manifest, PortsCollection } from "@/generated/schema";
import type { OfficialDataset } from "@/generated/official";
import type { PortIntelCollection } from "@/generated/port_intel";
import type { ObservationDataset } from "@/generated/observations";
import type { FisheriesDataset } from "@/generated/fisheries";
import { recordContext, type RecordContext, type RecordVariable } from "@/lib/history";
import fisheriesSchema from "@/generated/schemas/fisheries.schema.json";
import observationsSchema from "@/generated/schemas/observations.schema.json";
import manifestSchema from "@/generated/schemas/manifest.schema.json";
import officialSchema from "@/generated/schemas/official.schema.json";
import portIntelSchema from "@/generated/schemas/port_intel.schema.json";
import portsSchema from "@/generated/schemas/ports.schema.json";

/**
 * Server-side loading of pipeline artifacts. The manifest is validated against the
 * pipeline's JSON Schema before anything is rendered; an invalid or missing manifest
 * produces an explicit "data unavailable" state instead of a broken or guessed map.
 */
export type DataLoad =
  | {
      ok: true;
      manifest: Manifest;
      ports: PortsCollection | null;
      portsError: string | null;
      official: OfficialDataset | null;
      officialError: string | null;
      portIntel: PortIntelCollection | null;
      portIntelError: string | null;
      /** measured monitoring stations, latest value per variable only (map inspector) */
      stations: StationLite[] | null;
      stationsError: string | null;
      baseUrl: string;
    }
  | { ok: false; error: string; baseUrl: string };

/** What the map inspector needs from the measured observations: where each station is and its
 *  latest value per variable with the sample date. The full series stay on the Bloom page. */
export type StationLite = {
  station_id: string;
  name: string;
  lat: number;
  lon: number;
  status: string;
  source_url: string;
  /** the station's original-data request on SCCOOS ERDDAP (lib/observations sourceCsvUrl) */
  request_url?: string | null;
  latest: { variable: string; label: string; units: string; kind: string; value: number | null; date: string; qualifier: string | null; n_last_365d: number }[];
  /** the last 370 days of samples of the HAB measurements (My Coast's evidence timelines): [date, value] */
  recent?: Record<string, [string, number | null][]>;
  /** the newest toxin and Pseudo-nitzschia samples placed in the station's own record (lib/history) */
  record?: Partial<Record<RecordVariable, RecordContext>>;
};
/** HAB measurements kept as recent timelines for the evidence cards */
const RECENT_VARIABLES = ["pDA", "pn_seriata", "pn_delicatissima", "chl_extracted"];
const RECENT_DAYS = 370;
const MAP_VARIABLES = ["pDA", "pn_seriata", "pn_delicatissima"];

export function slimStations(obs: ObservationDataset): StationLite[] {
  const meta = new Map(obs.variables.map((v) => [v.id, v]));
  // the timelines end at the dataset's newest sample (not each station's), so a long-silent station sends none
  const newest = Math.max(0, ...obs.stations.map((st) => (st.sample_times.length ? Date.parse(st.sample_times[st.sample_times.length - 1]) : 0)));
  return obs.stations.flatMap((st) => (st.lat == null || st.lon == null ? [] : [{
    station_id: st.station_id,
    name: st.name,
    lat: st.lat,
    lon: st.lon,
    status: st.status,
    source_url: st.source_url,
    request_url: st.request_url ?? null,
    latest: st.summaries
      .filter((sm) => MAP_VARIABLES.includes(sm.variable) && sm.last_date)
      .map((sm) => {
        const v = meta.get(sm.variable);
        return { variable: sm.variable, label: v?.label ?? sm.variable, units: v?.units ?? "", kind: v?.kind ?? "", value: sm.last_value ?? null, date: sm.last_date!, qualifier: sm.last_qualifier ?? null, n_last_365d: sm.n_last_365d };
      }),
    recent: recentSeries(st, newest),
    record: stationRecord(st, obs.window_start ?? null),
  }]));
}

function stationRecord(st: ObservationDataset["stations"][number], windowStart: string | null): Partial<Record<RecordVariable, RecordContext>> {
  const out: Partial<Record<RecordVariable, RecordContext>> = {};
  for (const v of ["pDA", "pn"] as const) {
    const r = recordContext(st as Parameters<typeof recordContext>[0], v, windowStart);
    if (r) out[v] = r;
  }
  return out;
}

/** Samples in the RECENT_DAYS days before the dataset's newest sample, per HAB variable (measured values only, 3 significant digits). */
function recentSeries(st: ObservationDataset["stations"][number], newest: number): Record<string, [string, number | null][]> {
  const times = st.sample_times;
  const from = newest - RECENT_DAYS * 86400000;
  const out: Record<string, [string, number | null][]> = {};
  for (const se of st.series as { variable: string; values: (number | null)[] }[]) {
    if (!RECENT_VARIABLES.includes(se.variable)) continue;
    const pts: [string, number | null][] = [];
    se.values.forEach((v, i) => {
      if (v == null) return; // not measured that visit: no point (a gap is not a zero)
      const t = Date.parse(times[i]);
      if (t >= from) pts.push([times[i].slice(0, 10), Number(v.toPrecision(3))]);
    });
    if (pts.length) out[se.variable] = pts;
  }
  return out;
}

const ajv = new Ajv2020({ allErrors: true, strict: false });
const validateManifest = ajv.compile<Manifest>(manifestSchema);
const validatePorts = ajv.compile<PortsCollection>(portsSchema);
const validateOfficial = ajv.compile<OfficialDataset>(officialSchema);
const validatePortIntel = ajv.compile<PortIntelCollection>(portIntelSchema);
const validateObservations = ajv.compile<ObservationDataset>(observationsSchema);
const validateFisheries = ajv.compile<FisheriesDataset>(fisheriesSchema);

export function dataBaseUrl(): string {
  return (process.env.CW_DATA_BASE_URL || "/data/v1").replace(/\/$/, "");
}

async function readArtifact(base: string, rel: string): Promise<unknown> {
  if (/^https?:\/\//.test(base)) {
    const r = await fetch(`${base}/${rel}`, { next: { revalidate: 300 } });
    if (!r.ok) throw new Error(`HTTP ${r.status} fetching ${rel}`);
    return r.json();
  }
  // CW_DATA_DIR lets the server read artifacts from any directory (tests, local runs);
  // otherwise they are read from the matching path under public/.
  const dir = process.env.CW_DATA_DIR || path.join(process.cwd(), "public", base);
  const file = path.join(dir, rel);
  return JSON.parse(await readFile(file, "utf8"));
}

function schemaErrors(errors: typeof validateManifest.errors): string {
  return (errors ?? [])
    .slice(0, 5)
    .map((e) => `${e.instancePath || "/"} ${e.message}`)
    .join("; ");
}

export function checkManifest(raw: unknown): { ok: true; manifest: Manifest } | { ok: false; error: string } {
  if (!validateManifest(raw)) return { ok: false, error: `manifest failed schema validation: ${schemaErrors(validateManifest.errors)}` };
  return { ok: true, manifest: raw };
}

export function checkPorts(raw: unknown): { ok: true; ports: PortsCollection } | { ok: false; error: string } {
  if (!validatePorts(raw)) return { ok: false, error: `ports failed schema validation: ${schemaErrors(validatePorts.errors)}` };
  return { ok: true, ports: raw };
}

export function checkOfficial(raw: unknown): { ok: true; official: OfficialDataset } | { ok: false; error: string } {
  if (!validateOfficial(raw)) return { ok: false, error: `official notices failed schema validation: ${schemaErrors(validateOfficial.errors)}` };
  return { ok: true, official: raw };
}

export function checkPortIntel(raw: unknown): { ok: true; portIntel: PortIntelCollection } | { ok: false; error: string } {
  if (!validatePortIntel(raw)) return { ok: false, error: `port summaries failed schema validation: ${schemaErrors(validatePortIntel.errors)}` };
  return { ok: true, portIntel: raw };
}

export function checkObservations(raw: unknown): { ok: true; observations: ObservationDataset } | { ok: false; error: string } {
  if (!validateObservations(raw)) return { ok: false, error: `observations failed schema validation: ${schemaErrors(validateObservations.errors)}` };
  return { ok: true, observations: raw };
}

export function checkFisheries(raw: unknown): { ok: true; fisheries: FisheriesDataset } | { ok: false; error: string } {
  if (!validateFisheries(raw)) return { ok: false, error: `fisheries failed schema validation: ${schemaErrors(validateFisheries.errors)}` };
  return { ok: true, fisheries: raw };
}

async function optional<T>(
  base: string,
  rel: string | null | undefined,
  check: (raw: unknown) => { ok: true } & Record<string, unknown> | { ok: false; error: string },
  key: string,
): Promise<[T | null, string | null]> {
  if (!rel) return [null, "not published"];
  try {
    const r = check(await readArtifact(base, rel));
    return r.ok ? [(r as Record<string, unknown>)[key] as T, null] : [null, (r as { error: string }).error];
  } catch (e) {
    return [null, (e as Error).message];
  }
}

export async function loadData(): Promise<DataLoad> {
  const baseUrl = dataBaseUrl();
  let raw: unknown;
  try {
    raw = await readArtifact(baseUrl, "manifest.json");
  } catch (e) {
    return { ok: false, error: `manifest not available (${(e as Error).message})`, baseUrl };
  }
  const m0 = checkManifest(raw);
  if (!m0.ok) return { ok: false, error: m0.error, baseUrl };
  // the published manifest, with C-HARM's colours swapped for the display classes (lib/palette)
  const m = { ok: true as const, manifest: displayManifest(m0.manifest) };
  let ports: PortsCollection | null = null;
  let portsError: string | null = null;
  if (m.manifest.ports_url) {
    try {
      const p = checkPorts(await readArtifact(baseUrl, m.manifest.ports_url));
      if (p.ok) ports = p.ports;
      else portsError = p.error;
    } catch (e) {
      portsError = (e as Error).message;
    }
  }
  const [official, officialError] = await optional<OfficialDataset>(baseUrl, m.manifest.official_url, checkOfficial, "official");
  const [portIntel, portIntelError] = await optional<PortIntelCollection>(baseUrl, m.manifest.port_intel_url, checkPortIntel, "portIntel");
  // manifests written before M3 have no observations; the inspector then says so
  const [observations, stationsError] = await optional<ObservationDataset>(baseUrl, m.manifest.observations_url, checkObservations, "observations");
  const stations = observations ? slimStations(observations) : null;
  return { ok: true, manifest: m.manifest, ports, portsError, official, officialError, portIntel, portIntelError, stations, stationsError, baseUrl };
}

/** Manifest plus selected artifacts, for pages that do not need everything. */
export type PageLoad<T> =
  | { ok: true; manifest: Manifest; baseUrl: string } & T
  | { ok: false; error: string; baseUrl: string };

async function loadManifest(): Promise<{ ok: true; manifest: Manifest; baseUrl: string } | { ok: false; error: string; baseUrl: string }> {
  const baseUrl = dataBaseUrl();
  let raw: unknown;
  try {
    raw = await readArtifact(baseUrl, "manifest.json");
  } catch (e) {
    return { ok: false, error: `manifest not available (${(e as Error).message})`, baseUrl };
  }
  const m = checkManifest(raw);
  return m.ok ? { ok: true, manifest: m.manifest, baseUrl } : { ok: false, error: m.error, baseUrl };
}

export type BloomData = {
  observations: ObservationDataset | null;
  observationsError: string | null;
  official: OfficialDataset | null;
  officialError: string | null;
  portIntel: PortIntelCollection | null;
};

export async function loadBloomData(): Promise<PageLoad<BloomData>> {
  const m = await loadManifest();
  if (!m.ok) return m;
  // Manifests written before M3 have no observations artifact: an explicit unavailable state.
  const [observations, observationsError] = await optional<ObservationDataset>(m.baseUrl, m.manifest.observations_url, checkObservations, "observations");
  const [official, officialError] = await optional<OfficialDataset>(m.baseUrl, m.manifest.official_url, checkOfficial, "official");
  const [portIntel] = await optional<PortIntelCollection>(m.baseUrl, m.manifest.port_intel_url, checkPortIntel, "portIntel");
  return { ...m, observations, observationsError, official, officialError, portIntel };
}

export type FisheriesData = {
  fisheries: FisheriesDataset | null;
  fisheriesError: string | null;
  official: OfficialDataset | null;
};

export async function loadFisheriesData(): Promise<PageLoad<FisheriesData>> {
  const m = await loadManifest();
  if (!m.ok) return m;
  const [fisheries, fisheriesError] = await optional<FisheriesDataset>(m.baseUrl, m.manifest.fisheries_url, checkFisheries, "fisheries");
  const [official] = await optional<OfficialDataset>(m.baseUrl, m.manifest.official_url, checkOfficial, "official");
  return { ...m, fisheries, fisheriesError, official };
}
