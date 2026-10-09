import { readFile } from "node:fs/promises";
import path from "node:path";
import Ajv2020 from "ajv/dist/2020";
import type { Manifest, PortsCollection } from "@/generated/schema";
import type { OfficialDataset } from "@/generated/official";
import type { PortIntelCollection } from "@/generated/port_intel";
import type { ObservationDataset } from "@/generated/observations";
import type { FisheriesDataset } from "@/generated/fisheries";
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
      baseUrl: string;
    }
  | { ok: false; error: string; baseUrl: string };

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
  const m = checkManifest(raw);
  if (!m.ok) return { ok: false, error: m.error, baseUrl };
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
  return { ok: true, manifest: m.manifest, ports, portsError, official, officialError, portIntel, portIntelError, baseUrl };
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
