import { readFile } from "node:fs/promises";
import path from "node:path";
import Ajv2020 from "ajv/dist/2020";
import type { Manifest, PortsCollection } from "@/generated/schema";
import manifestSchema from "@/generated/schemas/manifest.schema.json";
import portsSchema from "@/generated/schemas/ports.schema.json";

/**
 * Server-side loading of pipeline artifacts. The manifest is validated against the
 * pipeline's JSON Schema before anything is rendered; an invalid or missing manifest
 * produces an explicit "data unavailable" state instead of a broken or guessed map.
 */
export type DataLoad =
  | { ok: true; manifest: Manifest; ports: PortsCollection | null; portsError: string | null; baseUrl: string }
  | { ok: false; error: string; baseUrl: string };

const ajv = new Ajv2020({ allErrors: true, strict: false });
const validateManifest = ajv.compile<Manifest>(manifestSchema);
const validatePorts = ajv.compile<PortsCollection>(portsSchema);

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
  return { ok: true, manifest: m.manifest, ports, portsError, baseUrl };
}
