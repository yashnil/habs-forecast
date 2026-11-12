import { readFile } from "node:fs/promises";
import path from "node:path";
import { loadData } from "@/lib/data";
import { parseDerived, type DerivedIndex } from "@/lib/derived";
import { buildExploreModel, type ExploreModel } from "./model";

/** The derived index (outlook, temperature) beside the manifest; null when it is missing or invalid. */
async function derivedIndex(base: string): Promise<DerivedIndex | null> {
  try {
    if (/^https?:\/\//.test(base)) {
      const r = await fetch(`${base}/derived/index.json`, { next: { revalidate: 300 } });
      return r.ok ? parseDerived(await r.json()) : null;
    }
    const dir = process.env.CW_DATA_DIR || path.join(process.cwd(), "public", base);
    return parseDerived(JSON.parse(await readFile(path.join(dir, "derived/index.json"), "utf8")));
  } catch {
    return null;
  }
}

/** Explore's model from the published data, or why it could not be read. */
export async function loadExploreModel(): Promise<{ ok: true; model: ExploreModel } | { ok: false; error: string }> {
  const d = await loadData();
  if (!d.ok) return { ok: false, error: d.error };
  const now = process.env.CW_FIXED_NOW ? new Date(process.env.CW_FIXED_NOW) : new Date();
  return { ok: true, model: buildExploreModel({ manifest: d.manifest, derived: await derivedIndex(d.baseUrl), stations: d.stations, baseUrl: d.baseUrl, now }) };
}
