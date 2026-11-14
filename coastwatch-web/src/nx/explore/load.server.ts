import { readFile } from "node:fs/promises";
import path from "node:path";
import { loadData } from "@/lib/data";
import { parseDerived, type DerivedIndex } from "@/lib/derived";
import { BUOYS } from "@/lib/noaaStations";
import { buildExploreModel, type ExploreBuoy, type ExploreModel } from "./model";

/**
 * Are the derived products (sea temperature, the 8-day chlorophyll outlook) read? They are published only to the
 * staging data channel, not to production (docs/coastwatch/v15.1/science/datasets.md), so on deployed data they
 * stay off until the owner publishes them and sets CW_DERIVED_LAYERS=1. A local fixture (a path, not a URL) keeps
 * them, so the tests of those layers still run. CW_DERIVED_LAYERS=0 turns them off everywhere.
 */
export function derivedEnabled(base: string, flag = process.env.CW_DERIVED_LAYERS): boolean {
  if (flag === "1") return true;
  if (flag === "0") return false;
  return !/^https?:\/\//.test(base);
}

/** The bundled NDBC list as Explore's buoy points (positions only). */
const buoys = (): ExploreBuoy[] => BUOYS.map((b) => ({ id: b.id, name: b.name, lat: b.lat, lon: b.lon, owner: (b as { owner?: string }).owner ?? null, type: (b as { type?: string }).type ?? null }));

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
  const on = derivedEnabled(d.baseUrl);
  return { ok: true, model: buildExploreModel({ manifest: d.manifest, derived: on ? await derivedIndex(d.baseUrl) : null, stations: d.stations, baseUrl: d.baseUrl, now, buoys: buoys(), derivedOn: on }) };
}
