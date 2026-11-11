import { readFile } from "node:fs/promises";
import path from "node:path";
import { loadData } from "@/lib/data";
import { parseDerived, type DerivedIndex } from "@/lib/derived";
import { buildExploreModel } from "@/nx/explore/model";
import { Explore, ExploreUnavailable } from "@/nx/explore/Explore";

export const metadata = { title: "Explore" };

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

/** Explore: observations and models in their place and time, on the full sheet (journeys.md Journey 3). */
export default async function ExplorePage() {
  const d = await loadData();
  if (!d.ok) return <ExploreUnavailable why={d.error} />;
  const now = process.env.CW_FIXED_NOW ? new Date(process.env.CW_FIXED_NOW) : new Date();
  const model = buildExploreModel({ manifest: d.manifest, derived: await derivedIndex(d.baseUrl), stations: d.stations, baseUrl: d.baseUrl, now });
  return <Explore model={model} />;
}
