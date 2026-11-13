import { loadTrust } from "@/lib/data";

export type NoticeBands = Record<string, { south: number; north: number }>;

/**
 * The official latitude band of each notice whose area is one (the registry's own `lat_band`), for the notices Today
 * lists: a Fisheries row can then hatch an advisory's area on the chart (rock crab's "do not eat internal organs").
 * Only what the registry states; a county or a named area has no band here and is never drawn as one.
 */
export async function noticeBands(ids: string[]): Promise<NoticeBands> {
  if (!ids.length) return {};
  const t = await loadTrust().catch(() => null);
  const out: NoticeBands = {};
  for (const r of t?.official?.registry.records ?? []) {
    const a = r.area as { type?: string; lat_south?: number | null; lat_north?: number | null };
    if (ids.includes(r.id) && a.type === "lat_band" && a.lat_south != null && a.lat_north != null) out[r.id] = { south: a.lat_south, north: a.lat_north };
  }
  return out;
}
