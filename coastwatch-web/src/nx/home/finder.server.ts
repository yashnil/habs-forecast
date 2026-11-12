import GEO from "@/content/fisheries-geo.json";
import { shellPorts } from "@/nx/shell/ports.server";
import type { PortEntry } from "@/nx/ds/PortFinder";

/**
 * The ports for Home's finder (ds PortFinder): every port the shell can open, grouped by CDFW port area
 * (the "Marine Landing Data System Ports" areas, north to south, ds3081), so a reader can read down the
 * coast. A port CDFW's list does not place in an area is grouped under "Other ports", last.
 */
export type FinderData = { ports: PortEntry[]; regions: Array<{ id: string; label: string }>; keys: Record<number, string> };

type GeoPort = { code: number; area: string; area_code: number };

export async function finderData(): Promise<FinderData> {
  const ports = await shellPorts();
  const geo = new Map((GEO.ports as GeoPort[]).map((p) => [p.code, p]));
  const areas = new Map<string, number>();
  for (const p of GEO.ports as GeoPort[]) areas.set(p.area, p.area_code);
  const entries: PortEntry[] = [];
  const keys: Record<number, string> = {};
  for (const p of ports) {
    const code = Number(p.key.replace(/^port-/, ""));
    if (!Number.isFinite(code)) continue;
    const g = geo.get(code);
    entries.push({ code, name: p.name, county: p.county ?? "Unknown", region: g?.area ?? "other", lat: p.lat });
    keys[code] = p.key;
  }
  const used = new Set(entries.map((e) => e.region));
  const regions = [...areas.entries()]
    .sort((a, b) => a[1] - b[1])
    .filter(([a]) => used.has(a))
    .map(([a]) => ({ id: a, label: `${a} area` }));
  if (used.has("other")) regions.push({ id: "other", label: "Other ports" });
  return { ports: entries, regions, keys };
}
