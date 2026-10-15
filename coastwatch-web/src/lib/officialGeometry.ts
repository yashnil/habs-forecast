import type { OfficialDataset } from "@/generated/official";

/**
 * The official geometry as the map draws it. The pipeline labels a latitude limit by its latitude
 * alone ("36°31.46′ N · south limit"), which on a map says nothing about whose limit it is; here
 * each limit line is named by the notices it bounds ("Northern anchovy notices · south limit
 * 36°31.46′ N"), from the records' own species lists. Geometry and values are unchanged.
 */
export function labelledGeometry(ds: OfficialDataset | null | undefined): GeoJSON.FeatureCollection | null {
  const fc = ds?.geometry as unknown as GeoJSON.FeatureCollection | null | undefined;
  if (!fc) return null;
  const byId = new Map((ds?.registry?.records ?? []).map((r) => [r.id, r]));
  return {
    ...fc,
    features: fc.features.map((f) => {
      const p = (f.properties ?? {}) as { kind?: string; label?: string; record_ids?: string[] };
      if (p.kind !== "lat_limit" || !p.label) return f;
      const species = [...new Set((p.record_ids ?? []).flatMap((id) => byId.get(id)?.species ?? []))];
      const who = species.length === 1 ? `${cap(species[0])} notices` : species.length > 1 ? `${cap(species[0])} and ${species.length - 1} other notice${species.length > 2 ? "s" : ""}` : (p.record_ids?.length ?? 0) > 0 ? "Agency notice" : null;
      const m = p.label.match(/^(.*?)\s*·\s*(north|south) limit$/i);
      const label = who && m ? `${who} · ${m[2].toLowerCase()} limit ${m[1]}` : p.label;
      return { ...f, properties: { ...p, label, pipeline_label: p.label } };
    }),
  };
}

const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);
