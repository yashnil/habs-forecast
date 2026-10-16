import type { OfficialDataset } from "@/generated/official";
import { distanceKm } from "@/lib/grid";
import type { WaterGeometry } from "@/lib/waterFallback";

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

// ---------------------------------------------------------------- limits over water only

/** Sample spacing along a limit line, in km (about 150 m). */
export const LIMIT_SAMPLE_KM = 0.15;

const mercX = (lon: number) => (lon + 180) / 360;
const mercY = (lat: number) => 0.5 - Math.log(Math.tan(Math.PI / 4 + (lat * Math.PI) / 360)) / (2 * Math.PI);

/** Even-odd crossings of a ring (flat x,y pairs in normalised Mercator) by a horizontal ray from (x, y). */
function crossings(ring: Float64Array, x: number, y: number): number {
  let n = 0;
  const len = ring.length;
  for (let i = 0; i < len; i += 2) {
    const j = (i + 2) % len;
    const [ax, ay, bx, by] = [ring[i], ring[i + 1], ring[j], ring[j + 1]];
    if (ay > y === by > y) continue;
    const xi = ax + ((y - ay) * (bx - ax)) / (by - ay);
    if (x < xi) n++;
  }
  return n;
}

/**
 * Is a point over the bundled water? Even-odd against every polygon's rings (outer ring and holes,
 * so an island inside the sea reads as land). Outside the geometry's domain the answer is water:
 * there is no coastline there to clip against, as in `rasterWater`.
 */
export function overWater(g: WaterGeometry, lon: number, lat: number): boolean {
  const x = mercX(lon);
  const y = mercY(lat);
  const [dx0, dy0, dx1, dy1] = g.domain;
  if (x < dx0 || x > dx1 || y < dy0 || y > dy1) return true;
  let n = 0;
  for (let r = 0; r < g.rings.length; r++) {
    const b = r * 4;
    if (x < g.bbox[b] || x > g.bbox[b + 2] || y < g.bbox[b + 1] || y > g.bbox[b + 3]) continue;
    n += crossings(g.rings[r], x, y);
  }
  return n % 2 === 1;
}

/**
 * The parts of a line that lie over water, sampled every `stepKm` along it: consecutive water
 * samples form one line; a lone water sample between land samples is dropped (narrower than the
 * sampling). Line ends over water are kept exactly.
 */
export function clipLineToWater(coords: number[][], g: WaterGeometry, stepKm = LIMIT_SAMPLE_KM): number[][][] {
  const samples: number[][] = [];
  for (let i = 0; i + 1 < coords.length; i++) {
    const [ax, ay] = coords[i];
    const [bx, by] = coords[i + 1];
    const n = Math.max(1, Math.ceil(distanceKm(ay, ax, by, bx) / stepKm));
    for (let k = i === 0 ? 0 : 1; k <= n; k++) samples.push([ax + ((bx - ax) * k) / n, ay + ((by - ay) * k) / n]);
  }
  const out: number[][][] = [];
  let run: number[][] = [];
  const flush = () => {
    if (run.length >= 2) out.push(run.map(([x, y]) => [+x.toFixed(5), +y.toFixed(5)]));
    run = [];
  };
  for (const s of samples) {
    if (overWater(g, s[0], s[1])) run.push(s);
    else flush();
  }
  flush();
  return out;
}

/**
 * Official latitude limits clipped to the sea: each `lat_limit` LineString becomes the
 * MultiLineString of its parts over the bundled water (lib/waterFallback), so a limit drawn to the
 * shore no longer runs across a headland or a town. Every other feature, every property (the
 * label, the pipeline's label, the record ids) and the feature count are unchanged; a limit with
 * no part over water keeps an empty geometry rather than an invented one.
 */
export function clipLimitsToWater(fc: GeoJSON.FeatureCollection, water: WaterGeometry): GeoJSON.FeatureCollection {
  return {
    ...fc,
    features: fc.features.map((f) => {
      const p = (f.properties ?? {}) as { kind?: string };
      if (p.kind !== "lat_limit" || f.geometry.type !== "LineString") return f;
      const parts = clipLineToWater(f.geometry.coordinates as number[][], water);
      return { ...f, geometry: { type: "MultiLineString", coordinates: parts } };
    }),
  };
}
