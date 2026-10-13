import { useSyncExternalStore } from "react";
import { fetchJSON, fetchResource, NET, type ResourceKey, SourceError } from "./net";
import type { GridSpec, Layer, Manifest, Observations, Official, Port } from "./types";

/**
 * Loading model. Each resource is one of:
 *   loading      — requested, nothing to show yet (the layout is already in place)
 *   ready        — the published data
 *   cached       — the network failed; showing the last copy this device kept, with its own dates
 *   failed       — nothing to show; the reason is known and said
 * "Stale" and "never observed" are properties of the *data* (its dates), not of loading, and are
 * computed from timestamps (lib/freshness), never from how long a request took.
 */
export type Status = "loading" | "ready" | "cached" | "failed";
export interface Res<T> { status: Status; data?: T; error?: string; startedAt: number; settledAt?: number }

const BASE = "/data/v1";
type Listener = () => void;

class Store {
  private res = new Map<string, Res<unknown>>();
  private listeners = new Set<Listener>();
  private version = 0;
  subscribe = (l: Listener) => (this.listeners.add(l), () => this.listeners.delete(l));
  getVersion = () => this.version;
  private emit() { this.version++; this.listeners.forEach((l) => l()); }
  get<T>(id: string) { return this.res.get(id) as Res<T> | undefined; }
  load<T>(id: string, key: ResourceKey, run: () => Promise<T>): Res<T> {
    const have = this.res.get(id) as Res<T> | undefined;
    if (have) return have;
    const r: Res<T> = { status: "loading", startedAt: performance.now() };
    this.res.set(id, r);
    run().then(
      (data) => { this.res.set(id, { ...r, status: "ready", data, settledAt: performance.now() }); this.emit(); },
      async (e: unknown) => {
        const msg = e instanceof SourceError ? e.message : String(e);
        if (NET.cache && e instanceof SourceError) {
          // the prototype's "device cache" is the same snapshot, flagged as cached
          try {
            const data = await runUncached<T>(id);
            this.res.set(id, { ...r, status: "cached", data, error: msg, settledAt: performance.now() });
            this.emit();
            return;
          } catch { /* fall through */ }
        }
        this.res.set(id, { ...r, status: "failed", error: msg, settledAt: performance.now() });
        this.emit();
        void key;
      },
    );
    return r;
  }
  reset() { this.res.clear(); this.emit(); }
}
export const store = new Store();

const uncached = new Map<string, () => Promise<unknown>>();
async function runUncached<T>(id: string): Promise<T> {
  const f = uncached.get(id);
  if (!f) throw new Error("no cache");
  return f() as Promise<T>;
}

function useRes<T>(id: string, key: ResourceKey, run: () => Promise<T>, plain: () => Promise<T>): Res<T> {
  useSyncExternalStore(store.subscribe, store.getVersion);
  uncached.set(id, plain);
  return store.load(id, key, run);
}

/* ---------------------------------------------------------------- resources */
export const useManifest = () =>
  useRes<Manifest>("manifest", "manifest", () => fetchJSON("manifest", `${BASE}/manifest.json`), () => fetch(`${BASE}/manifest.json`).then((r) => r.json()));

export function useOfficial(m?: Manifest) {
  const url = m ? `${BASE}/${m.official_url}` : "";
  return useRes<Official>("official", "official", () => (m ? fetchJSON<Official>("official", url).then(withFixtureConflicts) : never()), () => fetch(url).then((r) => r.json()).then(withFixtureConflicts));
}
export function useObservations(m?: Manifest) {
  const url = m ? `${BASE}/${m.observations_url}` : "";
  return useRes<Observations>("observations", "observations", () => (m ? fetchJSON("observations", url) : never()), () => fetch(url).then((r) => r.json()));
}
export function usePorts(m?: Manifest) {
  const url = m ? `${BASE}/${m.ports_url}` : "";
  const toPorts = (g: GeoJSON.FeatureCollection<GeoJSON.Point>) =>
    g.features.map((f) => ({ ...(f.properties as Omit<Port, "lon" | "lat">), lon: f.geometry.coordinates[0], lat: f.geometry.coordinates[1] }) as Port);
  return useRes<Port[]>("ports", "ports", () => (m ? fetchJSON<GeoJSON.FeatureCollection<GeoJSON.Point>>("ports", url).then(toPorts) : never()), () => fetch(url).then((r) => r.json()).then(toPorts));
}
export function useGrid(layer?: Layer) {
  const g = layer?.grid;
  const id = g ? `grid:${g.url}` : "grid:none";
  const key: ResourceKey = layer?.group_id === "charm" ? "charm" : "satellite";
  return useRes<Grid>(id, key, () => (g && !g.chunks ? loadGrid(key, g) : never()), () => (g ? loadGrid(null, g) : never()));
}
const never = <T,>() => new Promise<T>(() => {});

/* ---------------------------------------------------------------- grids */
export interface Grid { spec: GridSpec; values: Float32Array } // NaN = no value
export async function loadGrid(key: ResourceKey | null, spec: GridSpec): Promise<Grid> {
  const url = `${BASE}/${spec.url}`;
  const res = key ? await fetchResource(key, url) : await fetch(url);
  const stream = res.body!.pipeThrough(new DecompressionStream("gzip"));
  const buf = await new Response(stream).arrayBuffer();
  const u16 = new Uint16Array(buf);
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const v = u16[i];
    if (v === spec.nodata) { out[i] = NaN; continue; }
    const x = v * spec.scale_factor + spec.add_offset;
    out[i] = spec.transform === "log10" ? 10 ** x : x;
  }
  return { spec, values: out };
}
/** Exact cell containing a point (cells are centred on lat_first + i*lat_step), or null outside. */
export function cellAt(grid: Grid, lon: number, lat: number) {
  const { spec } = grid;
  const col = Math.round((lon - spec.lon_first) / spec.lon_step);
  const row = Math.round((lat - spec.lat_first) / spec.lat_step);
  if (col < 0 || row < 0 || col >= spec.width || row >= spec.height) return null;
  const value = grid.values[row * spec.width + col];
  const clat = spec.lat_first + row * spec.lat_step;
  const clon = spec.lon_first + col * spec.lon_step;
  const hy = Math.abs(spec.lat_step) / 2, hx = Math.abs(spec.lon_step) / 2;
  return { row, col, value, center: [clon, clat] as [number, number], bounds: [clon - hx, clat - hy, clon + hx, clat + hy] as [number, number, number, number] };
}
/** Nearest cell that has a value, searched in growing rings (max ring r cells). */
export function nearestValued(grid: Grid, lon: number, lat: number, r = 12) {
  const c0 = cellAt(grid, lon, lat);
  if (!c0) return null;
  if (!Number.isNaN(c0.value)) return c0;
  const { spec } = grid;
  let best: ReturnType<typeof cellAt> = null; let bestD = Infinity;
  for (let k = 1; k <= r && !best; k++) {
    for (let dr = -k; dr <= k; dr++) for (let dc = -k; dc <= k; dc++) {
      if (Math.max(Math.abs(dr), Math.abs(dc)) !== k) continue;
      const row = c0.row + dr, col = c0.col + dc;
      if (row < 0 || col < 0 || row >= spec.height || col >= spec.width) continue;
      const v = grid.values[row * spec.width + col];
      if (Number.isNaN(v)) continue;
      const lo = spec.lon_first + col * spec.lon_step, la = spec.lat_first + row * spec.lat_step;
      const d = (lo - lon) ** 2 * Math.cos((lat * Math.PI) / 180) ** 2 + (la - lat) ** 2;
      if (d < bestD) { bestD = d; best = cellAt(grid, lo, la); }
    }
  }
  return best;
}

/* ---------------------------------------------------------------- notice discrepancy (fixture) */
/**
 * FIXTURE, labelled in the UI: PR #40 publishes agency releases that appear to lift an active record
 * among `conflicts`. The pinned production snapshot predates that pipeline change, so the
 * prototypes add the one discrepancy PR #40's live check found (CDPH SN26-021 vs SN26-018), in
 * PR #40's exact format. No record is edited; the record is shown as disputed, never as lifted.
 */
export const FIXTURE_CONFLICT =
  'Agency source discrepancy: CDPH release SN26-021 ("CDPH has lifted the shellfish safety notification related to sport-harvested bivalve shellfish from Monterey County", https://www.cdph.ca.gov/Programs/OPA/Pages/SN26-021.aspx) appears to lift active record cdph-2026-sn26-018-monterey-bivalves.';
function withFixtureConflicts(o: Official): Official {
  return o.conflicts.some((c) => c.includes("SN26-021")) ? o : { ...o, conflicts: [...o.conflicts, FIXTURE_CONFLICT] };
}
export function disputedBy(o: Official, id: string) {
  for (const c of o.conflicts) {
    const m = c.match(/^Agency source discrepancy: (\w+) release (\S+) \("(.*)", (https:\/\/\S+)\) appears to lift active record ([a-z0-9-]+)\./);
    if (m && m[5] === id) return { agency: m[1], release: m[2], title: m[3], url: m[4] };
  }
  return null;
}
