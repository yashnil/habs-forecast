/**
 * Explore California Waters' layer switch, DOM-free (unit tests read it). Every entry is a real published or live
 * dataset (docs/coastwatch/v15.1/science/datasets.md): what it is (an observation or a model), its units, its own
 * dates and freshness, where it comes from and how fine it is. Nothing here is a fishing, safety or legal
 * statement; chlorophyll is algae, not toxin; C-HARM is a model probability, not a measured toxin.
 */
import { day, range, sig, time } from "@/nx/ds/format";
import type { FreshnessState } from "@/lib/freshness";
import type { ExploreModel, ProductId } from "./model";
import { isOld, oldMark } from "./words";
import { KNOTS } from "./read";

export type ProductClass = "observation" | "model";
export type SwitchId = "chl" | "charm" | "sst" | "outlook" | "currents" | "stations";

export type LayerMeta = {
  id: SwitchId;
  label: string;
  productClass: ProductClass;
  /** the units a reading is given in */
  units: string;
  /** the layer's own dates, in a few words ("Nov 5–10", "run of Nov 3, valid Nov 2–5") */
  when: string;
  fresh: FreshnessState | null;
  source: string;
  /** how fine the data is */
  resolution: string;
  /** the words that must travel with it */
  note: string;
  /** a surface layer (one at a time) or an overlay (switched on and off over the surface) */
  role: "surface" | "overlay";
};

const PRODUCT_LABEL: Record<ProductId, string> = { chl: "Chlorophyll", charm: "C-HARM", sst: "Sea temperature", outlook: "Outlook" };

/** The live layers this data run holds, surface layers first, then the overlays. */
export function layerMetas(model: ExploreModel): LayerMeta[] {
  const out: LayerMeta[] = [];
  for (const p of model.products) {
    const f = p.frames.find((x) => x.id === p.defaultFrame) ?? p.frames[0];
    if (!f) continue;
    if (p.id === "chl") {
      const days = p.frames.filter((x) => x.track === f.track && x.label !== "Latest").map((x) => x.end).sort();
      out.push({
        id: "chl",
        label: PRODUCT_LABEL.chl,
        productClass: "observation",
        units: "mg/m³",
        when: `${f.start && f.end && f.start !== f.end ? range(f.start, f.end) : day(f.end)}${oldMark(f.fresh)}`,
        fresh: f.fresh,
        source: p.source,
        resolution: `${f.sensor.replace(/^.*,\s*/, "")} cells${days.length ? `, ${days.length} daily views` : ""}`,
        note: "Algae, not toxin. Cloud and fog are gaps, never filled.",
        role: "surface",
      });
    } else if (p.id === "charm") {
      const valid = p.frames.map((x) => x.end).sort();
      out.push({
        id: "charm",
        label: PRODUCT_LABEL.charm,
        productClass: "model",
        units: "probability (%)",
        when: `${p.runIssued ? `run of ${day(p.runIssued)}, ` : ""}valid ${valid.length > 1 ? range(valid[0], valid[valid.length - 1]) : day(valid[0])}${p.pastRun ? " (expired)" : ""}`,
        fresh: p.pastRun ? "historical" : f.fresh,
        source: p.source,
        resolution: "3 km cells, 4 daily leads",
        note: "Model probability, not a measured toxin.",
        role: "surface",
      });
    } else if (p.id === "sst") {
      out.push({ id: "sst", label: PRODUCT_LABEL.sst, productClass: f.kind === "analysis" ? "model" : "observation", units: "°F", when: day(f.end), fresh: f.fresh, source: p.source, resolution: "1 km analysis", note: "An analysis, not a measurement at a point.", role: "surface" });
    } else {
      out.push({ id: "outlook", label: "Chlorophyll outlook (experimental)", productClass: "model", units: "mg/m³", when: `mean of ${range(f.start, f.end)}`, fresh: f.fresh, source: p.source, resolution: "300 m cells", note: "Experimental chlorophyll outlook. Algae, not toxin; it predicts no bloom.", role: "surface" });
    }
  }
  const c = model.currents;
  if (c) {
    const hours = c.frames.filter((f) => !f.mean).sort((a, b) => a.time.localeCompare(b.time));
    const last = hours[hours.length - 1];
    out.push({
      id: "currents",
      label: "Currents",
      productClass: "observation",
      units: "kn · cm/s",
      when: last ? `${hours.length} hours to ${day(last.time)}, ${time(last.time)}${oldMark(last.fresh)}` : "no hourly frames",
      fresh: last?.fresh ?? null,
      source: c.source,
      resolution: "2 km cells, hourly",
      note: "Measured surface currents. No streak is no data, not calm water.",
      role: "overlay",
    });
  }
  if (model.stations.length || model.buoys.length) {
    const newest = model.stations.flatMap((s) => s.latest.map((x) => x.date)).sort().pop() ?? null;
    out.push({
      id: "stations",
      label: "Stations",
      productClass: "observation",
      units: "ng/mL · cells/L · kn · ft · °F",
      when: `${model.stations.length} shore stations${newest ? `, newest sample ${day(newest)}` : ""}; ${model.buoys.length} buoys read live`,
      fresh: null,
      source: "CalHABMAP (SCCOOS) · NOAA NDBC",
      resolution: "one point each",
      note: "Each value is one dated sample or reading. An old sample says nothing about today.",
      role: "overlay",
    });
  }
  return out;
}

/** What Explore does not show because it is not published (yet), for the switch's one quiet line. */
export function notPublished(model: ExploreModel): string[] {
  const out: string[] = [];
  if (!model.products.some((p) => p.id === "sst")) out.push("sea temperature");
  if (!model.products.some((p) => p.id === "outlook")) out.push("the experimental chlorophyll outlook (not validated)");
  return out;
}

/** The surface layer's chip line: "Observation · Nov 5–10" / "Model · past run, Nov 3". */
export function chipWords(m: LayerMeta): string {
  return `${m.productClass === "observation" ? "Observation" : "Model"} · ${m.when}`;
}

export const isStale = (m: Pick<LayerMeta, "fresh">) => !!m.fresh && isOld(m.fresh);

// ---------------------------------------------------------------- currents: speeds from the real u/v

export type SpeedStats = { cells: number; median: number; p90: number; max: number } | null;

/** The measured speeds of one hour (m/s), from every cell with both components; null when none has. */
export function speedStats(u: ArrayLike<number>, v: ArrayLike<number>): SpeedStats {
  const s: number[] = [];
  for (let i = 0; i < u.length; i++) {
    const a = u[i], b = v[i];
    if (Number.isFinite(a) && Number.isFinite(b)) s.push(Math.hypot(a, b));
  }
  if (!s.length) return null;
  s.sort((a, b) => a - b);
  const q = (p: number) => s[Math.min(s.length - 1, Math.floor(p * (s.length - 1) + 0.5))];
  return { cells: s.length, median: q(0.5), p90: q(0.9), max: s[s.length - 1] };
}

/** A speed in m/s as "0.6 kn (31 cm/s)". */
export const speedWords = (ms: number) => `${sig(ms * KNOTS, 2)} kn (${Math.round(ms * 100)} cm/s)`;

// ---------------------------------------------------------------- resolution at the present zoom

/** Kilometres one CSS pixel spans at `zoom` and `lat` (MapLibre's 512 px tiles). */
export const kmPerPixel = (zoom: number, lat: number) => (40075.016686 * Math.cos((lat * Math.PI) / 180)) / (512 * 2 ** zoom);

/**
 * A line for the legend when the view is coarser than the data: at statewide scale a screen pixel covers many of
 * a fine grid's cells, so the picture shows one of them (the nearest) and says so. Null once the cells show.
 */
export function resolutionNote(zoom: number, lat: number, cellKm: number): string | null {
  const px = kmPerPixel(zoom, lat);
  if (px < cellKm * 1.5) return null;
  const n = Math.round((px / cellKm) ** 2);
  const cell = cellKm < 1 ? `${Math.round(cellKm * 1000)} m` : `${sig(cellKm, 2)} km`;
  return `At this scale one screen pixel spans about ${sig(px, 2)} km, ${n > 1 ? `about ${n} cells of ${cell}` : `more than one ${cell} cell`}: zoom in toward the coast for every cell.`;
}

/** A surface frame's cell size in km, from its grid's step (0.0025° → 0.28 km). */
export const cellKmOf = (g: { lat_step: number } | null | undefined) => (g ? Math.abs(g.lat_step) * 111.32 : null);

// ---------------------------------------------------------------- chlorophyll's three views (round 2, B)

/**
 * Chlorophyll is shown three ways, never mixed up: Observed (one day's pixels, gaps kept), Recent composite (each
 * pixel its own latest clear day within the week, with its age), and a Gap-filled estimate (a reconstruction, drawn
 * and worded as one). The estimate is offered only when its source is wired; until then it is "not available".
 */
export type ChlView = "observed" | "composite" | "estimate";
export const chlViewOf = (f: { id: string; start: string; end: string }): ChlView => (f.id.endsWith(":latest") ? "composite" : "observed");

/** The frame a view opens on: the newest single day for Observed, the latest view of the first sensor for Composite. */
export function frameForView(p: { frames: { id: string; track: string; start: string; end: string }[] }, v: ChlView, track?: string): string | null {
  const fs = p.frames.filter((f) => !track || f.track === track);
  if (v === "composite") return (fs.find((f) => f.id.endsWith(":latest")) ?? p.frames.find((f) => f.id.endsWith(":latest")))?.id ?? null;
  if (v === "observed") return fs.filter((f) => !f.id.endsWith(":latest") && f.start === f.end).sort((a, b) => a.start.localeCompare(b.start)).pop()?.id ?? null;
  return null;
}
