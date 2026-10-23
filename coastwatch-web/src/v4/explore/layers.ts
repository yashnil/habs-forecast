import type { LayerArtifact, Manifest } from "@/generated/schema";
import { CHARM_LEADS, CHARM_VARIABLES, artifactUrl, charmLayer, charmRun, satelliteDays, satelliteLatest, tileIndexUrl, type CharmVariable, type SatProduct } from "@/lib/layers";
import { currentsHourly, currentsMean, defaultHour } from "@/lib/currents";
import { supportsGridTiles } from "@/lib/gridTiles";
import { charmDays, type CharmDay } from "@/lib/charmDays";
import type { DerivedRaster, ImageRaster, TileRaster } from "@/components/map/MapCanvas";
import { dayLabel } from "@/v4/ui/time";
import { CHL_PALETTE, SST_PALETTE, derivedEstimate, derivedOutlook, derivedSst, loadDerived, type DerivedIndex, type EstimateLayer, type OutlookLayer, type SstLayer } from "@/lib/derived";
import { useEffect, useState } from "react";

/**
 * What the map shows: one primary field at a time (or none: the seafloor, the coast, harbours,
 * stations and official limits). Encoded in the URL with the v2 parameter names, so v2 links open
 * the same view (layer=, var=, lead=).
 */
export type LayerChoice =
  | { kind: "none" }
  | { kind: "forecast"; variable: CharmVariable; lead: number }
  /**
   * view: the chlorophyll colours, or the age of each pixel / which sensor (latest views only);
   * estimate: the gap-filling estimate under the observed pixels (latest chlorophyll view only; on
   * unless turned off, `est=0`), when the derived index offers one
   */
  | { kind: "satellite"; product: Exclude<SatProduct, "multi"> | "multi"; day: string | null; view?: "chl" | "age" | "sensor"; estimate?: boolean }
  /** the 8-day chlorophyll outlook of the derived index (lib/derived) */
  | { kind: "outlook" }
  /** observed (or analysed) sea-surface temperature of the derived index (lib/derived) */
  | { kind: "sst" }
  /** chl: the newest clear chlorophyll view underneath, to compare the two (v2 `chl=1`) */
  | { kind: "currents"; hour: string | null; mean: boolean; chl?: boolean };

export const VARIABLE_WORDS: Record<CharmVariable, { name: string; short: string; what: string }> = {
  particulate_domoic: { name: "Domoic acid in seawater", short: "Domoic acid (pDA)", what: "Probability that particulate domoic acid exceeds 500 ng per litre" },
  pseudo_nitzschia: { name: "Pseudo-nitzschia bloom", short: "Pseudo-nitzschia", what: "Probability that Pseudo-nitzschia exceeds 10,000 cells per litre" },
  cellular_domoic: { name: "Domoic acid per cell", short: "Cellular DA", what: "Probability that cellular domoic acid exceeds 10 pg per cell" },
};

export const SAT_WORDS: Record<string, { name: string; native: string }> = {
  olci300: { name: "Sentinel-3 OLCI", native: "300 m" },
  viirs750: { name: "VIIRS (NOAA-21, NOAA-20)", native: "750 m" },
  multi: { name: "Multi-sensor view", native: "300 m / 750 m" },
};

export function parseChoice(q: URLSearchParams): LayerChoice | null {
  const v = q.get("layer");
  const variable = (CHARM_VARIABLES as readonly string[]).includes(q.get("var") ?? "") ? (q.get("var") as CharmVariable) : "particulate_domoic";
  const leadQ = Number(q.get("lead"));
  const lead = q.has("lead") && (CHARM_LEADS as readonly number[]).includes(leadQ) ? leadQ : -1;
  if (v == null) return null;
  if (v === "none") return { kind: "none" };
  if (v === "outlook") return { kind: "outlook" };
  if (v === "sst") return { kind: "sst" };
  if (v === "forecast") return { kind: "forecast", variable, lead };
  if (v.startsWith("currents")) {
    const rest = v.split(":")[1] ?? null;
    const chl = q.get("chl") === "1";
    return rest === "mean" ? { kind: "currents", hour: null, mean: true, chl } : { kind: "currents", hour: rest && /^\d{8}T\d{2}Z$/.test(rest) ? rest : null, mean: false, chl };
  }
  const [prod, day] = v.split(":");
  const view = q.get("age") === "1" ? "age" : q.get("sensor") === "1" ? "sensor" : "chl";
  if (prod === "olci300" || prod === "viirs750" || prod === "multi") return { kind: "satellite", product: prod, day: prod === "multi" ? null : (day ?? null), view, ...(q.get("est") === "0" ? { estimate: false } : {}) };
  return null;
}

export function choiceParams(c: LayerChoice): Record<string, string> {
  switch (c.kind) {
    case "none":
      return { layer: "none" };
    case "outlook":
      return { layer: "outlook" };
    case "sst":
      return { layer: "sst" };
    case "forecast":
      return { layer: "forecast", var: c.variable, lead: String(c.lead) };
    case "satellite":
      return { layer: c.day ? `${c.product}:${c.day}` : c.product, ...(c.view === "age" ? { age: "1" } : c.view === "sensor" ? { sensor: "1" } : {}), ...(c.estimate === false ? { est: "0" } : {}) };
    case "currents":
      return { layer: c.mean ? "currents:mean" : c.hour ? `currents:${c.hour}` : "currents", ...(c.chl ? { chl: "1" } : {}) };
  }
}

/** The four days of the current C-HARM run, labelled by date (lib/charmDays). */
export function runDays(m: Manifest, now: Date | null): CharmDay[] {
  return charmDays(
    CHARM_LEADS.map((l) => charmLayer(m, "particulate_domoic", l) ?? charmLayer(m, "pseudo_nitzschia", l)),
    now,
  );
}

export type Resolved = {
  forecastLayer: LayerArtifact | null;
  forecast: ImageRaster | null;
  satLayer: LayerArtifact | null;
  satellite: TileRaster | null;
  /** categorical age or sensor tiles that replace the chlorophyll colours (latest views) */
  satelliteAge: TileRaster | null;
  curLayer: LayerArtifact | null;
  /** derived products (lib/derived): the outlook as the field, or the estimate under the observed pixels */
  outlookLayer: OutlookLayer | null;
  outlook: DerivedRaster | null;
  estimateLayer: EstimateLayer | null;
  estimate: DerivedRaster | null;
  sstLayer: SstLayer | null;
  sst: DerivedRaster | null;
};

/** The derived index for this dataset: undefined while loading, null when missing or invalid (then nothing derived is offered). */
export function useDerivedIndex(baseUrl: string): DerivedIndex | null | undefined {
  // undefined while loading; null when there is none
  const [d, setD] = useState<DerivedIndex | null | undefined>(undefined);
  useEffect(() => {
    let live = true;
    loadDerived(baseUrl).then((x) => live && setD(x));
    return () => {
      live = false;
    };
  }, [baseUrl]);
  return d;
}

/** Does the estimate belong under this satellite view: the latest clear chlorophyll view (of its own product, when it names one)? */
export function estimateFits(est: EstimateLayer | null, c: LayerChoice, sat: LayerArtifact | null): boolean {
  if (!est || c.kind !== "satellite" || c.day || (c.view ?? "chl") !== "chl" || !sat) return false;
  const own = est.observed_layer_id ?? est.observed_layer;
  return !own || own === sat.layer_id;
}

/** `region`: the region in view or selected, for the currents hour the map opens on (lib/currents chooseHour). */
export function resolve(m: Manifest, c: LayerChoice, baseUrl: string, derived: DerivedIndex | null = null, opts: { region?: string | null } = {}): Resolved {
  const out: Resolved = { forecastLayer: null, forecast: null, satLayer: null, satellite: null, satelliteAge: null, curLayer: null, outlookLayer: null, outlook: null, estimateLayer: null, estimate: null, sstLayer: null, sst: null };
  if (c.kind === "sst") {
    const t = derivedSst(derived);
    out.sstLayer = t;
    if (t) out.sst = { id: `${t.id}@${t.observed_dates[t.observed_dates.length - 1]}`, grid: t.grid, baseUrl, palette: SST_PALETTE, style: "observed" };
  }
  const palette = satelliteLatest(m, "olci300")?.palette ?? satelliteLatest(m, "viirs750")?.palette ?? CHL_PALETTE;
  if (c.kind === "outlook") {
    const o = derivedOutlook(derived);
    out.outlookLayer = o;
    if (o) out.outlook = { id: `${o.id}@${o.issued}`, grid: o.grid, baseUrl, palette };
  }
  if (c.kind === "forecast") {
    const l = charmLayer(m, c.variable, c.lead);
    out.forecastLayer = l;
    if (l?.image)
      out.forecast = {
        id: l.layer_id,
        url: artifactUrl(baseUrl, l.image.url),
        corners: l.image.corners_lnglat,
        grid: supportsGridTiles(l) ? { layer: l, url: artifactUrl(baseUrl, l.grid!.url) } : null,
      };
  }
  if (c.kind === "satellite") {
    const l = c.day ? (satelliteDays(m, c.product as SatProduct).find((d) => d.time.observed_date === c.day) ?? null) : satelliteLatest(m, c.product as SatProduct);
    out.satLayer = l;
    const t = l?.tiles;
    const at = c.view === "age" ? (l?.composite?.age_tiles ?? l?.multisensor?.age_tiles) : c.view === "sensor" ? l?.multisensor?.sensor_tiles : null;
    if (l && at) out.satelliteAge = { id: `${l.layer_id}-${c.view}`, template: artifactUrl(baseUrl, at.url_template), minzoom: at.min_zoom ?? 0, maxzoom: at.max_native_zoom, bounds: at.bounds_lnglat, indexUrl: tileIndexUrl(baseUrl, at) };
    else if (l && t) out.satellite = { id: l.layer_id, template: t.relative ? artifactUrl(baseUrl, t.url_template) : t.url_template, minzoom: t.min_zoom ?? 0, maxzoom: t.max_native_zoom, bounds: t.bounds_lnglat, indexUrl: tileIndexUrl(baseUrl, t) };
    const est = derivedEstimate(derived);
    if (est && estimateFits(est, c, l)) {
      out.estimateLayer = est;
      if (c.estimate !== false) out.estimate = { id: `${est.id}@${est.issued}`, grid: est.grid, baseUrl, palette, classGrid: est.class_grid ?? null };
    }
  }
  if (c.kind === "currents") {
    const hourly = currentsHourly(m);
    // no hour chosen: the newest hour with usable coverage of the region in view (an hour still filling in,
    // or one that saw nothing here, stays one tap away)
    const dflt = () => defaultHour(m, opts.region ?? null);
    out.curLayer = c.mean ? currentsMean(m) : c.hour ? (hourly.find((l) => l.layer_id.endsWith(c.hour!)) ?? dflt()) : dflt();
    // compare: the newest clear chlorophyll view underneath (each pixel keeps its own date)
    if (c.chl) {
      const l = satelliteLatest(m, "olci300") ?? satelliteLatest(m, "viirs750");
      const t = l?.tiles;
      out.satLayer = l;
      if (l && t) out.satellite = { id: l.layer_id, template: t.relative ? artifactUrl(baseUrl, t.url_template) : t.url_template, minzoom: t.min_zoom ?? 0, maxzoom: t.max_native_zoom, bounds: t.bounds_lnglat, indexUrl: tileIndexUrl(baseUrl, t) };
    }
  }
  return out;
}

export function hasForecast(m: Manifest) {
  return !!charmRun(m);
}

/** The dates a coverage figure belongs to. A composite or multi-sensor view counts pixels from
 *  several days, so it names the span; only a single-day view is "the overpass of" one date. */
export function coverageWhen(layer: Pick<LayerArtifact, "composite" | "multisensor" | "time">): string {
  const comp = layer.composite;
  if (comp) return span(comp.oldest_observed_date, comp.newest_observed_date);
  const days = [...new Set((layer.time.observed_times ?? []).map((t) => t.slice(0, 10)))].sort();
  if (layer.multisensor || days.length > 1) {
    const first = days[0], last = days[days.length - 1] ?? layer.time.observed_date;
    return first && last ? span(first, last) : last ? `pixels observed up to ${dayLabel(last)}` : "";
  }
  return layer.time.observed_date ? `overpass of ${dayLabel(layer.time.observed_date)}` : "";
}
const span = (a: string, b: string) => (a === b ? `pixels observed ${dayLabel(a)}` : `pixels observed ${dayLabel(a)}–${dayLabel(b)}, each its latest clear day`);
