import type { LayerArtifact, Manifest } from "@/generated/schema";
import { classifySource, classifyTime, type Freshness } from "@/lib/freshness";
import { charmLayer, charmRun, regionCoverage, satelliteLatest, sourceStatus, type SatProduct } from "@/lib/layers";

/**
 * Which layer the Ocean Map opens on when the link does not name one (M5).
 *
 * 1. C-HARM, when its latest run is current by the source's own published freshness policy
 *    (issued within `current_max_age_days`, 1 day) and the opening variable and lead exist.
 *    It is the only layer that speaks to toxin.
 * 2. Otherwise satellite chlorophyll, when a latest-clear-view layer is current by its
 *    published policy (newest observation within 3 days) and it observed at least
 *    MIN_REGION_COVERAGE of the opening region's ocean. Below half, the map would show more
 *    gap than observation. Products are tried in order: multi-sensor, Sentinel-3, VIIRS.
 * 3. Otherwise no data layer: the bathymetric map with a message saying what is missing and
 *    offering the latest (dated) forecast and satellite view.
 *
 * A layer named in the URL, or chosen by the user, always wins; this runs only for the
 * opening view. Thresholds come from the published FreshnessPolicy, not from this file,
 * except the coverage share.
 */
export const MIN_REGION_COVERAGE = 0.5;
const SAT_ORDER: SatProduct[] = ["multi", "olci300", "viirs750"];

export type Opening =
  | { group: "forecast"; reason: "forecast-current"; forecast: Freshness }
  | { group: "satellite"; product: SatProduct; reason: "forecast-stale" | "forecast-unavailable"; forecast: Freshness; coverage: number; layer: LayerArtifact }
  | { group: null; reason: "nothing-current"; forecast: Freshness; satellite: SatelliteCandidate | null };

export type SatelliteCandidate = { product: SatProduct; layer: LayerArtifact; freshness: Freshness; coverage: number | null };

/** Share of the region's reference ocean cells the layer observed (the whole domain for statewide). */
export function coverageOf(l: LayerArtifact, regionId: string | null): number | null {
  if (!l.coverage) return null;
  if (!regionId) return l.coverage.domain_observed_fraction;
  return regionCoverage(l, regionId)?.observed_fraction ?? null;
}

export function forecastFreshness(m: Manifest, now: Date): Freshness {
  const st = sourceStatus(m, "charm");
  const run = charmRun(m);
  if (!run || !st) return { state: "unavailable", basisDate: null, ageDays: null };
  return classifySource(st, now);
}

export function satelliteCandidates(m: Manifest, now: Date, regionId: string | null): SatelliteCandidate[] {
  return SAT_ORDER.flatMap((product) => {
    const layer = satelliteLatest(m, product);
    if (!layer || !layer.tiles) return [];
    return [{ product, layer, freshness: classifyTime(layer.freshness, layer.time, now), coverage: coverageOf(layer, regionId) }];
  });
}

export function openingLayer(m: Manifest, now: Date, opts: { regionId: string | null; variable: string; lead: number }): Opening {
  const forecast = forecastFreshness(m, now);
  const fl = charmLayer(m, opts.variable, opts.lead);
  if (forecast.state === "current" && fl?.image) return { group: "forecast", reason: "forecast-current", forecast };
  const cands = satelliteCandidates(m, now, opts.regionId);
  const usable = cands.find((c) => c.freshness.state === "current" && c.coverage != null && c.coverage >= MIN_REGION_COVERAGE);
  const reason = forecast.state === "unavailable" || !fl?.image ? "forecast-unavailable" : "forecast-stale";
  if (usable) return { group: "satellite", product: usable.product, reason, forecast, coverage: usable.coverage!, layer: usable.layer };
  // the best partial view, for the message: the most covered current layer, else the newest
  const best = [...cands].sort((a, b) => Number(b.freshness.state === "current") - Number(a.freshness.state === "current") || (b.coverage ?? 0) - (a.coverage ?? 0))[0] ?? null;
  return { group: null, reason: "nothing-current", forecast, satellite: best };
}
