import type { ForecastRun, LayerArtifact, Manifest, SourceStatus } from "@/generated/schema";

export const CHARM_VARIABLES = ["pseudo_nitzschia", "particulate_domoic", "cellular_domoic"] as const;
export type CharmVariable = (typeof CHARM_VARIABLES)[number];
export const CHARM_LEADS = [0, 1, 2, 3] as const;

export function charmRun(m: Manifest): ForecastRun | null {
  return m.forecast_runs.find((r) => r.group_id === "charm") ?? null;
}

export function charmLayers(m: Manifest): LayerArtifact[] {
  return m.layers.filter((l) => l.group_id === "charm");
}

export function charmLayer(m: Manifest, variable: string, lead: number): LayerArtifact | null {
  return charmLayers(m).find((l) => l.variable === variable && l.time.lead_days === lead) ?? null;
}

export function chlorophyllLayers(m: Manifest): LayerArtifact[] {
  return m.layers.filter((l) => l.group_id === "satellite_chlorophyll");
}

export function sourceStatus(m: Manifest, id: string): SourceStatus | null {
  return m.sources.find((s) => s.source_id === id) ?? null;
}

export function leadLabel(lead: number): string {
  return lead === 0 ? "Nowcast" : `+${lead} day${lead > 1 ? "s" : ""}`;
}

export function artifactUrl(base: string, rel: string): string {
  return `${base.replace(/\/$/, "")}/${rel}`;
}

export function isFixture(m: Manifest): boolean {
  return m.pipeline_version === "fixture" || m.pipeline_run_id === "fixture";
}

// ---------------------------------------------------------------- satellite chlorophyll (P1)
export const SAT_GROUP = "satellite_chlorophyll_hr";
export const SAT_PRODUCTS = ["olci300", "viirs750"] as const;
export type SatProduct = (typeof SAT_PRODUCTS)[number];

/** "Latest clear view" composite for a product (each pixel its newest observation, dated). */
export function satelliteLatest(m: Manifest, p: SatProduct): LayerArtifact | null {
  return m.layers.find((l) => l.group_id === SAT_GROUP && l.layer_id === `${p}_chl_latest`) ?? null;
}

/** Single-day layers for a product, oldest first. Days without a clear pixel are included (no grid). */
export function satelliteDays(m: Manifest, p: SatProduct): LayerArtifact[] {
  return m.layers
    .filter((l) => l.group_id === SAT_GROUP && l.layer_id.startsWith(`${p}_chl_2`))
    .sort((a, b) => (a.time.observed_date ?? "").localeCompare(b.time.observed_date ?? ""));
}

export function regionCoverage(l: LayerArtifact | null | undefined, regionId: string) {
  return l?.coverage?.regions.find((r) => r.region_id === regionId) ?? null;
}

/** Human label for a layer's native cell size, e.g. "300 m" or "3 km". */
export function nativeLabel(l: LayerArtifact | null | undefined): string | null {
  const m = l?.native_resolution_m ?? (l?.resolution_deg ? l.resolution_deg * 111_000 : null);
  if (!m) return null;
  return m >= 1000 ? `${Math.round(m / 1000)} km` : `${Math.round(m / 50) * 50} m`;
}
