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
