import type { LayerArtifact, Manifest, SourceStatus } from "@/generated/schema";
import { classifyTime, type Freshness } from "@/lib/freshness";
import { currentsHourly } from "@/lib/currents";
import { charmRun, satelliteLatest, sourceStatus } from "@/lib/layers";
import { forecastFreshness } from "@/lib/opening";
import type { SourceEntry, SourceState } from "./store";

/**
 * The freshness of the published sources a page shows, in the status centre's terms: one rule for every
 * route, so the top bar's chip never says "Ready" over a model run or a satellite view that is days old.
 * A source that loaded is "ready" only while it is current; stale and historical data, and a failed
 * update, are said (the same policy as Explore's layer panel).
 */
export function fieldState(fresh: Freshness | null, status: SourceStatus | null): { state: SourceState; note: string | null } {
  if (fresh?.state === "historical") return { state: "stale", note: `Out of date (${fresh.ageDays} days): does not describe current conditions` };
  if (fresh?.state === "stale") return { state: "stale", note: `Older than usual (${fresh.ageDays} days)` };
  if (status && status.outcome !== "updated") return { state: "partial", note: "The last update did not complete; the newest published data is shown" };
  return { state: "ready", note: null };
}

/** An observed layer's freshness by its own policy; a latest-clear view is as fresh as its newest pixel. */
export function layerFreshness(l: LayerArtifact | null, now: Date | null): Freshness | null {
  if (!now || !l) return null;
  return classifyTime(l.freshness, l.composite ? { ...l.time, observed_date: l.composite.newest_observed_date } : l.time, now);
}

/** The newest satellite chlorophyll view the dataset holds, in Explore's product order. */
export function newestSatellite(m: Manifest): LayerArtifact | null {
  return satelliteLatest(m, "olci300") ?? satelliteLatest(m, "viirs750") ?? satelliteLatest(m, "multi") ?? null;
}

/** NOAA C-HARM's newest run, with its freshness; `where` says how this page uses it ("In the 60-hour lanes"). */
export function modelEntry(m: Manifest, now: Date | null, where: string): SourceEntry | null {
  const run = charmRun(m);
  if (!run) return null;
  const f = fieldState(now ? forecastFreshness(m, now) : null, sourceStatus(m, "charm"));
  return { id: "model", label: "NOAA C-HARM forecast", state: f.state, detail: [`Run issued ${run.issued_date}`, f.note, where].filter(Boolean).join(". "), asOf: run.issued_date };
}

/** A satellite chlorophyll view (the one shown, else the newest), with its freshness. */
export function satelliteEntry(m: Manifest, now: Date | null, where: string, layer: LayerArtifact | null = null): SourceEntry | null {
  const l = layer ?? newestSatellite(m);
  if (!l) return null;
  const f = fieldState(layerFreshness(l, now), sourceStatus(m, "satellite_chl"));
  const date = l.composite?.newest_observed_date ?? l.time.observed_date ?? null;
  return { id: "satellite", label: "Satellite chlorophyll", state: f.state, detail: [`Newest view ${date ?? ""}`, f.note, where].filter(Boolean).join(". "), asOf: date };
}

/** A radar hour (the one shown, else the newest), with its freshness. */
export function currentsEntry(m: Manifest, now: Date | null, where: string, layer: LayerArtifact | null = null): SourceEntry | null {
  const l = layer ?? currentsHourly(m).at(-1) ?? null;
  if (!l) return null;
  const f = fieldState(layerFreshness(l, now), sourceStatus(m, "hf_radar"));
  return { id: "currents", label: "Surface currents (HF radar)", state: f.state, detail: ["Hourly, 2 km cells", f.note, where].filter(Boolean).join(". "), asOf: l.time.valid_time ?? null };
}
